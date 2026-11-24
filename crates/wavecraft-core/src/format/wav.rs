use std::ops::Range;
use std::path::Path;

use crate::buffer::{AudioBuffer, SampleEncoding};
use crate::error::{CoreError, Result};
use crate::format::{
    AudioFormat, AudioMetadata, FormatNode, FormatNodeKind, InspectedFile, decode_interleaved_pcm,
    encode_pcm_sample,
};

const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

#[derive(Debug, Clone, Copy)]
struct WavFmt {
    audio_format: u16,
    effective_format: u16,
    num_channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn format_tag_label(tag: u16) -> &'static str {
    match tag {
        WAVE_FORMAT_PCM => "PCM",
        WAVE_FORMAT_IEEE_FLOAT => "IEEE Float",
        WAVE_FORMAT_EXTENSIBLE => "Extensible",
        _ => "Other",
    }
}

fn resolve_encoding(fmt: &WavFmt) -> Result<SampleEncoding> {
    match (fmt.effective_format, fmt.bits_per_sample) {
        (WAVE_FORMAT_PCM, 8) => Ok(SampleEncoding::PcmU8),
        (WAVE_FORMAT_PCM, 16) => Ok(SampleEncoding::PcmI16),
        (WAVE_FORMAT_PCM, 24) => Ok(SampleEncoding::PcmI24),
        (WAVE_FORMAT_PCM, 32) => Ok(SampleEncoding::PcmI32),
        (WAVE_FORMAT_IEEE_FLOAT, 32) => Ok(SampleEncoding::Float32),
        (tag, bps) => Err(CoreError::UnsupportedFormat(format!(
            "unsupported WAV format tag 0x{tag:04X} with {bps} bits/sample"
        ))),
    }
}

fn field_node(name: &str, summary: String, byte_range: Range<usize>) -> FormatNode {
    FormatNode {
        name: name.to_string(),
        summary,
        byte_range,
        kind: FormatNodeKind::Field,
        children: Vec::new(),
    }
}

fn parse_fmt_chunk(bytes: &[u8], payload: Range<usize>) -> Result<(WavFmt, Vec<FormatNode>)> {
    let len = payload.len();
    if len < 16 {
        return Err(CoreError::MalformedFile(format!(
            "WAV fmt chunk payload is {len} bytes (minimum 16)"
        )));
    }
    let p = payload.start;
    let audio_format = read_u16_le(bytes, p);
    let num_channels = read_u16_le(bytes, p + 2);
    let sample_rate = read_u32_le(bytes, p + 4);
    let byte_rate = read_u32_le(bytes, p + 8);
    let block_align = read_u16_le(bytes, p + 12);
    let bits_per_sample = read_u16_le(bytes, p + 14);

    let effective_format = if audio_format == WAVE_FORMAT_EXTENSIBLE && len >= 40 {
        read_u16_le(bytes, p + 24)
    } else {
        audio_format
    };

    let mut children = vec![
        field_node(
            "audio_format",
            format!("{audio_format} ({})", format_tag_label(audio_format)),
            p..p + 2,
        ),
        field_node("num_channels", format!("{num_channels}"), p + 2..p + 4),
        field_node("sample_rate", format!("{sample_rate} Hz"), p + 4..p + 8),
        field_node("byte_rate", format!("{byte_rate} B/s"), p + 8..p + 12),
        field_node(
            "block_align",
            format!("{block_align} bytes/frame"),
            p + 12..p + 14,
        ),
        field_node(
            "bits_per_sample",
            format!("{bits_per_sample} bits"),
            p + 14..p + 16,
        ),
    ];

    if len > 16 {
        children.push(FormatNode {
            name: "extra_format_bytes".to_string(),
            summary: format!("{} extra format bytes", len - 16),
            byte_range: p + 16..payload.end,
            kind: FormatNodeKind::Metadata,
            children: Vec::new(),
        });
    }

    Ok((
        WavFmt {
            audio_format,
            effective_format,
            num_channels,
            sample_rate,
            block_align,
            bits_per_sample,
        },
        children,
    ))
}

fn parse_list_chunk(bytes: &[u8], payload: Range<usize>) -> Vec<FormatNode> {
    if payload.len() < 4 {
        return vec![FormatNode {
            name: "list_payload".to_string(),
            summary: format!("{} bytes", payload.len()),
            byte_range: payload,
            kind: FormatNodeKind::Metadata,
            children: Vec::new(),
        }];
    }

    let list_type = String::from_utf8_lossy(&bytes[payload.start..payload.start + 4]).to_string();
    let mut children = vec![field_node(
        "list_type",
        list_type.clone(),
        payload.start..payload.start + 4,
    )];

    if list_type == "INFO" {
        let mut pos = payload.start + 4;
        while pos + 8 <= payload.end {
            let tag = String::from_utf8_lossy(&bytes[pos..pos + 4]).to_string();
            let size = read_u32_le(bytes, pos + 4) as usize;
            let val_start = pos + 8;
            let Some(val_end) = val_start.checked_add(size) else {
                break;
            };
            if val_end > payload.end {
                break;
            }
            let text = String::from_utf8_lossy(&bytes[val_start..val_end])
                .trim_end_matches('\0')
                .to_string();
            let next = val_end.saturating_add(size & 1).min(payload.end);
            children.push(FormatNode {
                name: format!("INFO:{tag}"),
                summary: text,
                byte_range: pos..next,
                kind: FormatNodeKind::Metadata,
                children: vec![
                    field_node("tag_id", tag, pos..pos + 4),
                    field_node("tag_size", format!("{size} bytes"), pos + 4..pos + 8),
                    FormatNode {
                        name: "value".to_string(),
                        summary: format!("{size} bytes"),
                        byte_range: val_start..val_end,
                        kind: FormatNodeKind::Metadata,
                        children: Vec::new(),
                    },
                ],
            });
            pos = next;
        }
    } else if payload.start + 4 < payload.end {
        children.push(FormatNode {
            name: "list_entries".to_string(),
            summary: format!("{} bytes", payload.end - (payload.start + 4)),
            byte_range: payload.start + 4..payload.end,
            kind: FormatNodeKind::Metadata,
            children: Vec::new(),
        });
    }

    children
}

/// Parses a RIFF/WAV byte stream into a byte-exact [`FormatNode`] tree and decoded [`AudioBuffer`].
pub fn inspect_wav(bytes: Vec<u8>) -> Result<InspectedFile> {
    let total_len = bytes.len();
    if total_len < 12 {
        return Err(CoreError::MalformedFile(format!(
            "WAV file too short: {total_len} bytes (minimum 12)"
        )));
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(CoreError::MalformedFile(
            "missing RIFF/WAVE header magic".to_string(),
        ));
    }

    let riff_size = read_u32_le(&bytes, 4);
    let mut root_children = vec![
        field_node("Chunk ID", "RIFF".to_string(), 0..4),
        field_node("Chunk Size", format!("{riff_size} bytes"), 4..8),
        field_node("Form Type", "WAVE".to_string(), 8..12),
    ];

    let mut fmt_opt: Option<WavFmt> = None;
    let mut data_range_opt: Option<Range<usize>> = None;
    let mut data_node_idx: Option<usize> = None;

    let mut offset = 12;
    while offset < total_len {
        if offset + 8 > total_len {
            return Err(CoreError::MalformedFile(format!(
                "truncated WAV subchunk header at byte {offset}"
            )));
        }

        let chunk_id = String::from_utf8_lossy(&bytes[offset..offset + 4]).to_string();
        let chunk_size = read_u32_le(&bytes, offset + 4) as usize;
        let payload_start = offset + 8;
        let payload_end = payload_start.checked_add(chunk_size).ok_or_else(|| {
            CoreError::MalformedFile(format!("WAV chunk '{chunk_id}' size overflow"))
        })?;
        if payload_end > total_len {
            return Err(CoreError::MalformedFile(format!(
                "truncated WAV chunk '{chunk_id}': needs {payload_end} bytes, file has {total_len}"
            )));
        }
        let chunk_end = payload_end.saturating_add(chunk_size & 1).min(total_len);

        let mut chunk_children = vec![
            field_node("id", format!("'{chunk_id}'"), offset..offset + 4),
            field_node(
                "size",
                format!("{chunk_size} bytes"),
                offset + 4..offset + 8,
            ),
        ];

        let (kind, summary) = match chunk_id.as_str() {
            "fmt " => {
                let (parsed_fmt, mut fmt_fields) =
                    parse_fmt_chunk(&bytes, payload_start..payload_end)?;
                let sum = format!(
                    "{}, {} ch, {} Hz, {}-bit",
                    format_tag_label(parsed_fmt.audio_format),
                    parsed_fmt.num_channels,
                    parsed_fmt.sample_rate,
                    parsed_fmt.bits_per_sample
                );
                fmt_opt = Some(parsed_fmt);
                chunk_children.append(&mut fmt_fields);
                (FormatNodeKind::Header, sum)
            }
            "data" => {
                data_range_opt = Some(payload_start..payload_end);
                data_node_idx = Some(root_children.len());
                chunk_children.push(FormatNode {
                    name: "Sample Data".to_string(),
                    summary: format!("{chunk_size} PCM payload bytes"),
                    byte_range: payload_start..payload_end,
                    kind: FormatNodeKind::Payload,
                    children: Vec::new(),
                });
                (
                    FormatNodeKind::Payload,
                    format!("{chunk_size} audio payload bytes"),
                )
            }
            "fact" => {
                let sum = if chunk_size >= 4 {
                    let sample_len = read_u32_le(&bytes, payload_start);
                    format!("sample_length = {sample_len}")
                } else {
                    format!("{chunk_size} bytes")
                };
                chunk_children.push(FormatNode {
                    name: "fact_payload".to_string(),
                    summary: sum.clone(),
                    byte_range: payload_start..payload_end,
                    kind: FormatNodeKind::Metadata,
                    children: Vec::new(),
                });
                (FormatNodeKind::Metadata, sum)
            }
            "LIST" => {
                let mut list_children = parse_list_chunk(&bytes, payload_start..payload_end);
                chunk_children.append(&mut list_children);
                (
                    FormatNodeKind::Metadata,
                    format!("LIST ({chunk_size} bytes)"),
                )
            }
            "cue " => {
                let sum = if chunk_size >= 4 {
                    let cue_points = read_u32_le(&bytes, payload_start);
                    format!("{cue_points} cue points")
                } else {
                    format!("{chunk_size} bytes")
                };
                chunk_children.push(FormatNode {
                    name: "cue_payload".to_string(),
                    summary: sum.clone(),
                    byte_range: payload_start..payload_end,
                    kind: FormatNodeKind::Metadata,
                    children: Vec::new(),
                });
                (FormatNodeKind::Metadata, sum)
            }
            _ => {
                chunk_children.push(FormatNode {
                    name: "chunk_payload".to_string(),
                    summary: format!("{chunk_size} bytes"),
                    byte_range: payload_start..payload_end,
                    kind: FormatNodeKind::Metadata,
                    children: Vec::new(),
                });
                (
                    FormatNodeKind::Metadata,
                    format!("Unknown chunk ({chunk_size} bytes)"),
                )
            }
        };

        root_children.push(FormatNode {
            name: format!("Chunk '{chunk_id}'"),
            summary,
            byte_range: offset..chunk_end,
            kind,
            children: chunk_children,
        });

        offset = chunk_end;
    }

    let fmt = fmt_opt.ok_or_else(|| CoreError::MalformedFile("missing WAV 'fmt ' chunk".into()))?;
    let data_range = data_range_opt
        .ok_or_else(|| CoreError::MalformedFile("missing WAV 'data' chunk".into()))?;

    if fmt.sample_rate == 0 {
        return Err(CoreError::InvalidSampleRate(0));
    }
    if fmt.num_channels == 0 {
        return Err(CoreError::InvalidChannelLayout(
            "WAV channel count must be at least 1".to_string(),
        ));
    }

    let encoding = resolve_encoding(&fmt)?;
    let bytes_per_sample = (encoding.bits_per_sample() / 8) as usize;
    let expected_block_align = bytes_per_sample * fmt.num_channels as usize;
    let frame_stride = if fmt.block_align == 0 {
        expected_block_align
    } else {
        if fmt.block_align as usize != expected_block_align {
            return Err(CoreError::MalformedFile(format!(
                "WAV block_align {} does not match channels*bytes_per_sample {}",
                fmt.block_align, expected_block_align
            )));
        }
        fmt.block_align as usize
    };

    let data_slice = &bytes[data_range.clone()];
    if !data_slice.len().is_multiple_of(frame_stride) {
        return Err(CoreError::MalformedFile(format!(
            "WAV data byte length {} is not divisible by frame stride {frame_stride}",
            data_slice.len()
        )));
    }
    let num_frames = data_slice.len() / frame_stride;

    if let Some(idx) = data_node_idx {
        let total_samples = num_frames * fmt.num_channels as usize;
        let summary = format!(
            "{} bytes ({num_frames} frames, {total_samples} samples)",
            data_slice.len()
        );
        root_children[idx].summary = summary.clone();
        if let Some(payload_node) = root_children[idx].children.last_mut() {
            payload_node.summary = summary;
        }
    }

    let channels = decode_interleaved_pcm(
        data_slice,
        fmt.num_channels as usize,
        num_frames,
        encoding,
        false,
    );
    let buffer = AudioBuffer::from_channels(fmt.sample_rate, channels, encoding)?;
    let duration_secs = buffer.duration_secs();

    let tree = FormatNode {
        name: "RIFF Container".to_string(),
        summary: format!(
            "WAVE ({total_len} bytes, {} Hz, {} ch, {})",
            fmt.sample_rate,
            fmt.num_channels,
            encoding.label()
        ),
        byte_range: 0..total_len,
        kind: FormatNodeKind::Container,
        children: root_children,
    };

    let metadata = AudioMetadata {
        format: AudioFormat::Wav,
        sample_rate: fmt.sample_rate,
        num_channels: fmt.num_channels,
        num_frames,
        bits_per_sample: Some(encoding.bits_per_sample()),
        encoding_label: encoding.label().to_string(),
        file_size_bytes: total_len,
        duration_secs,
    };

    Ok(InspectedFile {
        format: AudioFormat::Wav,
        raw_bytes: bytes,
        tree,
        metadata,
        buffer,
    })
}

/// Encodes `buffer` into canonical RIFF/WAV bytes using `encoding`.
pub fn encode_wav(buffer: &AudioBuffer, encoding: SampleEncoding) -> Result<Vec<u8>> {
    if buffer.sample_rate == 0 {
        return Err(CoreError::InvalidSampleRate(0));
    }
    let num_channels = buffer.num_channels();
    if num_channels == 0 {
        return Err(CoreError::EmptyBuffer);
    }
    let num_channels_u16 = u16::try_from(num_channels).map_err(|_| {
        CoreError::InvalidChannelLayout(format!("too many channels: {num_channels}"))
    })?;

    let num_frames = buffer.num_frames();
    let bits_per_sample = encoding.bits_per_sample();
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let block_align = num_channels_u16
        .checked_mul(bytes_per_sample as u16)
        .ok_or_else(|| CoreError::InvalidChannelLayout("block_align overflow".into()))?;
    let byte_rate = buffer
        .sample_rate
        .checked_mul(block_align as u32)
        .ok_or(CoreError::InvalidSampleRate(buffer.sample_rate))?;

    let data_len = num_frames
        .checked_mul(block_align as usize)
        .ok_or_else(|| CoreError::MalformedFile("WAV payload size overflow".into()))?;
    let data_len_u32 = u32::try_from(data_len)
        .map_err(|_| CoreError::MalformedFile("WAV payload exceeds 4 GiB limit".into()))?;
    let pad_byte = data_len & 1;

    let fmt_chunk_size: u32 = 16;
    let riff_payload_len = 4u32
        .checked_add(8 + fmt_chunk_size)
        .and_then(|v| v.checked_add(8 + data_len_u32 + pad_byte as u32))
        .ok_or_else(|| CoreError::MalformedFile("WAV RIFF size exceeds 4 GiB limit".into()))?;

    let format_tag = match encoding {
        SampleEncoding::Float32 => WAVE_FORMAT_IEEE_FLOAT,
        _ => WAVE_FORMAT_PCM,
    };

    let mut out = Vec::with_capacity(8 + riff_payload_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_payload_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt subchunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&fmt_chunk_size.to_le_bytes());
    out.extend_from_slice(&format_tag.to_le_bytes());
    out.extend_from_slice(&num_channels_u16.to_le_bytes());
    out.extend_from_slice(&buffer.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data subchunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len_u32.to_le_bytes());
    for frame in 0..num_frames {
        for ch in &buffer.channels {
            encode_pcm_sample(ch[frame], encoding, false, &mut out);
        }
    }
    if pad_byte == 1 {
        out.push(0);
    }

    Ok(out)
}

/// Encodes `buffer` as WAV and writes it to `path`.
pub fn write_wav_file(path: &Path, buffer: &AudioBuffer, encoding: SampleEncoding) -> Result<()> {
    let bytes = encode_wav(buffer, encoding)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_node<'a>(node: &'a FormatNode, name: &str) -> Option<&'a FormatNode> {
        if node.name == name {
            return Some(node);
        }
        for child in &node.children {
            if let Some(found) = find_node(child, name) {
                return Some(found);
            }
        }
        None
    }

    #[test]
    fn wav_roundtrip_all_five_encodings() {
        let cases = [
            (SampleEncoding::PcmU8, 0.015_f32),
            (SampleEncoding::PcmI16, 1e-4_f32),
            (SampleEncoding::PcmI24, 1e-6_f32),
            (SampleEncoding::PcmI32, 1e-7_f32),
            (SampleEncoding::Float32, 1e-7_f32),
        ];

        let left = vec![0.0, 0.5, -0.5, 0.99, -0.99, 0.25, -0.75];
        let right = vec![-0.25, 0.75, 0.0, -0.5, 0.5, -0.99, 0.99];

        for (encoding, tol) in cases {
            let buf =
                AudioBuffer::from_channels(48_000, vec![left.clone(), right.clone()], encoding)
                    .unwrap();
            let wav_bytes = encode_wav(&buf, encoding).unwrap();
            let inspected = inspect_wav(wav_bytes.clone()).unwrap();

            assert_eq!(inspected.format, AudioFormat::Wav);
            assert!(inspected.tree.validate_ranges(wav_bytes.len()));
            assert_eq!(inspected.metadata.sample_rate, 48_000);
            assert_eq!(inspected.metadata.num_channels, 2);
            assert_eq!(inspected.metadata.num_frames, 7);
            assert_eq!(
                inspected.metadata.bits_per_sample,
                Some(encoding.bits_per_sample())
            );
            assert_eq!(inspected.buffer.source_encoding, encoding);

            // Verify exact byte ranges inside fmt chunk (starts at byte 12: id 12..16, size 16..20, payload at 20)
            let audio_fmt_node = find_node(&inspected.tree, "audio_format").unwrap();
            assert_eq!(audio_fmt_node.byte_range, 20..22);
            let sr_node = find_node(&inspected.tree, "sample_rate").unwrap();
            assert_eq!(sr_node.byte_range, 24..28);
            let bps_node = find_node(&inspected.tree, "bits_per_sample").unwrap();
            assert_eq!(bps_node.byte_range, 34..36);

            for ch_idx in 0..2 {
                for (i, (&orig, &dec)) in buf.channels[ch_idx]
                    .iter()
                    .zip(&inspected.buffer.channels[ch_idx])
                    .enumerate()
                {
                    assert!(
                        (orig - dec).abs() <= tol,
                        "encoding {encoding:?} ch {ch_idx} frame {i}: orig={orig}, dec={dec}, tol={tol}"
                    );
                }
            }
        }
    }

    #[test]
    fn wav_parses_list_info_and_fact_chunks() {
        let buf = AudioBuffer::from_channels(44_100, vec![vec![0.1, -0.2]], SampleEncoding::PcmI16)
            .unwrap();
        let mut wav = encode_wav(&buf, SampleEncoding::PcmI16).unwrap();

        // Append a 'fact' chunk and a 'LIST' INFO chunk, updating RIFF chunk size.
        wav.extend_from_slice(b"fact");
        wav.extend_from_slice(&4u32.to_le_bytes());
        wav.extend_from_slice(&2u32.to_le_bytes());

        let title = b"TestTone\0"; // 9 bytes -> padded to 10
        let list_payload_len = 4 + 8 + title.len() + 1;
        wav.extend_from_slice(b"LIST");
        wav.extend_from_slice(&(list_payload_len as u32).to_le_bytes());
        wav.extend_from_slice(b"INFO");
        wav.extend_from_slice(b"INAM");
        wav.extend_from_slice(&(title.len() as u32).to_le_bytes());
        wav.extend_from_slice(title);
        wav.push(0); // pad byte for odd INAM size

        let riff_size = (wav.len() - 8) as u32;
        wav[4..8].copy_from_slice(&riff_size.to_le_bytes());

        let inspected = inspect_wav(wav.clone()).unwrap();
        assert!(inspected.tree.validate_ranges(wav.len()));
        assert!(find_node(&inspected.tree, "Chunk 'fact'").is_some());
        let info_title = find_node(&inspected.tree, "INFO:INAM").unwrap();
        assert_eq!(info_title.summary, "TestTone");
    }

    #[test]
    fn wav_malformed_and_truncated_errors() {
        let too_short = inspect_wav(vec![b'R', b'I', b'F', b'F']);
        assert!(matches!(too_short, Err(CoreError::MalformedFile(_))));

        let bad_magic = inspect_wav(b"NOTARIFFWAVE".to_vec());
        assert!(matches!(bad_magic, Err(CoreError::MalformedFile(_))));

        let buf = AudioBuffer::from_channels(44_100, vec![vec![0.5; 16]], SampleEncoding::PcmI16)
            .unwrap();
        let valid = encode_wav(&buf, SampleEncoding::PcmI16).unwrap();
        let truncated = valid[..valid.len() - 5].to_vec();
        let err = inspect_wav(truncated);
        assert!(matches!(err, Err(CoreError::MalformedFile(_))));
    }
}
