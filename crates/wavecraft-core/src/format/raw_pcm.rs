use std::path::Path;

use crate::buffer::AudioBuffer;
use crate::error::{CoreError, Result};
use crate::format::{
    AudioFormat, AudioMetadata, FormatNode, FormatNodeKind, InspectedFile, RawPcmConfig,
    decode_interleaved_pcm, encode_pcm_sample,
};

const PREVIEW_FRAMES: usize = 8;

/// Parses a headerless raw PCM byte stream using `cfg`, building a frame-level [`FormatNode`] tree.
pub fn inspect_raw_pcm(bytes: Vec<u8>, cfg: RawPcmConfig) -> Result<InspectedFile> {
    if cfg.sample_rate == 0 {
        return Err(CoreError::InvalidSampleRate(0));
    }
    if cfg.num_channels == 0 {
        return Err(CoreError::InvalidChannelLayout(
            "raw PCM channel count must be at least 1".to_string(),
        ));
    }

    let total_len = bytes.len();
    let bytes_per_sample = (cfg.encoding.bits_per_sample() / 8) as usize;
    let frame_stride = bytes_per_sample * cfg.num_channels as usize;

    if !total_len.is_multiple_of(frame_stride) {
        return Err(CoreError::MalformedFile(format!(
            "raw PCM byte length {total_len} is not a multiple of frame stride {frame_stride} bytes"
        )));
    }

    let num_frames = total_len / frame_stride;
    let channels = decode_interleaved_pcm(
        &bytes,
        cfg.num_channels as usize,
        num_frames,
        cfg.encoding,
        cfg.big_endian,
    );
    let buffer = AudioBuffer::from_channels(cfg.sample_rate, channels, cfg.encoding)?;
    let duration_secs = buffer.duration_secs();
    let endian_label = if cfg.big_endian { "BE" } else { "LE" };

    let mut frame_children = Vec::new();
    let shown_frames = num_frames.min(PREVIEW_FRAMES);
    for frame in 0..shown_frames {
        let f_start = frame * frame_stride;
        let f_end = f_start + frame_stride;
        let mut ch_nodes = Vec::with_capacity(cfg.num_channels as usize);
        let mut vals = Vec::with_capacity(cfg.num_channels as usize);
        for ch_idx in 0..cfg.num_channels as usize {
            let s_start = f_start + ch_idx * bytes_per_sample;
            let s_end = s_start + bytes_per_sample;
            let val = buffer.channels[ch_idx][frame];
            vals.push(format!("{val:.4}"));
            ch_nodes.push(FormatNode {
                name: format!("ch{ch_idx}"),
                summary: format!("{val:.6}"),
                byte_range: s_start..s_end,
                kind: FormatNodeKind::Field,
                children: Vec::new(),
            });
        }
        frame_children.push(FormatNode {
            name: format!("Frame {frame}"),
            summary: format!("[{}]", vals.join(", ")),
            byte_range: f_start..f_end,
            kind: FormatNodeKind::Frame,
            children: ch_nodes,
        });
    }

    if num_frames > shown_frames {
        let rest_start = shown_frames * frame_stride;
        frame_children.push(FormatNode {
            name: format!("Frames {shown_frames}..{num_frames}"),
            summary: format!(
                "{} remaining frames ({} bytes)",
                num_frames - shown_frames,
                total_len - rest_start
            ),
            byte_range: rest_start..total_len,
            kind: FormatNodeKind::Payload,
            children: Vec::new(),
        });
    }

    let tree = FormatNode {
        name: "Raw PCM Stream".to_string(),
        summary: format!(
            "{} {endian_label}, {} Hz, {} ch, stride {frame_stride} B, {num_frames} frames",
            cfg.encoding.label(),
            cfg.sample_rate,
            cfg.num_channels
        ),
        byte_range: 0..total_len,
        kind: FormatNodeKind::Container,
        children: vec![
            FormatNode {
                name: "Stream Config".to_string(),
                summary: format!(
                    "Headerless: {} Hz, {} ch, {}, {endian_label}, {frame_stride} bytes/frame",
                    cfg.sample_rate,
                    cfg.num_channels,
                    cfg.encoding.label()
                ),
                byte_range: 0..0,
                kind: FormatNodeKind::Header,
                children: Vec::new(),
            },
            FormatNode {
                name: "Sample Payload".to_string(),
                summary: format!("{num_frames} frames ({total_len} bytes)"),
                byte_range: 0..total_len,
                kind: FormatNodeKind::Payload,
                children: frame_children,
            },
        ],
    };

    let metadata = AudioMetadata {
        format: AudioFormat::RawPcm,
        sample_rate: cfg.sample_rate,
        num_channels: cfg.num_channels,
        num_frames,
        bits_per_sample: Some(cfg.encoding.bits_per_sample()),
        encoding_label: format!("{} ({endian_label})", cfg.encoding.label()),
        file_size_bytes: total_len,
        duration_secs,
    };

    Ok(InspectedFile {
        format: AudioFormat::RawPcm,
        raw_bytes: bytes,
        tree,
        metadata,
        buffer,
    })
}

/// Encodes `buffer` as headerless interleaved raw PCM using `cfg`.
pub fn encode_raw_pcm(buffer: &AudioBuffer, cfg: RawPcmConfig) -> Result<Vec<u8>> {
    if buffer.num_channels() == 0 {
        return Err(CoreError::EmptyBuffer);
    }
    if cfg.num_channels as usize != buffer.num_channels() {
        return Err(CoreError::InvalidChannelLayout(format!(
            "raw PCM config specifies {} channels, but buffer has {}",
            cfg.num_channels,
            buffer.num_channels()
        )));
    }
    let bytes_per_sample = (cfg.encoding.bits_per_sample() / 8) as usize;
    let num_frames = buffer.num_frames();
    let mut out = Vec::with_capacity(num_frames * buffer.num_channels() * bytes_per_sample);
    for frame in 0..num_frames {
        for ch in &buffer.channels {
            encode_pcm_sample(ch[frame], cfg.encoding, cfg.big_endian, &mut out);
        }
    }
    Ok(out)
}

/// Encodes `buffer` as raw PCM using `cfg` and writes it to `path`.
pub fn write_raw_pcm_file(path: &Path, buffer: &AudioBuffer, cfg: RawPcmConfig) -> Result<()> {
    let bytes = encode_raw_pcm(buffer, cfg)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::SampleEncoding;

    #[test]
    fn raw_pcm_roundtrip_little_and_big_endian() {
        let encodings = [
            (SampleEncoding::PcmU8, 0.015_f32),
            (SampleEncoding::PcmI16, 1e-4_f32),
            (SampleEncoding::PcmI24, 1e-6_f32),
            (SampleEncoding::PcmI32, 1e-7_f32),
            (SampleEncoding::Float32, 1e-7_f32),
        ];

        let ch0 = vec![0.0, 0.4, -0.4, 0.85, -0.85, 0.1, -0.2, 0.3, -0.5, 0.6];
        let ch1 = vec![-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 0.0];

        for big_endian in [false, true] {
            for (encoding, tol) in encodings {
                let cfg = RawPcmConfig {
                    sample_rate: 44_100,
                    num_channels: 2,
                    encoding,
                    big_endian,
                };
                let buf = AudioBuffer::from_channels(
                    cfg.sample_rate,
                    vec![ch0.clone(), ch1.clone()],
                    encoding,
                )
                .unwrap();

                let raw = encode_raw_pcm(&buf, cfg).unwrap();
                let inspected = inspect_raw_pcm(raw.clone(), cfg).unwrap();

                assert_eq!(inspected.format, AudioFormat::RawPcm);
                assert!(inspected.tree.validate_ranges(raw.len()));
                assert_eq!(inspected.metadata.num_frames, 10);
                assert_eq!(inspected.metadata.num_channels, 2);

                // Payload node should have 8 preview frames + 1 remaining frames node = 9 children.
                let payload = &inspected.tree.children[1];
                assert_eq!(payload.children.len(), 9);
                assert_eq!(payload.children[0].name, "Frame 0");

                for ch_idx in 0..2 {
                    for (orig, dec) in buf.channels[ch_idx]
                        .iter()
                        .zip(&inspected.buffer.channels[ch_idx])
                    {
                        assert!((orig - dec).abs() <= tol);
                    }
                }
            }
        }
    }

    #[test]
    fn raw_pcm_rejects_odd_byte_length() {
        let cfg = RawPcmConfig {
            sample_rate: 44_100,
            num_channels: 1,
            encoding: SampleEncoding::PcmI16,
            big_endian: false,
        };
        let err = inspect_raw_pcm(vec![0x01, 0x02, 0x03], cfg);
        assert!(matches!(err, Err(CoreError::MalformedFile(_))));
    }

    #[test]
    fn encode_raw_pcm_rejects_channel_mismatch() {
        let buf = AudioBuffer::new_silent(44_100, 2, 16);
        let cfg = RawPcmConfig {
            sample_rate: 44_100,
            num_channels: 1,
            encoding: SampleEncoding::PcmI16,
            big_endian: false,
        };
        let err = encode_raw_pcm(&buf, cfg);
        assert!(matches!(err, Err(CoreError::InvalidChannelLayout(_))));
    }
}
