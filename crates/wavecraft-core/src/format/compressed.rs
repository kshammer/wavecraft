use std::io::Cursor;
use std::ops::Range;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::buffer::{AudioBuffer, SampleEncoding};
use crate::error::{CoreError, Result};
use crate::format::{AudioFormat, AudioMetadata, FormatNode, FormatNodeKind, InspectedFile};

const MAX_DETAILED_FRAMES: usize = 64;
const MAX_DETAILED_PAGES: usize = 64;

fn field_node(
    name: impl Into<String>,
    summary: impl Into<String>,
    range: Range<usize>,
) -> FormatNode {
    FormatNode {
        name: name.into(),
        summary: summary.into(),
        byte_range: range,
        kind: FormatNodeKind::Field,
        children: Vec::new(),
    }
}

fn flac_block_name(block_type: u8) -> &'static str {
    match block_type {
        0 => "STREAMINFO",
        1 => "PADDING",
        2 => "APPLICATION",
        3 => "SEEKTABLE",
        4 => "VORBIS_COMMENT",
        5 => "CUESHEET",
        6 => "PICTURE",
        _ => "METADATA_BLOCK",
    }
}

fn format_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

/// Parses a FLAC stream's magic header, metadata blocks (`STREAMINFO`, etc.), and audio frame region.
pub fn parse_flac_tree(bytes: &[u8]) -> FormatNode {
    let total_len = bytes.len();
    let mut children = Vec::new();

    if total_len < 4 || &bytes[0..4] != b"fLaC" {
        children.push(FormatNode {
            name: "Invalid Magic".to_string(),
            summary: format!("expected 'fLaC', got {} bytes", total_len.min(4)),
            byte_range: 0..total_len.min(4),
            kind: FormatNodeKind::Header,
            children: Vec::new(),
        });
        return FormatNode {
            name: "FLAC Stream".to_string(),
            summary: format!("Malformed FLAC ({total_len} bytes)"),
            byte_range: 0..total_len,
            kind: FormatNodeKind::Container,
            children,
        };
    }

    children.push(FormatNode {
        name: "Magic".to_string(),
        summary: "fLaC".to_string(),
        byte_range: 0..4,
        kind: FormatNodeKind::Header,
        children: Vec::new(),
    });

    let mut pos = 4;
    let mut stream_summary = format!("FLAC ({total_len} bytes)");

    while pos + 4 <= total_len {
        let header_byte = bytes[pos];
        let is_last = (header_byte & 0x80) != 0;
        let block_type = header_byte & 0x7F;
        let block_len = ((bytes[pos + 1] as usize) << 16)
            | ((bytes[pos + 2] as usize) << 8)
            | (bytes[pos + 3] as usize);
        let payload_start = pos + 4;
        let payload_end = payload_start.saturating_add(block_len).min(total_len);
        let truncated = payload_start.saturating_add(block_len) > total_len;
        let block_name = flac_block_name(block_type);

        let mut block_children = vec![field_node(
            "block_header",
            format!("type={block_type} ({block_name}), last={is_last}, len={block_len}"),
            pos..pos + 4,
        )];

        let mut summary = if truncated {
            format!("{block_len} bytes (truncated)")
        } else {
            format!("{block_len} bytes")
        };

        if block_type == 0 && payload_end.saturating_sub(payload_start) >= 34 {
            let p = &bytes[payload_start..payload_start + 34];
            let min_block = u16::from_be_bytes([p[0], p[1]]);
            let max_block = u16::from_be_bytes([p[2], p[3]]);
            let min_frame = ((p[4] as u32) << 16) | ((p[5] as u32) << 8) | (p[6] as u32);
            let max_frame = ((p[7] as u32) << 16) | ((p[8] as u32) << 8) | (p[9] as u32);
            let packed =
                u64::from_be_bytes([p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17]]);
            let sample_rate = (packed >> 44) as u32;
            let num_channels = (((packed >> 41) & 0x07) + 1) as u16;
            let bits_per_sample = (((packed >> 36) & 0x1F) + 1) as u16;
            let total_samples = packed & 0x000F_FFFF_FFFF;
            let md5_hex = format_hex(&p[18..34]);

            summary = format!(
                "{sample_rate} Hz, {num_channels} ch, {bits_per_sample}-bit, {total_samples} samples"
            );
            stream_summary = format!(
                "FLAC ({sample_rate} Hz, {num_channels} ch, {bits_per_sample}-bit, {total_len} bytes)"
            );

            block_children.push(field_node(
                "min_block_size",
                format!("{min_block} samples"),
                payload_start..payload_start + 2,
            ));
            block_children.push(field_node(
                "max_block_size",
                format!("{max_block} samples"),
                payload_start + 2..payload_start + 4,
            ));
            block_children.push(field_node(
                "min_frame_size",
                format!("{min_frame} bytes"),
                payload_start + 4..payload_start + 7,
            ));
            block_children.push(field_node(
                "max_frame_size",
                format!("{max_frame} bytes"),
                payload_start + 7..payload_start + 10,
            ));
            block_children.push(field_node(
                "stream_params",
                format!("{sample_rate} Hz, {num_channels} ch, {bits_per_sample} bps, {total_samples} total samples"),
                payload_start + 10..payload_start + 18,
            ));
            block_children.push(field_node(
                "md5",
                md5_hex,
                payload_start + 18..payload_start + 34,
            ));
        }

        children.push(FormatNode {
            name: format!("Metadata: {block_name}"),
            summary,
            byte_range: pos..payload_end,
            kind: FormatNodeKind::Metadata,
            children: block_children,
        });

        pos = payload_end;
        if is_last || truncated {
            break;
        }
    }

    if pos < total_len {
        let sync_offset = (pos..total_len.saturating_sub(1))
            .find(|&i| bytes[i] == 0xFF && (bytes[i + 1] & 0xFE) == 0xF8);
        let frame_start = sync_offset.unwrap_or(pos);
        if frame_start > pos {
            children.push(FormatNode {
                name: "Pre-frame Gap".to_string(),
                summary: format!("{} bytes before audio sync", frame_start - pos),
                byte_range: pos..frame_start,
                kind: FormatNodeKind::Metadata,
                children: Vec::new(),
            });
        }
        let sync_note = match sync_offset {
            Some(off) => format!(
                "{} bytes (sync 0x{:02X}{:02X} at {off})",
                total_len - frame_start,
                bytes[off],
                bytes[off + 1]
            ),
            None => format!(
                "{} bytes (no 0xFFF8/0xFFF9 sync found)",
                total_len - frame_start
            ),
        };
        children.push(FormatNode {
            name: "Audio Frames".to_string(),
            summary: sync_note,
            byte_range: frame_start..total_len,
            kind: FormatNodeKind::Payload,
            children: Vec::new(),
        });
    }

    FormatNode {
        name: "FLAC Stream".to_string(),
        summary: stream_summary,
        byte_range: 0..total_len,
        kind: FormatNodeKind::Container,
        children,
    }
}

#[derive(Debug, Clone, Copy)]
struct MpegHeaderInfo {
    version_label: &'static str,
    layer_label: &'static str,
    bitrate_kbps: u32,
    sample_rate_hz: u32,
    padding: usize,
    channel_mode: &'static str,
    frame_size: usize,
}

fn decode_mpeg_header(hdr: &[u8]) -> Option<MpegHeaderInfo> {
    if hdr.len() < 4 || hdr[0] != 0xFF || (hdr[1] & 0xE0) != 0xE0 {
        return None;
    }

    let version_bits = (hdr[1] >> 3) & 0x03;
    let layer_bits = (hdr[1] >> 1) & 0x03;
    let bitrate_idx = ((hdr[2] >> 4) & 0x0F) as usize;
    let sr_idx = ((hdr[2] >> 2) & 0x03) as usize;
    let padding = ((hdr[2] >> 1) & 0x01) as usize;
    let ch_mode_bits = (hdr[3] >> 6) & 0x03;

    if version_bits == 1 || layer_bits == 0 || bitrate_idx == 0 || bitrate_idx == 15 || sr_idx == 3
    {
        return None;
    }

    let version_label = match version_bits {
        0 => "MPEG 2.5",
        2 => "MPEG 2",
        3 => "MPEG 1",
        _ => return None,
    };

    let layer_label = match layer_bits {
        1 => "Layer III",
        2 => "Layer II",
        3 => "Layer I",
        _ => return None,
    };

    let sample_rate_hz = match (version_bits, sr_idx) {
        (3, 0) => 44_100,
        (3, 1) => 48_000,
        (3, 2) => 32_000,
        (2, 0) => 22_050,
        (2, 1) => 24_000,
        (2, 2) => 16_000,
        (0, 0) => 11_025,
        (0, 1) => 12_000,
        (0, 2) => 8_000,
        _ => return None,
    };

    const V1_L1: [u32; 15] = [
        0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
    ];
    const V1_L2: [u32; 15] = [
        0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
    ];
    const V1_L3: [u32; 15] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ];
    const V2_L1: [u32; 15] = [
        0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256,
    ];
    const V2_L23: [u32; 15] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];

    let bitrate_kbps = match (version_bits, layer_bits) {
        (3, 3) => V1_L1[bitrate_idx],
        (3, 2) => V1_L2[bitrate_idx],
        (3, 1) => V1_L3[bitrate_idx],
        (0 | 2, 3) => V2_L1[bitrate_idx],
        (0 | 2, 1 | 2) => V2_L23[bitrate_idx],
        _ => return None,
    };
    if bitrate_kbps == 0 {
        return None;
    }

    let bitrate_bps = bitrate_kbps as usize * 1000;
    let sr = sample_rate_hz as usize;
    let frame_size = match (version_bits, layer_bits) {
        (_, 3) => (12 * bitrate_bps / sr + padding) * 4,
        (3, _) => 144 * bitrate_bps / sr + padding,
        (_, 1) => 72 * bitrate_bps / sr + padding,
        (_, 2) => 144 * bitrate_bps / sr + padding,
        _ => return None,
    };
    if frame_size < 4 {
        return None;
    }

    let channel_mode = match ch_mode_bits {
        0 => "Stereo",
        1 => "Joint Stereo",
        2 => "Dual Channel",
        _ => "Mono",
    };

    Some(MpegHeaderInfo {
        version_label,
        layer_label,
        bitrate_kbps,
        sample_rate_hz,
        padding,
        channel_mode,
        frame_size,
    })
}

/// Returns `true` if `hdr` begins with a valid 4-byte MPEG audio frame sync header.
pub(crate) fn is_valid_mpeg_header(hdr: &[u8]) -> bool {
    decode_mpeg_header(hdr).is_some()
}

fn read_synchsafe_u32(b: &[u8]) -> usize {
    (((b[0] & 0x7F) as usize) << 21)
        | (((b[1] & 0x7F) as usize) << 14)
        | (((b[2] & 0x7F) as usize) << 7)
        | ((b[3] & 0x7F) as usize)
}

fn parse_id3v2_frames(bytes: &[u8], start: usize, end: usize, ver_major: u8) -> Vec<FormatNode> {
    let mut frames = Vec::new();
    let mut pos = start;
    while pos + 10 <= end && frames.len() < MAX_DETAILED_FRAMES {
        if bytes[pos] == 0 {
            break;
        }
        let id_bytes = &bytes[pos..pos + 4];
        if !id_bytes
            .iter()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        {
            break;
        }
        let frame_id = String::from_utf8_lossy(id_bytes).to_string();
        let frame_size = if ver_major >= 4 {
            read_synchsafe_u32(&bytes[pos + 4..pos + 8])
        } else {
            u32::from_be_bytes([
                bytes[pos + 4],
                bytes[pos + 5],
                bytes[pos + 6],
                bytes[pos + 7],
            ]) as usize
        };
        let payload_start = pos + 10;
        let payload_end = payload_start.saturating_add(frame_size).min(end);
        let text_preview = if frame_id.starts_with('T') && payload_end > payload_start + 1 {
            let raw = String::from_utf8_lossy(&bytes[payload_start + 1..payload_end])
                .trim_matches('\0')
                .to_string();
            format!("'{raw}' ({frame_size} B)")
        } else {
            format!("{frame_size} bytes")
        };

        frames.push(FormatNode {
            name: format!("ID3 Frame {frame_id}"),
            summary: text_preview,
            byte_range: pos..payload_end,
            kind: FormatNodeKind::Metadata,
            children: Vec::new(),
        });

        if payload_end <= pos {
            break;
        }
        pos = payload_end;
    }
    frames
}

/// Parses an MP3 stream's ID3v2 header, MPEG audio frame headers, and trailing ID3v1 tag.
pub fn parse_mp3_tree(bytes: &[u8]) -> FormatNode {
    let total_len = bytes.len();
    let mut children = Vec::new();
    let mut audio_start = 0usize;

    if total_len >= 10 && &bytes[0..3] == b"ID3" {
        let ver_major = bytes[3];
        let ver_minor = bytes[4];
        let flags = bytes[5];
        let tag_size = read_synchsafe_u32(&bytes[6..10]);
        let tag_end = 10usize.saturating_add(tag_size).min(total_len);
        let mut id3_children = vec![
            field_node("version", format!("ID3v2.{ver_major}.{ver_minor}"), 3..5),
            field_node("flags", format!("0x{flags:02X}"), 5..6),
            field_node("size", format!("{tag_size} bytes"), 6..10),
        ];
        if ver_major >= 3 {
            id3_children.extend(parse_id3v2_frames(bytes, 10, tag_end, ver_major));
        }
        children.push(FormatNode {
            name: "ID3v2 Header".to_string(),
            summary: format!("ID3v2.{ver_major}.{ver_minor} ({tag_end} bytes)"),
            byte_range: 0..tag_end,
            kind: FormatNodeKind::Metadata,
            children: id3_children,
        });
        audio_start = tag_end;
    }

    let mut id3v1_node: Option<FormatNode> = None;
    let mut audio_end = total_len;
    if total_len >= 128 && total_len - 128 >= audio_start {
        let tag_start = total_len - 128;
        if &bytes[tag_start..tag_start + 3] == b"TAG" {
            audio_end = tag_start;
            let title = String::from_utf8_lossy(&bytes[tag_start + 3..tag_start + 33])
                .trim_end_matches(['\0', ' '])
                .to_string();
            let artist = String::from_utf8_lossy(&bytes[tag_start + 33..tag_start + 63])
                .trim_end_matches(['\0', ' '])
                .to_string();
            id3v1_node = Some(FormatNode {
                name: "ID3v1 Tag".to_string(),
                summary: format!("title='{title}', artist='{artist}'"),
                byte_range: tag_start..total_len,
                kind: FormatNodeKind::Metadata,
                children: Vec::new(),
            });
        }
    }

    let mut frame_count = 0usize;
    let mut first_info: Option<MpegHeaderInfo> = None;
    let mut pos = audio_start;

    while pos + 4 <= audio_end {
        if let Some(info) = decode_mpeg_header(&bytes[pos..audio_end]) {
            if first_info.is_none() {
                first_info = Some(info);
            }
            let frame_end = pos.saturating_add(info.frame_size).min(audio_end);
            if frame_count < MAX_DETAILED_FRAMES {
                children.push(FormatNode {
                    name: format!("MPEG Frame {frame_count}"),
                    summary: format!(
                        "{} {}, {} kbps, {} Hz, {}, pad={}, {} B",
                        info.version_label,
                        info.layer_label,
                        info.bitrate_kbps,
                        info.sample_rate_hz,
                        info.channel_mode,
                        info.padding,
                        frame_end - pos
                    ),
                    byte_range: pos..frame_end,
                    kind: FormatNodeKind::Frame,
                    children: Vec::new(),
                });
                frame_count += 1;
                pos = frame_end;
                if frame_count == MAX_DETAILED_FRAMES && pos < audio_end {
                    let rest_start = pos;
                    let mut remaining_frames = 0usize;
                    let mut scan = rest_start;
                    while scan + 4 <= audio_end {
                        if let Some(rem) = decode_mpeg_header(&bytes[scan..audio_end]) {
                            remaining_frames += 1;
                            scan = scan.saturating_add(rem.frame_size).min(audio_end);
                        } else {
                            scan += 1;
                        }
                    }
                    children.push(FormatNode {
                        name: "Remaining MPEG Frames".to_string(),
                        summary: format!(
                            "~{remaining_frames} more frames ({} bytes)",
                            audio_end - rest_start
                        ),
                        byte_range: rest_start..audio_end,
                        kind: FormatNodeKind::Payload,
                        children: Vec::new(),
                    });
                    break;
                }
            } else {
                break;
            }
        } else {
            pos += 1;
        }
    }

    if let Some(tag) = id3v1_node {
        children.push(tag);
    }

    let root_summary = match first_info {
        Some(info) => format!(
            "{} {}, {} kbps, {} Hz, {} ({total_len} bytes)",
            info.version_label,
            info.layer_label,
            info.bitrate_kbps,
            info.sample_rate_hz,
            info.channel_mode
        ),
        None => format!("MP3 Stream ({total_len} bytes, no valid MPEG sync found)"),
    };

    FormatNode {
        name: "MP3 Stream".to_string(),
        summary: root_summary,
        byte_range: 0..total_len,
        kind: FormatNodeKind::Container,
        children,
    }
}

fn ogg_flag_summary(flags: u8) -> String {
    let mut parts = Vec::new();
    if flags & 0x01 != 0 {
        parts.push("CONT");
    }
    if flags & 0x02 != 0 {
        parts.push("BOS");
    }
    if flags & 0x04 != 0 {
        parts.push("EOS");
    }
    if parts.is_empty() {
        "NONE".to_string()
    } else {
        parts.join("|")
    }
}

fn inspect_vorbis_packets(
    payload: &[u8],
    base_offset: usize,
    children: &mut Vec<FormatNode>,
) -> Option<String> {
    let mut summary = None;
    if let Some(id_pos) = payload.windows(7).position(|w| w == b"\x01vorbis") {
        let abs = base_offset + id_pos;
        if id_pos + 30 <= payload.len() {
            let p = &payload[id_pos..];
            let version = u32::from_le_bytes([p[7], p[8], p[9], p[10]]);
            let channels = p[11];
            let sample_rate = u32::from_le_bytes([p[12], p[13], p[14], p[15]]);
            let bitrate_nom = i32::from_le_bytes([p[20], p[21], p[22], p[23]]);
            let desc = format!(
                "v{version}, {channels} ch, {sample_rate} Hz, nominal {} kbps",
                bitrate_nom / 1000
            );
            summary = Some(desc.clone());
            children.push(FormatNode {
                name: "Vorbis Identification".to_string(),
                summary: desc,
                byte_range: abs..abs + 30,
                kind: FormatNodeKind::Metadata,
                children: vec![
                    field_node("channels", format!("{channels}"), abs + 11..abs + 12),
                    field_node(
                        "sample_rate",
                        format!("{sample_rate} Hz"),
                        abs + 12..abs + 16,
                    ),
                    field_node(
                        "bitrate_nominal",
                        format!("{bitrate_nom} bps"),
                        abs + 20..abs + 24,
                    ),
                ],
            });
        }
    }

    if let Some(cmt_pos) = payload.windows(7).position(|w| w == b"\x03vorbis") {
        let abs = base_offset + cmt_pos;
        let mut cmt_summary = "Vorbis comment header".to_string();
        if cmt_pos + 11 <= payload.len() {
            let vendor_len = u32::from_le_bytes([
                payload[cmt_pos + 7],
                payload[cmt_pos + 8],
                payload[cmt_pos + 9],
                payload[cmt_pos + 10],
            ]) as usize;
            let vendor_end = (cmt_pos + 11).saturating_add(vendor_len).min(payload.len());
            let vendor = String::from_utf8_lossy(&payload[cmt_pos + 11..vendor_end]).to_string();
            if !vendor.is_empty() {
                cmt_summary = format!("vendor='{vendor}'");
            }
        }
        let end = (abs + 7).min(base_offset + payload.len());
        children.push(FormatNode {
            name: "Vorbis Comment".to_string(),
            summary: cmt_summary,
            byte_range: abs..end,
            kind: FormatNodeKind::Metadata,
            children: Vec::new(),
        });
    }

    summary
}

/// Parses an OGG page stream (`OggS` headers, segment tables, and early Vorbis packet headers).
pub fn parse_ogg_tree(bytes: &[u8]) -> FormatNode {
    let total_len = bytes.len();
    let mut children = Vec::new();
    let mut pos = 0usize;
    let mut page_count = 0usize;
    let mut vorbis_desc: Option<String> = None;

    while pos + 27 <= total_len {
        if &bytes[pos..pos + 4] != b"OggS" {
            if let Some(rel) = bytes[pos..].windows(4).position(|w| w == b"OggS") {
                pos += rel;
                if pos + 27 > total_len {
                    break;
                }
            } else {
                break;
            }
        }

        let version = bytes[pos + 4];
        let header_type = bytes[pos + 5];
        let granule = u64::from_le_bytes([
            bytes[pos + 6],
            bytes[pos + 7],
            bytes[pos + 8],
            bytes[pos + 9],
            bytes[pos + 10],
            bytes[pos + 11],
            bytes[pos + 12],
            bytes[pos + 13],
        ]);
        let serial = u32::from_le_bytes([
            bytes[pos + 14],
            bytes[pos + 15],
            bytes[pos + 16],
            bytes[pos + 17],
        ]);
        let page_seq = u32::from_le_bytes([
            bytes[pos + 18],
            bytes[pos + 19],
            bytes[pos + 20],
            bytes[pos + 21],
        ]);
        let checksum = u32::from_le_bytes([
            bytes[pos + 22],
            bytes[pos + 23],
            bytes[pos + 24],
            bytes[pos + 25],
        ]);
        let num_segments = bytes[pos + 26] as usize;

        let seg_start = pos + 27;
        let seg_end = seg_start.saturating_add(num_segments).min(total_len);
        let payload_len: usize = bytes[seg_start..seg_end].iter().map(|&b| b as usize).sum();
        let payload_start = seg_end;
        let payload_end = payload_start.saturating_add(payload_len).min(total_len);

        if page_count < MAX_DETAILED_PAGES {
            let mut page_children = vec![
                field_node("capture_pattern", "OggS".to_string(), pos..pos + 4),
                field_node("version", format!("{version}"), pos + 4..pos + 5),
                field_node(
                    "header_type",
                    ogg_flag_summary(header_type),
                    pos + 5..pos + 6,
                ),
                field_node("granule_position", format!("{granule}"), pos + 6..pos + 14),
                field_node(
                    "serial_number",
                    format!("0x{serial:08X}"),
                    pos + 14..pos + 18,
                ),
                field_node("page_sequence", format!("{page_seq}"), pos + 18..pos + 22),
                field_node("checksum", format!("0x{checksum:08X}"), pos + 22..pos + 26),
                field_node(
                    "segment_table",
                    format!("{num_segments} segments ({payload_len} B payload)"),
                    pos + 26..seg_end,
                ),
            ];

            if payload_start < payload_end {
                if page_seq <= 2
                    && let Some(desc) = inspect_vorbis_packets(
                        &bytes[payload_start..payload_end],
                        payload_start,
                        &mut page_children,
                    )
                    && vorbis_desc.is_none()
                {
                    vorbis_desc = Some(desc);
                }
                page_children.push(FormatNode {
                    name: "Page Payload".to_string(),
                    summary: format!("{} bytes", payload_end - payload_start),
                    byte_range: payload_start..payload_end,
                    kind: FormatNodeKind::Payload,
                    children: Vec::new(),
                });
            }

            children.push(FormatNode {
                name: format!("OGG Page {page_seq}"),
                summary: format!(
                    "flags={}, granule={granule}, {} B",
                    ogg_flag_summary(header_type),
                    payload_end - pos
                ),
                byte_range: pos..payload_end,
                kind: FormatNodeKind::Frame,
                children: page_children,
            });
            page_count += 1;
            pos = payload_end;

            if page_count == MAX_DETAILED_PAGES && pos < total_len {
                let remaining_pages = bytes[pos..].windows(4).filter(|w| *w == b"OggS").count();
                children.push(FormatNode {
                    name: "Remaining OGG Pages".to_string(),
                    summary: format!("{remaining_pages} more pages ({} bytes)", total_len - pos),
                    byte_range: pos..total_len,
                    kind: FormatNodeKind::Payload,
                    children: Vec::new(),
                });
                break;
            }
        } else {
            break;
        }
    }

    if children.is_empty() && total_len > 0 {
        children.push(FormatNode {
            name: "Unrecognized Data".to_string(),
            summary: format!("{total_len} bytes (no OggS pages found)"),
            byte_range: 0..total_len,
            kind: FormatNodeKind::Metadata,
            children: Vec::new(),
        });
    }

    let summary = match vorbis_desc {
        Some(desc) => format!("OGG Vorbis ({desc}, {total_len} bytes)"),
        None => format!("OGG ({page_count} pages inspected, {total_len} bytes)"),
    };

    FormatNode {
        name: "OGG Container".to_string(),
        summary,
        byte_range: 0..total_len,
        kind: FormatNodeKind::Container,
        children,
    }
}

fn decode_with_symphonia(
    bytes: Vec<u8>,
    format: AudioFormat,
) -> Result<(AudioBuffer, Option<u16>)> {
    let mut hint = Hint::new();
    hint.with_extension(format.extension());

    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| {
            CoreError::MalformedFile(format!("failed to probe {}: {e}", format.label()))
        })?;

    let mut reader = probed.format;
    let track = reader
        .default_track()
        .or_else(|| {
            reader
                .tracks()
                .iter()
                .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        })
        .ok_or_else(|| {
            CoreError::MalformedFile(format!(
                "no decodable audio track in {} stream",
                format.label()
            ))
        })?
        .clone();

    let track_id = track.id;
    let codec_sample_rate = track.codec_params.sample_rate;
    let codec_channels = track.codec_params.channels.map(|c| c.count());
    let bits_per_sample = track.codec_params.bits_per_sample.map(|b| b as u16);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| CoreError::DecodeError(format!("codec init failed: {e}")))?;

    let mut channels: Vec<Vec<f32>> = Vec::new();
    let mut detected_sr = codec_sample_rate;

    loop {
        let packet = match reader.next_packet() {
            Ok(pkt) => pkt,
            Err(SymphoniaError::IoError(ref err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => {
                return Err(CoreError::DecodeError(format!("packet read failed: {e}")));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                detected_sr.get_or_insert(spec.rate);
                let ch_count = spec.channels.count();
                if ch_count == 0 {
                    continue;
                }
                if channels.is_empty() {
                    channels = vec![Vec::new(); ch_count];
                }
                let frames_in_pkt = decoded.frames();
                if frames_in_pkt == 0 {
                    continue;
                }
                let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sample_buf.copy_interleaved_ref(decoded);
                let interleaved = sample_buf.samples();
                for frame in interleaved.chunks_exact(ch_count) {
                    for (ch_idx, &sample) in frame.iter().enumerate() {
                        if let Some(ch_vec) = channels.get_mut(ch_idx) {
                            let clean = if sample.is_nan() {
                                0.0
                            } else {
                                sample.clamp(-1.0, 1.0)
                            };
                            ch_vec.push(clean);
                        }
                    }
                }
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => {
                return Err(CoreError::DecodeError(format!("audio decode failed: {e}")));
            }
        }
    }

    let sample_rate = detected_sr.ok_or_else(|| {
        CoreError::MalformedFile(format!("missing sample rate in {} stream", format.label()))
    })?;

    if channels.is_empty() {
        let ch_count = codec_channels.ok_or_else(|| {
            CoreError::MalformedFile(format!(
                "missing channel count in {} stream",
                format.label()
            ))
        })?;
        channels = vec![Vec::new(); ch_count];
    }

    let source_encoding = match (format, bits_per_sample) {
        (AudioFormat::Flac, Some(8)) => SampleEncoding::PcmU8,
        (AudioFormat::Flac, Some(16)) => SampleEncoding::PcmI16,
        (AudioFormat::Flac, Some(24)) => SampleEncoding::PcmI24,
        (AudioFormat::Flac, Some(32)) => SampleEncoding::PcmI32,
        (AudioFormat::Flac, _) => SampleEncoding::PcmI16,
        _ => SampleEncoding::Float32,
    };

    let buffer = AudioBuffer::from_channels(sample_rate, channels, source_encoding)?;
    Ok((buffer, bits_per_sample))
}

/// Inspects a compressed audio stream (`Flac`, `Mp3`, or `Ogg`), parsing its structural tree
/// and decoding PCM samples via `symphonia`.
pub fn inspect_compressed(bytes: Vec<u8>, format: AudioFormat) -> Result<InspectedFile> {
    if !matches!(
        format,
        AudioFormat::Flac | AudioFormat::Mp3 | AudioFormat::Ogg
    ) {
        return Err(CoreError::UnsupportedFormat(format!(
            "{} is not a compressed format",
            format.label()
        )));
    }

    let tree = match format {
        AudioFormat::Flac => parse_flac_tree(&bytes),
        AudioFormat::Mp3 => parse_mp3_tree(&bytes),
        AudioFormat::Ogg => parse_ogg_tree(&bytes),
        _ => unreachable!(),
    };
    let file_size_bytes = bytes.len();
    let (buffer, bits_per_sample) = decode_with_symphonia(bytes.clone(), format)?;

    let encoding_label = match format {
        AudioFormat::Flac => match bits_per_sample {
            Some(bps) => format!("FLAC ({bps}-bit lossless)"),
            None => "FLAC (lossless)".to_string(),
        },
        AudioFormat::Mp3 => "MPEG Layer III (MP3)".to_string(),
        AudioFormat::Ogg => "Ogg Vorbis".to_string(),
        _ => unreachable!(),
    };

    let metadata = AudioMetadata {
        format,
        sample_rate: buffer.sample_rate,
        num_channels: buffer.num_channels() as u16,
        num_frames: buffer.num_frames(),
        bits_per_sample,
        encoding_label,
        file_size_bytes,
        duration_secs: buffer.duration_secs(),
    };

    Ok(InspectedFile {
        format,
        raw_bytes: bytes,
        tree,
        metadata,
        buffer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_flac_tree_valid_streaminfo_and_sync() {
        let mut flac = Vec::new();
        flac.extend_from_slice(b"fLaC");
        // Last metadata block (0x80 | 0 = 0x80), length = 34
        flac.push(0x80);
        flac.extend_from_slice(&[0x00, 0x00, 34]);
        // STREAMINFO: min_block=4096, max_block=4096, min_frame=16, max_frame=8192
        flac.extend_from_slice(&4096u16.to_be_bytes());
        flac.extend_from_slice(&4096u16.to_be_bytes());
        flac.extend_from_slice(&[0x00, 0x00, 16]);
        flac.extend_from_slice(&[0x00, 0x20, 0x00]);
        // Packed 64 bits: sr=44100 (20b), ch-1=1 (3b -> 2ch), bps-1=15 (5b -> 16-bit), total=44100 (36b)
        let packed: u64 = (44_100u64 << 44) | (1u64 << 41) | (15u64 << 36) | 44_100u64;
        flac.extend_from_slice(&packed.to_be_bytes());
        flac.extend_from_slice(&[0xAB; 16]); // MD5
        // Audio frame sync
        flac.extend_from_slice(&[0xFF, 0xF8, 0x69, 0x10]);

        let tree = parse_flac_tree(&flac);
        assert!(tree.validate_ranges(flac.len()));
        assert!(tree.summary.contains("44100 Hz"));
        assert!(tree.summary.contains("2 ch"));
        assert!(tree.summary.contains("16-bit"));

        let streaminfo = &tree.children[1];
        assert_eq!(streaminfo.name, "Metadata: STREAMINFO");
        assert!(streaminfo.summary.contains("44100 samples"));
        let frames_node = tree.children.last().unwrap();
        assert_eq!(frames_node.name, "Audio Frames");
        assert!(frames_node.summary.contains("sync 0xFFF8"));
    }

    #[test]
    fn parse_mp3_tree_id3v2_mpeg_frame_and_id3v1() {
        let mut mp3 = Vec::new();
        // ID3v2.3 header with one TIT2 frame
        let tit2_payload = b"\x00Hello";
        let frame_total = 10 + tit2_payload.len();
        mp3.extend_from_slice(b"ID3");
        mp3.extend_from_slice(&[3, 0, 0]); // v2.3.0
        mp3.extend_from_slice(&[0, 0, 0, frame_total as u8]); // synchsafe size
        mp3.extend_from_slice(b"TIT2");
        mp3.extend_from_slice(&(tit2_payload.len() as u32).to_be_bytes());
        mp3.extend_from_slice(&[0, 0]);
        mp3.extend_from_slice(tit2_payload);

        // MPEG 1 Layer III, 128 kbps (idx 9), 44100 Hz (idx 0), padding=0, stereo (0)
        // Frame size = 144 * 128000 / 44100 = 417 bytes
        let frame_start = mp3.len();
        mp3.extend_from_slice(&[0xFF, 0xFB, 0x90, 0x00]);
        mp3.resize(frame_start + 417, 0x00);

        // Trailing 128-byte ID3v1 tag
        let id3v1_start = mp3.len();
        mp3.resize(id3v1_start + 128, 0x00);
        mp3[id3v1_start..id3v1_start + 3].copy_from_slice(b"TAG");
        mp3[id3v1_start + 3..id3v1_start + 7].copy_from_slice(b"Song");
        mp3[id3v1_start + 33..id3v1_start + 37].copy_from_slice(b"Band");

        let tree = parse_mp3_tree(&mp3);
        assert!(tree.validate_ranges(mp3.len()));
        assert!(tree.summary.contains("MPEG 1 Layer III"));
        assert!(tree.summary.contains("128 kbps"));
        assert!(tree.summary.contains("44100 Hz"));

        let id3v2 = &tree.children[0];
        assert_eq!(id3v2.name, "ID3v2 Header");
        assert!(
            id3v2
                .children
                .iter()
                .any(|c| c.name == "ID3 Frame TIT2" && c.summary.contains("Hello"))
        );

        let frame0 = &tree.children[1];
        assert_eq!(frame0.name, "MPEG Frame 0");
        assert_eq!(frame0.byte_range, frame_start..frame_start + 417);

        let id3v1 = tree.children.last().unwrap();
        assert_eq!(id3v1.name, "ID3v1 Tag");
        assert!(id3v1.summary.contains("Song"));
        assert!(id3v1.summary.contains("Band"));
    }

    #[test]
    fn parse_ogg_tree_vorbis_ident_packet() {
        let mut ogg = Vec::new();
        // OggS page header (BOS) with 1 segment of 30 bytes (\x01vorbis ident packet)
        ogg.extend_from_slice(b"OggS");
        ogg.push(0); // version
        ogg.push(0x02); // BOS
        ogg.extend_from_slice(&0u64.to_le_bytes()); // granule
        ogg.extend_from_slice(&0x1234_5678u32.to_le_bytes()); // serial
        ogg.extend_from_slice(&0u32.to_le_bytes()); // seq 0
        ogg.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes()); // crc
        ogg.push(1); // page_segments = 1
        ogg.push(30); // segment length = 30

        // \x01vorbis identification packet (30 bytes)
        ogg.extend_from_slice(b"\x01vorbis");
        ogg.extend_from_slice(&0u32.to_le_bytes()); // vorbis version 0
        ogg.push(2); // channels = 2
        ogg.extend_from_slice(&48_000u32.to_le_bytes()); // sample rate
        ogg.extend_from_slice(&0i32.to_le_bytes()); // bitrate max
        ogg.extend_from_slice(&128_000i32.to_le_bytes()); // bitrate nominal
        ogg.extend_from_slice(&0i32.to_le_bytes()); // bitrate min
        ogg.push(0xB8); // blocksize
        ogg.push(0x01); // framing flag

        let tree = parse_ogg_tree(&ogg);
        assert!(tree.validate_ranges(ogg.len()));
        assert!(tree.summary.contains("2 ch"));
        assert!(tree.summary.contains("48000 Hz"));
        assert!(tree.summary.contains("128 kbps"));
        let page0 = &tree.children[0];
        assert_eq!(page0.name, "OGG Page 0");
        assert!(
            page0
                .children
                .iter()
                .any(|c| c.name == "Vorbis Identification")
        );
    }

    #[test]
    fn compressed_parsers_never_panic_on_truncated_or_garbage_bytes() {
        let garbage_inputs: [&[u8]; 6] = [
            &[],
            &[0xFF],
            b"fLaC\x80\xFF\xFF\xFF",
            b"ID3\x04\x00\x00\x7F\x7F\x7F\x7F\xFF\xFB",
            b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\xFF\xFF",
            &[0xAA; 64],
        ];

        for input in garbage_inputs {
            let flac_tree = parse_flac_tree(input);
            assert!(flac_tree.validate_ranges(input.len()));

            let mp3_tree = parse_mp3_tree(input);
            assert!(mp3_tree.validate_ranges(input.len()));

            let ogg_tree = parse_ogg_tree(input);
            assert!(ogg_tree.validate_ranges(input.len()));
        }
    }

    /// Valid 16-bit mono 44100 Hz 64-frame 440 Hz sine FLAC generated via `ffmpeg`
    /// with padding stripped and STREAMINFO marked as the last metadata block.
    const VALID_MINI_FLAC: &[u8] = &[
        0x66, 0x4C, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x12, 0x00, 0x12, 0x00, 0x00, 0x00, 0x55,
        0x00, 0x00, 0x55, 0x0A, 0xC4, 0x40, 0xF0, 0x00, 0x00, 0x00, 0x40, 0x3B, 0xAE, 0x50, 0xDB,
        0xDE, 0xCE, 0xA9, 0x75, 0x9C, 0x86, 0x6B, 0x14, 0xA0, 0xB0, 0xA0, 0x93, 0xFF, 0xF8, 0x69,
        0x08, 0x00, 0x3F, 0xA0, 0x4E, 0x00, 0x00, 0x01, 0x00, 0x01, 0xFF, 0x02, 0xFD, 0x03, 0xF8,
        0x04, 0xEE, 0x05, 0xE0, 0x06, 0xCC, 0xE6, 0x2C, 0x9F, 0x96, 0xE4, 0xAC, 0x22, 0xF0, 0x49,
        0x93, 0x1D, 0xEF, 0x21, 0xFA, 0xFF, 0x1C, 0x00, 0x8F, 0x12, 0x20, 0x40, 0x81, 0x02, 0x04,
        0x1C, 0x24, 0xF3, 0x44, 0x16, 0x20, 0xA3, 0x4A, 0x2C, 0x92, 0x48, 0x7B, 0xDC, 0xA7, 0x23,
        0xAA, 0xDB, 0x30, 0xCE, 0x6F, 0xDF, 0xFD, 0x3D, 0xBB, 0xB7, 0xD1, 0x6D, 0x96, 0xD9, 0x75,
        0xD6, 0xDF, 0x6D, 0xF6, 0xC0, 0x63, 0x81,
    ];

    #[test]
    fn inspect_compressed_and_inspect_bytes_decode_valid_mini_flac() {
        let inspected = inspect_compressed(VALID_MINI_FLAC.to_vec(), AudioFormat::Flac).unwrap();
        assert_eq!(inspected.format, AudioFormat::Flac);
        assert_eq!(inspected.metadata.sample_rate, 44_100);
        assert_eq!(inspected.metadata.num_channels, 1);
        assert_eq!(inspected.buffer.num_frames(), 64);
        assert!(
            inspected.buffer.channels[0].iter().any(|&s| s.abs() > 1e-4),
            "decoded FLAC sine should contain non-zero PCM samples"
        );
        assert!(inspected.tree.validate_ranges(inspected.raw_bytes.len()));

        let via_bytes = crate::format::inspect_bytes(VALID_MINI_FLAC.to_vec(), None, None).unwrap();
        assert_eq!(via_bytes.format, AudioFormat::Flac);
        assert_eq!(via_bytes.metadata.sample_rate, 44_100);
        assert_eq!(via_bytes.metadata.num_channels, 1);
        assert_eq!(via_bytes.buffer.num_frames(), 64);
        assert!(via_bytes.tree.validate_ranges(via_bytes.raw_bytes.len()));
    }

    #[test]
    fn inspect_compressed_returns_err_on_truncated_and_garbage_inputs() {
        let invalid_inputs: [&[u8]; 4] = [
            &[],
            b"fLaC\x80\x00\x00\x22\x00\x00",
            b"ID3\x03\x00\x00\x00\x00\x00\x00",
            &[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04],
        ];

        for format in [AudioFormat::Flac, AudioFormat::Mp3, AudioFormat::Ogg] {
            for input in invalid_inputs {
                let res = inspect_compressed(input.to_vec(), format);
                assert!(
                    res.is_err(),
                    "expected Err for {format:?} on {} invalid bytes",
                    input.len()
                );
            }
        }
    }
}
