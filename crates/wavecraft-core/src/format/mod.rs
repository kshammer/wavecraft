pub mod compressed;
pub mod raw_pcm;
pub mod wav;

use std::fmt::Write as _;
use std::ops::Range;
use std::path::Path;

use crate::buffer::{AudioBuffer, PeakBucket, SampleEncoding};
use crate::error::{CoreError, Result};

/// Supported audio container and stream formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioFormat {
    Wav,
    RawPcm,
    Flac,
    Mp3,
    Ogg,
}

impl AudioFormat {
    /// Human-readable short format label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Wav => "WAV",
            Self::RawPcm => "Raw PCM",
            Self::Flac => "FLAC",
            Self::Mp3 => "MP3",
            Self::Ogg => "OGG",
        }
    }

    /// Canonical lowercase file extension without a leading dot.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::RawPcm => "pcm",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::Ogg => "ogg",
        }
    }

    /// Returns `true` if `wavecraft-core` supports encoding and writing this format.
    pub fn is_writable(&self) -> bool {
        matches!(self, Self::Wav | Self::RawPcm)
    }
}

/// Semantic category of a node inside a parsed binary format tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatNodeKind {
    Container,
    Header,
    Field,
    Frame,
    Payload,
    Metadata,
}

impl FormatNodeKind {
    fn tag(self) -> &'static str {
        match self {
            Self::Container => "CONTAINER",
            Self::Header => "HEADER",
            Self::Field => "FIELD",
            Self::Frame => "FRAME",
            Self::Payload => "PAYLOAD",
            Self::Metadata => "META",
        }
    }
}

/// Tree node representing a byte-exact structure within an inspected audio file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatNode {
    pub name: String,
    pub summary: String,
    pub byte_range: Range<usize>,
    pub kind: FormatNodeKind,
    pub children: Vec<FormatNode>,
}

impl FormatNode {
    /// Recursively verifies that `byte_range.start <= byte_range.end <= total_len`
    /// for this node and every descendant.
    pub fn validate_ranges(&self, total_len: usize) -> bool {
        self.byte_range.start <= self.byte_range.end
            && self.byte_range.end <= total_len
            && self
                .children
                .iter()
                .all(|child| child.validate_ranges(total_len))
    }

    /// Formats the format tree as an indented human-readable CLI tree.
    pub fn format_tree_text(&self) -> String {
        let mut out = String::new();
        self.write_tree_lines(&mut out, "", true, true);
        out
    }

    fn write_tree_lines(&self, out: &mut String, prefix: &str, is_last: bool, is_root: bool) {
        let connector = if is_root {
            ""
        } else if is_last {
            "└─ "
        } else {
            "├─ "
        };
        let _ = writeln!(
            out,
            "{prefix}{connector}[{}..{}] [{}] {}: {}",
            self.byte_range.start,
            self.byte_range.end,
            self.kind.tag(),
            self.name,
            self.summary
        );
        let child_prefix = if is_root {
            String::new()
        } else if is_last {
            format!("{prefix}   ")
        } else {
            format!("{prefix}│  ")
        };
        let count = self.children.len();
        for (idx, child) in self.children.iter().enumerate() {
            child.write_tree_lines(out, &child_prefix, idx + 1 == count, false);
        }
    }
}

/// High-level technical summary extracted from an audio file.
#[derive(Debug, Clone)]
pub struct AudioMetadata {
    pub format: AudioFormat,
    pub sample_rate: u32,
    pub num_channels: u16,
    pub num_frames: usize,
    pub bits_per_sample: Option<u16>,
    pub encoding_label: String,
    pub file_size_bytes: usize,
    pub duration_secs: f64,
}

/// Complete inspection result containing raw bytes, byte-mapped structure tree, metadata, and decoded PCM.
#[derive(Debug, Clone)]
pub struct InspectedFile {
    pub format: AudioFormat,
    pub raw_bytes: Vec<u8>,
    pub tree: FormatNode,
    pub metadata: AudioMetadata,
    pub buffer: AudioBuffer,
}

/// Stream layout configuration required to inspect or write headerless raw PCM files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawPcmConfig {
    pub sample_rate: u32,
    pub num_channels: u16,
    pub encoding: SampleEncoding,
    pub big_endian: bool,
}

impl Default for RawPcmConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            num_channels: 1,
            encoding: SampleEncoding::PcmI16,
            big_endian: false,
        }
    }
}

/// Detects the audio format from magic bytes first, falling back to file extension.
pub fn detect_format(bytes: &[u8], path: Option<&Path>) -> Option<AudioFormat> {
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return Some(AudioFormat::Wav);
    }
    if bytes.len() >= 4 && &bytes[0..4] == b"fLaC" {
        return Some(AudioFormat::Flac);
    }
    if bytes.len() >= 4 && &bytes[0..4] == b"OggS" {
        return Some(AudioFormat::Ogg);
    }
    if bytes.len() >= 3 && &bytes[0..3] == b"ID3" {
        return Some(AudioFormat::Mp3);
    }
    if bytes.len() >= 4 && compressed::is_valid_mpeg_header(&bytes[0..4]) {
        return Some(AudioFormat::Mp3);
    }

    let ext = path
        .and_then(|p| p.extension())
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)?;

    match ext.as_str() {
        "wav" => Some(AudioFormat::Wav),
        "flac" => Some(AudioFormat::Flac),
        "mp3" => Some(AudioFormat::Mp3),
        "ogg" | "oga" => Some(AudioFormat::Ogg),
        "pcm" | "raw" => Some(AudioFormat::RawPcm),
        _ => None,
    }
}

/// Inspects in-memory bytes, parsing their byte-exact structure and decoding PCM samples.
pub fn inspect_bytes(
    bytes: Vec<u8>,
    path: Option<&Path>,
    raw_cfg: Option<RawPcmConfig>,
) -> Result<InspectedFile> {
    let format = detect_format(&bytes, path).ok_or_else(|| {
        let hint = path
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<bytes>".to_string());
        CoreError::UnsupportedFormat(format!("unable to detect audio format for {hint}"))
    })?;

    match format {
        AudioFormat::Wav => wav::inspect_wav(bytes),
        AudioFormat::RawPcm => raw_pcm::inspect_raw_pcm(bytes, raw_cfg.unwrap_or_default()),
        AudioFormat::Flac | AudioFormat::Mp3 | AudioFormat::Ogg => {
            compressed::inspect_compressed(bytes, format)
        }
    }
}

/// Reads a file from disk and inspects its binary structure and decoded audio.
pub fn inspect_file(path: &Path, raw_cfg: Option<RawPcmConfig>) -> Result<InspectedFile> {
    let bytes = std::fs::read(path)?;
    inspect_bytes(bytes, Some(path), raw_cfg)
}

/// Lightweight explorer preview helper returning file metadata and overview waveform peak buckets.
pub fn probe_preview(path: &Path, num_buckets: usize) -> Result<(AudioMetadata, Vec<PeakBucket>)> {
    let inspected = inspect_file(path, None)?;
    let peaks = if num_buckets == 0 || inspected.buffer.is_empty() {
        Vec::new()
    } else {
        inspected
            .buffer
            .compute_peaks(0, 0..inspected.buffer.num_frames(), num_buckets)
    };
    Ok((inspected.metadata, peaks))
}

/// Decodes a single PCM sample from `src` according to `encoding` and `big_endian`.
pub(crate) fn decode_pcm_sample(src: &[u8], encoding: SampleEncoding, big_endian: bool) -> f32 {
    match encoding {
        SampleEncoding::PcmU8 => (src[0] as f32 - 128.0) / 128.0,
        SampleEncoding::PcmI16 => {
            let raw = if big_endian {
                i16::from_be_bytes([src[0], src[1]])
            } else {
                i16::from_le_bytes([src[0], src[1]])
            };
            raw as f32 / 32768.0
        }
        SampleEncoding::PcmI24 => {
            let b = if big_endian {
                [src[2], src[1], src[0]]
            } else {
                [src[0], src[1], src[2]]
            };
            let raw = (((b[0] as i32) | ((b[1] as i32) << 8) | ((b[2] as i32) << 16)) << 8) >> 8;
            raw as f32 / 8_388_608.0
        }
        SampleEncoding::PcmI32 => {
            let raw = if big_endian {
                i32::from_be_bytes([src[0], src[1], src[2], src[3]])
            } else {
                i32::from_le_bytes([src[0], src[1], src[2], src[3]])
            };
            (raw as f64 / 2_147_483_648.0) as f32
        }
        SampleEncoding::Float32 => {
            let val = if big_endian {
                f32::from_be_bytes([src[0], src[1], src[2], src[3]])
            } else {
                f32::from_le_bytes([src[0], src[1], src[2], src[3]])
            };
            if val.is_nan() {
                0.0
            } else {
                val.clamp(-1.0, 1.0)
            }
        }
    }
}

/// Encodes a normalized `[-1.0, 1.0]` `f32` sample into `dst` according to `encoding` and `big_endian`.
pub(crate) fn encode_pcm_sample(
    sample: f32,
    encoding: SampleEncoding,
    big_endian: bool,
    dst: &mut Vec<u8>,
) {
    let clamped = if sample.is_nan() {
        0.0
    } else {
        sample.clamp(-1.0, 1.0)
    };
    match encoding {
        SampleEncoding::PcmU8 => {
            let v = (clamped * 128.0 + 128.0).round().clamp(0.0, 255.0) as u8;
            dst.push(v);
        }
        SampleEncoding::PcmI16 => {
            let scaled = if clamped < 0.0 {
                (clamped * 32768.0).round()
            } else {
                (clamped * 32767.0).round()
            }
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            let bytes = if big_endian {
                scaled.to_be_bytes()
            } else {
                scaled.to_le_bytes()
            };
            dst.extend_from_slice(&bytes);
        }
        SampleEncoding::PcmI24 => {
            let scaled = if clamped < 0.0 {
                (clamped as f64 * 8_388_608.0).round()
            } else {
                (clamped as f64 * 8_388_607.0).round()
            }
            .clamp(-8_388_608.0, 8_388_607.0) as i32;
            let le = scaled.to_le_bytes();
            if big_endian {
                dst.extend_from_slice(&[le[2], le[1], le[0]]);
            } else {
                dst.extend_from_slice(&[le[0], le[1], le[2]]);
            }
        }
        SampleEncoding::PcmI32 => {
            let scaled = if clamped < 0.0 {
                (clamped as f64 * 2_147_483_648.0).round()
            } else {
                (clamped as f64 * 2_147_483_647.0).round()
            }
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
            let bytes = if big_endian {
                scaled.to_be_bytes()
            } else {
                scaled.to_le_bytes()
            };
            dst.extend_from_slice(&bytes);
        }
        SampleEncoding::Float32 => {
            let bytes = if big_endian {
                clamped.to_be_bytes()
            } else {
                clamped.to_le_bytes()
            };
            dst.extend_from_slice(&bytes);
        }
    }
}

/// Decodes interleaved PCM bytes into channel vectors.
pub(crate) fn decode_interleaved_pcm(
    data: &[u8],
    num_channels: usize,
    num_frames: usize,
    encoding: SampleEncoding,
    big_endian: bool,
) -> Vec<Vec<f32>> {
    let bps = (encoding.bits_per_sample() / 8) as usize;
    let mut channels = vec![Vec::with_capacity(num_frames); num_channels];
    let mut pos = 0;
    for _ in 0..num_frames {
        for ch in &mut channels {
            ch.push(decode_pcm_sample(
                &data[pos..pos + bps],
                encoding,
                big_endian,
            ));
            pos += bps;
        }
    }
    channels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_format_properties_and_detection() {
        assert_eq!(AudioFormat::Wav.label(), "WAV");
        assert_eq!(AudioFormat::Wav.extension(), "wav");
        assert!(AudioFormat::Wav.is_writable());
        assert!(AudioFormat::RawPcm.is_writable());
        assert!(!AudioFormat::Flac.is_writable());
        assert!(!AudioFormat::Mp3.is_writable());
        assert!(!AudioFormat::Ogg.is_writable());

        assert_eq!(
            detect_format(b"RIFF\x00\x00\x00\x00WAVE", None),
            Some(AudioFormat::Wav)
        );
        assert_eq!(
            detect_format(b"fLaC\x80\x00\x00\x22", None),
            Some(AudioFormat::Flac)
        );
        assert_eq!(detect_format(b"OggS\x00\x02", None), Some(AudioFormat::Ogg));
        assert_eq!(detect_format(b"ID3\x03\x00", None), Some(AudioFormat::Mp3));
        assert_eq!(
            detect_format(&[0xFF, 0xFB, 0x90, 0x00], None),
            Some(AudioFormat::Mp3)
        );

        assert_eq!(
            detect_format(&[], Some(Path::new("a.WAV"))),
            Some(AudioFormat::Wav)
        );
        assert_eq!(
            detect_format(&[], Some(Path::new("a.oga"))),
            Some(AudioFormat::Ogg)
        );
        assert_eq!(
            detect_format(&[], Some(Path::new("a.raw"))),
            Some(AudioFormat::RawPcm)
        );
        assert_eq!(
            detect_format(&[], Some(Path::new("a.pcm"))),
            Some(AudioFormat::RawPcm)
        );
        assert_eq!(detect_format(&[], None), None);
    }

    #[test]
    fn format_node_tree_text_and_probe_preview() {
        let dir = std::env::temp_dir().join(format!("wavecraft_probe_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav_path = dir.join("probe.wav");
        let buf = AudioBuffer::from_channels(
            44_100,
            vec![vec![0.0, 0.5, -0.5, 0.25, -0.25, 0.8]],
            SampleEncoding::PcmI16,
        )
        .unwrap();
        wav::write_wav_file(&wav_path, &buf, SampleEncoding::PcmI16).unwrap();

        let inspected = inspect_file(&wav_path, None).unwrap();
        let text = inspected.tree.format_tree_text();
        assert!(text.contains("RIFF Container"));
        assert!(text.contains("audio_format"));

        let (meta, peaks) = probe_preview(&wav_path, 3).unwrap();
        assert_eq!(meta.format, AudioFormat::Wav);
        assert_eq!(meta.num_frames, 6);
        assert_eq!(peaks.len(), 3);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
