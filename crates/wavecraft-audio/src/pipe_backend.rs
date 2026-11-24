use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use thiserror::Error;

/// Errors returned by audio output backends.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PlaybackError {
    #[error("no supported system audio output backend available")]
    NoBackendAvailable,

    #[error("failed to spawn audio output backend: {0}")]
    SpawnFailed(String),

    #[error("failed to write audio samples to backend: {0}")]
    WriteFailed(String),

    #[error("invalid audio stream configuration: {0}")]
    InvalidConfig(String),
}

/// Sink interface for streaming interleaved `f32` audio samples to an output device or test double.
pub trait AudioBackend: Send + 'static {
    fn open(&mut self, sample_rate: u32, channels: u16) -> Result<(), PlaybackError>;
    fn write_interleaved_f32(&mut self, samples: &[f32]) -> Result<(), PlaybackError>;
    fn stop(&mut self);
    fn name(&self) -> &'static str;
}

/// Supported external CLI audio pipe tools on Linux.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipeTool {
    PwPlay,
    PaPlay,
    APlay,
}

impl PipeTool {
    /// Binary executable name searched on `PATH`.
    pub fn binary_name(self) -> &'static str {
        match self {
            Self::PwPlay => "pw-play",
            Self::PaPlay => "paplay",
            Self::APlay => "aplay",
        }
    }

    /// Human-readable backend display name.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::PwPlay => "PipeWire (pw-play)",
            Self::PaPlay => "PulseAudio (paplay)",
            Self::APlay => "ALSA (aplay)",
        }
    }

    /// CLI arguments for streaming raw audio over stdin.
    pub fn args(self, sample_rate: u32, channels: u16) -> Vec<String> {
        let sr = sample_rate.to_string();
        let ch = channels.to_string();
        match self {
            Self::PwPlay => vec![
                "--rate".to_string(),
                sr,
                "--channels".to_string(),
                ch,
                "--format".to_string(),
                "f32".to_string(),
                "--latency".to_string(),
                "20ms".to_string(),
                "-".to_string(),
            ],
            Self::PaPlay => vec![
                "--raw".to_string(),
                format!("--rate={sr}"),
                format!("--channels={ch}"),
                "--format=float32le".to_string(),
            ],
            Self::APlay => vec![
                "-q".to_string(),
                "-t".to_string(),
                "raw".to_string(),
                "-f".to_string(),
                "S16_LE".to_string(),
                "-r".to_string(),
                sr,
                "-c".to_string(),
                ch,
                "-".to_string(),
            ],
        }
    }
}

/// Converts normalized `f32` samples in `[-1.0, 1.0]` to little-endian signed 16-bit PCM bytes.
pub fn f32_to_s16_le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        bytes.extend_from_slice(&f32_to_s16_sample(sample).to_le_bytes());
    }
    bytes
}

fn f32_to_s16_sample(sample: f32) -> i16 {
    if sample.is_nan() {
        return 0;
    }
    let clamped = sample.clamp(-1.0, 1.0);
    let scaled = if clamped >= 0.0 {
        clamped * f32::from(i16::MAX)
    } else {
        clamped * (f32::from(i16::MAX) + 1.0)
    };
    scaled
        .round()
        .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn find_in_path(binary: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(binary);
        if is_executable_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// Audio backend streaming raw PCM into `pw-play`, `paplay`, or `aplay` via piped stdin.
pub struct SystemPipeBackend {
    tool: PipeTool,
    binary_path: PathBuf,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    scratch_bytes: Vec<u8>,
}

impl SystemPipeBackend {
    /// Detects the first available audio pipe binary (`pw-play`, `paplay`, `aplay`) on `PATH`.
    pub fn detect() -> Option<Self> {
        const PRIORITY: [PipeTool; 3] = [PipeTool::PwPlay, PipeTool::PaPlay, PipeTool::APlay];
        for tool in PRIORITY {
            if let Some(binary_path) = find_in_path(tool.binary_name()) {
                return Some(Self {
                    tool,
                    binary_path,
                    child: None,
                    stdin: None,
                    scratch_bytes: Vec::new(),
                });
            }
        }
        None
    }

    /// Returns the selected system pipe tool.
    pub fn tool(&self) -> PipeTool {
        self.tool
    }
}

impl AudioBackend for SystemPipeBackend {
    fn open(&mut self, sample_rate: u32, channels: u16) -> Result<(), PlaybackError> {
        if sample_rate == 0 || channels == 0 {
            return Err(PlaybackError::InvalidConfig(format!(
                "sample_rate ({sample_rate}) and channels ({channels}) must be non-zero"
            )));
        }

        // Ensure any prior stream process is terminated and reaped before opening a new one.
        self.stop();

        let args = self.tool.args(sample_rate, channels);
        let mut child = Command::new(&self.binary_path)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| {
                PlaybackError::SpawnFailed(format!(
                    "failed to launch {}: {e}",
                    self.binary_path.display()
                ))
            })?;

        let Some(stdin) = child.stdin.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(PlaybackError::SpawnFailed(format!(
                "{} did not expose piped stdin",
                self.tool.binary_name()
            )));
        };

        self.child = Some(child);
        self.stdin = Some(stdin);
        Ok(())
    }

    fn write_interleaved_f32(&mut self, samples: &[f32]) -> Result<(), PlaybackError> {
        if samples.is_empty() {
            return Ok(());
        }
        let Some(stdin) = self.stdin.as_mut() else {
            return Err(PlaybackError::WriteFailed(
                "audio stream is not open".to_string(),
            ));
        };

        self.scratch_bytes.clear();
        match self.tool {
            PipeTool::PwPlay | PipeTool::PaPlay => {
                self.scratch_bytes.reserve(samples.len() * 4);
                for &sample in samples {
                    let clean = if sample.is_nan() { 0.0 } else { sample };
                    self.scratch_bytes.extend_from_slice(&clean.to_le_bytes());
                }
            }
            PipeTool::APlay => {
                self.scratch_bytes.reserve(samples.len() * 2);
                for &sample in samples {
                    self.scratch_bytes
                        .extend_from_slice(&f32_to_s16_sample(sample).to_le_bytes());
                }
            }
        }

        stdin
            .write_all(&self.scratch_bytes)
            .map_err(|e| PlaybackError::WriteFailed(e.to_string()))
    }

    fn stop(&mut self) {
        // Drop stdin first so the child sees EOF, then kill/reap to avoid leaking zombies.
        drop(self.stdin.take());
        if let Some(mut child) = self.child.take() {
            match child.try_wait() {
                Ok(Some(_)) => {}
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        self.tool.display_name()
    }
}

impl Drop for SystemPipeBackend {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Recorded state and audio chunks captured by [`NullAudioBackend`].
#[derive(Debug, Default, Clone, PartialEq)]
pub struct NullBackendLog {
    pub chunks: Vec<Vec<f32>>,
    pub open_count: usize,
    pub stop_count: usize,
    pub last_config: Option<(u32, u16)>,
}

impl NullBackendLog {
    /// Flattens all recorded audio chunks into a single interleaved `f32` vector.
    pub fn all_samples(&self) -> Vec<f32> {
        let total: usize = self.chunks.iter().map(Vec::len).sum();
        let mut out = Vec::with_capacity(total);
        for chunk in &self.chunks {
            out.extend_from_slice(chunk);
        }
        out
    }
}

/// Silent in-memory [`AudioBackend`] for deterministic tests and headless fallback.
#[derive(Debug, Clone)]
pub struct NullAudioBackend {
    log: Arc<Mutex<NullBackendLog>>,
    paced: bool,
    sample_rate: u32,
    channels: u16,
}

impl NullAudioBackend {
    /// Name returned by [`NullAudioBackend::name`].
    pub const NAME: &'static str = "Null (silent)";

    /// Creates an unpaced [`NullAudioBackend`] with a fresh [`NullBackendLog`].
    pub fn new() -> Self {
        Self {
            log: Arc::new(Mutex::new(NullBackendLog::default())),
            paced: false,
            sample_rate: 0,
            channels: 0,
        }
    }

    /// Creates an unpaced [`NullAudioBackend`] sharing an existing [`NullBackendLog`].
    pub fn with_log(log: Arc<Mutex<NullBackendLog>>) -> Self {
        Self {
            log,
            paced: false,
            sample_rate: 0,
            channels: 0,
        }
    }

    /// Enables or disables real-time sleep pacing per written chunk.
    pub fn with_paced(mut self, paced: bool) -> Self {
        self.paced = paced;
        self
    }

    /// Returns a shared handle to the recorded backend log.
    pub fn log(&self) -> Arc<Mutex<NullBackendLog>> {
        Arc::clone(&self.log)
    }
}

impl Default for NullAudioBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioBackend for NullAudioBackend {
    fn open(&mut self, sample_rate: u32, channels: u16) -> Result<(), PlaybackError> {
        if sample_rate == 0 || channels == 0 {
            return Err(PlaybackError::InvalidConfig(format!(
                "sample_rate ({sample_rate}) and channels ({channels}) must be non-zero"
            )));
        }
        self.sample_rate = sample_rate;
        self.channels = channels;
        if let Ok(mut guard) = self.log.lock() {
            guard.open_count += 1;
            guard.last_config = Some((sample_rate, channels));
        }
        Ok(())
    }

    fn write_interleaved_f32(&mut self, samples: &[f32]) -> Result<(), PlaybackError> {
        if let Ok(mut guard) = self.log.lock() {
            guard.chunks.push(samples.to_vec());
        }
        if self.paced && self.sample_rate > 0 && self.channels > 0 && !samples.is_empty() {
            let frames = samples.len() / usize::from(self.channels);
            if frames > 0 {
                let dur = Duration::from_secs_f64(frames as f64 / f64::from(self.sample_rate));
                std::thread::sleep(dur);
            }
        }
        Ok(())
    }

    fn stop(&mut self) {
        if let Ok(mut guard) = self.log.lock() {
            guard.stop_count += 1;
        }
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_to_s16_le_bytes_clamps_and_encodes() {
        let input = [-1.0_f32, 0.0, 1.0, -1.5, 2.5, 0.5, f32::NAN];
        let bytes = f32_to_s16_le_bytes(&input);

        let decoded: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        assert_eq!(decoded[0], i16::MIN);
        assert_eq!(decoded[1], 0);
        assert_eq!(decoded[2], i16::MAX);
        assert_eq!(decoded[3], i16::MIN);
        assert_eq!(decoded[4], i16::MAX);
        assert!((decoded[5] - 16384).abs() <= 1);
        assert_eq!(decoded[6], 0);
    }

    #[test]
    fn pipe_tool_args_match_expected_flags() {
        assert_eq!(
            PipeTool::PwPlay.args(48000, 2),
            vec![
                "--rate",
                "48000",
                "--channels",
                "2",
                "--format",
                "f32",
                "--latency",
                "20ms",
                "-"
            ]
        );
        assert_eq!(
            PipeTool::PaPlay.args(44100, 1),
            vec![
                "--raw",
                "--rate=44100",
                "--channels=1",
                "--format=float32le"
            ]
        );
        assert_eq!(
            PipeTool::APlay.args(48000, 2),
            vec![
                "-q", "-t", "raw", "-f", "S16_LE", "-r", "48000", "-c", "2", "-"
            ]
        );
    }

    #[test]
    fn null_backend_records_open_write_stop_and_validates_config() {
        let mut backend = NullAudioBackend::new();
        let log = backend.log();

        assert!(matches!(
            backend.open(0, 2),
            Err(PlaybackError::InvalidConfig(_))
        ));
        assert!(matches!(
            backend.open(44100, 0),
            Err(PlaybackError::InvalidConfig(_))
        ));

        backend.open(48000, 2).unwrap();
        backend.write_interleaved_f32(&[0.25, -0.25]).unwrap();
        backend.stop();

        let guard = log.lock().unwrap();
        assert_eq!(guard.open_count, 1);
        assert_eq!(guard.stop_count, 1);
        assert_eq!(guard.last_config, Some((48000, 2)));
        assert_eq!(guard.chunks, vec![vec![0.25, -0.25]]);
    }
}
