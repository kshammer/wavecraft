use thiserror::Error;

/// Convenience result type for `wavecraft-core` operations.
pub type Result<T> = std::result::Result<T, CoreError>;

/// Errors returned by `wavecraft-core` buffer, DSP, and audio I/O operations.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid channel index {channel} (buffer has {num_channels} channels)")]
    InvalidChannel { channel: usize, num_channels: usize },

    #[error("invalid frame index {frame} (buffer has {num_frames} frames)")]
    InvalidFrame { frame: usize, num_frames: usize },

    #[error("invalid frame range {start}..{end} (buffer has {num_frames} frames)")]
    InvalidRange {
        start: usize,
        end: usize,
        num_frames: usize,
    },

    #[error("invalid channel layout: {0}")]
    InvalidChannelLayout(String),

    #[error("unsupported audio format or encoding: {0}")]
    UnsupportedFormat(String),

    #[error("malformed file or header: {0}")]
    MalformedFile(String),

    #[error("audio decode failure: {0}")]
    DecodeError(String),

    #[error("audio buffer is empty")]
    EmptyBuffer,

    #[error("invalid sample rate: {0}")]
    InvalidSampleRate(u32),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
