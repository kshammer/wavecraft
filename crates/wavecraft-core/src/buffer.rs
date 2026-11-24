use std::ops::Range;

use crate::error::{CoreError, Result};

/// Maximum number of undo snapshots retained by [`EditableAudio`].
pub const MAX_UNDO_HISTORY: usize = 64;

/// Audio sample encoding of a loaded or exported stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleEncoding {
    PcmU8,
    PcmI16,
    PcmI24,
    PcmI32,
    Float32,
}

impl SampleEncoding {
    /// Returns the bit depth per sample for this encoding.
    pub fn bits_per_sample(&self) -> u16 {
        match self {
            Self::PcmU8 => 8,
            Self::PcmI16 => 16,
            Self::PcmI24 => 24,
            Self::PcmI32 => 32,
            Self::Float32 => 32,
        }
    }

    /// Returns a human-readable label describing the encoding.
    pub fn label(&self) -> &'static str {
        match self {
            Self::PcmU8 => "8-bit Unsigned PCM",
            Self::PcmI16 => "16-bit Signed PCM",
            Self::PcmI24 => "24-bit Signed PCM",
            Self::PcmI32 => "32-bit Signed PCM",
            Self::Float32 => "32-bit Float",
        }
    }
}

/// Min/max amplitude pair representing a single waveform overview bucket.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakBucket {
    pub min: f32,
    pub max: f32,
}

/// Multi-channel non-interleaved `f32` audio buffer normalized to `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f32>>,
    pub source_encoding: SampleEncoding,
}

fn sanitize_sample(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(-1.0, 1.0)
    }
}

impl AudioBuffer {
    /// Creates a silent buffer with `num_channels` channels and `num_frames` frames per channel.
    pub fn new_silent(sample_rate: u32, num_channels: usize, num_frames: usize) -> Self {
        Self {
            sample_rate,
            channels: vec![vec![0.0; num_frames]; num_channels],
            source_encoding: SampleEncoding::Float32,
        }
    }

    /// Creates a buffer from existing per-channel sample vectors after validating layout.
    pub fn from_channels(
        sample_rate: u32,
        channels: Vec<Vec<f32>>,
        encoding: SampleEncoding,
    ) -> Result<Self> {
        if sample_rate == 0 {
            return Err(CoreError::InvalidSampleRate(sample_rate));
        }
        if let Some(first_len) = channels.first().map(Vec::len) {
            for (idx, ch) in channels.iter().enumerate() {
                if ch.len() != first_len {
                    return Err(CoreError::InvalidChannelLayout(format!(
                        "channel {idx} has {} frames, expected {first_len}",
                        ch.len()
                    )));
                }
            }
        }
        Ok(Self {
            sample_rate,
            channels,
            source_encoding: encoding,
        })
    }

    /// Number of channels in the buffer.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of sample frames per channel.
    pub fn num_frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    /// Duration of the buffer in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.num_frames() as f64 / self.sample_rate as f64
        }
    }

    /// Returns `true` if the buffer has zero channels or zero frames.
    pub fn is_empty(&self) -> bool {
        self.num_channels() == 0 || self.num_frames() == 0
    }

    /// Clamps `range` to valid frame bounds `[0, num_frames()]`.
    pub(crate) fn clamp_range(&self, range: Range<usize>) -> Range<usize> {
        let total = self.num_frames();
        let start = range.start.min(total);
        let end = range.end.clamp(start, total);
        start..end
    }

    /// Returns the sample at `(ch, frame)` if within bounds.
    pub fn get_sample(&self, ch: usize, frame: usize) -> Option<f32> {
        self.channels.get(ch)?.get(frame).copied()
    }

    /// Sets the sample at `(ch, frame)` clamped to `[-1.0, 1.0]`.
    /// Returns `true` if the indices were in bounds.
    pub fn set_sample(&mut self, ch: usize, frame: usize, value: f32) -> bool {
        if let Some(slot) = self.channels.get_mut(ch).and_then(|c| c.get_mut(frame)) {
            *slot = sanitize_sample(value);
            true
        } else {
            false
        }
    }

    /// Returns a new [`AudioBuffer`] containing the clamped frame range across all channels.
    pub fn slice(&self, range: Range<usize>) -> Self {
        let clamped = self.clamp_range(range);
        let channels = self
            .channels
            .iter()
            .map(|ch| ch[clamped.clone()].to_vec())
            .collect();
        Self {
            sample_rate: self.sample_rate,
            channels,
            source_encoding: self.source_encoding,
        }
    }

    /// Converts the clamped frame range into interleaved `f32` samples scaled by `volume`
    /// and clamped to `[-1.0, 1.0]`.
    pub fn to_interleaved_f32(&self, range: Range<usize>, volume: f32) -> Vec<f32> {
        let clamped = self.clamp_range(range);
        let num_channels = self.num_channels();
        let len = clamped.len();
        if num_channels == 0 || len == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(len * num_channels);
        for frame in clamped {
            for ch in &self.channels {
                out.push(sanitize_sample(ch[frame] * volume));
            }
        }
        out
    }

    /// Averages all channels across the clamped frame range into a single mono slice.
    pub fn mono_mix_slice(&self, range: Range<usize>) -> Vec<f32> {
        let clamped = self.clamp_range(range);
        let num_channels = self.num_channels();
        let len = clamped.len();
        if num_channels == 0 || len == 0 {
            return Vec::new();
        }
        let inv_channels = 1.0 / num_channels as f32;
        let mut out = Vec::with_capacity(len);
        for frame in clamped {
            let sum: f32 = self.channels.iter().map(|ch| ch[frame]).sum();
            out.push(sum * inv_channels);
        }
        out
    }

    /// Computes min/max peak buckets for `channel` over the clamped `range` partitioned into
    /// `num_buckets` buckets for waveform rendering at arbitrary zoom levels.
    pub fn compute_peaks(
        &self,
        channel: usize,
        range: Range<usize>,
        num_buckets: usize,
    ) -> Vec<PeakBucket> {
        let Some(ch) = self.channels.get(channel) else {
            return Vec::new();
        };
        let clamped = self.clamp_range(range);
        let len = clamped.len();
        if len == 0 || num_buckets == 0 {
            return Vec::new();
        }

        let start = clamped.start;
        let end = clamped.end;
        let mut buckets = Vec::with_capacity(num_buckets);

        for b in 0..num_buckets {
            let b_start = start + (b * len) / num_buckets;
            let b_end = start + ((b + 1) * len) / num_buckets;
            if b_start >= b_end {
                let idx = b_start.min(end - 1);
                let val = ch[idx];
                buckets.push(PeakBucket { min: val, max: val });
            } else {
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;
                for &sample in &ch[b_start..b_end] {
                    min_val = min_val.min(sample);
                    max_val = max_val.max(sample);
                }
                buckets.push(PeakBucket {
                    min: min_val,
                    max: max_val,
                });
            }
        }

        buckets
    }
}

/// Editable audio document wrapping an [`AudioBuffer`] with undo/redo stacks,
/// clipboard support, and dirty-state tracking.
#[derive(Debug, Clone)]
pub struct EditableAudio {
    buffer: AudioBuffer,
    undo_stack: Vec<AudioBuffer>,
    redo_stack: Vec<AudioBuffer>,
    clipboard: Option<AudioBuffer>,
    dirty: bool,
}

impl EditableAudio {
    /// Creates a new [`EditableAudio`] starting in a clean state.
    pub fn new(buffer: AudioBuffer) -> Self {
        Self {
            buffer,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            clipboard: None,
            dirty: false,
        }
    }

    /// Returns a shared reference to the underlying [`AudioBuffer`].
    pub fn buffer(&self) -> &AudioBuffer {
        &self.buffer
    }

    /// Returns a mutable reference to the underlying [`AudioBuffer`] without recording undo history.
    pub fn buffer_mut_raw(&mut self) -> &mut AudioBuffer {
        &mut self.buffer
    }

    /// Returns `true` if the buffer has unsaved edits.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clears the dirty flag (e.g., after saving).
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Returns `true` if at least one undo snapshot is available.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Returns `true` if at least one redo snapshot is available.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Returns the number of stored undo snapshots.
    pub fn undo_len(&self) -> usize {
        self.undo_stack.len()
    }

    /// Returns the number of stored redo snapshots.
    pub fn redo_len(&self) -> usize {
        self.redo_stack.len()
    }

    /// Returns a reference to the current clipboard buffer, if any.
    pub fn clipboard(&self) -> Option<&AudioBuffer> {
        self.clipboard.as_ref()
    }

    /// Pushes the current buffer onto the undo stack (capped at [`MAX_UNDO_HISTORY`])
    /// and clears the redo stack.
    pub fn push_undo_snapshot(&mut self) {
        if self.undo_stack.len() >= MAX_UNDO_HISTORY {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(self.buffer.clone());
        self.redo_stack.clear();
    }

    /// Reverts to the most recent undo snapshot. Returns `true` if an undo was performed.
    pub fn undo(&mut self) -> bool {
        let Some(previous) = self.undo_stack.pop() else {
            return false;
        };
        self.redo_stack.push(self.buffer.clone());
        self.buffer = previous;
        self.dirty = true;
        true
    }

    /// Reapplies the most recent undone snapshot. Returns `true` if a redo was performed.
    pub fn redo(&mut self) -> bool {
        let Some(next) = self.redo_stack.pop() else {
            return false;
        };
        if self.undo_stack.len() >= MAX_UNDO_HISTORY {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(self.buffer.clone());
        self.buffer = next;
        self.dirty = true;
        true
    }

    /// Replaces the current buffer after pushing an undo snapshot.
    pub fn replace_buffer(&mut self, new_buf: AudioBuffer) {
        self.push_undo_snapshot();
        self.buffer = new_buf;
        self.dirty = true;
    }

    /// Directly sets a single sample without pushing an undo snapshot (for interactive strokes).
    pub fn set_sample_raw(&mut self, ch: usize, frame: usize, val: f32) {
        if self.buffer.set_sample(ch, frame, val) {
            self.dirty = true;
        }
    }

    /// Draws a pencil stroke onto `ch` by linearly interpolating sample values between
    /// consecutive `(frame, value)` points so fast pointer movements never skip samples.
    /// Does not push an undo snapshot; callers should call [`Self::push_undo_snapshot`] on stroke start.
    pub fn draw_stroke_raw(&mut self, ch: usize, points: &[(usize, f32)]) {
        let num_frames = self.buffer.num_frames();
        if ch >= self.buffer.num_channels() || num_frames == 0 || points.is_empty() {
            return;
        }

        let channel = &mut self.buffer.channels[ch];

        if points.len() == 1 {
            let (frame, val) = points[0];
            if frame < num_frames {
                channel[frame] = sanitize_sample(val);
                self.dirty = true;
            }
            return;
        }

        let mut modified = false;
        for window in points.windows(2) {
            let (f0, v0) = window[0];
            let (f1, v1) = window[1];

            if f0 == f1 {
                if f0 < num_frames {
                    channel[f0] = sanitize_sample(v1);
                    modified = true;
                }
                continue;
            }

            let lo = f0.min(f1);
            let hi = f0.max(f1).min(num_frames - 1);
            if lo > hi {
                continue;
            }

            let denom = f1 as f64 - f0 as f64;
            let dv = v1 as f64 - v0 as f64;
            for (offset, slot) in channel[lo..=hi].iter_mut().enumerate() {
                let frame = lo + offset;
                let t = (frame as f64 - f0 as f64) / denom;
                let interp = (v0 as f64 + dv * t) as f32;
                *slot = sanitize_sample(interp);
                modified = true;
            }
        }

        if modified {
            self.dirty = true;
        }
    }

    /// Applies local moving-average smoothing over `range` on channel `ch` using window `[-radius, +radius]`.
    /// Does not push an undo snapshot.
    pub fn smooth_range_raw(&mut self, ch: usize, range: Range<usize>, radius: usize) {
        let num_frames = self.buffer.num_frames();
        if ch >= self.buffer.num_channels() || num_frames == 0 || radius == 0 {
            return;
        }
        let clamped = self.buffer.clamp_range(range);
        if clamped.is_empty() {
            return;
        }

        let snapshot = self.buffer.channels[ch].clone();
        let target = &mut self.buffer.channels[ch];

        for i in clamped {
            let win_start = i.saturating_sub(radius);
            let win_end = (i + radius + 1).min(num_frames);
            let count = (win_end - win_start) as f64;
            let sum: f64 = snapshot[win_start..win_end].iter().map(|&s| s as f64).sum();
            target[i] = sanitize_sample((sum / count) as f32);
        }

        self.dirty = true;
    }

    /// Copies the clamped `range` into the internal clipboard without modifying the buffer.
    pub fn copy(&mut self, range: Range<usize>) {
        let clamped = self.buffer.clamp_range(range);
        if !clamped.is_empty() {
            self.clipboard = Some(self.buffer.slice(clamped));
        }
    }

    /// Copies the clamped `range` to the clipboard and removes it from the buffer.
    pub fn cut(&mut self, range: Range<usize>) {
        let clamped = self.buffer.clamp_range(range);
        if clamped.is_empty() {
            return;
        }
        self.clipboard = Some(self.buffer.slice(clamped.clone()));
        self.delete_range(clamped);
    }

    /// Inserts the clipboard audio at `at_frame`, preserving equal channel lengths across all channels.
    pub fn paste(&mut self, at_frame: usize) {
        let Some(clip) = self.clipboard.clone() else {
            return;
        };
        if clip.is_empty() || self.buffer.num_channels() == 0 {
            return;
        }

        self.push_undo_snapshot();
        let insert_at = at_frame.min(self.buffer.num_frames());
        let clip_frames = clip.num_frames();
        let clip_channels = clip.num_channels();

        for (ch_idx, ch) in self.buffer.channels.iter_mut().enumerate() {
            if ch_idx < clip_channels {
                ch.splice(insert_at..insert_at, clip.channels[ch_idx].iter().copied());
            } else if clip_channels == 1 {
                ch.splice(insert_at..insert_at, clip.channels[0].iter().copied());
            } else {
                ch.splice(insert_at..insert_at, std::iter::repeat_n(0.0, clip_frames));
            }
        }

        self.dirty = true;
    }

    /// Deletes the clamped `range` from all channels after pushing an undo snapshot.
    pub fn delete_range(&mut self, range: Range<usize>) {
        let clamped = self.buffer.clamp_range(range);
        if clamped.is_empty() {
            return;
        }
        self.push_undo_snapshot();
        for ch in &mut self.buffer.channels {
            ch.drain(clamped.clone());
        }
        self.dirty = true;
    }

    /// Trims the buffer to keep only the clamped `range`.
    pub fn trim(&mut self, range: Range<usize>) {
        let clamped = self.buffer.clamp_range(range);
        if clamped.start == 0 && clamped.end == self.buffer.num_frames() {
            return;
        }
        self.push_undo_snapshot();
        self.buffer = self.buffer.slice(clamped);
        self.dirty = true;
    }

    /// Pushes an undo snapshot, runs `f` on the buffer, and marks the document dirty.
    pub fn apply_transform<F>(&mut self, f: F)
    where
        F: FnOnce(&mut AudioBuffer),
    {
        self.push_undo_snapshot();
        f(&mut self.buffer);
        self.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_encoding_metadata() {
        assert_eq!(SampleEncoding::PcmU8.bits_per_sample(), 8);
        assert_eq!(SampleEncoding::PcmI16.bits_per_sample(), 16);
        assert_eq!(SampleEncoding::PcmI24.bits_per_sample(), 24);
        assert_eq!(SampleEncoding::PcmI32.bits_per_sample(), 32);
        assert_eq!(SampleEncoding::Float32.bits_per_sample(), 32);
        assert!(!SampleEncoding::PcmI16.label().is_empty());
    }

    #[test]
    fn buffer_creation_clamping_and_slicing() {
        let mut buf = AudioBuffer::new_silent(44100, 2, 100);
        assert_eq!(buf.num_channels(), 2);
        assert_eq!(buf.num_frames(), 100);
        assert!(!buf.is_empty());
        assert!((buf.duration_secs() - 100.0 / 44100.0).abs() < 1e-9);

        assert!(buf.set_sample(0, 10, 1.75));
        assert!(buf.set_sample(1, 10, -2.5));
        assert!(!buf.set_sample(2, 10, 0.5));
        assert_eq!(buf.get_sample(0, 10), Some(1.0));
        assert_eq!(buf.get_sample(1, 10), Some(-1.0));
        assert_eq!(buf.get_sample(0, 100), None);

        let sliced = buf.slice(5..15);
        assert_eq!(sliced.num_frames(), 10);
        assert_eq!(sliced.get_sample(0, 5), Some(1.0));
        assert_eq!(sliced.get_sample(1, 5), Some(-1.0));
    }

    #[test]
    fn from_channels_validates_layout() {
        let ok = AudioBuffer::from_channels(
            48000,
            vec![vec![0.1, 0.2], vec![-0.1, -0.2]],
            SampleEncoding::PcmI16,
        );
        assert!(ok.is_ok());

        let mismatch = AudioBuffer::from_channels(
            48000,
            vec![vec![0.1, 0.2], vec![-0.1]],
            SampleEncoding::PcmI16,
        );
        assert!(matches!(mismatch, Err(CoreError::InvalidChannelLayout(_))));

        let zero_sr = AudioBuffer::from_channels(0, vec![vec![0.0]], SampleEncoding::Float32);
        assert!(matches!(zero_sr, Err(CoreError::InvalidSampleRate(0))));
    }

    #[test]
    fn interleaved_and_mono_mix() {
        let buf = AudioBuffer::from_channels(
            44100,
            vec![vec![0.5, -0.5, 1.0], vec![-0.5, 0.5, 0.0]],
            SampleEncoding::Float32,
        )
        .unwrap();

        let interleaved = buf.to_interleaved_f32(0..3, 0.5);
        assert_eq!(interleaved, vec![0.25, -0.25, -0.25, 0.25, 0.5, 0.0]);

        let mono = buf.mono_mix_slice(0..3);
        assert_eq!(mono, vec![0.0, 0.0, 0.5]);
    }

    #[test]
    fn compute_peaks_exact_min_max() {
        let data: Vec<f32> = vec![-0.8, 0.2, 0.9, -0.1, -0.5, 0.4, 0.1, 0.7];
        let buf = AudioBuffer::from_channels(44100, vec![data], SampleEncoding::Float32).unwrap();

        let peaks = buf.compute_peaks(0, 0..8, 2);
        assert_eq!(peaks.len(), 2);
        assert_eq!(
            peaks[0],
            PeakBucket {
                min: -0.8,
                max: 0.9
            }
        );
        assert_eq!(
            peaks[1],
            PeakBucket {
                min: -0.5,
                max: 0.7
            }
        );

        let upsampled_peaks = buf.compute_peaks(0, 0..4, 8);
        assert_eq!(upsampled_peaks.len(), 8);
        assert!(buf.compute_peaks(5, 0..8, 2).is_empty());
        assert!(buf.compute_peaks(0, 0..8, 0).is_empty());
    }

    #[test]
    fn editable_draw_stroke_interpolates_forward_and_backward() {
        let buf = AudioBuffer::new_silent(44100, 1, 16);
        let mut editable = EditableAudio::new(buf);
        editable.push_undo_snapshot();
        editable.draw_stroke_raw(0, &[(0, 0.0), (4, 1.0), (8, -1.0)]);

        let ch = &editable.buffer().channels[0];
        assert!((ch[0] - 0.0).abs() < 1e-6);
        assert!((ch[1] - 0.25).abs() < 1e-6);
        assert!((ch[2] - 0.5).abs() < 1e-6);
        assert!((ch[3] - 0.75).abs() < 1e-6);
        assert!((ch[4] - 1.0).abs() < 1e-6);
        assert!((ch[6] - 0.0).abs() < 1e-6);
        assert!((ch[8] - (-1.0)).abs() < 1e-6);
        assert!(editable.is_dirty());

        // Backward stroke
        editable.draw_stroke_raw(0, &[(14, 0.6), (10, -0.2)]);
        let ch = &editable.buffer().channels[0];
        assert!((ch[14] - 0.6).abs() < 1e-6);
        assert!((ch[12] - 0.2).abs() < 1e-6);
        assert!((ch[10] - (-0.2)).abs() < 1e-6);
    }

    #[test]
    fn editable_smooth_range_averages_neighbors() {
        let buf = AudioBuffer::from_channels(
            44100,
            vec![vec![0.0, 0.0, 1.0, 0.0, 0.0]],
            SampleEncoding::Float32,
        )
        .unwrap();
        let mut editable = EditableAudio::new(buf);
        editable.smooth_range_raw(0, 1..4, 1);
        let ch = &editable.buffer().channels[0];
        assert!((ch[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((ch[2] - 1.0 / 3.0).abs() < 1e-6);
        assert!((ch[3] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn undo_redo_stack_cap_and_clear_on_new_edit() {
        let buf = AudioBuffer::new_silent(44100, 1, 4);
        let mut editable = EditableAudio::new(buf);

        for i in 1..=70 {
            editable.apply_transform(|b| {
                b.set_sample(0, 0, (i as f32) / 100.0);
            });
        }
        assert_eq!(editable.undo_len(), MAX_UNDO_HISTORY);
        assert!(editable.can_undo());
        assert!(!editable.can_redo());

        assert!(editable.undo());
        assert!(editable.can_redo());
        assert!((editable.buffer().get_sample(0, 0).unwrap() - 0.69).abs() < 1e-6);

        assert!(editable.redo());
        assert!((editable.buffer().get_sample(0, 0).unwrap() - 0.70).abs() < 1e-6);

        // Undo once, then make a new edit -> redo stack clears
        assert!(editable.undo());
        assert!(editable.can_redo());
        editable.apply_transform(|b| {
            b.set_sample(0, 1, 0.5);
        });
        assert!(!editable.can_redo());
    }

    #[test]
    fn cut_copy_paste_delete_trim_preserve_channel_lengths() {
        let buf = AudioBuffer::from_channels(
            44100,
            vec![
                vec![0.1, 0.2, 0.3, 0.4, 0.5],
                vec![-0.1, -0.2, -0.3, -0.4, -0.5],
            ],
            SampleEncoding::Float32,
        )
        .unwrap();
        let mut editable = EditableAudio::new(buf);

        editable.copy(1..3);
        editable.paste(5);
        assert_eq!(editable.buffer().num_frames(), 7);
        assert_eq!(
            editable.buffer().channels[0].len(),
            editable.buffer().channels[1].len()
        );
        assert_eq!(editable.buffer().get_sample(0, 5), Some(0.2));
        assert_eq!(editable.buffer().get_sample(1, 6), Some(-0.3));

        editable.cut(1..3);
        assert_eq!(editable.buffer().num_frames(), 5);
        assert_eq!(editable.buffer().channels[0], vec![0.1, 0.4, 0.5, 0.2, 0.3]);
        assert_eq!(
            editable.buffer().channels[1],
            vec![-0.1, -0.4, -0.5, -0.2, -0.3]
        );

        editable.delete_range(3..5);
        assert_eq!(editable.buffer().num_frames(), 3);
        assert_eq!(editable.buffer().channels[0], vec![0.1, 0.4, 0.5]);

        editable.trim(1..3);
        assert_eq!(editable.buffer().num_frames(), 2);
        assert_eq!(editable.buffer().channels[0], vec![0.4, 0.5]);
        assert_eq!(editable.buffer().channels[1], vec![-0.4, -0.5]);
    }
}
