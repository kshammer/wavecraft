pub mod pipe_backend;

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use wavecraft_core::AudioBuffer;

pub use pipe_backend::{
    AudioBackend, NullAudioBackend, NullBackendLog, PipeTool, PlaybackError, SystemPipeBackend,
    f32_to_s16_le_bytes,
};

const CHUNK_FRAMES: usize = 512;

/// Lock-free observable playback state shared between [`PlaybackEngine`] and UI callers.
#[derive(Debug, Default)]
pub struct PlaybackState {
    playing: AtomicBool,
    playhead_frame: AtomicUsize,
    total_frames: AtomicUsize,
}

impl PlaybackState {
    /// Returns `true` if audio is currently playing.
    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Acquire)
    }

    /// Returns the current playhead position in frames.
    pub fn playhead_frame(&self) -> usize {
        self.playhead_frame.load(Ordering::Acquire)
    }

    /// Returns the total frame count of the loaded [`AudioBuffer`].
    pub fn total_frames(&self) -> usize {
        self.total_frames.load(Ordering::Acquire)
    }
}

enum PlaybackCommand {
    SetBuffer(Arc<AudioBuffer>),
    Play,
    Pause,
    Stop,
    TogglePlayPause,
    Seek(usize),
    SetLoopRange(Option<Range<usize>>),
    SetVolume(f32),
    Shutdown,
}

/// Threaded real-time audio playback controller streaming [`AudioBuffer`] chunks to an [`AudioBackend`].
pub struct PlaybackEngine {
    cmd_tx: Sender<PlaybackCommand>,
    worker: Option<JoinHandle<()>>,
    state: Arc<PlaybackState>,
    backend_name: &'static str,
    is_real_output: bool,
}

impl PlaybackEngine {
    /// Creates a [`PlaybackEngine`] using [`SystemPipeBackend::detect`], falling back to a
    /// real-time paced [`NullAudioBackend`] if no system audio tool is installed.
    pub fn new_system() -> Self {
        if let Some(backend) = SystemPipeBackend::detect() {
            Self::spawn(backend, true)
        } else {
            Self::spawn(NullAudioBackend::new().with_paced(true), false)
        }
    }

    /// Creates a [`PlaybackEngine`] backed by the provided [`AudioBackend`].
    pub fn with_backend<B: AudioBackend>(backend: B) -> Self {
        let is_real_output = backend.name() != NullAudioBackend::NAME;
        Self::spawn(backend, is_real_output)
    }

    fn spawn<B: AudioBackend>(backend: B, is_real_output: bool) -> Self {
        let backend_name = backend.name();
        let state = Arc::new(PlaybackState::default());
        let worker_state = Arc::clone(&state);
        let (cmd_tx, cmd_rx) = mpsc::channel();

        let worker = thread::Builder::new()
            .name("wavecraft-playback".to_string())
            .spawn(move || {
                let mut runner = PlaybackWorker::new(backend, worker_state);
                runner.run(cmd_rx);
            })
            .expect("failed to spawn audio playback thread");

        Self {
            cmd_tx,
            worker: Some(worker),
            state,
            backend_name,
            is_real_output,
        }
    }

    /// Replaces the active audio buffer, clamping the playhead and reopening the stream if the
    /// sample rate or channel count changed.
    pub fn set_buffer(&self, buffer: Arc<AudioBuffer>) {
        self.state
            .total_frames
            .store(buffer.num_frames(), Ordering::Release);
        let _ = self.cmd_tx.send(PlaybackCommand::SetBuffer(buffer));
    }

    /// Starts or resumes playback from the current playhead.
    pub fn play(&self) {
        let _ = self.cmd_tx.send(PlaybackCommand::Play);
    }

    /// Pauses playback at the current playhead.
    pub fn pause(&self) {
        let _ = self.cmd_tx.send(PlaybackCommand::Pause);
    }

    /// Stops playback and resets the playhead to the loop start (if set) or frame `0`.
    pub fn stop(&self) {
        let _ = self.cmd_tx.send(PlaybackCommand::Stop);
    }

    /// Toggles between playing and paused states.
    pub fn toggle_play_pause(&self) {
        let _ = self.cmd_tx.send(PlaybackCommand::TogglePlayPause);
    }

    /// Seeks the playhead to `frame` (clamped to `[0, total_frames]`).
    pub fn seek(&self, frame: usize) {
        let _ = self.cmd_tx.send(PlaybackCommand::Seek(frame));
    }

    /// Sets or clears an optional loop range (`start..end` with `start < end <= num_frames`).
    pub fn set_loop_range(&self, range: Option<Range<usize>>) {
        let _ = self.cmd_tx.send(PlaybackCommand::SetLoopRange(range));
    }

    /// Sets the playback output gain, clamped to `0.0..=2.0`.
    pub fn set_volume(&self, volume: f32) {
        let _ = self.cmd_tx.send(PlaybackCommand::SetVolume(volume));
    }

    /// Returns a shared handle to the lock-free [`PlaybackState`].
    pub fn state(&self) -> Arc<PlaybackState> {
        Arc::clone(&self.state)
    }

    /// Returns the display name of the active [`AudioBackend`].
    pub fn backend_name(&self) -> &'static str {
        self.backend_name
    }

    /// Returns `true` if audio is routed to a real system audio device rather than the null sink.
    pub fn is_real_output(&self) -> bool {
        self.is_real_output
    }
}

impl Drop for PlaybackEngine {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(PlaybackCommand::Shutdown);
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

struct PlaybackWorker<B: AudioBackend> {
    backend: B,
    state: Arc<PlaybackState>,
    buffer: Option<Arc<AudioBuffer>>,
    stream_open: bool,
    open_config: Option<(u32, u16)>,
    playing: bool,
    draining: bool,
    playhead: usize,
    loop_range: Option<Range<usize>>,
    volume: f32,
}

impl<B: AudioBackend> PlaybackWorker<B> {
    fn new(backend: B, state: Arc<PlaybackState>) -> Self {
        Self {
            backend,
            state,
            buffer: None,
            stream_open: false,
            open_config: None,
            playing: false,
            draining: false,
            playhead: 0,
            loop_range: None,
            volume: 1.0,
        }
    }

    fn run(&mut self, rx: Receiver<PlaybackCommand>) {
        loop {
            if self.draining {
                match rx.recv_timeout(Duration::from_millis(10)) {
                    Ok(cmd) => {
                        if matches!(self.handle_command(cmd), LoopControl::Shutdown) {
                            break;
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {
                        self.shutdown();
                        break;
                    }
                }
                if self.draining && self.backend.poll_drained() {
                    self.draining = false;
                    self.stream_open = false;
                    self.open_config = None;
                    self.set_playing(false);
                    self.set_playhead(0);
                }
            } else if self.playing {
                match self.drain_commands(&rx) {
                    LoopControl::Continue => {}
                    LoopControl::Shutdown => break,
                }
                if self.playing && !self.draining {
                    self.step_playback();
                }
            } else {
                match rx.recv() {
                    Ok(cmd) => {
                        if matches!(self.handle_command(cmd), LoopControl::Shutdown) {
                            break;
                        }
                    }
                    Err(_) => {
                        self.shutdown();
                        break;
                    }
                }
            }
        }
    }

    fn drain_commands(&mut self, rx: &Receiver<PlaybackCommand>) -> LoopControl {
        loop {
            match rx.try_recv() {
                Ok(cmd) => {
                    if matches!(self.handle_command(cmd), LoopControl::Shutdown) {
                        return LoopControl::Shutdown;
                    }
                }
                Err(TryRecvError::Empty) => return LoopControl::Continue,
                Err(TryRecvError::Disconnected) => {
                    self.shutdown();
                    return LoopControl::Shutdown;
                }
            }
        }
    }

    fn handle_command(&mut self, cmd: PlaybackCommand) -> LoopControl {
        if self.draining
            && matches!(
                cmd,
                PlaybackCommand::Pause
                    | PlaybackCommand::Stop
                    | PlaybackCommand::TogglePlayPause
                    | PlaybackCommand::Seek(_)
                    | PlaybackCommand::Play
                    | PlaybackCommand::SetBuffer(_)
            )
        {
            self.draining = false;
            self.close_stream();
            if matches!(cmd, PlaybackCommand::Play) {
                self.set_playing(false);
            }
        }

        match cmd {
            PlaybackCommand::SetBuffer(buf) => self.on_set_buffer(buf),
            PlaybackCommand::Play => self.on_play(),
            PlaybackCommand::Pause => self.on_pause(),
            PlaybackCommand::Stop => self.on_stop(),
            PlaybackCommand::TogglePlayPause => {
                if self.playing {
                    self.on_pause();
                } else {
                    self.on_play();
                }
            }
            PlaybackCommand::Seek(frame) => self.on_seek(frame),
            PlaybackCommand::SetLoopRange(range) => {
                self.loop_range = range;
            }
            PlaybackCommand::SetVolume(vol) => {
                self.volume = if vol.is_nan() {
                    0.0
                } else {
                    vol.clamp(0.0, 2.0)
                };
            }
            PlaybackCommand::Shutdown => {
                self.shutdown();
                return LoopControl::Shutdown;
            }
        }
        LoopControl::Continue
    }

    fn shutdown(&mut self) {
        self.draining = false;
        self.set_playing(false);
        self.stream_open = false;
        self.open_config = None;
        self.backend.stop();
    }

    fn set_playing(&mut self, playing: bool) {
        self.playing = playing;
        self.state.playing.store(playing, Ordering::Release);
    }

    fn set_playhead(&mut self, frame: usize) {
        self.playhead = frame;
        self.state.playhead_frame.store(frame, Ordering::Release);
    }

    fn close_stream(&mut self) {
        if self.stream_open {
            self.backend.stop();
            self.stream_open = false;
            self.open_config = None;
        }
    }

    fn valid_loop_range(&self, num_frames: usize) -> Option<Range<usize>> {
        self.loop_range
            .as_ref()
            .filter(|r| r.start < r.end && r.end <= num_frames)
            .cloned()
    }

    fn buffer_config(buf: &AudioBuffer) -> Option<(u32, u16)> {
        if buf.is_empty() || buf.sample_rate == 0 {
            return None;
        }
        let channels = u16::try_from(buf.num_channels()).ok()?;
        Some((buf.sample_rate, channels))
    }

    fn ensure_stream_open(&mut self, buf: &AudioBuffer) -> bool {
        let Some(cfg) = Self::buffer_config(buf) else {
            self.close_stream();
            return false;
        };
        if self.stream_open && self.open_config == Some(cfg) {
            return true;
        }
        self.close_stream();
        match self.backend.open(cfg.0, cfg.1) {
            Ok(()) => {
                self.stream_open = true;
                self.open_config = Some(cfg);
                true
            }
            Err(_) => false,
        }
    }

    fn on_set_buffer(&mut self, new_buf: Arc<AudioBuffer>) {
        let new_frames = new_buf.num_frames();
        let clamped_playhead = self.playhead.min(new_frames);
        self.set_playhead(clamped_playhead);

        let new_cfg = Self::buffer_config(&new_buf);
        let config_changed = self.open_config != new_cfg;
        self.buffer = Some(new_buf);

        if new_cfg.is_none() {
            self.set_playing(false);
            self.close_stream();
            return;
        }

        if self.stream_open && config_changed {
            self.close_stream();
            if self.playing
                && let Some(buf) = self.buffer.clone()
                && !self.ensure_stream_open(&buf)
            {
                self.set_playing(false);
            }
        }
    }

    fn on_play(&mut self) {
        if self.playing {
            return;
        }
        let Some(buf) = self.buffer.clone() else {
            return;
        };
        let num_frames = buf.num_frames();
        if Self::buffer_config(&buf).is_none() {
            return;
        }
        if self.playhead >= num_frames {
            let reset = self.valid_loop_range(num_frames).map_or(0, |r| r.start);
            self.set_playhead(reset);
        }
        if self.ensure_stream_open(&buf) {
            self.set_playing(true);
        } else {
            self.set_playing(false);
        }
    }

    fn on_pause(&mut self) {
        if self.playing {
            self.set_playing(false);
            self.close_stream();
        }
    }

    fn on_stop(&mut self) {
        self.set_playing(false);
        self.close_stream();
        let num_frames = self.buffer.as_ref().map_or(0, |b| b.num_frames());
        let reset = self.valid_loop_range(num_frames).map_or(0, |r| r.start);
        self.set_playhead(reset);
    }

    fn on_seek(&mut self, frame: usize) {
        let num_frames = self.buffer.as_ref().map_or(0, |b| b.num_frames());
        self.set_playhead(frame.min(num_frames));
    }

    fn step_playback(&mut self) {
        let Some(buf) = self.buffer.clone() else {
            self.set_playing(false);
            self.close_stream();
            return;
        };

        let num_frames = buf.num_frames();
        if Self::buffer_config(&buf).is_none() {
            self.set_playing(false);
            self.close_stream();
            return;
        }

        if !self.ensure_stream_open(&buf) {
            self.set_playing(false);
            return;
        }

        let valid_loop = self.valid_loop_range(num_frames);
        let start = self.playhead;

        let (limit, wrap_target) = if let Some(loop_r) = valid_loop
            && start < loop_r.end
        {
            (loop_r.end, Some(loop_r.start))
        } else {
            (num_frames, None)
        };

        if start >= limit {
            self.set_playing(false);
            self.close_stream();
            self.set_playhead(0);
            return;
        }

        let end = (start + CHUNK_FRAMES).min(limit);
        let samples = buf.to_interleaved_f32(start..end, self.volume);
        if self.backend.write_interleaved_f32(&samples).is_err() {
            self.set_playing(false);
            self.close_stream();
            return;
        }

        if end >= limit {
            if let Some(loop_start) = wrap_target {
                self.set_playhead(loop_start);
            } else {
                self.set_playhead(end);
                self.draining = true;
                self.backend.begin_drain();
            }
        } else {
            self.set_playhead(end);
        }
    }
}

enum LoopControl {
    Continue,
    Shutdown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    use wavecraft_core::SampleEncoding;

    fn wait_until(timeout: Duration, mut pred: impl FnMut() -> bool) -> bool {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if pred() {
                return true;
            }
            thread::sleep(Duration::from_millis(2));
        }
        pred()
    }

    fn make_stereo_buffer(sample_rate: u32, frames: usize) -> Arc<AudioBuffer> {
        let mut left = Vec::with_capacity(frames);
        let mut right = Vec::with_capacity(frames);
        for i in 0..frames {
            let v = (i as f32 / frames as f32) * 2.0 - 1.0;
            left.push(v);
            right.push(-v * 0.5);
        }
        Arc::new(
            AudioBuffer::from_channels(sample_rate, vec![left, right], SampleEncoding::Float32)
                .unwrap(),
        )
    }

    #[test]
    fn playback_streams_exact_volume_scaled_samples_and_auto_stops() {
        let backend = NullAudioBackend::new();
        let log = backend.log();
        let engine = PlaybackEngine::with_backend(backend);

        assert!(!engine.is_real_output());
        assert_eq!(engine.backend_name(), NullAudioBackend::NAME);

        let buf = make_stereo_buffer(44100, 700);
        let expected = buf.to_interleaved_f32(0..700, 0.5);

        engine.set_buffer(Arc::clone(&buf));
        assert_eq!(engine.state().total_frames(), 700);

        engine.set_volume(0.5);
        engine.play();

        let finished = wait_until(Duration::from_secs(2), || {
            let guard = log.lock().unwrap();
            guard.drain_count >= 1 && !engine.state().is_playing()
        });
        assert!(finished, "playback did not finish within timeout");

        assert!(!engine.state().is_playing());
        assert_eq!(engine.state().playhead_frame(), 0);

        let guard = log.lock().unwrap();
        assert_eq!(guard.open_count, 1);
        assert_eq!(guard.drain_count, 1);
        assert_eq!(guard.stop_count, 0);
        assert_eq!(guard.last_config, Some((44100, 2)));
        assert_eq!(guard.chunks.len(), 2);
        assert_eq!(guard.chunks[0].len(), 512 * 2);
        assert_eq!(guard.chunks[1].len(), (700 - 512) * 2);
        assert_eq!(guard.all_samples(), expected);
    }

    #[test]
    fn seek_starts_playback_from_exact_requested_frame() {
        let backend = NullAudioBackend::new();
        let log = backend.log();
        let engine = PlaybackEngine::with_backend(backend);

        let buf = make_stereo_buffer(48000, 600);
        let expected = buf.to_interleaved_f32(250..600, 1.0);

        engine.set_buffer(buf);
        engine.seek(250);
        engine.play();

        let finished = wait_until(Duration::from_secs(2), || {
            let guard = log.lock().unwrap();
            guard.drain_count >= 1 && !engine.state().is_playing()
        });
        assert!(finished, "seek playback did not finish within timeout");

        let guard = log.lock().unwrap();
        assert_eq!(guard.drain_count, 1);
        assert_eq!(guard.stop_count, 0);
        assert_eq!(guard.all_samples(), expected);
    }

    #[test]
    fn loop_range_wraps_across_multiple_cycles_until_pause_or_stop() {
        let backend = NullAudioBackend::new();
        let log = backend.log();
        let engine = PlaybackEngine::with_backend(backend);

        let buf = make_stereo_buffer(44100, 1000);
        let expected_cycle = buf.to_interleaved_f32(100..300, 1.0);

        engine.set_buffer(buf);
        engine.set_loop_range(Some(100..300));
        engine.seek(100);
        engine.play();

        let reached_cycles = wait_until(Duration::from_secs(2), || {
            log.lock().unwrap().chunks.len() >= 4
        });
        assert!(reached_cycles, "loop did not emit >= 4 cycles");

        engine.pause();
        let paused = wait_until(Duration::from_secs(2), || !engine.state().is_playing());
        assert!(paused, "engine did not pause");

        {
            let guard = log.lock().unwrap();
            assert!(guard.chunks.len() >= 4);
            for chunk in &guard.chunks {
                assert_eq!(chunk, &expected_cycle);
            }
        }

        engine.stop();
        let reset = wait_until(Duration::from_secs(2), || {
            engine.state().playhead_frame() == 100
        });
        assert!(reset, "stop did not reset playhead to loop start");
    }

    #[test]
    fn set_buffer_hot_swaps_and_reopens_on_format_change() {
        let backend = NullAudioBackend::new();
        let log = backend.log();
        let engine = PlaybackEngine::with_backend(backend);

        let buf_a = Arc::new(
            AudioBuffer::from_channels(44100, vec![vec![0.25; 200]], SampleEncoding::Float32)
                .unwrap(),
        );
        let buf_b = make_stereo_buffer(48000, 600);

        engine.set_buffer(buf_a);
        engine.set_loop_range(Some(0..100));
        engine.play();

        assert!(wait_until(Duration::from_secs(2), || {
            log.lock().unwrap().open_count >= 1 && !log.lock().unwrap().chunks.is_empty()
        }));

        // Hot-swap to a buffer with a different sample rate and channel count while playing.
        engine.set_loop_range(None);
        engine.set_buffer(Arc::clone(&buf_b));
        assert_eq!(engine.state().total_frames(), 600);

        let reopened_and_finished = wait_until(Duration::from_secs(2), || {
            let guard = log.lock().unwrap();
            guard.open_count >= 2
                && guard.last_config == Some((48000, 2))
                && !engine.state().is_playing()
        });
        assert!(
            reopened_and_finished,
            "hot swap did not reopen stream with new format and finish"
        );
    }

    #[test]
    fn drop_shuts_down_worker_and_calls_backend_stop() {
        let backend = NullAudioBackend::new();
        let log = backend.log();
        let engine = PlaybackEngine::with_backend(backend);

        assert_eq!(log.lock().unwrap().stop_count, 0);
        drop(engine);
        assert_eq!(log.lock().unwrap().stop_count, 1);
    }

    #[test]
    fn toggle_play_pause_and_volume_clamping() {
        let backend = NullAudioBackend::new();
        let log = backend.log();
        let engine = PlaybackEngine::with_backend(backend);

        let buf = Arc::new(
            AudioBuffer::from_channels(44100, vec![vec![0.8; 100]], SampleEncoding::Float32)
                .unwrap(),
        );
        engine.set_buffer(buf);
        engine.set_volume(5.0); // clamps to 2.0 -> 0.8 * 2.0 = 1.6 -> sanitized to 1.0 by AudioBuffer
        engine.set_loop_range(Some(0..50));

        engine.toggle_play_pause();
        assert!(wait_until(Duration::from_secs(2), || {
            engine.state().is_playing()
        }));

        engine.toggle_play_pause();
        assert!(wait_until(Duration::from_secs(2), || {
            !engine.state().is_playing()
        }));

        let guard = log.lock().unwrap();
        assert!(!guard.chunks.is_empty());
        assert!(guard.chunks[0].iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn natural_eof_drains_without_stop_until_poll_drained_completes() {
        let backend = NullAudioBackend::new().with_drain_polls_before_done(3);
        let log = backend.log();
        let engine = PlaybackEngine::with_backend(backend);

        let buf = make_stereo_buffer(44100, 200);
        engine.set_buffer(buf);
        engine.play();

        let drained = wait_until(Duration::from_secs(2), || {
            let guard = log.lock().unwrap();
            guard.drain_count == 1 && !engine.state().is_playing()
        });
        assert!(drained, "natural EOF did not finish draining");

        let guard = log.lock().unwrap();
        assert_eq!(guard.drain_count, 1);
        assert_eq!(guard.stop_count, 0);
        assert_eq!(engine.state().playhead_frame(), 0);
    }

    #[test]
    fn stop_and_pause_while_draining_abort_immediately_via_stop() {
        // Explicit Stop interrupts draining immediately.
        {
            let backend = NullAudioBackend::new().with_drain_polls_before_done(1000);
            let log = backend.log();
            let engine = PlaybackEngine::with_backend(backend);

            engine.set_buffer(make_stereo_buffer(44100, 200));
            engine.play();

            assert!(
                wait_until(Duration::from_secs(2), || {
                    log.lock().unwrap().drain_count == 1
                }),
                "engine did not enter draining state"
            );
            assert!(engine.state().is_playing());

            engine.stop();
            assert!(
                wait_until(Duration::from_secs(1), || {
                    let guard = log.lock().unwrap();
                    guard.stop_count == 1 && !engine.state().is_playing()
                }),
                "stop did not immediately abort draining"
            );
            assert_eq!(engine.state().playhead_frame(), 0);
        }

        // Explicit Pause interrupts draining immediately.
        {
            let backend = NullAudioBackend::new().with_drain_polls_before_done(1000);
            let log = backend.log();
            let engine = PlaybackEngine::with_backend(backend);

            engine.set_buffer(make_stereo_buffer(44100, 200));
            engine.play();

            assert!(
                wait_until(Duration::from_secs(2), || {
                    log.lock().unwrap().drain_count == 1
                }),
                "engine did not enter draining state"
            );
            assert!(engine.state().is_playing());

            engine.pause();
            assert!(
                wait_until(Duration::from_secs(1), || {
                    let guard = log.lock().unwrap();
                    guard.stop_count == 1 && !engine.state().is_playing()
                }),
                "pause did not immediately abort draining"
            );
        }
    }
}
