# Wavecraft

Wavecraft is a native Rust audio waveform editor, wavelength-locked harmonic synthesizer, and binary format inspector built with `egui` / `eframe`.

## Key Features

- **Granular Waveform Editor**:
  - Multi-channel adaptive rendering: min/max peak envelopes when zoomed out, transitioning smoothly to sample-to-sample polylines, amplitude stems, and interactive circular sample dots when zoomed in (`>= 4 px/sample`).
  - Smooth cursor-anchored mouse-wheel zoom (down to ~33 px/sample), `Shift + Scroll` / middle-mouse horizontal panning, and an overview minimap scrollbar.
  - **Select / Scrub** (click to seek, drag to select, double-click to select one wavelength cycle), **Pencil (Draw)** (interpolated per-sample drawing directly onto the buffer), and **Smooth** (local moving-average brush).
  - **Wavelength Grid Overlay** aligned to the active tuner wavelength and selection start.
  - Live hover HUD (`Sample #`, `Time (ms)`, `f32` / `i16` amplitude) and selection readout (`[start..end]`, sample count, duration, λ frequency).
- **Live FFT Spectrum Analyzer**:
  - Hann-windowed `rustfft` analyzer (`512` / `1024` / `2048` / `4096` points) over selection or playhead window.
  - Log-frequency (`20 Hz..Nyquist`) vs magnitude (`-100..0 dB`) plot with parabolic peak detection, musical note + cents readout, and one-click `Lock Tuner to Peak λ`.
- **Wavelength & 16-Harmonic Tuner**:
  - Linked **Wavelength (samples) ↔ Frequency (Hz) ↔ Musical Note** controls with octave/semitone shift and `A4 (440 Hz)` presets.
  - Shapes: `Sine`, `Square`, `Sawtooth`, `Triangle`, `Pulse` (adjustable duty cycle), `Noise`, and interactive 16-partial `Harmonics` bar editor with `Fund`, `Saw`, `Square`, `Organ`, and `Clear` presets plus a live single-cycle preview.
  - `Apply to Selection` (`Replace` / `Mix`), `Generate New 1s Buffer`, and `Stretch Selection to λ` (wavelength-targeted resampling).
- **Region DSP & Undo/Redo**:
  - Gain (`-24..+24 dB`), `Normalize (0.95)`, `Fade In`, `Fade Out`, `Reverse`, `Remove DC`, `Trim`, `Cut`, `Copy`, `Paste at Playhead`, and `Delete Selection` with up to 64 levels of undo/redo.
- **File Explorer & Binary Format Inspector**:
  - Directory browser with format filter chips (`WAV`, `PCM`, `FLAC`, `MP3`, `OGG`) and lazy waveform thumbnails.
  - Deep format structure tree + synchronized Hex/ASCII viewer (`RIFF`/`WAVE` chunks, FLAC metadata blocks, MP3 frames/ID3, Ogg Vorbis pages) and Raw PCM import/export.

## Architecture

- `crates/wavecraft-core`: Pure audio buffer model (`AudioBuffer`, `EditableAudio`), DSP & FFT engine (`compute_spectrum`, `generate_wave`, `resample_region_to_wavelength`, region transforms), format parsers/encoders (`wav`, `raw_pcm`, `compressed`), and sample fixture generator.
- `crates/wavecraft-audio`: Threaded `PlaybackEngine` with lock-free `PlaybackState`, looping, seeking, volume scaling, and `AudioBackend` trait (`SystemPipeBackend` + `NullAudioBackend`).
- `crates/wavecraft-app`: Native `eframe`/`egui` workbench UI and CLI entrypoint.

## Running

### GUI Workbench

```bash
cargo run -p wavecraft-app
# Or open a specific file directly:
cargo run -p wavecraft-app -- samples/sine_440hz.wav
```

### CLI Tools

```bash
# Generate demo WAV & raw PCM fixtures into ./samples
cargo run -p wavecraft-app -- --generate-samples samples

# Inspect a file's metadata and binary chunk/frame tree
cargo run -p wavecraft-app -- --inspect samples/sine_440hz.wav
```

## Keyboard Shortcuts

| Shortcut | Action |
| --- | --- |
| `Space` | Play / Pause |
| `Ctrl+Z` / `Cmd+Z` | Undo |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |
| `Ctrl+C` / `Cmd+C` | Copy selection |
| `Ctrl+X` / `Cmd+X` | Cut selection |
| `Ctrl+V` / `Cmd+V` | Paste at playhead |
| `Delete` / `Backspace` | Delete selection |
| `Ctrl+S` / `Cmd+S` | Save WAV |
| `=` / `+` | Zoom in |
| `-` | Zoom out |

## Linux Audio Playback Note

`wavecraft-audio` automatically detects and streams 32-bit float PCM (`pw-play` / `paplay`) or 16-bit LE PCM (`aplay`) over `stdin` using whichever tool is available on `PATH`:

1. **PipeWire**: `pw-play`
2. **PulseAudio**: `paplay`
3. **ALSA**: `aplay`

If none of these CLI tools is present (e.g., headless CI containers), `PlaybackEngine` falls back cleanly to a real-time paced `NullAudioBackend` so transport controls, playhead tracking, and looping remain functional without crashing.

## Testing & Verification

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --workspace
```
