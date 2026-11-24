use std::f64::consts::TAU;
use std::ops::Range;

use rustfft::{FftPlanner, num_complex::Complex};

use crate::buffer::AudioBuffer;

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

const SPECTRUM_FLOOR_DB: f32 = -120.0;

fn clamp_sample(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(-1.0, 1.0) }
}

/// Returns `true` if `ch_idx` is included by `channel_mask`.
/// `None` selects all channels; `Some(mask)` treats `mask` as a bitmask where bit `ch_idx` selects channel `ch_idx`.
fn channel_selected(channel_mask: Option<usize>, ch_idx: usize) -> bool {
    match channel_mask {
        None => true,
        Some(mask) => ch_idx < usize::BITS as usize && ((mask >> ch_idx) & 1) != 0,
    }
}

/// Converts a cycle length in samples to frequency in Hertz at `sample_rate`.
/// Returns `0.0` for non-positive or non-finite inputs.
pub fn wavelength_to_hz(wavelength_samples: f64, sample_rate: u32) -> f64 {
    if wavelength_samples <= 0.0 || !wavelength_samples.is_finite() || sample_rate == 0 {
        0.0
    } else {
        sample_rate as f64 / wavelength_samples
    }
}

/// Converts a frequency in Hertz to cycle length in samples at `sample_rate`.
/// Returns `0.0` for non-positive or non-finite inputs.
pub fn hz_to_wavelength(hz: f64, sample_rate: u32) -> f64 {
    if hz <= 0.0 || !hz.is_finite() || sample_rate == 0 {
        0.0
    } else {
        sample_rate as f64 / hz
    }
}

/// Converts a fractional MIDI note number to frequency in Hertz (`A4 = 69 -> 440.0 Hz`).
pub fn midi_note_to_hz(midi_note: f64) -> f64 {
    440.0 * 2.0_f64.powf((midi_note - 69.0) / 12.0)
}

/// Nearest musical pitch information for a given frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct NoteInfo {
    pub name: String,
    pub midi_note: i32,
    pub cents_offset: f64,
}

/// Finds the nearest 12-TET musical note and cent offset for `hz`.
/// Returns `None` if `hz` is non-positive or non-finite.
pub fn hz_to_nearest_note(hz: f64) -> Option<NoteInfo> {
    if hz <= 0.0 || !hz.is_finite() {
        return None;
    }
    let midi_float = 69.0 + 12.0 * (hz / 440.0).log2();
    if !midi_float.is_finite() {
        return None;
    }
    let midi_note = midi_float.round() as i32;
    let cents_offset = (midi_float - midi_note as f64) * 100.0;
    let note_idx = midi_note.rem_euclid(12) as usize;
    let octave = midi_note.div_euclid(12) - 1;
    Some(NoteInfo {
        name: format!("{}{}", NOTE_NAMES[note_idx], octave),
        midi_note,
        cents_offset,
    })
}

/// A single partial for additive harmonic synthesis (`1` = fundamental, typically up to `16`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicPartial {
    pub harmonic: usize,
    pub amplitude: f32,
    pub phase_rad: f32,
}

/// Waveform generator shape.
#[derive(Debug, Clone, PartialEq)]
pub enum WaveShape {
    Sine,
    Square,
    Sawtooth,
    Triangle,
    Pulse { duty: f32 },
    Noise,
    Harmonics(Vec<HarmonicPartial>),
}

/// Parameters controlling wavelength-locked waveform synthesis.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthParams {
    pub shape: WaveShape,
    pub wavelength_samples: f64,
    pub amplitude: f32,
    pub phase_offset_rad: f32,
}

/// How synthesized audio is written into an existing buffer selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyMode {
    Replace,
    Mix,
}

/// Deterministic splitmix64 step returning a value in `[-1.0, 1.0]`.
fn next_deterministic_noise(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let unit = (z >> 11) as f64 / ((1u64 << 53) as f64);
    (unit * 2.0 - 1.0) as f32
}

fn eval_wave_at_cycle(shape: &WaveShape, frac: f64, noise_state: &mut u64) -> f32 {
    let phase = frac * TAU;
    match shape {
        WaveShape::Sine => phase.sin() as f32,
        WaveShape::Square => {
            if frac < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        WaveShape::Sawtooth => (2.0 * frac - 1.0) as f32,
        WaveShape::Triangle => {
            let val = if frac < 0.25 {
                4.0 * frac
            } else if frac < 0.75 {
                2.0 - 4.0 * frac
            } else {
                4.0 * frac - 4.0
            };
            val as f32
        }
        WaveShape::Pulse { duty } => {
            let d = if duty.is_nan() {
                0.5
            } else {
                (*duty as f64).clamp(0.0, 1.0)
            };
            if frac < d { 1.0 } else { -1.0 }
        }
        WaveShape::Noise => next_deterministic_noise(noise_state),
        WaveShape::Harmonics(partials) => {
            let sum: f64 = partials
                .iter()
                .filter(|p| p.harmonic >= 1)
                .map(|p| {
                    p.amplitude as f64 * (phase * p.harmonic as f64 + p.phase_rad as f64).sin()
                })
                .sum();
            sum as f32
        }
    }
}

/// Generates `num_frames` samples with deterministic phase progression `2 * PI * i / wavelength_samples`.
/// Output samples are clamped to `[-1.0, 1.0]`.
pub fn generate_wave(params: &SynthParams, num_frames: usize) -> Vec<f32> {
    if num_frames == 0 {
        return Vec::new();
    }
    if !matches!(params.shape, WaveShape::Noise)
        && (params.wavelength_samples <= 0.0 || !params.wavelength_samples.is_finite())
    {
        return vec![0.0; num_frames];
    }

    let phase_cycles_offset = if params.phase_offset_rad.is_finite() {
        params.phase_offset_rad as f64 / TAU
    } else {
        0.0
    };
    let amp = if params.amplitude.is_finite() {
        params.amplitude
    } else {
        0.0
    };
    let mut noise_state = 0x517C_C1B7_2722_0A95_u64 ^ (params.phase_offset_rad.to_bits() as u64);

    let mut out = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let frac = if matches!(params.shape, WaveShape::Noise) {
            0.0
        } else {
            ((i as f64 / params.wavelength_samples) + phase_cycles_offset).rem_euclid(1.0)
        };
        let raw = eval_wave_at_cycle(&params.shape, frac, &mut noise_state);
        out.push(clamp_sample(raw * amp));
    }
    out
}

/// Generates a single-cycle preview of `params.shape` sampled at `points` positions.
pub fn generate_single_cycle(params: &SynthParams, points: usize) -> Vec<f32> {
    if points == 0 {
        return Vec::new();
    }
    let cycle_params = SynthParams {
        shape: params.shape.clone(),
        wavelength_samples: points as f64,
        amplitude: params.amplitude,
        phase_offset_rad: params.phase_offset_rad,
    };
    generate_wave(&cycle_params, points)
}

/// Synthesizes a waveform into `buf` over `range` on channels selected by `channel_mask`.
pub fn apply_synth_to_buffer(
    buf: &mut AudioBuffer,
    range: Range<usize>,
    channel_mask: Option<usize>,
    params: &SynthParams,
    mode: ApplyMode,
) {
    let clamped = buf.clamp_range(range);
    if clamped.is_empty() {
        return;
    }
    let wave = generate_wave(params, clamped.len());
    for (ch_idx, ch) in buf.channels.iter_mut().enumerate() {
        if !channel_selected(channel_mask, ch_idx) {
            continue;
        }
        for (offset, frame) in clamped.clone().enumerate() {
            let next = match mode {
                ApplyMode::Replace => wave[offset],
                ApplyMode::Mix => ch[frame] + wave[offset],
            };
            ch[frame] = clamp_sample(next);
        }
    }
}

/// Maximum number of frames allocated by [`resample_region_to_wavelength`] (~6 minutes at 44.1 kHz).
pub const MAX_RESAMPLED_FRAMES: usize = 16_000_000;

/// Computes the clamped output frame count for resampling `old_len` frames from
/// `current_wavelength` to `target_wavelength`, returning `None` for invalid or non-finite inputs.
pub fn capped_resampled_len(
    old_len: usize,
    current_wavelength: f64,
    target_wavelength: f64,
) -> Option<usize> {
    if old_len == 0
        || current_wavelength <= 0.0
        || target_wavelength <= 0.0
        || !current_wavelength.is_finite()
        || !target_wavelength.is_finite()
    {
        return None;
    }

    let ratio = target_wavelength / current_wavelength;
    if !ratio.is_finite() || ratio <= 0.0 {
        return None;
    }

    let scaled = (old_len as f64) * ratio;
    if !scaled.is_finite() || scaled >= MAX_RESAMPLED_FRAMES as f64 {
        Some(MAX_RESAMPLED_FRAMES)
    } else {
        Some((scaled.round() as usize).clamp(1, MAX_RESAMPLED_FRAMES))
    }
}

/// Linearly resamples the selected `range` across all channels to shift its wavelength
/// from `current_wavelength` to `target_wavelength`, replacing `range` in the buffer.
pub fn resample_region_to_wavelength(
    buf: &mut AudioBuffer,
    range: Range<usize>,
    current_wavelength: f64,
    target_wavelength: f64,
) {
    let clamped = buf.clamp_range(range);
    let old_len = clamped.len();
    let Some(new_len) = capped_resampled_len(old_len, current_wavelength, target_wavelength) else {
        return;
    };
    let ratio = target_wavelength / current_wavelength;

    for ch in &mut buf.channels {
        let src = &ch[clamped.clone()];
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_pos = (i as f64 / ratio).min((old_len - 1) as f64);
            let idx0 = (src_pos.floor() as usize).min(old_len - 1);
            let idx1 = (idx0 + 1).min(old_len - 1);
            let frac = (src_pos - idx0 as f64).clamp(0.0, 1.0);
            let val = src[idx0] as f64 + (src[idx1] as f64 - src[idx0] as f64) * frac;
            resampled.push(clamp_sample(val as f32));
        }
        ch.splice(clamped.clone(), resampled);
    }
}

/// Applies gain in decibels (`10^(db/20)`) to `range` on selected channels, clamping to `[-1.0, 1.0]`.
pub fn apply_gain_db(
    buf: &mut AudioBuffer,
    range: Range<usize>,
    channel_mask: Option<usize>,
    db: f32,
) {
    let clamped = buf.clamp_range(range);
    if clamped.is_empty() || !db.is_finite() {
        return;
    }
    let gain = 10.0_f32.powf(db / 20.0);
    for (ch_idx, ch) in buf.channels.iter_mut().enumerate() {
        if !channel_selected(channel_mask, ch_idx) {
            continue;
        }
        for sample in &mut ch[clamped.clone()] {
            *sample = clamp_sample(*sample * gain);
        }
    }
}

/// Scales selected channels in `range` so their peak absolute amplitude reaches `target_peak`.
pub fn normalize(
    buf: &mut AudioBuffer,
    range: Range<usize>,
    channel_mask: Option<usize>,
    target_peak: f32,
) {
    let clamped = buf.clamp_range(range);
    if clamped.is_empty() || !target_peak.is_finite() {
        return;
    }
    let target = target_peak.abs().clamp(0.0, 1.0);

    let mut max_abs = 0.0_f32;
    for (ch_idx, ch) in buf.channels.iter().enumerate() {
        if !channel_selected(channel_mask, ch_idx) {
            continue;
        }
        for &sample in &ch[clamped.clone()] {
            max_abs = max_abs.max(sample.abs());
        }
    }
    if max_abs <= 0.0 {
        return;
    }

    let scale = target / max_abs;
    for (ch_idx, ch) in buf.channels.iter_mut().enumerate() {
        if !channel_selected(channel_mask, ch_idx) {
            continue;
        }
        for sample in &mut ch[clamped.clone()] {
            *sample = clamp_sample(*sample * scale);
        }
    }
}

fn apply_linear_fade(
    buf: &mut AudioBuffer,
    range: Range<usize>,
    channel_mask: Option<usize>,
    fade_in_dir: bool,
) {
    let clamped = buf.clamp_range(range);
    let len = clamped.len();
    if len == 0 {
        return;
    }
    for (ch_idx, ch) in buf.channels.iter_mut().enumerate() {
        if !channel_selected(channel_mask, ch_idx) {
            continue;
        }
        for (i, sample) in ch[clamped.clone()].iter_mut().enumerate() {
            let factor = if len == 1 {
                0.0
            } else {
                let t = i as f32 / (len - 1) as f32;
                if fade_in_dir { t } else { 1.0 - t }
            };
            *sample = clamp_sample(*sample * factor);
        }
    }
}

/// Applies a linear fade-in from `0.0` to `1.0` across `range` on selected channels.
pub fn fade_in(buf: &mut AudioBuffer, range: Range<usize>, channel_mask: Option<usize>) {
    apply_linear_fade(buf, range, channel_mask, true);
}

/// Applies a linear fade-out from `1.0` to `0.0` across `range` on selected channels.
pub fn fade_out(buf: &mut AudioBuffer, range: Range<usize>, channel_mask: Option<usize>) {
    apply_linear_fade(buf, range, channel_mask, false);
}

/// Reverses the sample order in `range` on selected channels.
pub fn reverse(buf: &mut AudioBuffer, range: Range<usize>, channel_mask: Option<usize>) {
    let clamped = buf.clamp_range(range);
    if clamped.len() <= 1 {
        return;
    }
    for (ch_idx, ch) in buf.channels.iter_mut().enumerate() {
        if !channel_selected(channel_mask, ch_idx) {
            continue;
        }
        ch[clamped.clone()].reverse();
    }
}

/// Subtracts the per-channel mean value across `range` to eliminate DC offset.
pub fn remove_dc_offset(buf: &mut AudioBuffer, range: Range<usize>, channel_mask: Option<usize>) {
    let clamped = buf.clamp_range(range);
    let len = clamped.len();
    if len == 0 {
        return;
    }
    for (ch_idx, ch) in buf.channels.iter_mut().enumerate() {
        if !channel_selected(channel_mask, ch_idx) {
            continue;
        }
        let sum: f64 = ch[clamped.clone()].iter().map(|&s| s as f64).sum();
        let mean = (sum / len as f64) as f32;
        for sample in &mut ch[clamped.clone()] {
            *sample = clamp_sample(*sample - mean);
        }
    }
}

/// Single-sided FFT magnitude spectrum and detected fundamental peak.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectrumData {
    pub sample_rate: u32,
    pub fft_size: usize,
    pub freqs_hz: Vec<f32>,
    pub magnitudes_db: Vec<f32>,
    pub peak_freq_hz: Option<f64>,
    pub peak_wavelength_samples: Option<f64>,
}

/// Computes a Hann-windowed FFT magnitude spectrum (in dB, floored at `-120.0`) and
/// estimates the dominant peak frequency and wavelength using parabolic interpolation.
pub fn compute_spectrum(samples: &[f32], sample_rate: u32, fft_size: usize) -> SpectrumData {
    let fft_size = fft_size.max(64).next_power_of_two();
    let num_bins = fft_size / 2 + 1;
    let bin_hz = if sample_rate > 0 {
        sample_rate as f64 / fft_size as f64
    } else {
        0.0
    };
    let freqs_hz: Vec<f32> = (0..num_bins).map(|k| (k as f64 * bin_hz) as f32).collect();

    if samples.is_empty() || sample_rate == 0 {
        return SpectrumData {
            sample_rate,
            fft_size,
            freqs_hz,
            magnitudes_db: vec![SPECTRUM_FLOOR_DB; num_bins],
            peak_freq_hz: None,
            peak_wavelength_samples: None,
        };
    }

    let active_len = samples.len().min(fft_size);
    let mut buffer = vec![Complex::new(0.0_f32, 0.0_f32); fft_size];
    let mut window_sum = 0.0_f64;

    for (i, &sample) in samples[..active_len].iter().enumerate() {
        let w = if active_len <= 1 {
            1.0
        } else {
            0.5 * (1.0 - (TAU * i as f64 / active_len as f64).cos())
        };
        window_sum += w;
        let clean = if sample.is_finite() { sample } else { 0.0 };
        buffer[i] = Complex::new((clean as f64 * w) as f32, 0.0);
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut buffer);

    let norm_base = if window_sum > 0.0 { window_sum } else { 1.0 };
    let mut magnitudes_db = Vec::with_capacity(num_bins);
    let mut linear_mags = Vec::with_capacity(num_bins);

    for (k, bin) in buffer.iter().take(num_bins).enumerate() {
        let scale = if k == 0 || k == fft_size / 2 {
            1.0 / norm_base
        } else {
            2.0 / norm_base
        };
        let mag = (bin.norm() as f64 * scale) as f32;
        linear_mags.push(mag);
        let db = if mag > 0.0 {
            (20.0 * mag.log10()).max(SPECTRUM_FLOOR_DB)
        } else {
            SPECTRUM_FLOOR_DB
        };
        magnitudes_db.push(db);
    }

    // Search non-DC bins for peak magnitude.
    let mut peak_bin = None;
    let mut best_mag = 1e-7_f32;
    for (k, &mag) in linear_mags.iter().enumerate().skip(1) {
        if mag > best_mag {
            best_mag = mag;
            peak_bin = Some(k);
        }
    }

    let (peak_freq_hz, peak_wavelength_samples) = match peak_bin {
        Some(k) => {
            let delta = if k >= 1 && k + 1 < num_bins {
                let y0 = magnitudes_db[k - 1] as f64;
                let y1 = magnitudes_db[k] as f64;
                let y2 = magnitudes_db[k + 1] as f64;
                let denom = y0 - 2.0 * y1 + y2;
                if denom.abs() > 1e-12 {
                    (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let freq = (k as f64 + delta) * bin_hz;
            if freq > 0.0 {
                (Some(freq), Some(sample_rate as f64 / freq))
            } else {
                (None, None)
            }
        }
        None => (None, None),
    };

    SpectrumData {
        sample_rate,
        fft_size,
        freqs_hz,
        magnitudes_db,
        peak_freq_hz,
        peak_wavelength_samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::SampleEncoding;

    #[test]
    fn wavelength_hz_and_midi_conversions() {
        let wl = hz_to_wavelength(440.0, 44100);
        assert!((wl - 100.2272727).abs() < 1e-5);
        assert!((wavelength_to_hz(wl, 44100) - 440.0).abs() < 1e-9);
        assert_eq!(wavelength_to_hz(0.0, 44100), 0.0);
        assert_eq!(hz_to_wavelength(-10.0, 44100), 0.0);

        assert!((midi_note_to_hz(69.0) - 440.0).abs() < 1e-9);
        assert!((midi_note_to_hz(57.0) - 220.0).abs() < 1e-9);

        let a4 = hz_to_nearest_note(440.0).unwrap();
        assert_eq!(a4.name, "A4");
        assert_eq!(a4.midi_note, 69);
        assert!(a4.cents_offset.abs() < 1e-9);

        let c4 = hz_to_nearest_note(midi_note_to_hz(60.0)).unwrap();
        assert_eq!(c4.name, "C4");
        assert_eq!(c4.midi_note, 60);
        assert!(hz_to_nearest_note(0.0).is_none());
    }

    #[test]
    fn generate_wave_all_shapes_and_single_cycle() {
        let shapes = [
            WaveShape::Sine,
            WaveShape::Square,
            WaveShape::Sawtooth,
            WaveShape::Triangle,
            WaveShape::Pulse { duty: 0.25 },
            WaveShape::Noise,
            WaveShape::Harmonics(vec![
                HarmonicPartial {
                    harmonic: 1,
                    amplitude: 0.7,
                    phase_rad: 0.0,
                },
                HarmonicPartial {
                    harmonic: 3,
                    amplitude: 0.3,
                    phase_rad: 0.5,
                },
            ]),
        ];

        for shape in shapes {
            let params = SynthParams {
                shape: shape.clone(),
                wavelength_samples: 64.0,
                amplitude: 0.8,
                phase_offset_rad: 0.0,
            };
            let wave = generate_wave(&params, 128);
            assert_eq!(wave.len(), 128);
            assert!(wave.iter().all(|&s| (-1.0..=1.0).contains(&s)));
            assert!(wave.iter().any(|&s| s.abs() > 0.01));

            let cycle = generate_single_cycle(&params, 32);
            assert_eq!(cycle.len(), 32);
        }

        // Noise is deterministic for identical params
        let noise_params = SynthParams {
            shape: WaveShape::Noise,
            wavelength_samples: 64.0,
            amplitude: 0.5,
            phase_offset_rad: 0.0,
        };
        assert_eq!(
            generate_wave(&noise_params, 32),
            generate_wave(&noise_params, 32)
        );
    }

    #[test]
    fn apply_synth_replace_and_mix() {
        let mut buf = AudioBuffer::from_channels(
            44100,
            vec![vec![0.25; 100], vec![0.25; 100]],
            SampleEncoding::Float32,
        )
        .unwrap();

        let params = SynthParams {
            shape: WaveShape::Square,
            wavelength_samples: 20.0,
            amplitude: 0.5,
            phase_offset_rad: 0.0,
        };

        // Channel mask 0b01 -> only channel 0 replaced
        apply_synth_to_buffer(&mut buf, 0..20, Some(0b01), &params, ApplyMode::Replace);
        assert!((buf.channels[0][0] - 0.5).abs() < 1e-6);
        assert!((buf.channels[0][10] - (-0.5)).abs() < 1e-6);
        assert!((buf.channels[1][0] - 0.25).abs() < 1e-6);

        // Mix onto channel 1
        apply_synth_to_buffer(&mut buf, 0..20, Some(0b10), &params, ApplyMode::Mix);
        assert!((buf.channels[1][0] - 0.75).abs() < 1e-6);
        assert!((buf.channels[1][10] - (-0.25)).abs() < 1e-6);
    }

    #[test]
    fn resample_region_shifts_length_and_pitch() {
        let sr = 44100;
        let current_wl = 100.0;
        let target_wl = 50.0; // Double frequency (octave up) -> half the length
        let params = SynthParams {
            shape: WaveShape::Sine,
            wavelength_samples: current_wl,
            amplitude: 0.9,
            phase_offset_rad: 0.0,
        };
        let wave = generate_wave(&params, 4000);
        let mut buf =
            AudioBuffer::from_channels(sr, vec![wave.clone(), wave], SampleEncoding::Float32)
                .unwrap();

        resample_region_to_wavelength(&mut buf, 0..4000, current_wl, target_wl);
        assert_eq!(buf.num_frames(), 2000);
        assert_eq!(buf.channels[0].len(), buf.channels[1].len());

        let spec = compute_spectrum(&buf.channels[0], sr, 2048);
        let expected_hz = wavelength_to_hz(target_wl, sr);
        let detected = spec.peak_freq_hz.unwrap();
        assert!(
            (detected - expected_hz).abs() < 3.0,
            "expected ~{expected_hz} Hz, got {detected} Hz"
        );
    }

    #[test]
    fn resample_caps_extreme_ratios_and_guards_invalid_inputs() {
        assert_eq!(
            capped_resampled_len(100, 1e-9, 1e9),
            Some(MAX_RESAMPLED_FRAMES)
        );
        assert_eq!(
            capped_resampled_len(100, 1.0, f64::MAX),
            Some(MAX_RESAMPLED_FRAMES)
        );
        assert_eq!(capped_resampled_len(100, 1e9, 1e-9), Some(1));
        assert_eq!(capped_resampled_len(0, 100.0, 50.0), None);
        assert_eq!(capped_resampled_len(100, 0.0, 50.0), None);
        assert_eq!(capped_resampled_len(100, 100.0, -5.0), None);
        assert_eq!(capped_resampled_len(100, f64::NAN, 50.0), None);
        assert_eq!(capped_resampled_len(100, 1e-308, 1e308), None);

        // Resampling a small region with a moderate ratio and invalid ratio leaves buffer valid
        let mut buf = AudioBuffer::from_channels(
            44100,
            vec![vec![0.0, 0.5, 1.0, 0.5]],
            SampleEncoding::Float32,
        )
        .unwrap();
        resample_region_to_wavelength(&mut buf, 0..4, 10.0, 20.0);
        assert_eq!(buf.num_frames(), 8);

        resample_region_to_wavelength(&mut buf, 0..4, 1e-308, 1e308);
        assert_eq!(buf.num_frames(), 8);
    }

    #[test]
    fn region_dsp_transforms() {
        // Gain dB & Normalize
        let mut buf = AudioBuffer::from_channels(
            44100,
            vec![vec![0.25, -0.5, 0.25, 0.0]],
            SampleEncoding::Float32,
        )
        .unwrap();
        apply_gain_db(&mut buf, 0..4, None, 6.0206); // ~2x
        assert!((buf.channels[0][1] - (-1.0)).abs() < 1e-3);

        normalize(&mut buf, 0..4, None, 0.5);
        assert!((buf.channels[0][1] - (-0.5)).abs() < 1e-6);

        // Fade in / Fade out
        let mut fade_buf =
            AudioBuffer::from_channels(44100, vec![vec![1.0; 5]], SampleEncoding::Float32).unwrap();
        fade_in(&mut fade_buf, 0..5, None);
        assert_eq!(fade_buf.channels[0], vec![0.0, 0.25, 0.5, 0.75, 1.0]);

        let mut fade_out_buf =
            AudioBuffer::from_channels(44100, vec![vec![1.0; 5]], SampleEncoding::Float32).unwrap();
        fade_out(&mut fade_out_buf, 0..5, None);
        assert_eq!(fade_out_buf.channels[0], vec![1.0, 0.75, 0.5, 0.25, 0.0]);

        // Reverse
        let mut rev_buf = AudioBuffer::from_channels(
            44100,
            vec![vec![0.1, 0.2, 0.3, 0.4]],
            SampleEncoding::Float32,
        )
        .unwrap();
        reverse(&mut rev_buf, 1..4, None);
        assert_eq!(rev_buf.channels[0], vec![0.1, 0.4, 0.3, 0.2]);

        // Remove DC offset
        let mut dc_buf = AudioBuffer::from_channels(
            44100,
            vec![vec![0.6, 0.8, 0.6, 0.8]],
            SampleEncoding::Float32,
        )
        .unwrap();
        remove_dc_offset(&mut dc_buf, 0..4, None);
        for (got, expected) in dc_buf.channels[0].iter().zip([-0.1, 0.1, -0.1, 0.1]) {
            assert!((got - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn compute_spectrum_detects_440hz_sine_accurately() {
        let sr = 44100;
        let expected_wl = hz_to_wavelength(440.0, sr); // ~100.227 samples
        let params = SynthParams {
            shape: WaveShape::Sine,
            wavelength_samples: expected_wl,
            amplitude: 1.0,
            phase_offset_rad: 0.0,
        };
        let samples = generate_wave(&params, 4096);
        let spec = compute_spectrum(&samples, sr, 4096);

        assert_eq!(spec.fft_size, 4096);
        assert_eq!(spec.freqs_hz.len(), 2049);
        assert_eq!(spec.magnitudes_db.len(), 2049);

        let peak_hz = spec.peak_freq_hz.expect("should detect peak frequency");
        let peak_wl = spec
            .peak_wavelength_samples
            .expect("should detect peak wavelength");

        assert!(
            (peak_hz - 440.0).abs() <= 2.0,
            "expected 440.0 +/- 2.0 Hz, got {peak_hz}"
        );
        assert!(
            (peak_wl - expected_wl).abs() <= 0.5,
            "expected {expected_wl} +/- 0.5 samples, got {peak_wl}"
        );
    }

    #[test]
    fn compute_spectrum_handles_silence() {
        let silence = vec![0.0_f32; 1024];
        let spec = compute_spectrum(&silence, 44100, 1024);
        assert!(spec.peak_freq_hz.is_none());
        assert!(spec.peak_wavelength_samples.is_none());
        assert!(spec.magnitudes_db.iter().all(|&db| db <= -120.0 + 1e-5));
    }
}
