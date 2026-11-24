use std::f64::consts::TAU;
use std::path::{Path, PathBuf};

use crate::buffer::{AudioBuffer, SampleEncoding};
use crate::dsp::{HarmonicPartial, SynthParams, WaveShape, generate_wave, hz_to_wavelength};
use crate::error::Result;
use crate::format::RawPcmConfig;
use crate::format::raw_pcm::write_raw_pcm_file;
use crate::format::wav::write_wav_file;

/// Generates deterministic demo audio fixtures in `dir` and returns their paths.
pub fn generate_sample_fixtures(dir: &Path) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let mut paths = Vec::with_capacity(6);

    // 1. sine_440hz.wav: 16-bit mono, 1.0s, 44100 Hz, pure A4 440 Hz sine
    {
        let sr = 44_100;
        let samples = generate_wave(
            &SynthParams {
                shape: WaveShape::Sine,
                wavelength_samples: hz_to_wavelength(440.0, sr),
                amplitude: 0.8,
                phase_offset_rad: 0.0,
            },
            sr as usize,
        );
        let buf = AudioBuffer::from_channels(sr, vec![samples], SampleEncoding::PcmI16)?;
        let path = dir.join("sine_440hz.wav");
        write_wav_file(&path, &buf, SampleEncoding::PcmI16)?;
        paths.push(path);
    }

    // 2. harmonics_chord.wav: 24-bit stereo, 1.5s, 44100 Hz, rich additive harmonic chord
    {
        let sr = 44_100;
        let frames = (sr as usize * 3) / 2;
        let left_partials = vec![
            HarmonicPartial {
                harmonic: 1,
                amplitude: 0.45,
                phase_rad: 0.0,
            },
            HarmonicPartial {
                harmonic: 2,
                amplitude: 0.22,
                phase_rad: 0.2,
            },
            HarmonicPartial {
                harmonic: 3,
                amplitude: 0.16,
                phase_rad: 0.0,
            },
            HarmonicPartial {
                harmonic: 5,
                amplitude: 0.10,
                phase_rad: 0.4,
            },
            HarmonicPartial {
                harmonic: 7,
                amplitude: 0.05,
                phase_rad: 0.1,
            },
        ];
        let right_partials = vec![
            HarmonicPartial {
                harmonic: 1,
                amplitude: 0.40,
                phase_rad: 0.3,
            },
            HarmonicPartial {
                harmonic: 3,
                amplitude: 0.24,
                phase_rad: 0.6,
            },
            HarmonicPartial {
                harmonic: 4,
                amplitude: 0.18,
                phase_rad: 0.1,
            },
            HarmonicPartial {
                harmonic: 6,
                amplitude: 0.11,
                phase_rad: 0.5,
            },
            HarmonicPartial {
                harmonic: 8,
                amplitude: 0.06,
                phase_rad: 0.2,
            },
        ];
        let left = generate_wave(
            &SynthParams {
                shape: WaveShape::Harmonics(left_partials),
                wavelength_samples: hz_to_wavelength(220.0, sr),
                amplitude: 0.85,
                phase_offset_rad: 0.0,
            },
            frames,
        );
        let right = generate_wave(
            &SynthParams {
                shape: WaveShape::Harmonics(right_partials),
                wavelength_samples: hz_to_wavelength(220.0, sr),
                amplitude: 0.85,
                phase_offset_rad: 0.25,
            },
            frames,
        );
        let buf = AudioBuffer::from_channels(sr, vec![left, right], SampleEncoding::PcmI24)?;
        let path = dir.join("harmonics_chord.wav");
        write_wav_file(&path, &buf, SampleEncoding::PcmI24)?;
        paths.push(path);
    }

    // 3. saw_sweep_float32.wav: 32-bit float mono, 1.0s, 44100 Hz, sawtooth frequency sweep
    {
        let sr = 44_100;
        let total_frames = sr as usize;
        let segments = 100usize;
        let seg_len = total_frames / segments;
        let mut sweep = Vec::with_capacity(total_frames);
        let mut phase_cycles = 0.0_f64;

        for seg in 0..segments {
            let len = if seg + 1 == segments {
                total_frames - sweep.len()
            } else {
                seg_len
            };
            let t = seg as f64 / (segments - 1) as f64;
            let freq = 110.0 * (8.0_f64).powf(t); // 110 Hz -> 880 Hz exponential sweep
            let wl = hz_to_wavelength(freq, sr);
            let block = generate_wave(
                &SynthParams {
                    shape: WaveShape::Sawtooth,
                    wavelength_samples: wl,
                    amplitude: 0.75,
                    phase_offset_rad: (phase_cycles * TAU) as f32,
                },
                len,
            );
            sweep.extend_from_slice(&block);
            phase_cycles = (phase_cycles + len as f64 / wl).rem_euclid(1.0);
        }

        let buf = AudioBuffer::from_channels(sr, vec![sweep], SampleEncoding::Float32)?;
        let path = dir.join("saw_sweep_float32.wav");
        write_wav_file(&path, &buf, SampleEncoding::Float32)?;
        paths.push(path);
    }

    // 4. pulse_beat_hit.wav: 16-bit stereo, 0.5s, 44100 Hz, percussive exponential-decay hit
    {
        let sr = 44_100;
        let frames = sr as usize / 2;
        let raw_l = generate_wave(
            &SynthParams {
                shape: WaveShape::Pulse { duty: 0.25 },
                wavelength_samples: hz_to_wavelength(90.0, sr),
                amplitude: 0.9,
                phase_offset_rad: 0.0,
            },
            frames,
        );
        let raw_r = generate_wave(
            &SynthParams {
                shape: WaveShape::Pulse { duty: 0.30 },
                wavelength_samples: hz_to_wavelength(90.5, sr),
                amplitude: 0.9,
                phase_offset_rad: 0.35,
            },
            frames,
        );
        let mut left = Vec::with_capacity(frames);
        let mut right = Vec::with_capacity(frames);
        for i in 0..frames {
            let t = i as f64 / sr as f64;
            let env = (-t * 14.0).exp() as f32;
            left.push((raw_l[i] * env).clamp(-1.0, 1.0));
            right.push((raw_r[i] * env).clamp(-1.0, 1.0));
        }
        let buf = AudioBuffer::from_channels(sr, vec![left, right], SampleEncoding::PcmI16)?;
        let path = dir.join("pulse_beat_hit.wav");
        write_wav_file(&path, &buf, SampleEncoding::PcmI16)?;
        paths.push(path);
    }

    // 5. triangle_8bit.wav: 8-bit unsigned mono, 0.75s, 22050 Hz
    {
        let sr = 22_050;
        let frames = (sr as usize * 3) / 4;
        let samples = generate_wave(
            &SynthParams {
                shape: WaveShape::Triangle,
                wavelength_samples: hz_to_wavelength(330.0, sr),
                amplitude: 0.85,
                phase_offset_rad: 0.0,
            },
            frames,
        );
        let buf = AudioBuffer::from_channels(sr, vec![samples], SampleEncoding::PcmU8)?;
        let path = dir.join("triangle_8bit.wav");
        write_wav_file(&path, &buf, SampleEncoding::PcmU8)?;
        paths.push(path);
    }

    // 6. raw_tri_16bit_le_mono_44100.pcm: raw 16-bit LE mono PCM, 0.5s, 44100 Hz
    {
        let sr = 44_100;
        let frames = sr as usize / 2;
        let samples = generate_wave(
            &SynthParams {
                shape: WaveShape::Triangle,
                wavelength_samples: hz_to_wavelength(261.625_565, sr),
                amplitude: 0.8,
                phase_offset_rad: 0.0,
            },
            frames,
        );
        let buf = AudioBuffer::from_channels(sr, vec![samples], SampleEncoding::PcmI16)?;
        let path = dir.join("raw_tri_16bit_le_mono_44100.pcm");
        write_raw_pcm_file(
            &path,
            &buf,
            RawPcmConfig {
                sample_rate: sr,
                num_channels: 1,
                encoding: SampleEncoding::PcmI16,
                big_endian: false,
            },
        )?;
        paths.push(path);
    }

    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{AudioFormat, inspect_file};

    #[test]
    fn generates_and_reinspects_all_sample_fixtures() {
        let dir =
            std::env::temp_dir().join(format!("wavecraft_sample_fixtures_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let paths = generate_sample_fixtures(&dir).unwrap();
        assert_eq!(paths.len(), 6);

        let expected_names = [
            "sine_440hz.wav",
            "harmonics_chord.wav",
            "saw_sweep_float32.wav",
            "pulse_beat_hit.wav",
            "triangle_8bit.wav",
            "raw_tri_16bit_le_mono_44100.pcm",
        ];

        for (path, expected_name) in paths.iter().zip(expected_names) {
            assert_eq!(path.file_name().unwrap().to_str().unwrap(), expected_name);
            assert!(path.exists());

            let inspected = inspect_file(path, None).unwrap();
            assert!(inspected.tree.validate_ranges(inspected.raw_bytes.len()));
            assert!(!inspected.buffer.is_empty());
            assert!(inspected.metadata.num_frames > 0);
            assert!(inspected.metadata.sample_rate > 0);
            assert!(inspected.metadata.duration_secs > 0.0);

            if expected_name.ends_with(".pcm") {
                assert_eq!(inspected.format, AudioFormat::RawPcm);
                assert_eq!(inspected.metadata.sample_rate, 44_100);
                assert_eq!(inspected.metadata.num_channels, 1);
            } else {
                assert_eq!(inspected.format, AudioFormat::Wav);
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
