use std::ops::Range;

use egui::Color32;
use wavecraft_core::EditableAudio;

/// Active tab in the bottom workbench dock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BottomTab {
    SpectrumFft,
    TunerSynth,
    RegionDsp,
}

impl BottomTab {
    pub const ALL: [Self; 3] = [Self::SpectrumFft, Self::TunerSynth, Self::RegionDsp];

    pub fn label(self) -> &'static str {
        match self {
            Self::SpectrumFft => "Spectrum / FFT",
            Self::TunerSynth => "Tuner & Harmonics",
            Self::RegionDsp => "Region DSP",
        }
    }
}

/// Persistent state for the bottom Spectrum FFT / Tuner / Region DSP dock.
#[derive(Debug, Clone)]
pub struct TunerState {
    pub active_tab: BottomTab,
    pub fft_size: usize,
    pub harmonic_amps: [f32; 16],
    pub wavelength_samples: f64,
    pub target_hz: f64,
    pub gain_db: f32,
    pub normalize_target: f32,
}

impl Default for TunerState {
    fn default() -> Self {
        let mut harmonic_amps = [0.0; 16];
        harmonic_amps[0] = 1.0;
        Self {
            active_tab: BottomTab::SpectrumFft,
            fft_size: 2048,
            harmonic_amps,
            wavelength_samples: 44_100.0 / 440.0,
            target_hz: 440.0,
            gain_db: 0.0,
            normalize_target: 0.95,
        }
    }
}

/// Renders the bottom Spectrum / Tuner / Region DSP panel scaffold.
/// Returns `true` if any action modified `editable`.
pub fn show_tuner_fft_panel(
    ui: &mut egui::Ui,
    editable: &mut EditableAudio,
    selection: &Option<Range<usize>>,
    active_channel: Option<usize>,
    state: &mut TunerState,
) -> bool {
    ui.horizontal(|ui| {
        for tab in BottomTab::ALL {
            if ui
                .selectable_label(state.active_tab == tab, tab.label())
                .clicked()
            {
                state.active_tab = tab;
            }
        }
        ui.separator();
        let ch_text =
            active_channel.map_or_else(|| "All Channels".to_string(), |c| format!("Ch {}", c + 1));
        let sel_text = selection.as_ref().map_or_else(
            || "Full Buffer".to_string(),
            |r| format!("Frames {}..{}", r.start, r.end),
        );
        ui.label(
            egui::RichText::new(format!("{ch_text}  •  {sel_text}"))
                .size(11.5)
                .color(Color32::from_rgb(148, 163, 184)),
        );
    });

    ui.separator();

    let buf = editable.buffer();
    match state.active_tab {
        BottomTab::SpectrumFft => {
            ui.label(
                egui::RichText::new(format!(
                    "FFT Analyzer ready ({}-point Hann window @ {} Hz).",
                    state.fft_size, buf.sample_rate
                ))
                .size(12.0)
                .color(Color32::from_rgb(186, 230, 253)),
            );
        }
        BottomTab::TunerSynth => {
            let active_harmonics = state.harmonic_amps.iter().filter(|&&a| a > 0.0).count();
            ui.label(
                egui::RichText::new(format!(
                    "Wavelength-locked Tuner & 16-Harmonic Synthesizer ({:.2} Hz • {:.2} samples/cycle • {active_harmonics} active partials).",
                    state.target_hz, state.wavelength_samples
                ))
                .size(12.0)
                .color(Color32::from_rgb(167, 243, 208)),
            );
        }
        BottomTab::RegionDsp => {
            ui.label(
                egui::RichText::new(format!(
                    "Region DSP ready (Gain {:.1} dB • Normalize peak {:.2}).",
                    state.gain_db, state.normalize_target
                ))
                .size(12.0)
                .color(Color32::from_rgb(253, 230, 138)),
            );
        }
    }

    false
}
