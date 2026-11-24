use std::f32::consts::TAU as TAU_F32;
use std::ops::Range;

use egui::epaint::{Mesh, Vertex};
use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Shape, Stroke, Vec2};
use wavecraft_core::dsp::{
    ApplyMode, HarmonicPartial, SpectrumData, SynthParams, WaveShape, apply_gain_db,
    apply_synth_to_buffer, compute_spectrum, fade_in, fade_out, generate_single_cycle,
    generate_wave, hz_to_nearest_note, hz_to_wavelength, normalize, remove_dc_offset,
    resample_region_to_wavelength, reverse, wavelength_to_hz,
};
use wavecraft_core::{AudioBuffer, EditableAudio};

const MIN_PLOT_FREQ_HZ: f64 = 20.0;
const MIN_PLOT_DB: f32 = -100.0;
const MAX_PLOT_DB: f32 = 0.0;
const FFT_SIZES: [usize; 4] = [512, 1024, 2048, 4096];
const FREQ_GRID_LINES: [f64; 9] = [
    50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 20_000.0,
];
const DB_GRID_LINES: [f32; 4] = [-20.0, -40.0, -60.0, -80.0];

/// Converts an optional single-channel index into a bitmask (`1 << ch`),
/// returning `Some(0)` if `ch >= usize::BITS as usize` so shifting never overflows.
pub(crate) fn channel_bitmask(active_channel: Option<usize>) -> Option<usize> {
    active_channel.map(|ch| {
        u32::try_from(ch)
            .ok()
            .and_then(|s| 1usize.checked_shl(s))
            .unwrap_or(0)
    })
}

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

/// Selectable waveform generator shape in the Tuner & Harmonics tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SynthShapeKind {
    Sine,
    Square,
    Sawtooth,
    Triangle,
    Pulse,
    Noise,
    Harmonics,
}

impl SynthShapeKind {
    pub const ALL: [Self; 7] = [
        Self::Sine,
        Self::Square,
        Self::Sawtooth,
        Self::Triangle,
        Self::Pulse,
        Self::Noise,
        Self::Harmonics,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Sine => "Sine",
            Self::Square => "Square",
            Self::Sawtooth => "Sawtooth",
            Self::Triangle => "Triangle",
            Self::Pulse => "Pulse",
            Self::Noise => "Noise",
            Self::Harmonics => "Harmonics",
        }
    }
}

/// Built-in 16-harmonic partial presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HarmonicPreset {
    Fundamental,
    Saw,
    Square,
    Organ,
    Clear,
}

/// Action emitted by [`show_tuner_fft_panel`].
#[derive(Debug, Clone, PartialEq)]
pub enum TunerAction {
    None,
    BufferEditedLive,
    BufferEdited,
    ReplaceEntireBuffer(AudioBuffer),
    StatusMessage(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SpectrumCacheKey {
    fft_size: usize,
    use_sel_or_playhead: bool,
    selection: Option<Range<usize>>,
    active_channel: Option<usize>,
    playhead_window_start: usize,
    num_frames: usize,
    sample_rate: u32,
}

/// Persistent state for the bottom Spectrum FFT / Tuner / Region DSP dock.
#[derive(Debug, Clone)]
pub struct TunerState {
    pub active_tab: BottomTab,
    pub fft_size: usize,
    pub fft_use_selection_or_playhead: bool,
    pub fft_dirty: bool,
    pub cached_spectrum: Option<SpectrumData>,
    cache_key: Option<SpectrumCacheKey>,
    pub wavelength_samples: f64,
    pub target_hz: f64,
    pub shape_kind: SynthShapeKind,
    pub pulse_duty: f32,
    pub harmonic_amps: [f32; 16],
    pub amplitude: f32,
    pub phase_deg: f32,
    pub apply_mode: ApplyMode,
    pub gain_db: f32,
    pub normalize_target: f32,
    pub live_apply_while_looping: bool,
    live_base: Option<AudioBuffer>,
    live_base_range: Option<Range<usize>>,
    live_base_channel: Option<usize>,
}

impl Default for TunerState {
    fn default() -> Self {
        Self {
            active_tab: BottomTab::SpectrumFft,
            fft_size: 2048,
            fft_use_selection_or_playhead: true,
            fft_dirty: true,
            cached_spectrum: None,
            cache_key: None,
            wavelength_samples: 44_100.0 / 440.0,
            target_hz: 440.0,
            shape_kind: SynthShapeKind::Sine,
            pulse_duty: 0.5,
            harmonic_amps: harmonic_preset_amps(HarmonicPreset::Fundamental),
            amplitude: 0.8,
            phase_deg: 0.0,
            apply_mode: ApplyMode::Replace,
            gain_db: 0.0,
            normalize_target: 0.95,
            live_apply_while_looping: true,
            live_base: None,
            live_base_range: None,
            live_base_channel: None,
        }
    }
}

impl TunerState {
    /// Clears the active live-edit base snapshot so the next live edit starts a fresh undo group.
    pub fn reset_live_session(&mut self) {
        self.live_base = None;
        self.live_base_range = None;
        self.live_base_channel = None;
    }

    /// Builds [`SynthParams`] from the current tuner controls.
    pub fn synth_params(&self) -> SynthParams {
        let shape = match self.shape_kind {
            SynthShapeKind::Sine => WaveShape::Sine,
            SynthShapeKind::Square => WaveShape::Square,
            SynthShapeKind::Sawtooth => WaveShape::Sawtooth,
            SynthShapeKind::Triangle => WaveShape::Triangle,
            SynthShapeKind::Pulse => WaveShape::Pulse {
                duty: self.pulse_duty.clamp(0.05, 0.95),
            },
            SynthShapeKind::Noise => WaveShape::Noise,
            SynthShapeKind::Harmonics => {
                let partials = self
                    .harmonic_amps
                    .iter()
                    .enumerate()
                    .filter(|(_, amp)| **amp > 0.0)
                    .map(|(idx, &amp)| HarmonicPartial {
                        harmonic: idx + 1,
                        amplitude: amp,
                        phase_rad: 0.0,
                    })
                    .collect();
                WaveShape::Harmonics(partials)
            }
        };
        SynthParams {
            shape,
            wavelength_samples: self.wavelength_samples,
            amplitude: self.amplitude,
            phase_offset_rad: (self.phase_deg / 360.0) * TAU_F32,
        }
    }
}

/// Builds 16-partial amplitude weights for the given [`HarmonicPreset`].
pub fn harmonic_preset_amps(preset: HarmonicPreset) -> [f32; 16] {
    let mut amps = [0.0_f32; 16];
    match preset {
        HarmonicPreset::Fundamental => {
            amps[0] = 1.0;
        }
        HarmonicPreset::Saw => {
            for (i, slot) in amps.iter_mut().enumerate() {
                *slot = 1.0 / (i + 1) as f32;
            }
        }
        HarmonicPreset::Square => {
            for (i, slot) in amps.iter_mut().enumerate() {
                if i % 2 == 0 {
                    *slot = 1.0 / (i + 1) as f32;
                }
            }
        }
        HarmonicPreset::Organ => {
            let weights = [(0, 1.0), (1, 0.8), (2, 0.6), (3, 0.5), (5, 0.35), (7, 0.25)];
            for (idx, val) in weights {
                amps[idx] = val;
            }
        }
        HarmonicPreset::Clear => {}
    }
    amps
}

/// Synchronizes `(wavelength_samples, target_hz)` from an edited wavelength value.
pub fn sync_from_wavelength(wavelength_samples: f64, sample_rate: u32) -> (f64, f64) {
    let max_wl = (sample_rate.max(2) as f64).max(2.0);
    let wl = wavelength_samples.clamp(2.0, max_wl);
    let hz = wavelength_to_hz(wl, sample_rate.max(1));
    (wl, hz)
}

/// Synchronizes `(wavelength_samples, target_hz)` from an edited frequency value.
pub fn sync_from_hz(hz: f64, sample_rate: u32) -> (f64, f64) {
    let max_hz = (sample_rate.max(2) as f64 / 2.0).max(1.0);
    let clamped_hz = hz.clamp(1.0, max_hz);
    let wl = hz_to_wavelength(clamped_hz, sample_rate.max(1));
    (wl, clamped_hz)
}

/// Maps a frequency in `[20 Hz, nyquist_hz]` onto a logarithmic X coordinate inside `[left, left + width]`.
pub fn freq_to_log_x(freq_hz: f64, nyquist_hz: f64, left: f32, width: f32) -> f32 {
    let max_hz = nyquist_hz.max(MIN_PLOT_FREQ_HZ + 1.0);
    let clamped = freq_hz.clamp(MIN_PLOT_FREQ_HZ, max_hz);
    let min_ln = MIN_PLOT_FREQ_HZ.ln();
    let max_ln = max_hz.ln();
    let norm = ((clamped.ln() - min_ln) / (max_ln - min_ln)).clamp(0.0, 1.0);
    left + (norm as f32) * width
}

/// Inverse of [`freq_to_log_x`]: converts an X coordinate back to frequency in Hz.
pub fn log_x_to_freq(x: f32, nyquist_hz: f64, left: f32, width: f32) -> f64 {
    let max_hz = nyquist_hz.max(MIN_PLOT_FREQ_HZ + 1.0);
    let norm = ((x - left) / width.max(1.0)).clamp(0.0, 1.0) as f64;
    let min_ln = MIN_PLOT_FREQ_HZ.ln();
    let max_ln = max_hz.ln();
    (min_ln + norm * (max_ln - min_ln)).exp()
}

/// Maps a magnitude in `[-100, 0] dB` to a Y pixel coordinate inside `[top, top + height]`.
pub fn db_to_y(db: f32, top: f32, height: f32) -> f32 {
    let t = ((db.clamp(MIN_PLOT_DB, MAX_PLOT_DB) - MIN_PLOT_DB) / (MAX_PLOT_DB - MIN_PLOT_DB))
        .clamp(0.0, 1.0);
    top + (1.0 - t) * height
}

fn format_freq_label(hz: f64) -> String {
    if hz >= 1000.0 {
        format!("{:.0}k", hz / 1000.0)
    } else {
        format!("{hz:.0}")
    }
}

fn format_note_summary(hz: f64) -> String {
    match hz_to_nearest_note(hz) {
        Some(n) => format!("{} ({:+.1}¢)", n.name, n.cents_offset),
        None => "—".to_string(),
    }
}

fn resolve_fft_window(
    selection: &Option<Range<usize>>,
    playhead_frame: usize,
    fft_size: usize,
    num_frames: usize,
    use_sel_or_playhead: bool,
) -> Range<usize> {
    if num_frames == 0 {
        return 0..0;
    }
    if use_sel_or_playhead {
        if let Some(sel) = selection.as_ref()
            && sel.start < sel.end
        {
            let start = sel.start.min(num_frames);
            let end = sel.end.min(num_frames);
            if start < end {
                return start..end;
            }
        }
        let half = fft_size / 2;
        let center = playhead_frame.min(num_frames);
        let mut start = center.saturating_sub(half);
        let mut end = start.saturating_add(fft_size);
        if end > num_frames {
            end = num_frames;
            start = end.saturating_sub(fft_size);
        }
        start..end
    } else {
        0..fft_size.min(num_frames)
    }
}

fn extract_fft_samples(
    buf: &AudioBuffer,
    range: Range<usize>,
    active_channel: Option<usize>,
) -> Vec<f32> {
    if range.is_empty() || buf.is_empty() {
        return Vec::new();
    }
    match active_channel {
        Some(ch) if ch < buf.num_channels() => buf.channels[ch][range].to_vec(),
        _ => buf.mono_mix_slice(range),
    }
}

fn ensure_spectrum_cached(
    buf: &AudioBuffer,
    selection: &Option<Range<usize>>,
    active_channel: Option<usize>,
    playhead_frame: usize,
    state: &mut TunerState,
) {
    let window = resolve_fft_window(
        selection,
        playhead_frame,
        state.fft_size,
        buf.num_frames(),
        state.fft_use_selection_or_playhead,
    );
    let key = SpectrumCacheKey {
        fft_size: state.fft_size,
        use_sel_or_playhead: state.fft_use_selection_or_playhead,
        selection: selection.clone(),
        active_channel,
        playhead_window_start: window.start,
        num_frames: buf.num_frames(),
        sample_rate: buf.sample_rate,
    };

    if !state.fft_dirty && state.cache_key.as_ref() == Some(&key) && state.cached_spectrum.is_some()
    {
        return;
    }

    let samples = extract_fft_samples(buf, window, active_channel);
    state.cached_spectrum = Some(compute_spectrum(&samples, buf.sample_rate, state.fft_size));
    state.cache_key = Some(key);
    state.fft_dirty = false;
}

/// Renders the bottom Spectrum / Tuner / Region DSP panel.
pub fn show_tuner_fft_panel(
    ui: &mut egui::Ui,
    editable: &mut EditableAudio,
    selection: &mut Option<Range<usize>>,
    active_channel: Option<usize>,
    playhead_frame: usize,
    is_looping_and_playing: bool,
    state: &mut TunerState,
) -> TunerAction {
    let mut action = TunerAction::None;
    if !is_looping_and_playing || !state.live_apply_while_looping {
        state.reset_live_session();
    }

    // Header tab strip
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
            |r| format!("Frames {}..{} ({})", r.start, r.end, r.len()),
        );
        ui.label(
            egui::RichText::new(format!("{ch_text}  •  {sel_text}"))
                .size(11.5)
                .color(Color32::from_rgb(148, 163, 184)),
        );
    });

    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| match state.active_tab {
            BottomTab::SpectrumFft => {
                render_spectrum_tab(
                    ui,
                    editable,
                    selection,
                    active_channel,
                    playhead_frame,
                    state,
                );
            }
            BottomTab::TunerSynth => {
                action = render_tuner_tab(
                    ui,
                    editable,
                    selection,
                    active_channel,
                    is_looping_and_playing,
                    state,
                );
            }
            BottomTab::RegionDsp => {
                action = render_region_dsp_tab(
                    ui,
                    editable,
                    selection,
                    active_channel,
                    playhead_frame,
                    state,
                );
            }
        });

    action
}

fn render_spectrum_tab(
    ui: &mut egui::Ui,
    editable: &EditableAudio,
    selection: &Option<Range<usize>>,
    active_channel: Option<usize>,
    playhead_frame: usize,
    state: &mut TunerState,
) {
    let buf = editable.buffer();
    ensure_spectrum_cached(buf, selection, active_channel, playhead_frame, state);

    let sr = buf.sample_rate.max(1);
    let nyquist = (sr as f64 / 2.0).max(MIN_PLOT_FREQ_HZ + 1.0);

    // Controls + Peak summary row
    ui.horizontal_wrapped(|ui| {
        ui.label("FFT:");
        for sz in FFT_SIZES {
            if ui
                .selectable_label(state.fft_size == sz, format!("{sz}"))
                .clicked()
            {
                state.fft_size = sz;
                state.fft_dirty = true;
            }
        }
        ui.separator();
        if ui
            .checkbox(
                &mut state.fft_use_selection_or_playhead,
                "Selection or Playhead Window",
            )
            .changed()
        {
            state.fft_dirty = true;
        }

        ui.separator();
        if let Some(spec) = state.cached_spectrum.as_ref() {
            if let (Some(pk_hz), Some(pk_wl)) = (spec.peak_freq_hz, spec.peak_wavelength_samples) {
                let note = format_note_summary(pk_hz);
                ui.label(
                    egui::RichText::new(format!(
                        "Peak: {pk_hz:.2} Hz ({note}) | λ = {pk_wl:.2} samples"
                    ))
                    .monospace()
                    .size(11.5)
                    .color(Color32::from_rgb(125, 211, 252)),
                );
                if ui.button("Lock Tuner to Peak λ").clicked() {
                    let (wl, hz) = sync_from_wavelength(pk_wl, sr);
                    state.wavelength_samples = wl;
                    state.target_hz = hz;
                }
            } else {
                ui.label(
                    egui::RichText::new("Peak: Silence / No dominant peak")
                        .size(11.5)
                        .color(Color32::from_rgb(148, 163, 184)),
                );
            }
        }
    });

    ui.add_space(4.0);

    // Spectrum plot canvas
    let plot_w = ui.available_width().max(200.0);
    let plot_h = ui.available_height().clamp(110.0, 220.0);
    let (plot_rect, response) = ui.allocate_exact_size(Vec2::new(plot_w, plot_h), Sense::hover());
    let painter = ui.painter_at(plot_rect);

    painter.rect_filled(
        plot_rect,
        CornerRadius::same(4),
        Color32::from_rgb(13, 17, 24),
    );
    painter.rect_stroke(
        plot_rect,
        CornerRadius::same(4),
        Stroke::new(1.0, Color32::from_rgb(38, 45, 56)),
        egui::StrokeKind::Inside,
    );

    let inner = plot_rect.shrink2(Vec2::new(34.0, 14.0));
    if inner.width() <= 10.0 || inner.height() <= 10.0 {
        return;
    }

    // Frequency grid lines
    for &freq in &FREQ_GRID_LINES {
        if freq >= nyquist {
            continue;
        }
        let x = freq_to_log_x(freq, nyquist, inner.left(), inner.width());
        painter.line_segment(
            [Pos2::new(x, inner.top()), Pos2::new(x, inner.bottom())],
            Stroke::new(1.0, Color32::from_rgb(28, 35, 46)),
        );
        painter.text(
            Pos2::new(x, inner.bottom() + 2.0),
            Align2::CENTER_TOP,
            format_freq_label(freq),
            FontId::monospace(9.5),
            Color32::from_rgb(100, 116, 139),
        );
    }

    // dB grid lines
    for &db in &DB_GRID_LINES {
        let y = db_to_y(db, inner.top(), inner.height());
        painter.line_segment(
            [Pos2::new(inner.left(), y), Pos2::new(inner.right(), y)],
            Stroke::new(1.0, Color32::from_rgb(28, 35, 46)),
        );
        painter.text(
            Pos2::new(inner.left() - 4.0, y),
            Align2::RIGHT_CENTER,
            format!("{db:.0}"),
            FontId::monospace(9.5),
            Color32::from_rgb(100, 116, 139),
        );
    }

    if let Some(spec) = state.cached_spectrum.as_ref() {
        let mut points: Vec<Pos2> = Vec::new();
        for (&freq, &db) in spec.freqs_hz.iter().zip(spec.magnitudes_db.iter()) {
            let f = freq as f64;
            if f < MIN_PLOT_FREQ_HZ || f > nyquist {
                continue;
            }
            let x = freq_to_log_x(f, nyquist, inner.left(), inner.width());
            let y = db_to_y(db, inner.top(), inner.height());
            points.push(Pos2::new(x, y));
        }

        // Translucent fill mesh under curve
        if points.len() >= 2 {
            let fill_color = Color32::from_rgba_unmultiplied(56, 189, 248, 45);
            let mut mesh = Mesh::default();
            let base_y = inner.bottom();
            for win in points.windows(2) {
                let p0 = win[0];
                let p1 = win[1];
                let idx = mesh.vertices.len() as u32;
                mesh.vertices.push(Vertex {
                    pos: p0,
                    uv: Pos2::ZERO,
                    color: fill_color,
                });
                mesh.vertices.push(Vertex {
                    pos: p1,
                    uv: Pos2::ZERO,
                    color: fill_color,
                });
                mesh.vertices.push(Vertex {
                    pos: Pos2::new(p1.x, base_y),
                    uv: Pos2::ZERO,
                    color: fill_color,
                });
                mesh.vertices.push(Vertex {
                    pos: Pos2::new(p0.x, base_y),
                    uv: Pos2::ZERO,
                    color: fill_color,
                });
                mesh.indices
                    .extend_from_slice(&[idx, idx + 1, idx + 2, idx, idx + 2, idx + 3]);
            }
            painter.add(Shape::mesh(mesh));
            painter.add(Shape::line(
                points,
                Stroke::new(1.5, Color32::from_rgb(56, 189, 248)),
            ));
        }

        // Peak marker
        if let Some(pk_hz) = spec.peak_freq_hz
            && pk_hz >= MIN_PLOT_FREQ_HZ
            && pk_hz <= nyquist
        {
            let pk_x = freq_to_log_x(pk_hz, nyquist, inner.left(), inner.width());
            let bin_idx = spec
                .freqs_hz
                .partition_point(|&f| (f as f64) < pk_hz)
                .min(spec.magnitudes_db.len().saturating_sub(1));
            let pk_db = spec.magnitudes_db.get(bin_idx).copied().unwrap_or(-60.0);
            let pk_y = db_to_y(pk_db, inner.top(), inner.height());
            let pk_color = Color32::from_rgb(245, 158, 11);
            painter.line_segment(
                [
                    Pos2::new(pk_x, inner.top()),
                    Pos2::new(pk_x, inner.bottom()),
                ],
                Stroke::new(1.0, Color32::from_rgba_unmultiplied(245, 158, 11, 140)),
            );
            painter.circle_filled(Pos2::new(pk_x, pk_y), 3.5, pk_color);
        }

        // Hover readout
        if response.hovered()
            && let Some(pos) = response.hover_pos()
            && inner.contains(pos)
        {
            let hover_hz = log_x_to_freq(pos.x, nyquist, inner.left(), inner.width());
            let bin_idx = spec
                .freqs_hz
                .partition_point(|&f| (f as f64) < hover_hz)
                .min(spec.magnitudes_db.len().saturating_sub(1));
            let hover_db = spec
                .magnitudes_db
                .get(bin_idx)
                .copied()
                .unwrap_or(MIN_PLOT_DB);
            let note = format_note_summary(hover_hz);
            painter.line_segment(
                [
                    Pos2::new(pos.x, inner.top()),
                    Pos2::new(pos.x, inner.bottom()),
                ],
                Stroke::new(1.0, Color32::from_rgba_unmultiplied(226, 232, 240, 90)),
            );
            painter.text(
                Pos2::new(inner.right() - 6.0, inner.top() + 4.0),
                Align2::RIGHT_TOP,
                format!("{hover_hz:.1} Hz ({note})  |  {hover_db:.1} dB"),
                FontId::monospace(11.0),
                Color32::from_rgb(226, 232, 240),
            );
        }
    }
}

fn resolve_tuner_range(
    num_frames: usize,
    selection: Option<&Range<usize>>,
) -> Option<Range<usize>> {
    if num_frames == 0 {
        return None;
    }
    Some(
        selection
            .and_then(|s| {
                let lo = s.start.min(num_frames);
                let hi = s.end.min(num_frames);
                (lo < hi).then_some(lo..hi)
            })
            .unwrap_or(0..num_frames),
    )
}

pub(crate) fn apply_live_tuner_edit(
    editable: &mut EditableAudio,
    selection: &Option<Range<usize>>,
    active_channel: Option<usize>,
    state: &mut TunerState,
) -> bool {
    let Some(range) = resolve_tuner_range(editable.buffer().num_frames(), selection.as_ref())
    else {
        return false;
    };

    if state.live_base.is_none()
        || state.live_base_range.as_ref() != Some(&range)
        || state.live_base_channel != active_channel
    {
        state.live_base = Some(editable.buffer().clone());
        state.live_base_range = Some(range.clone());
        state.live_base_channel = active_channel;
        editable.push_undo_snapshot();
    }

    let mut base = state
        .live_base
        .as_ref()
        .expect("live_base initialized above")
        .clone();
    let channel_mask = channel_bitmask(active_channel);
    let params = state.synth_params();
    let mode = state.apply_mode;
    apply_synth_to_buffer(&mut base, range, channel_mask, &params, mode);
    editable.set_buffer_raw(base);
    state.fft_dirty = true;
    true
}

pub(crate) fn commit_tuner_apply(
    editable: &mut EditableAudio,
    selection: Option<&Range<usize>>,
    active_channel: Option<usize>,
    state: &mut TunerState,
) -> bool {
    let Some(range) = resolve_tuner_range(editable.buffer().num_frames(), selection) else {
        return false;
    };
    let channel_mask = channel_bitmask(active_channel);
    let params = state.synth_params();
    let mode = state.apply_mode;

    if state.live_base.is_some()
        && state.live_base_range.as_ref() == Some(&range)
        && state.live_base_channel == active_channel
    {
        let mut working = state.live_base.take().expect("live_base checked Some");
        apply_synth_to_buffer(&mut working, range, channel_mask, &params, mode);
        editable.set_buffer_raw(working);
        state.reset_live_session();
    } else {
        state.reset_live_session();
        editable.apply_transform(|b| {
            apply_synth_to_buffer(b, range, channel_mask, &params, mode);
        });
    }

    state.fft_dirty = true;
    true
}

fn render_tuner_tab(
    ui: &mut egui::Ui,
    editable: &mut EditableAudio,
    selection: &mut Option<Range<usize>>,
    active_channel: Option<usize>,
    is_looping_and_playing: bool,
    state: &mut TunerState,
) -> TunerAction {
    let mut action = TunerAction::None;
    let mut params_changed = false;
    let sr = editable.buffer().sample_rate.max(2);
    let max_wl = sr as f64;
    let max_hz = (sr as f64 / 2.0).max(1.0);

    // Row 1: Linked Wavelength <-> Hz <-> Note tuner
    ui.horizontal_wrapped(|ui| {
        ui.label("λ (samples):");
        let mut wl = state.wavelength_samples.clamp(2.0, max_wl);
        if ui
            .add(
                egui::DragValue::new(&mut wl)
                    .range(2.0..=max_wl)
                    .speed(0.25)
                    .max_decimals(2),
            )
            .changed()
        {
            let (new_wl, new_hz) = sync_from_wavelength(wl, sr);
            state.wavelength_samples = new_wl;
            state.target_hz = new_hz;
            params_changed = true;
        }

        ui.label("Freq (Hz):");
        let mut hz = state.target_hz.clamp(1.0, max_hz);
        if ui
            .add(
                egui::DragValue::new(&mut hz)
                    .range(1.0..=max_hz)
                    .speed(0.5)
                    .max_decimals(2)
                    .suffix(" Hz"),
            )
            .changed()
        {
            let (new_wl, new_hz) = sync_from_hz(hz, sr);
            state.wavelength_samples = new_wl;
            state.target_hz = new_hz;
            params_changed = true;
        }

        ui.separator();
        ui.label(
            egui::RichText::new(format!("Note: {}", format_note_summary(state.target_hz)))
                .monospace()
                .strong()
                .color(Color32::from_rgb(167, 243, 208)),
        );

        let shift_steps = [
            ("-1 Oct", 0.5_f64),
            ("-1 Semi", 2.0_f64.powf(-1.0 / 12.0)),
            ("+1 Semi", 2.0_f64.powf(1.0 / 12.0)),
            ("+1 Oct", 2.0_f64),
        ];
        for (lbl, mult) in shift_steps {
            if ui.small_button(lbl).clicked() {
                let (new_wl, new_hz) = sync_from_hz(state.target_hz * mult, sr);
                state.wavelength_samples = new_wl;
                state.target_hz = new_hz;
                params_changed = true;
            }
        }
        if ui.small_button("A4 (440 Hz)").clicked() {
            let (new_wl, new_hz) = sync_from_hz(440.0, sr);
            state.wavelength_samples = new_wl;
            state.target_hz = new_hz;
            params_changed = true;
        }
    });

    ui.add_space(4.0);

    // Row 2: Shape selector + Amp / Phase / Mode
    ui.horizontal_wrapped(|ui| {
        ui.label("Shape:");
        for kind in SynthShapeKind::ALL {
            if ui
                .selectable_label(state.shape_kind == kind, kind.label())
                .clicked()
            {
                state.shape_kind = kind;
                params_changed = true;
            }
        }

        if state.shape_kind == SynthShapeKind::Pulse {
            ui.separator();
            ui.label("Duty");
            if ui
                .add(egui::Slider::new(&mut state.pulse_duty, 0.05..=0.95))
                .changed()
            {
                params_changed = true;
            }
        }

        ui.separator();
        ui.label("Amp");
        if ui
            .add(egui::Slider::new(&mut state.amplitude, 0.0..=1.0))
            .changed()
        {
            params_changed = true;
        }
        ui.label("Phase");
        if ui
            .add(egui::Slider::new(&mut state.phase_deg, 0.0..=360.0).suffix("°"))
            .changed()
        {
            params_changed = true;
        }

        ui.separator();
        let prev_mode = state.apply_mode;
        ui.selectable_value(&mut state.apply_mode, ApplyMode::Replace, "Replace");
        ui.selectable_value(&mut state.apply_mode, ApplyMode::Mix, "Mix");
        if state.apply_mode != prev_mode {
            params_changed = true;
        }
    });

    ui.add_space(4.0);

    // Row 3: Interactive 16-Harmonic Partial Bar Editor + Live Single-Cycle Preview
    let avail_w = ui.available_width().max(260.0);
    let bars_w = (avail_w * 0.62).max(180.0);
    let preview_w = (avail_w - bars_w - 16.0).max(100.0);

    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("16-Harmonic Partials (H1..H16)").size(11.5));
                for (lbl, preset) in [
                    ("Fund", HarmonicPreset::Fundamental),
                    ("Saw", HarmonicPreset::Saw),
                    ("Square", HarmonicPreset::Square),
                    ("Organ", HarmonicPreset::Organ),
                    ("Clear", HarmonicPreset::Clear),
                ] {
                    if ui.small_button(lbl).clicked() {
                        state.harmonic_amps = harmonic_preset_amps(preset);
                        state.shape_kind = SynthShapeKind::Harmonics;
                        params_changed = true;
                    }
                }
            });

            let (bar_rect, bar_resp) =
                ui.allocate_exact_size(Vec2::new(bars_w, 72.0), Sense::click_and_drag());
            let bp = ui.painter_at(bar_rect);
            bp.rect_filled(
                bar_rect,
                CornerRadius::same(4),
                Color32::from_rgb(13, 17, 24),
            );
            bp.rect_stroke(
                bar_rect,
                CornerRadius::same(4),
                Stroke::new(1.0, Color32::from_rgb(38, 45, 56)),
                egui::StrokeKind::Inside,
            );

            let inner = bar_rect.shrink2(Vec2::new(6.0, 6.0));
            let col_w = inner.width() / 16.0;

            if (bar_resp.clicked() || bar_resp.dragged())
                && let Some(pos) = bar_resp.interact_pointer_pos()
                && inner.width() > 1.0
                && inner.height() > 1.0
            {
                let idx = (((pos.x - inner.left()) / col_w).floor() as isize).clamp(0, 15) as usize;
                let amp = ((inner.bottom() - pos.y) / inner.height()).clamp(0.0, 1.0);
                if (state.harmonic_amps[idx] - amp).abs() > 1e-6
                    || state.shape_kind != SynthShapeKind::Harmonics
                {
                    state.harmonic_amps[idx] = amp;
                    state.shape_kind = SynthShapeKind::Harmonics;
                    params_changed = true;
                }
            }

            for (i, &amp) in state.harmonic_amps.iter().enumerate() {
                let x0 = inner.left() + i as f32 * col_w + 2.0;
                let x1 = inner.left() + (i + 1) as f32 * col_w - 2.0;
                let h = amp.clamp(0.0, 1.0) * (inner.height() - 12.0);
                let top_y = (inner.bottom() - 12.0) - h;
                let r = Rect::from_min_max(
                    Pos2::new(x0, top_y),
                    Pos2::new(x1.max(x0 + 2.0), inner.bottom() - 12.0),
                );
                let bar_color = if state.shape_kind == SynthShapeKind::Harmonics {
                    Color32::from_rgb(45, 212, 191)
                } else {
                    Color32::from_rgb(71, 85, 105)
                };
                bp.rect_filled(r, CornerRadius::same(2), bar_color);
                bp.text(
                    Pos2::new((x0 + x1) * 0.5, inner.bottom()),
                    Align2::CENTER_BOTTOM,
                    format!("{}", i + 1),
                    FontId::monospace(8.5),
                    Color32::from_rgb(148, 163, 184),
                );
            }
        });

        ui.vertical(|ui| {
            ui.label(egui::RichText::new("Single-Cycle Preview").size(11.5));
            let (prev_rect, _) = ui.allocate_exact_size(Vec2::new(preview_w, 72.0), Sense::hover());
            let pp = ui.painter_at(prev_rect);
            pp.rect_filled(
                prev_rect,
                CornerRadius::same(4),
                Color32::from_rgb(13, 17, 24),
            );
            pp.rect_stroke(
                prev_rect,
                CornerRadius::same(4),
                Stroke::new(1.0, Color32::from_rgb(38, 45, 56)),
                egui::StrokeKind::Inside,
            );

            let inner = prev_rect.shrink2(Vec2::new(6.0, 6.0));
            let mid_y = inner.center().y;
            pp.line_segment(
                [
                    Pos2::new(inner.left(), mid_y),
                    Pos2::new(inner.right(), mid_y),
                ],
                Stroke::new(1.0, Color32::from_rgb(40, 48, 61)),
            );

            let params = state.synth_params();
            let cycle = generate_single_cycle(&params, 128);
            let n = cycle.len().max(1);
            let half_h = inner.height() * 0.45;
            let pts: Vec<Pos2> = cycle
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let x = inner.left() + (i as f32 / (n - 1).max(1) as f32) * inner.width();
                    let y = mid_y - s.clamp(-1.0, 1.0) * half_h;
                    Pos2::new(x, y)
                })
                .collect();
            if pts.len() >= 2 {
                pp.add(Shape::line(
                    pts,
                    Stroke::new(1.5, Color32::from_rgb(52, 211, 153)),
                ));
            }
        });
    });

    ui.add_space(4.0);

    // Row 4: Synthesis & Stretch action buttons
    let num_frames = editable.buffer().num_frames();
    let has_sel = selection.as_ref().is_some_and(|s| s.start < s.end);
    ui.horizontal_wrapped(|ui| {
        if ui
            .add_enabled(num_frames > 0, egui::Button::new("Apply to Selection"))
            .clicked()
            && commit_tuner_apply(editable, selection.as_ref(), active_channel, state)
        {
            action = TunerAction::BufferEdited;
        }

        ui.checkbox(
            &mut state.live_apply_while_looping,
            "Live apply while looping",
        )
        .on_hover_text(
            "Immediately apply wavelength & harmonic changes to the selection while Loop playback is active (grouped as one Undo step)",
        );

        if ui.button("Generate New 1s Buffer").clicked() {
            state.reset_live_session();
            let cur_buf = editable.buffer();
            let gen_sr = cur_buf.sample_rate.max(1);
            let ch_count = cur_buf.num_channels().max(1);
            let enc = cur_buf.source_encoding;
            let params = state.synth_params();
            let wave = generate_wave(&params, gen_sr as usize);
            let new_buf = AudioBuffer::from_channels(gen_sr, vec![wave; ch_count], enc)
                .unwrap_or_else(|_| AudioBuffer::new_silent(44_100, 1, 44_100));
            state.fft_dirty = true;
            action = TunerAction::ReplaceEntireBuffer(new_buf);
        }

        if ui
            .add_enabled(
                has_sel && num_frames > 0,
                egui::Button::new("Stretch Selection to λ"),
            )
            .clicked()
            && let Some(sel) = selection.clone()
        {
            state.reset_live_session();
            let start = sel.start.min(num_frames);
            let end = sel.end.min(num_frames);
            if start < end {
                let clamped = start..end;
                let samples =
                    extract_fft_samples(editable.buffer(), clamped.clone(), active_channel);
                let spec = compute_spectrum(&samples, sr, state.fft_size);
                let current_wl = spec
                    .peak_wavelength_samples
                    .filter(|&w| w > 0.0 && w.is_finite())
                    .unwrap_or(clamped.len() as f64);
                let target_wl = state.wavelength_samples.max(1.0);
                let total_before = num_frames;
                let sel_len = clamped.len();
                editable.apply_transform(|b| {
                    resample_region_to_wavelength(b, clamped.clone(), current_wl, target_wl);
                });
                let total_after = editable.buffer().num_frames();
                let actual_new_len =
                    total_after.saturating_sub(total_before.saturating_sub(sel_len));
                let new_end = start.saturating_add(actual_new_len).min(total_after);
                *selection = if start < new_end {
                    Some(start..new_end)
                } else {
                    None
                };
                state.fft_dirty = true;
                action = TunerAction::BufferEdited;
            }
        }
    });

    if !is_looping_and_playing || !state.live_apply_while_looping {
        state.reset_live_session();
    } else if params_changed
        && matches!(action, TunerAction::None)
        && apply_live_tuner_edit(editable, selection, active_channel, state)
    {
        action = TunerAction::BufferEditedLive;
    }

    action
}

fn render_region_dsp_tab(
    ui: &mut egui::Ui,
    editable: &mut EditableAudio,
    selection: &mut Option<Range<usize>>,
    active_channel: Option<usize>,
    playhead_frame: usize,
    state: &mut TunerState,
) -> TunerAction {
    let mut action = TunerAction::None;
    let num_frames = editable.buffer().num_frames();
    let target_range = selection.clone().unwrap_or(0..num_frames);
    let channel_mask = channel_bitmask(active_channel);
    let has_sel = selection
        .as_ref()
        .is_some_and(|s| s.start < s.end && s.end <= num_frames);
    let has_clip = editable.clipboard().is_some();

    ui.horizontal_wrapped(|ui| {
        ui.label("Gain (dB):");
        ui.add(egui::Slider::new(&mut state.gain_db, -24.0..=24.0).suffix(" dB"));
        if ui
            .add_enabled(num_frames > 0, egui::Button::new("Apply Gain"))
            .clicked()
        {
            let db = state.gain_db;
            let r = target_range.clone();
            editable.apply_transform(|b| apply_gain_db(b, r, channel_mask, db));
            state.fft_dirty = true;
            action = TunerAction::BufferEdited;
        }
    });

    ui.add_space(6.0);

    ui.horizontal_wrapped(|ui| {
        if ui
            .add_enabled(
                num_frames > 0,
                egui::Button::new(format!("Normalize ({:.2})", state.normalize_target)),
            )
            .clicked()
        {
            let r = target_range.clone();
            let target = state.normalize_target;
            editable.apply_transform(|b| normalize(b, r, channel_mask, target));
            state.fft_dirty = true;
            action = TunerAction::BufferEdited;
        }
        if ui
            .add_enabled(num_frames > 0, egui::Button::new("Fade In"))
            .clicked()
        {
            let r = target_range.clone();
            editable.apply_transform(|b| fade_in(b, r, channel_mask));
            state.fft_dirty = true;
            action = TunerAction::BufferEdited;
        }
        if ui
            .add_enabled(num_frames > 0, egui::Button::new("Fade Out"))
            .clicked()
        {
            let r = target_range.clone();
            editable.apply_transform(|b| fade_out(b, r, channel_mask));
            state.fft_dirty = true;
            action = TunerAction::BufferEdited;
        }
        if ui
            .add_enabled(num_frames > 0, egui::Button::new("Reverse"))
            .clicked()
        {
            let r = target_range.clone();
            editable.apply_transform(|b| reverse(b, r, channel_mask));
            state.fft_dirty = true;
            action = TunerAction::BufferEdited;
        }
        if ui
            .add_enabled(num_frames > 0, egui::Button::new("Remove DC"))
            .clicked()
        {
            let r = target_range.clone();
            editable.apply_transform(|b| remove_dc_offset(b, r, channel_mask));
            state.fft_dirty = true;
            action = TunerAction::BufferEdited;
        }
    });

    ui.add_space(6.0);

    ui.horizontal_wrapped(|ui| {
        if ui.add_enabled(has_sel, egui::Button::new("Trim")).clicked()
            && let Some(sel) = selection.clone()
        {
            editable.trim(sel);
            *selection = None;
            state.fft_dirty = true;
            action = TunerAction::BufferEdited;
        }
        if ui.add_enabled(has_sel, egui::Button::new("Cut")).clicked()
            && let Some(sel) = selection.clone()
        {
            editable.cut(sel);
            *selection = None;
            state.fft_dirty = true;
            action = TunerAction::BufferEdited;
        }
        if ui.add_enabled(has_sel, egui::Button::new("Copy")).clicked()
            && let Some(sel) = selection.as_ref()
        {
            let count = sel.len();
            editable.copy(sel.clone());
            action = TunerAction::StatusMessage(format!("Copied {count} frames to clipboard"));
        }
        if ui
            .add_enabled(has_clip, egui::Button::new("Paste at Playhead"))
            .clicked()
        {
            let clip_len = editable.clipboard().map_or(0, |c| c.num_frames());
            let at = playhead_frame.min(num_frames);
            editable.paste(at);
            let total_after = editable.buffer().num_frames();
            *selection = Some(at..(at + clip_len).min(total_after));
            state.fft_dirty = true;
            action = TunerAction::BufferEdited;
        }
        if ui
            .add_enabled(has_sel, egui::Button::new("Delete Selection"))
            .clicked()
            && let Some(sel) = selection.clone()
        {
            editable.delete_range(sel);
            *selection = None;
            state.fft_dirty = true;
            action = TunerAction::BufferEdited;
        }
    });

    action
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_frequency_x_mapping_roundtrip_and_endpoints() {
        let nyquist = 22_050.0;
        let left = 50.0_f32;
        let width = 400.0_f32;

        let x_min = freq_to_log_x(20.0, nyquist, left, width);
        let x_max = freq_to_log_x(nyquist, nyquist, left, width);
        assert!((x_min - left).abs() < 1e-4);
        assert!((x_max - (left + width)).abs() < 1e-4);

        let x_440 = freq_to_log_x(440.0, nyquist, left, width);
        let roundtrip = log_x_to_freq(x_440, nyquist, left, width);
        assert!((roundtrip - 440.0).abs() < 0.25);

        // Equal ratio octaves occupy equal horizontal spans on log axis
        let x_200 = freq_to_log_x(200.0, nyquist, left, width);
        let x_400 = freq_to_log_x(400.0, nyquist, left, width);
        let x_800 = freq_to_log_x(800.0, nyquist, left, width);
        assert!(((x_400 - x_200) - (x_800 - x_400)).abs() < 1e-3);
    }

    #[test]
    fn linked_wavelength_and_hz_sync_math() {
        let sr = 44_100;
        let (wl, hz) = sync_from_hz(440.0, sr);
        assert!((hz - 440.0).abs() < 1e-9);
        assert!((wl - 44_100.0 / 440.0).abs() < 1e-9);

        let (wl2, hz2) = sync_from_wavelength(wl, sr);
        assert!((wl2 - wl).abs() < 1e-9);
        assert!((hz2 - 440.0).abs() < 1e-9);

        // Clamping behavior
        let (wl_clamped, hz_clamped) = sync_from_wavelength(0.5, sr);
        assert!((wl_clamped - 2.0).abs() < 1e-9);
        assert!((hz_clamped - 22_050.0).abs() < 1e-9);

        let (wl_max_hz, hz_max) = sync_from_hz(99_000.0, sr);
        assert!((hz_max - 22_050.0).abs() < 1e-9);
        assert!((wl_max_hz - 2.0).abs() < 1e-9);
    }

    #[test]
    fn harmonic_preset_builders_produce_expected_weights() {
        let fund = harmonic_preset_amps(HarmonicPreset::Fundamental);
        assert_eq!(fund[0], 1.0);
        assert!(fund[1..].iter().all(|&a| a == 0.0));

        let saw = harmonic_preset_amps(HarmonicPreset::Saw);
        assert!((saw[0] - 1.0).abs() < 1e-6);
        assert!((saw[1] - 0.5).abs() < 1e-6);
        assert!((saw[15] - 1.0 / 16.0).abs() < 1e-6);

        let sq = harmonic_preset_amps(HarmonicPreset::Square);
        assert!((sq[0] - 1.0).abs() < 1e-6);
        assert_eq!(sq[1], 0.0);
        assert!((sq[2] - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(sq[3], 0.0);

        let organ = harmonic_preset_amps(HarmonicPreset::Organ);
        assert!(organ[0] > 0.0 && organ[1] > 0.0 && organ[7] > 0.0);

        let cleared = harmonic_preset_amps(HarmonicPreset::Clear);
        assert!(cleared.iter().all(|&a| a == 0.0));
    }

    #[test]
    fn channel_bitmask_guards_shift_overflow() {
        assert_eq!(channel_bitmask(Some(0)), Some(1));
        assert_eq!(channel_bitmask(Some(1)), Some(2));
        assert_eq!(channel_bitmask(Some(64)), Some(0));
        assert_eq!(channel_bitmask(Some(100)), Some(0));
        assert_eq!(channel_bitmask(None), None);
    }

    #[test]
    fn live_tuner_edit_snapshots_once_and_undoes_cleanly() {
        use wavecraft_core::SampleEncoding;

        let original_buf = AudioBuffer::from_channels(
            44_100,
            vec![vec![0.25; 200], vec![-0.25; 200]],
            SampleEncoding::Float32,
        )
        .unwrap();
        let mut editable = EditableAudio::new(original_buf.clone());
        let mut state = TunerState::default();
        let selection = Some(20..120);

        // First live edit pushes one undo snapshot and saves live_base.
        let (wl1, hz1) = sync_from_hz(220.0, 44_100);
        state.wavelength_samples = wl1;
        state.target_hz = hz1;
        assert!(apply_live_tuner_edit(
            &mut editable,
            &selection,
            None,
            &mut state
        ));
        assert_eq!(editable.undo_len(), 1);
        assert!(editable.is_dirty());

        // Second & third consecutive live edits reuse live_base and push NO extra undo snapshots.
        state.shape_kind = SynthShapeKind::Harmonics;
        state.harmonic_amps = harmonic_preset_amps(HarmonicPreset::Saw);
        state.amplitude = 0.6;
        assert!(apply_live_tuner_edit(
            &mut editable,
            &selection,
            None,
            &mut state
        ));

        let (wl2, hz2) = sync_from_hz(550.0, 44_100);
        state.wavelength_samples = wl2;
        state.target_hz = hz2;
        state.apply_mode = ApplyMode::Mix;
        assert!(apply_live_tuner_edit(
            &mut editable,
            &selection,
            None,
            &mut state
        ));

        assert_eq!(
            editable.undo_len(),
            1,
            "consecutive live tuner edits must share one undo snapshot"
        );
        assert!(state.fft_dirty);

        // Verify the final edit was applied against original live_base (not stacked repeatedly).
        let mut expected_buf = original_buf.clone();
        apply_synth_to_buffer(
            &mut expected_buf,
            20..120,
            None,
            &state.synth_params(),
            ApplyMode::Mix,
        );
        assert_eq!(editable.buffer(), &expected_buf);
        // Outside selection must be untouched.
        assert_eq!(editable.buffer().get_sample(0, 10), Some(0.25));
        assert_eq!(editable.buffer().get_sample(1, 150), Some(-0.25));

        // A single undo restores the exact original audio buffer.
        assert!(editable.undo());
        assert_eq!(editable.buffer(), &original_buf);
        assert!(!editable.can_undo());
    }

    #[test]
    fn explicit_apply_after_live_mix_does_not_double_mix_or_duplicate_undo() {
        use wavecraft_core::SampleEncoding;

        let original_buf = AudioBuffer::from_channels(
            44_100,
            vec![vec![0.25; 200], vec![-0.25; 200]],
            SampleEncoding::Float32,
        )
        .unwrap();
        let mut editable = EditableAudio::new(original_buf.clone());
        let mut state = TunerState::default();
        let selection = Some(20..120);

        let (wl, hz) = sync_from_hz(330.0, 44_100);
        state.wavelength_samples = wl;
        state.target_hz = hz;
        state.amplitude = 0.4;
        state.apply_mode = ApplyMode::Mix;

        assert!(apply_live_tuner_edit(
            &mut editable,
            &selection,
            Some(0),
            &mut state
        ));
        assert_eq!(editable.undo_len(), 1);
        assert!(state.live_base.is_some());

        // Clicking "Apply to Selection" while live session is active must reuse live_base
        // rather than double-mixing onto the already-mixed buffer or pushing a second undo step.
        assert!(commit_tuner_apply(
            &mut editable,
            selection.as_ref(),
            Some(0),
            &mut state
        ));
        assert!(state.live_base.is_none());
        assert!(state.live_base_range.is_none());
        assert!(state.live_base_channel.is_none());
        assert!(state.fft_dirty);
        assert_eq!(editable.undo_len(), 1);

        let mut expected_buf = original_buf.clone();
        apply_synth_to_buffer(
            &mut expected_buf,
            20..120,
            channel_bitmask(Some(0)),
            &state.synth_params(),
            ApplyMode::Mix,
        );
        assert_eq!(editable.buffer(), &expected_buf);

        // One undo restores the original buffer cleanly.
        assert!(editable.undo());
        assert_eq!(editable.buffer(), &original_buf);
        assert!(!editable.can_undo());
    }
}
