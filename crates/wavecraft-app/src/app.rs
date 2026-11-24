use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use egui::{Color32, CornerRadius, Key, Modifiers, Stroke};
use wavecraft_audio::PlaybackEngine;
use wavecraft_core::dsp::{SynthParams, WaveShape, generate_wave, hz_to_wavelength};
use wavecraft_core::fixtures::generate_sample_fixtures;
use wavecraft_core::format::inspect_file;
use wavecraft_core::format::raw_pcm::write_raw_pcm_file;
use wavecraft_core::format::wav::write_wav_file;
use wavecraft_core::{AudioBuffer, EditableAudio, InspectedFile, RawPcmConfig, SampleEncoding};

use crate::ui::explorer::{ExplorerAction, ExplorerState, show_explorer};
use crate::ui::inspector::{InspectorState, show_inspector};
use crate::ui::tuner_fft::{TunerState, show_tuner_fft_panel};
use crate::ui::waveform::{WaveformAction, WaveformViewState, show_waveform_panel};

const ALL_ENCODINGS: [SampleEncoding; 5] = [
    SampleEncoding::PcmU8,
    SampleEncoding::PcmI16,
    SampleEncoding::PcmI24,
    SampleEncoding::PcmI32,
    SampleEncoding::Float32,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NewWaveKind {
    Silent,
    Sine440,
}

#[derive(Debug, Clone)]
enum ActiveModal {
    NewBuffer {
        sample_rate: u32,
        channels: usize,
        duration_secs: f32,
        wave: NewWaveKind,
    },
    SaveWav {
        path_input: String,
        encoding: SampleEncoding,
        error: Option<String>,
    },
    ExportRawPcm {
        path_input: String,
        encoding: SampleEncoding,
        big_endian: bool,
        error: Option<String>,
    },
    OpenRawPcm {
        path: PathBuf,
        config: RawPcmConfig,
        error: Option<String>,
    },
}

/// Main Wavecraft native workbench application.
pub struct WavecraftApp {
    pub current_path: Option<PathBuf>,
    pub inspected: Option<InspectedFile>,
    pub editable: EditableAudio,
    pub engine: PlaybackEngine,
    pub selection: Option<Range<usize>>,
    pub active_channel: Option<usize>,
    pub volume: f32,
    pub loop_enabled: bool,
    pub show_explorer: bool,
    pub show_inspector: bool,
    pub explorer: ExplorerState,
    pub inspector: InspectorState,
    pub waveform: WaveformViewState,
    pub tuner: TunerState,
    modal: Option<ActiveModal>,
    pub status_message: String,
}

impl WavecraftApp {
    /// Creates the application, applies the dark audio-workbench theme, and loads an initial file or sine buffer.
    pub fn new(cc: &eframe::CreationContext<'_>, initial_file: Option<PathBuf>) -> Self {
        configure_theme(&cc.egui_ctx);

        let samples_dir = PathBuf::from("samples");
        let explorer_dir = if let Some(ref file) = initial_file {
            file.parent()
                .filter(|p| !p.as_os_str().is_empty() && p.is_dir())
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."))
        } else if samples_dir.is_dir() {
            samples_dir.clone()
        } else {
            PathBuf::from(".")
        };

        let default_buf = make_sine_440_buffer(44_100, 1, 1.0);
        let engine = PlaybackEngine::new_system();
        engine.set_buffer(Arc::new(default_buf.clone()));

        let mut app = Self {
            current_path: None,
            inspected: None,
            editable: EditableAudio::new(default_buf),
            engine,
            selection: None,
            active_channel: None,
            volume: 0.85,
            loop_enabled: false,
            show_explorer: true,
            show_inspector: true,
            explorer: ExplorerState::new(explorer_dir),
            inspector: InspectorState::default(),
            waveform: WaveformViewState::default(),
            tuner: TunerState::default(),
            modal: None,
            status_message: "Ready — 440 Hz 1.0s Sine Buffer".to_string(),
        };
        app.engine.set_volume(app.volume);

        if let Some(path) = initial_file {
            app.open_file(&path, None);
        } else {
            let demo_sine = samples_dir.join("sine_440hz.wav");
            if demo_sine.is_file() {
                app.open_file(&demo_sine, None);
            }
        }

        app
    }

    /// Pushes the current [`EditableAudio`] buffer to the [`PlaybackEngine`] and updates loop bounds.
    pub fn sync_engine_buffer(&self) {
        self.engine
            .set_buffer(Arc::new(self.editable.buffer().clone()));
        self.sync_loop_range();
    }

    fn sync_loop_range(&self) {
        if !self.loop_enabled {
            self.engine.set_loop_range(None);
            return;
        }
        let total = self.editable.buffer().num_frames();
        let range = match &self.selection {
            Some(sel) if sel.start < sel.end && sel.end <= total => Some(sel.clone()),
            _ if total > 0 => Some(0..total),
            _ => None,
        };
        self.engine.set_loop_range(range);
    }

    fn open_file(&mut self, path: &Path, raw_cfg: Option<RawPcmConfig>) -> bool {
        match inspect_file(path, raw_cfg) {
            Ok(file) => {
                self.engine.stop();
                let buf = file.buffer.clone();
                self.current_path = Some(path.to_path_buf());
                self.explorer.selected_file = Some(path.to_path_buf());
                self.inspected = Some(file);
                self.editable = EditableAudio::new(buf);
                self.selection = None;
                self.active_channel = None;
                self.inspector.selected_range = None;
                self.sync_engine_buffer();
                self.status_message = format!("Loaded {}", path.display());
                true
            }
            Err(err) => {
                self.status_message = format!("Failed to open {}: {err}", path.display());
                false
            }
        }
    }

    fn open_save_wav_modal(&mut self) {
        let default_path = self
            .current_path
            .as_ref()
            .map(|p| p.with_extension("wav").display().to_string())
            .unwrap_or_else(|| {
                self.explorer
                    .current_dir
                    .join("untitled.wav")
                    .display()
                    .to_string()
            });
        self.modal = Some(ActiveModal::SaveWav {
            path_input: default_path,
            encoding: self.editable.buffer().source_encoding,
            error: None,
        });
    }

    fn open_export_pcm_modal(&mut self) {
        let default_path = self
            .current_path
            .as_ref()
            .map(|p| p.with_extension("pcm").display().to_string())
            .unwrap_or_else(|| {
                self.explorer
                    .current_dir
                    .join("export.pcm")
                    .display()
                    .to_string()
            });
        self.modal = Some(ActiveModal::ExportRawPcm {
            path_input: default_path,
            encoding: self.editable.buffer().source_encoding,
            big_endian: false,
            error: None,
        });
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }

        let space_pressed = ctx.input(|i| i.key_pressed(Key::Space));
        if space_pressed {
            self.engine.toggle_play_pause();
        }

        let undo_pressed = ctx.input(|i| {
            i.key_pressed(Key::Z) && (i.modifiers.ctrl || i.modifiers.mac_cmd) && !i.modifiers.shift
        });
        if undo_pressed && self.editable.undo() {
            self.sync_engine_buffer();
            self.status_message = "Undo".to_string();
        }

        let redo_pressed = ctx.input(|i| {
            let mod_cmd = i.modifiers.ctrl || i.modifiers.mac_cmd;
            mod_cmd && (i.key_pressed(Key::Y) || (i.key_pressed(Key::Z) && i.modifiers.shift))
        });
        if redo_pressed && self.editable.redo() {
            self.sync_engine_buffer();
            self.status_message = "Redo".to_string();
        }

        let save_pressed = ctx.input(|i| {
            let mod_cmd = i.modifiers.ctrl || i.modifiers.mac_cmd;
            mod_cmd && i.modifiers.matches_logically(Modifiers::CTRL) && i.key_pressed(Key::S)
                || (mod_cmd && i.key_pressed(Key::S))
        });
        if save_pressed {
            self.open_save_wav_modal();
        }
    }

    fn render_top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_bar")
            .exact_height(42.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.label(
                        egui::RichText::new("WAVECRAFT")
                            .strong()
                            .size(14.0)
                            .color(Color32::from_rgb(45, 212, 191)),
                    );
                    ui.separator();

                    // File actions
                    if ui.button("New").clicked() {
                        self.modal = Some(ActiveModal::NewBuffer {
                            sample_rate: 44_100,
                            channels: 1,
                            duration_secs: 1.0,
                            wave: NewWaveKind::Sine440,
                        });
                    }
                    if ui.button("Save WAV").clicked() {
                        self.open_save_wav_modal();
                    }
                    if ui.button("Export PCM").clicked() {
                        self.open_export_pcm_modal();
                    }

                    ui.separator();

                    // Edit actions
                    if ui
                        .add_enabled(self.editable.can_undo(), egui::Button::new("Undo"))
                        .clicked()
                        && self.editable.undo()
                    {
                        self.sync_engine_buffer();
                        self.status_message = "Undo".to_string();
                    }
                    if ui
                        .add_enabled(self.editable.can_redo(), egui::Button::new("Redo"))
                        .clicked()
                        && self.editable.redo()
                    {
                        self.sync_engine_buffer();
                        self.status_message = "Redo".to_string();
                    }

                    ui.separator();

                    // Transport controls
                    let playback_state = self.engine.state();
                    let is_playing = playback_state.is_playing();
                    let play_label = if is_playing { "Pause" } else { "Play" };
                    if ui.button(play_label).clicked() {
                        self.engine.toggle_play_pause();
                    }
                    if ui.button("Stop").clicked() {
                        self.engine.stop();
                    }
                    if ui.selectable_label(self.loop_enabled, "Loop").clicked() {
                        self.loop_enabled = !self.loop_enabled;
                        self.sync_loop_range();
                    }

                    ui.label("Vol");
                    let vol_slider = ui.add(
                        egui::Slider::new(&mut self.volume, 0.0..=1.5)
                            .show_value(false)
                            .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                    );
                    if vol_slider.changed() {
                        self.engine.set_volume(self.volume);
                    }

                    let sr = self.editable.buffer().sample_rate.max(1) as f64;
                    let cur_secs = playback_state.playhead_frame() as f64 / sr;
                    let total_secs = self.editable.buffer().duration_secs();
                    ui.label(
                        egui::RichText::new(format!(
                            "{} / {}",
                            format_time_mm_ss(cur_secs),
                            format_time_mm_ss(total_secs)
                        ))
                        .monospace()
                        .size(12.0),
                    );

                    // Backend badge
                    let (bg, fg) = if self.engine.is_real_output() {
                        (
                            Color32::from_rgb(20, 83, 45),
                            Color32::from_rgb(134, 239, 172),
                        )
                    } else {
                        (
                            Color32::from_rgb(66, 52, 22),
                            Color32::from_rgb(253, 224, 71),
                        )
                    };
                    egui::Frame::NONE
                        .fill(bg)
                        .corner_radius(CornerRadius::same(4))
                        .inner_margin(egui::Margin::symmetric(6, 2))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new(self.engine.backend_name())
                                    .size(10.5)
                                    .strong()
                                    .color(fg),
                            );
                        });

                    // View toggles on right
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.toggle_value(&mut self.show_inspector, "Inspector");
                        ui.toggle_value(&mut self.show_explorer, "Explorer");
                    });
                });
            });
    }

    fn render_modals(&mut self, ctx: &egui::Context) {
        let Some(mut modal) = self.modal.take() else {
            return;
        };

        let mut keep_open = true;
        match &mut modal {
            ActiveModal::NewBuffer {
                sample_rate,
                channels,
                duration_secs,
                wave,
            } => {
                egui::Window::new("New Audio Buffer")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        egui::Grid::new("new_buffer_grid")
                            .num_columns(2)
                            .spacing([12.0, 8.0])
                            .show(ui, |ui| {
                                ui.label("Sample Rate:");
                                ui.horizontal(|ui| {
                                    for sr in [22_050u32, 44_100, 48_000] {
                                        ui.selectable_value(sample_rate, sr, format!("{sr} Hz"));
                                    }
                                });
                                ui.end_row();

                                ui.label("Channels:");
                                ui.horizontal(|ui| {
                                    ui.selectable_value(channels, 1, "1 (Mono)");
                                    ui.selectable_value(channels, 2, "2 (Stereo)");
                                });
                                ui.end_row();

                                ui.label("Duration (s):");
                                ui.add(
                                    egui::DragValue::new(duration_secs)
                                        .range(0.05..=60.0)
                                        .speed(0.05)
                                        .suffix(" s"),
                                );
                                ui.end_row();

                                ui.label("Initial Wave:");
                                ui.horizontal(|ui| {
                                    ui.selectable_value(wave, NewWaveKind::Silent, "Silent");
                                    ui.selectable_value(wave, NewWaveKind::Sine440, "440 Hz Sine");
                                });
                                ui.end_row();
                            });

                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            if ui.button("Create").clicked() {
                                let buf = match *wave {
                                    NewWaveKind::Silent => {
                                        let frames = ((*sample_rate as f64)
                                            * (*duration_secs as f64))
                                            .round()
                                            .max(1.0)
                                            as usize;
                                        AudioBuffer::new_silent(*sample_rate, *channels, frames)
                                    }
                                    NewWaveKind::Sine440 => make_sine_440_buffer(
                                        *sample_rate,
                                        *channels,
                                        *duration_secs as f64,
                                    ),
                                };
                                self.engine.stop();
                                self.current_path = None;
                                self.inspected = None;
                                self.editable = EditableAudio::new(buf);
                                self.selection = None;
                                self.active_channel = None;
                                self.inspector.selected_range = None;
                                self.sync_engine_buffer();
                                self.status_message = format!(
                                    "Created new {} Hz {}ch buffer ({:.2}s)",
                                    *sample_rate, *channels, *duration_secs
                                );
                                keep_open = false;
                            }
                            if ui.button("Cancel").clicked() {
                                keep_open = false;
                            }
                        });
                    });
            }

            ActiveModal::SaveWav {
                path_input,
                encoding,
                error,
            } => {
                egui::Window::new("Save WAV File")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        egui::Grid::new("save_wav_grid")
                            .num_columns(2)
                            .spacing([12.0, 8.0])
                            .show(ui, |ui| {
                                ui.label("Output Path:");
                                ui.add(egui::TextEdit::singleline(path_input).desired_width(300.0));
                                ui.end_row();

                                ui.label("Encoding:");
                                ui.vertical(|ui| {
                                    for enc in ALL_ENCODINGS {
                                        ui.selectable_value(encoding, enc, enc.label());
                                    }
                                });
                                ui.end_row();
                            });

                        if let Some(err) = error.as_ref() {
                            ui.colored_label(Color32::from_rgb(248, 113, 113), err);
                        }

                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            if ui.button("Save WAV").clicked() {
                                let out_path = PathBuf::from(path_input.trim());
                                match write_wav_file(&out_path, self.editable.buffer(), *encoding) {
                                    Ok(()) => {
                                        self.editable.mark_clean();
                                        self.current_path = Some(out_path.clone());
                                        match inspect_file(&out_path, None) {
                                            Ok(inspected) => {
                                                self.inspected = Some(inspected);
                                                self.status_message =
                                                    format!("Saved WAV: {}", out_path.display());
                                            }
                                            Err(e) => {
                                                self.status_message = format!(
                                                    "Saved WAV, but re-inspect failed: {e}"
                                                );
                                            }
                                        }
                                        self.explorer.refresh();
                                        self.explorer.selected_file = Some(out_path.clone());
                                        keep_open = false;
                                    }
                                    Err(err) => {
                                        *error = Some(err.to_string());
                                    }
                                }
                            }
                            if ui.button("Cancel").clicked() {
                                keep_open = false;
                            }
                        });
                    });
            }

            ActiveModal::ExportRawPcm {
                path_input,
                encoding,
                big_endian,
                error,
            } => {
                egui::Window::new("Export Raw PCM")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        egui::Grid::new("export_pcm_grid")
                            .num_columns(2)
                            .spacing([12.0, 8.0])
                            .show(ui, |ui| {
                                ui.label("Output Path:");
                                ui.add(egui::TextEdit::singleline(path_input).desired_width(300.0));
                                ui.end_row();

                                ui.label("Encoding:");
                                ui.vertical(|ui| {
                                    for enc in ALL_ENCODINGS {
                                        ui.selectable_value(encoding, enc, enc.label());
                                    }
                                });
                                ui.end_row();

                                ui.label("Byte Order:");
                                ui.horizontal(|ui| {
                                    ui.selectable_value(big_endian, false, "Little Endian");
                                    ui.selectable_value(big_endian, true, "Big Endian");
                                });
                                ui.end_row();
                            });

                        if let Some(err) = error.as_ref() {
                            ui.colored_label(Color32::from_rgb(248, 113, 113), err);
                        }

                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            if ui.button("Export PCM").clicked() {
                                let out_path = PathBuf::from(path_input.trim());
                                let cfg = RawPcmConfig {
                                    sample_rate: self.editable.buffer().sample_rate,
                                    num_channels: self.editable.buffer().num_channels() as u16,
                                    encoding: *encoding,
                                    big_endian: *big_endian,
                                };
                                match write_raw_pcm_file(&out_path, self.editable.buffer(), cfg) {
                                    Ok(()) => {
                                        self.explorer.refresh();
                                        self.status_message =
                                            format!("Exported Raw PCM: {}", out_path.display());
                                        keep_open = false;
                                    }
                                    Err(err) => {
                                        *error = Some(err.to_string());
                                    }
                                }
                            }
                            if ui.button("Cancel").clicked() {
                                keep_open = false;
                            }
                        });
                    });
            }

            ActiveModal::OpenRawPcm {
                path,
                config,
                error,
            } => {
                egui::Window::new("Open Raw PCM Configuration")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.label(
                            egui::RichText::new(format!("File: {}", path.display()))
                                .strong()
                                .color(Color32::from_rgb(226, 232, 240)),
                        );
                        ui.add_space(4.0);
                        egui::Grid::new("open_pcm_grid")
                            .num_columns(2)
                            .spacing([12.0, 8.0])
                            .show(ui, |ui| {
                                ui.label("Sample Rate:");
                                ui.horizontal(|ui| {
                                    for sr in [22_050u32, 44_100, 48_000] {
                                        ui.selectable_value(
                                            &mut config.sample_rate,
                                            sr,
                                            format!("{sr} Hz"),
                                        );
                                    }
                                });
                                ui.end_row();

                                ui.label("Channels:");
                                ui.horizontal(|ui| {
                                    ui.selectable_value(&mut config.num_channels, 1, "1 (Mono)");
                                    ui.selectable_value(&mut config.num_channels, 2, "2 (Stereo)");
                                });
                                ui.end_row();

                                ui.label("Encoding:");
                                ui.vertical(|ui| {
                                    for enc in ALL_ENCODINGS {
                                        ui.selectable_value(&mut config.encoding, enc, enc.label());
                                    }
                                });
                                ui.end_row();

                                ui.label("Byte Order:");
                                ui.horizontal(|ui| {
                                    ui.selectable_value(
                                        &mut config.big_endian,
                                        false,
                                        "Little Endian",
                                    );
                                    ui.selectable_value(&mut config.big_endian, true, "Big Endian");
                                });
                                ui.end_row();
                            });

                        if let Some(err) = error.as_ref() {
                            ui.colored_label(Color32::from_rgb(248, 113, 113), err);
                        }

                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            if ui.button("Open").clicked() {
                                let target = path.clone();
                                let cfg = *config;
                                match inspect_file(&target, Some(cfg)) {
                                    Ok(file) => {
                                        self.engine.stop();
                                        let buf = file.buffer.clone();
                                        self.current_path = Some(target.clone());
                                        self.explorer.selected_file = Some(target.clone());
                                        self.inspected = Some(file);
                                        self.editable = EditableAudio::new(buf);
                                        self.selection = None;
                                        self.active_channel = None;
                                        self.inspector.selected_range = None;
                                        self.sync_engine_buffer();
                                        self.status_message =
                                            format!("Opened Raw PCM {}", target.display());
                                        keep_open = false;
                                    }
                                    Err(err) => {
                                        *error = Some(err.to_string());
                                    }
                                }
                            }
                            if ui.button("Cancel").clicked() {
                                keep_open = false;
                            }
                        });
                    });
            }
        }

        if keep_open {
            self.modal = Some(modal);
        }
    }
}

impl eframe::App for WavecraftApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_shortcuts(ctx);

        if self.engine.state().is_playing() {
            ctx.request_repaint_after(Duration::from_millis(16));
        }

        self.render_top_bar(ctx);

        if self.show_explorer {
            egui::SidePanel::left("explorer_panel")
                .default_width(270.0)
                .width_range(210.0..=420.0)
                .show(ctx, |ui| match show_explorer(ui, &mut self.explorer) {
                    ExplorerAction::None => {}
                    ExplorerAction::OpenFile(path) => {
                        self.open_file(&path, None);
                    }
                    ExplorerAction::OpenRawPcm(path) => {
                        self.modal = Some(ActiveModal::OpenRawPcm {
                            path,
                            config: RawPcmConfig::default(),
                            error: None,
                        });
                    }
                    ExplorerAction::QuickPreviewPlay(path) => {
                        if self.open_file(&path, None) {
                            self.engine.seek(0);
                            self.engine.play();
                        }
                    }
                    ExplorerAction::GenerateSamplesHere(dir) => {
                        match generate_sample_fixtures(&dir) {
                            Ok(paths) => {
                                self.explorer.refresh();
                                self.status_message = format!(
                                    "Generated {} sample fixtures in {}",
                                    paths.len(),
                                    dir.display()
                                );
                            }
                            Err(err) => {
                                self.status_message = format!("Failed to generate samples: {err}");
                            }
                        }
                    }
                });
        }

        if self.show_inspector {
            egui::SidePanel::right("inspector_panel")
                .default_width(420.0)
                .width_range(300.0..=720.0)
                .show(ctx, |ui| {
                    show_inspector(
                        ui,
                        self.inspected.as_ref(),
                        self.editable.is_dirty(),
                        &mut self.inspector,
                    );
                });
        }

        egui::TopBottomPanel::bottom("bottom_panel")
            .resizable(true)
            .default_height(180.0)
            .min_height(110.0)
            .show(ctx, |ui| {
                let edited = show_tuner_fft_panel(
                    ui,
                    &mut self.editable,
                    &self.selection,
                    self.active_channel,
                    &mut self.tuner,
                );
                if edited {
                    self.sync_engine_buffer();
                }
                ui.separator();
                ui.horizontal(|ui| {
                    let dirty_tag = if self.editable.is_dirty() {
                        " [modified]"
                    } else {
                        ""
                    };
                    let file_label = self.current_path.as_ref().map_or_else(
                        || "<unsaved buffer>".to_string(),
                        |p| p.display().to_string(),
                    );
                    ui.label(
                        egui::RichText::new(format!(
                            "{file_label}{dirty_tag}  •  {}",
                            self.status_message
                        ))
                        .size(11.5)
                        .color(Color32::from_rgb(148, 163, 184)),
                    );
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let playhead = self.engine.state().playhead_frame();
            match show_waveform_panel(
                ui,
                &mut self.editable,
                &mut self.selection,
                &mut self.active_channel,
                playhead,
                &mut self.waveform,
            ) {
                WaveformAction::None => {}
                WaveformAction::Seek(frame) => {
                    self.engine.seek(frame);
                }
                WaveformAction::BufferEdited => {
                    self.sync_engine_buffer();
                }
            }
        });

        self.render_modals(ctx);
    }
}

/// Builds a deterministic 440 Hz sine [`AudioBuffer`] for initial startup or New Buffer modal.
pub fn make_sine_440_buffer(sample_rate: u32, channels: usize, duration_secs: f64) -> AudioBuffer {
    let frames = ((sample_rate as f64) * duration_secs).round().max(1.0) as usize;
    let samples = generate_wave(
        &SynthParams {
            shape: WaveShape::Sine,
            wavelength_samples: hz_to_wavelength(440.0, sample_rate),
            amplitude: 0.8,
            phase_offset_rad: 0.0,
        },
        frames,
    );
    let ch_count = channels.max(1);
    let ch_vecs = vec![samples; ch_count];
    AudioBuffer::from_channels(sample_rate, ch_vecs, SampleEncoding::PcmI16)
        .unwrap_or_else(|_| AudioBuffer::new_silent(44_100, 1, 44_100))
}

fn format_time_mm_ss(secs: f64) -> String {
    let clamped = secs.max(0.0);
    let mins = (clamped / 60.0).floor() as u64;
    let rem = clamped - (mins as f64 * 60.0);
    format!("{mins:02}:{rem:05.2}")
}

fn configure_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = Color32::from_rgb(20, 23, 30);
    visuals.window_fill = Color32::from_rgb(24, 28, 37);
    visuals.extreme_bg_color = Color32::from_rgb(13, 16, 22);
    visuals.faint_bg_color = Color32::from_rgb(27, 32, 42);
    visuals.selection.bg_fill = Color32::from_rgb(20, 110, 118);
    visuals.selection.stroke = Stroke::new(1.0, Color32::from_rgb(45, 212, 191));
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(40, 48, 61));
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(31, 37, 48);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(40, 48, 62);
    visuals.widgets.active.bg_fill = Color32::from_rgb(45, 132, 130);
    ctx.set_visuals(visuals);
}
