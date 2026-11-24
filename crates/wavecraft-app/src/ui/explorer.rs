use std::collections::HashMap;
use std::path::PathBuf;

use egui::{Color32, CornerRadius, Pos2, Sense, Stroke, Vec2};
use wavecraft_core::format::{detect_format, probe_preview};
use wavecraft_core::{AudioFormat, AudioMetadata, PeakBucket};

const MAX_PROBES_PER_FRAME: usize = 2;
const THUMBNAIL_BUCKETS: usize = 64;

/// Directory entry metadata shown in the file explorer list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntryInfo {
    pub path: PathBuf,
    pub name: String,
    pub is_dir: bool,
    pub size_bytes: u64,
    pub format: Option<AudioFormat>,
}

/// Format filter chip selection for the file explorer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatFilter {
    All,
    Wav,
    RawPcm,
    Flac,
    Mp3,
    Ogg,
}

impl FormatFilter {
    pub const ALL: [Self; 6] = [
        Self::All,
        Self::Wav,
        Self::RawPcm,
        Self::Flac,
        Self::Mp3,
        Self::Ogg,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Wav => "WAV",
            Self::RawPcm => "PCM",
            Self::Flac => "FLAC",
            Self::Mp3 => "MP3",
            Self::Ogg => "OGG",
        }
    }

    pub fn matches_format(self, format: Option<AudioFormat>) -> bool {
        match self {
            Self::All => format.is_some(),
            Self::Wav => format == Some(AudioFormat::Wav),
            Self::RawPcm => format == Some(AudioFormat::RawPcm),
            Self::Flac => format == Some(AudioFormat::Flac),
            Self::Mp3 => format == Some(AudioFormat::Mp3),
            Self::Ogg => format == Some(AudioFormat::Ogg),
        }
    }
}

/// Cached lazy-probed preview state for an audio file.
#[derive(Debug, Clone)]
pub enum PreviewState {
    Loading,
    Ready {
        metadata: AudioMetadata,
        peaks: Vec<PeakBucket>,
    },
    Error(String),
}

/// Action emitted by [`show_explorer`] for the parent application to handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplorerAction {
    None,
    OpenFile(PathBuf),
    OpenRawPcm(PathBuf),
    QuickPreviewPlay(PathBuf),
    GenerateSamplesHere(PathBuf),
}

/// Persistent state for the left-side audio File Explorer panel.
pub struct ExplorerState {
    pub current_dir: PathBuf,
    pub path_input: String,
    pub entries: Vec<DirEntryInfo>,
    pub filter: FormatFilter,
    pub previews: HashMap<PathBuf, PreviewState>,
    pub selected_file: Option<PathBuf>,
    pub last_error: Option<String>,
}

impl ExplorerState {
    /// Creates an [`ExplorerState`] rooted at `initial_dir` and scans its entries.
    pub fn new(initial_dir: PathBuf) -> Self {
        let resolved = if initial_dir.is_dir() {
            initial_dir
        } else {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        };
        let path_input = resolved.display().to_string();
        let mut state = Self {
            current_dir: resolved,
            path_input,
            entries: Vec::new(),
            filter: FormatFilter::All,
            previews: HashMap::new(),
            selected_file: None,
            last_error: None,
        };
        state.refresh();
        state
    }

    /// Changes `current_dir` to `dir` if it exists and reloads its directory entries.
    pub fn navigate_to(&mut self, dir: PathBuf) {
        if dir.is_dir() {
            self.current_dir = dir;
            self.path_input = self.current_dir.display().to_string();
            self.last_error = None;
            self.refresh();
        } else {
            self.last_error = Some(format!("Directory not found: {}", dir.display()));
        }
    }

    /// Rescans `current_dir` and clears cached previews so modified files re-probe cleanly.
    pub fn refresh(&mut self) {
        self.previews.clear();
        self.entries.clear();
        self.last_error = None;

        match std::fs::read_dir(&self.current_dir) {
            Ok(read_dir) => {
                for item in read_dir.flatten() {
                    let path = item.path();
                    let name = item.file_name().to_string_lossy().to_string();
                    if name.starts_with('.') {
                        continue;
                    }
                    let Ok(meta) = item.metadata() else {
                        continue;
                    };
                    let is_dir = meta.is_dir();
                    let format = if is_dir {
                        None
                    } else {
                        detect_format(&[], Some(&path))
                    };
                    if is_dir || format.is_some() {
                        self.entries.push(DirEntryInfo {
                            path,
                            name,
                            is_dir,
                            size_bytes: meta.len(),
                            format,
                        });
                    }
                }
                sort_entries(&mut self.entries);
            }
            Err(err) => {
                self.last_error = Some(format!("Failed to read directory: {err}"));
            }
        }
    }
}

/// Sorts directory entries with folders first, then files case-insensitively by name.
pub fn sort_entries(entries: &mut [DirEntryInfo]) {
    entries.sort_by(|a, b| {
        b.is_dir
            .cmp(&a.is_dir)
            .then_with(|| {
                a.name
                    .to_ascii_lowercase()
                    .cmp(&b.name.to_ascii_lowercase())
            })
            .then_with(|| a.name.cmp(&b.name))
    });
}

/// Returns `true` if `entry` should be displayed under `filter`.
/// Directories remain visible across all filters to allow folder navigation.
pub fn filter_entry(entry: &DirEntryInfo, filter: FormatFilter) -> bool {
    if entry.is_dir {
        true
    } else {
        filter.matches_format(entry.format)
    }
}

/// Formats a byte count into a compact human-readable size string.
pub fn format_human_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if bytes < 1024 {
        format!("{bytes} B")
    } else if b < MB {
        format!("{:.1} KB", b / KB)
    } else if b < GB {
        format!("{:.2} MB", b / MB)
    } else {
        format!("{:.2} GB", b / GB)
    }
}

/// Returns a distinct badge `(background, text)` color pair for an [`AudioFormat`].
pub fn format_badge_colors(format: AudioFormat) -> (Color32, Color32) {
    match format {
        AudioFormat::Wav => (
            Color32::from_rgb(20, 92, 108),
            Color32::from_rgb(132, 240, 250),
        ),
        AudioFormat::RawPcm => (
            Color32::from_rgb(112, 68, 22),
            Color32::from_rgb(253, 208, 128),
        ),
        AudioFormat::Flac => (
            Color32::from_rgb(24, 96, 56),
            Color32::from_rgb(134, 245, 180),
        ),
        AudioFormat::Mp3 => (
            Color32::from_rgb(84, 46, 118),
            Color32::from_rgb(218, 178, 255),
        ),
        AudioFormat::Ogg => (
            Color32::from_rgb(32, 70, 124),
            Color32::from_rgb(152, 204, 255),
        ),
    }
}

fn format_compact_metadata(meta: &AudioMetadata) -> String {
    let sr_khz = meta.sample_rate as f64 / 1000.0;
    let sr_str = if (sr_khz.fract()).abs() < 0.01 {
        format!("{:.0}kHz", sr_khz)
    } else {
        format!("{:.1}kHz", sr_khz)
    };
    format!(
        "{sr_str} • {}ch • {:.2}s",
        meta.num_channels, meta.duration_secs
    )
}

fn draw_badge(ui: &mut egui::Ui, label: &str, bg: Color32, fg: Color32) {
    egui::Frame::NONE
        .fill(bg)
        .corner_radius(CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(6, 2))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(label).size(10.0).strong().color(fg));
        });
}

fn draw_mini_waveform(ui: &mut egui::Ui, peaks: &[PeakBucket], selected: bool) -> egui::Response {
    let desired = Vec2::new(ui.available_width().max(60.0), 26.0);
    let (rect, response) = ui.allocate_exact_size(desired, Sense::click());
    let painter = ui.painter_at(rect);

    let bg = if selected {
        Color32::from_rgb(22, 34, 46)
    } else {
        Color32::from_rgb(18, 22, 29)
    };
    painter.rect_filled(rect, CornerRadius::same(3), bg);

    let mid_y = rect.center().y;
    painter.line_segment(
        [
            Pos2::new(rect.left(), mid_y),
            Pos2::new(rect.right(), mid_y),
        ],
        Stroke::new(1.0, Color32::from_rgb(45, 53, 66)),
    );

    if peaks.is_empty() {
        return response;
    }

    let half_h = (rect.height() * 0.44).max(1.0);
    let n = peaks.len();
    let wave_color = if selected {
        Color32::from_rgb(94, 234, 212)
    } else {
        Color32::from_rgb(45, 212, 191)
    };

    for (i, bucket) in peaks.iter().enumerate() {
        let x = rect.left() + ((i as f32 + 0.5) / n as f32) * rect.width();
        let y_top = mid_y - bucket.max.clamp(-1.0, 1.0) * half_h;
        let y_bot = mid_y - bucket.min.clamp(-1.0, 1.0) * half_h;
        let y0 = y_top.min(y_bot);
        let y1 = y_top.max(y_bot).max(y0 + 1.0);
        painter.line_segment(
            [Pos2::new(x, y0), Pos2::new(x, y1)],
            Stroke::new(1.2, wave_color),
        );
    }

    response
}

/// Renders the interactive File Explorer panel with format filters and cached mini-waveforms.
pub fn show_explorer(ui: &mut egui::Ui, state: &mut ExplorerState) -> ExplorerAction {
    let mut action = ExplorerAction::None;

    ui.horizontal(|ui| {
        ui.heading("Explorer");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .button("Generate Samples")
                .on_hover_text("Create demo WAV and PCM fixtures in the current folder")
                .clicked()
            {
                action = ExplorerAction::GenerateSamplesHere(state.current_dir.clone());
            }
        });
    });

    ui.add_space(4.0);

    // Path bar + navigation controls
    ui.horizontal(|ui| {
        if ui
            .button("..")
            .on_hover_text("Go to parent folder")
            .clicked()
            && let Some(parent) = state.current_dir.parent()
        {
            state.navigate_to(parent.to_path_buf());
        }
        if ui
            .button("Refresh")
            .on_hover_text("Rescan current folder")
            .clicked()
        {
            state.refresh();
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let go_clicked = ui.button("Go").clicked();
            let text_w = ui.available_width().max(40.0);
            let text_resp = ui.add_sized(
                Vec2::new(text_w, 22.0),
                egui::TextEdit::singleline(&mut state.path_input).desired_width(text_w),
            );
            if go_clicked
                || (text_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
            {
                let target = PathBuf::from(state.path_input.trim());
                state.navigate_to(target);
            }
        });
    });

    ui.add_space(4.0);

    // Format filter chips
    ui.horizontal_wrapped(|ui| {
        for filter in FormatFilter::ALL {
            let selected = state.filter == filter;
            if ui.selectable_label(selected, filter.label()).clicked() {
                state.filter = filter;
            }
        }
    });

    if let Some(err) = &state.last_error {
        ui.colored_label(Color32::from_rgb(248, 113, 113), err);
    }

    ui.separator();

    // Collect filtered indices so we can borrow entries & lazily probe at most N items per frame.
    let visible_indices: Vec<usize> = state
        .entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| filter_entry(entry, state.filter))
        .map(|(idx, _)| idx)
        .collect();

    let mut probed_this_frame = 0usize;
    let mut has_unprobed = false;
    for &idx in &visible_indices {
        let entry = &state.entries[idx];
        if entry.is_dir {
            continue;
        }
        let needs_probe = matches!(
            state.previews.get(&entry.path),
            None | Some(PreviewState::Loading)
        );
        if needs_probe {
            if probed_this_frame < MAX_PROBES_PER_FRAME {
                state
                    .previews
                    .insert(entry.path.clone(), PreviewState::Loading);
                let result = match probe_preview(&entry.path, THUMBNAIL_BUCKETS) {
                    Ok((metadata, peaks)) => PreviewState::Ready { metadata, peaks },
                    Err(err) => PreviewState::Error(err.to_string()),
                };
                state.previews.insert(entry.path.clone(), result);
                probed_this_frame += 1;
            } else {
                has_unprobed = true;
            }
        }
    }
    if has_unprobed {
        ui.ctx().request_repaint();
    }

    let mut pending_dir: Option<PathBuf> = None;

    egui::ScrollArea::vertical()
        .id_salt("explorer_entries")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if visible_indices.is_empty() {
                ui.label(
                    egui::RichText::new("No matching audio files in this folder.")
                        .color(Color32::GRAY),
                );
                return;
            }

            for &idx in &visible_indices {
                let entry = &state.entries[idx];
                if entry.is_dir {
                    let row_res = ui.selectable_label(false, format!("📁  {}", entry.name));
                    if row_res.clicked() {
                        pending_dir = Some(entry.path.clone());
                    }
                    continue;
                }

                let is_selected = state.selected_file.as_ref() == Some(&entry.path);
                let card_fill = if is_selected {
                    Color32::from_rgb(29, 44, 58)
                } else {
                    Color32::from_rgb(23, 27, 35)
                };
                let card_stroke = if is_selected {
                    Stroke::new(1.0, Color32::from_rgb(45, 212, 191))
                } else {
                    Stroke::new(1.0, Color32::from_rgb(38, 45, 56))
                };

                let mut open_clicked = false;
                egui::Frame::NONE
                    .fill(card_fill)
                    .stroke(card_stroke)
                    .corner_radius(CornerRadius::same(5))
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            if let Some(fmt) = entry.format {
                                let (bg, fg) = format_badge_colors(fmt);
                                draw_badge(ui, fmt.label(), bg, fg);
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .small_button("▶")
                                        .on_hover_text("Quick audition")
                                        .clicked()
                                    {
                                        action =
                                            ExplorerAction::QuickPreviewPlay(entry.path.clone());
                                    }
                                    ui.label(
                                        egui::RichText::new(format_human_size(entry.size_bytes))
                                            .size(11.0)
                                            .color(Color32::from_rgb(148, 163, 184)),
                                    );
                                    ui.with_layout(
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            let title_resp = ui.add(
                                                egui::Label::new(
                                                    egui::RichText::new(&entry.name)
                                                        .strong()
                                                        .color(Color32::from_rgb(226, 232, 240)),
                                                )
                                                .truncate()
                                                .sense(Sense::click()),
                                            );
                                            if title_resp.clicked() {
                                                open_clicked = true;
                                            }
                                        },
                                    );
                                },
                            );
                        });

                        ui.add_space(3.0);

                        match state.previews.get(&entry.path) {
                            Some(PreviewState::Ready { metadata, peaks }) => {
                                if draw_mini_waveform(ui, peaks, is_selected).clicked() {
                                    open_clicked = true;
                                }
                                let meta_resp = ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(format_compact_metadata(metadata))
                                            .size(11.0)
                                            .color(Color32::from_rgb(148, 163, 184)),
                                    )
                                    .sense(Sense::click()),
                                );
                                if meta_resp.clicked() {
                                    open_clicked = true;
                                }
                            }
                            Some(PreviewState::Error(err)) => {
                                let err_resp = ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(format!("Preview unavailable: {err}"))
                                            .size(11.0)
                                            .color(Color32::from_rgb(248, 113, 113)),
                                    )
                                    .sense(Sense::click()),
                                );
                                if err_resp.clicked() {
                                    open_clicked = true;
                                }
                            }
                            _ => {
                                if draw_mini_waveform(ui, &[], is_selected).clicked() {
                                    open_clicked = true;
                                }
                                ui.label(
                                    egui::RichText::new("Probing...")
                                        .size(11.0)
                                        .color(Color32::GRAY),
                                );
                            }
                        }
                    });

                if open_clicked && matches!(action, ExplorerAction::None) {
                    state.selected_file = Some(entry.path.clone());
                    action = if entry.format == Some(AudioFormat::RawPcm) {
                        ExplorerAction::OpenRawPcm(entry.path.clone())
                    } else {
                        ExplorerAction::OpenFile(entry.path.clone())
                    };
                }

                ui.add_space(4.0);
            }
        });

    if let Some(dir) = pending_dir {
        state.navigate_to(dir);
    }

    action
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(name: &str, is_dir: bool, format: Option<AudioFormat>) -> DirEntryInfo {
        DirEntryInfo {
            path: PathBuf::from(name),
            name: name.to_string(),
            is_dir,
            size_bytes: 1024,
            format,
        }
    }

    #[test]
    fn sort_entries_puts_folders_first_then_case_insensitive_alpha() {
        let mut entries = vec![
            make_entry("zeta.wav", false, Some(AudioFormat::Wav)),
            make_entry("Alpha.wav", false, Some(AudioFormat::Wav)),
            make_entry("sub_b", true, None),
            make_entry("Sub_A", true, None),
            make_entry("beat.mp3", false, Some(AudioFormat::Mp3)),
        ];

        sort_entries(&mut entries);
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["Sub_A", "sub_b", "Alpha.wav", "beat.mp3", "zeta.wav"]
        );
    }

    #[test]
    fn filter_entry_respects_format_chips_and_keeps_directories() {
        let dir = make_entry("samples", true, None);
        let wav = make_entry("tone.wav", false, Some(AudioFormat::Wav));
        let pcm = make_entry("raw.pcm", false, Some(AudioFormat::RawPcm));
        let flac = make_entry("lossless.flac", false, Some(AudioFormat::Flac));
        let mp3 = make_entry("track.mp3", false, Some(AudioFormat::Mp3));
        let ogg = make_entry("voice.ogg", false, Some(AudioFormat::Ogg));

        assert!(filter_entry(&dir, FormatFilter::Wav));
        assert!(filter_entry(&wav, FormatFilter::All));
        assert!(filter_entry(&wav, FormatFilter::Wav));
        assert!(!filter_entry(&wav, FormatFilter::Flac));

        assert!(filter_entry(&pcm, FormatFilter::RawPcm));
        assert!(!filter_entry(&pcm, FormatFilter::Wav));
        assert!(filter_entry(&flac, FormatFilter::Flac));
        assert!(filter_entry(&mp3, FormatFilter::Mp3));
        assert!(filter_entry(&ogg, FormatFilter::Ogg));
    }

    #[test]
    fn human_size_formatting_covers_bytes_kb_mb() {
        assert_eq!(format_human_size(512), "512 B");
        assert_eq!(format_human_size(2048), "2.0 KB");
        assert_eq!(format_human_size(1_572_864), "1.50 MB");
    }
}
