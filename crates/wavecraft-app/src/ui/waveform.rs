use std::ops::Range;

use egui::{Color32, CornerRadius, Pos2, Rect, Sense, Stroke, Vec2};
use wavecraft_core::EditableAudio;

/// Active interactive tool in the central waveform editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WaveformTool {
    Select,
    Pencil,
    Smooth,
}

impl WaveformTool {
    pub fn label(self) -> &'static str {
        match self {
            Self::Select => "Select",
            Self::Pencil => "Pencil",
            Self::Smooth => "Smooth",
        }
    }
}

/// Persistent view and interaction state for the waveform canvas.
#[derive(Debug, Clone)]
pub struct WaveformViewState {
    pub tool: WaveformTool,
    pub zoom: f32,
    #[allow(dead_code)] // Populated in interactive waveform zoom/scroll ticket
    pub scroll_offset_frames: f64,
    pub smooth_radius: usize,
    pub snap_zero_crossing: bool,
    pub show_grid: bool,
    #[allow(dead_code)] // Populated in interactive pencil stroke ticket
    pub active_stroke: Vec<(usize, f32)>,
}

impl Default for WaveformViewState {
    fn default() -> Self {
        Self {
            tool: WaveformTool::Select,
            zoom: 1.0,
            scroll_offset_frames: 0.0,
            smooth_radius: 3,
            snap_zero_crossing: false,
            show_grid: true,
            active_stroke: Vec::new(),
        }
    }
}

/// Action emitted by [`show_waveform_panel`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaveformAction {
    None,
    Seek(usize),
    #[allow(dead_code)] // Emitted by interactive waveform editing tools
    BufferEdited,
}

/// Renders the central waveform panel scaffold with a multi-channel `egui::Painter` overview,
/// playhead cursor, selection overlay, and click-to-seek.
pub fn show_waveform_panel(
    ui: &mut egui::Ui,
    editable: &mut EditableAudio,
    selection: &mut Option<Range<usize>>,
    active_channel: &mut Option<usize>,
    playhead_frame: usize,
    state: &mut WaveformViewState,
) -> WaveformAction {
    let mut action = WaveformAction::None;
    let buf = editable.buffer();
    let num_channels = buf.num_channels();
    let num_frames = buf.num_frames();

    // Tool & status strip
    ui.horizontal_wrapped(|ui| {
        for tool in [
            WaveformTool::Select,
            WaveformTool::Pencil,
            WaveformTool::Smooth,
        ] {
            if ui
                .selectable_label(state.tool == tool, tool.label())
                .clicked()
            {
                state.tool = tool;
            }
        }

        ui.separator();

        if num_channels > 1 {
            if ui
                .selectable_label(active_channel.is_none(), "All Ch")
                .clicked()
            {
                *active_channel = None;
            }
            for ch in 0..num_channels {
                if ui
                    .selectable_label(*active_channel == Some(ch), format!("Ch {}", ch + 1))
                    .clicked()
                {
                    *active_channel = Some(ch);
                }
            }
            ui.separator();
        }

        ui.checkbox(&mut state.show_grid, "Grid");
        ui.checkbox(&mut state.snap_zero_crossing, "Snap 0-X");
        if state.tool == WaveformTool::Smooth {
            ui.add(
                egui::DragValue::new(&mut state.smooth_radius)
                    .range(1..=32)
                    .prefix("r="),
            );
        }
        ui.separator();

        ui.label(
            egui::RichText::new(format!(
                "{} Hz • {} ch • {} frames • {:.3}s • Zoom {:.1}x",
                buf.sample_rate,
                num_channels,
                num_frames,
                buf.duration_secs(),
                state.zoom
            ))
            .size(12.0)
            .color(Color32::from_rgb(148, 163, 184)),
        );
    });

    ui.separator();

    let desired = Vec2::new(
        ui.available_width().max(120.0),
        ui.available_height().max(140.0),
    );
    let (rect, response) = ui.allocate_exact_size(desired, Sense::click_and_drag());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, CornerRadius::same(4), Color32::from_rgb(15, 19, 26));
    painter.rect_stroke(
        rect,
        CornerRadius::same(4),
        Stroke::new(1.0, Color32::from_rgb(38, 45, 56)),
        egui::StrokeKind::Inside,
    );

    if num_channels == 0 || num_frames == 0 {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Empty AudioBuffer",
            egui::FontId::proportional(14.0),
            Color32::GRAY,
        );
        return action;
    }

    if state.show_grid {
        let divisions = 8;
        for d in 1..divisions {
            let x = rect.left() + (d as f32 / divisions as f32) * rect.width();
            painter.line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                Stroke::new(1.0, Color32::from_rgb(26, 32, 42)),
            );
        }
    }

    let lane_h = rect.height() / num_channels as f32;
    let bucket_count = (rect.width().round() as usize).clamp(64, 2048);

    for ch in 0..num_channels {
        let lane_top = rect.top() + ch as f32 * lane_h;
        let lane_rect = Rect::from_min_size(
            Pos2::new(rect.left(), lane_top),
            Vec2::new(rect.width(), lane_h),
        );
        let mid_y = lane_rect.center().y;

        if ch > 0 {
            painter.line_segment(
                [
                    Pos2::new(rect.left(), lane_top),
                    Pos2::new(rect.right(), lane_top),
                ],
                Stroke::new(1.0, Color32::from_rgb(34, 41, 52)),
            );
        }

        // Zero amplitude baseline
        painter.line_segment(
            [
                Pos2::new(rect.left(), mid_y),
                Pos2::new(rect.right(), mid_y),
            ],
            Stroke::new(1.0, Color32::from_rgb(45, 55, 72)),
        );

        let peaks = buf.compute_peaks(ch, 0..num_frames, bucket_count);
        let half_h = (lane_h * 0.44).max(1.0);
        let is_active = active_channel.is_none_or(|c| c == ch);
        let wave_color = if is_active {
            Color32::from_rgb(45, 212, 191)
        } else {
            Color32::from_rgb(71, 85, 105)
        };

        let n = peaks.len().max(1);
        for (i, bucket) in peaks.iter().enumerate() {
            let x = lane_rect.left() + ((i as f32 + 0.5) / n as f32) * lane_rect.width();
            let y_top = mid_y - bucket.max.clamp(-1.0, 1.0) * half_h;
            let y_bot = mid_y - bucket.min.clamp(-1.0, 1.0) * half_h;
            let y0 = y_top.min(y_bot);
            let y1 = y_top.max(y_bot).max(y0 + 1.0);
            painter.line_segment(
                [Pos2::new(x, y0), Pos2::new(x, y1)],
                Stroke::new(1.2, wave_color),
            );
        }
    }

    // Selection overlay
    if let Some(sel) = selection.as_ref()
        && sel.start < sel.end
    {
        let x0 =
            rect.left() + (sel.start.min(num_frames) as f32 / num_frames as f32) * rect.width();
        let x1 = rect.left() + (sel.end.min(num_frames) as f32 / num_frames as f32) * rect.width();
        let sel_rect = Rect::from_min_max(Pos2::new(x0, rect.top()), Pos2::new(x1, rect.bottom()));
        painter.rect_filled(
            sel_rect,
            CornerRadius::ZERO,
            Color32::from_rgba_unmultiplied(56, 189, 248, 40),
        );
    }

    // Playhead cursor
    let ph_frac = (playhead_frame.min(num_frames) as f32) / (num_frames as f32);
    let ph_x = rect.left() + ph_frac * rect.width();
    painter.line_segment(
        [Pos2::new(ph_x, rect.top()), Pos2::new(ph_x, rect.bottom())],
        Stroke::new(1.5, Color32::from_rgb(245, 158, 11)),
    );

    // Click-to-seek scaffold
    if response.clicked()
        && let Some(pos) = response.interact_pointer_pos()
    {
        let frac = ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
        let frame = (frac * num_frames as f32).round() as usize;
        action = WaveformAction::Seek(frame.min(num_frames));
    }

    action
}
