use std::ops::Range;

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Shape, Stroke, Vec2};
use wavecraft_core::EditableAudio;

/// Minimum zoomed-in scale: ~33.3 pixels per sample.
pub const MIN_SAMPLES_PER_PIXEL: f64 = 0.03;

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
            Self::Select => "Select / Scrub",
            Self::Pencil => "Pencil (Draw)",
            Self::Smooth => "Smooth",
        }
    }
}

/// Persistent view and interaction state for the waveform canvas.
#[derive(Debug, Clone)]
pub struct WaveformViewState {
    pub tool: WaveformTool,
    pub samples_per_pixel: f64,
    pub scroll_offset_frames: f64,
    pub smooth_radius: usize,
    pub show_wavelength_grid: bool,
    pub active_stroke: Vec<(usize, f32)>,
    pub fit_to_view: bool,
    pub last_canvas_width: f32,
    active_channel_target: Option<usize>,
    drag_select_anchor: Option<usize>,
    last_smooth_frame: Option<usize>,
    middle_panning: bool,
    scrollbar_dragging: bool,
}

impl Default for WaveformViewState {
    fn default() -> Self {
        Self {
            tool: WaveformTool::Select,
            samples_per_pixel: 1.0,
            scroll_offset_frames: 0.0,
            smooth_radius: 3,
            show_wavelength_grid: true,
            active_stroke: Vec::new(),
            fit_to_view: true,
            last_canvas_width: 800.0,
            active_channel_target: None,
            drag_select_anchor: None,
            last_smooth_frame: None,
            middle_panning: false,
            scrollbar_dragging: false,
        }
    }
}

impl WaveformViewState {
    /// Resets zoom and scroll so the next rendered frame fits the full buffer.
    pub fn reset_view_for_new_buffer(&mut self) {
        self.fit_to_view = true;
        self.scroll_offset_frames = 0.0;
        self.active_stroke.clear();
        self.active_channel_target = None;
        self.drag_select_anchor = None;
        self.last_smooth_frame = None;
    }

    /// Zooms in around the horizontal center of the canvas.
    pub fn zoom_in(&mut self, num_frames: usize) {
        self.zoom_by_factor(1.0 / 1.5, num_frames);
    }

    /// Zooms out around the horizontal center of the canvas.
    pub fn zoom_out(&mut self, num_frames: usize) {
        self.zoom_by_factor(1.5, num_frames);
    }

    /// Fits the entire buffer into the canvas width.
    pub fn zoom_fit(&mut self, num_frames: usize) {
        let width = self.last_canvas_width.max(1.0);
        self.samples_per_pixel = fit_samples_per_pixel(num_frames, width);
        self.scroll_offset_frames = 0.0;
        self.fit_to_view = false;
    }

    /// Sets zoom to 1 sample per pixel anchored at the canvas center.
    pub fn zoom_one_to_one(&mut self, num_frames: usize) {
        let width = self.last_canvas_width.max(1.0);
        let center_x = width * 0.5;
        let (new_spp, new_scroll) = zoom_around_x(
            self.scroll_offset_frames,
            self.samples_per_pixel,
            1.0,
            center_x,
            0.0,
            num_frames,
            width,
        );
        self.samples_per_pixel = new_spp;
        self.scroll_offset_frames = new_scroll;
        self.fit_to_view = false;
    }

    fn zoom_by_factor(&mut self, factor: f64, num_frames: usize) {
        let width = self.last_canvas_width.max(1.0);
        if self.fit_to_view {
            self.samples_per_pixel = fit_samples_per_pixel(num_frames, width);
            self.fit_to_view = false;
        }
        let center_x = width * 0.5;
        let (new_spp, new_scroll) = zoom_around_x(
            self.scroll_offset_frames,
            self.samples_per_pixel,
            self.samples_per_pixel * factor,
            center_x,
            0.0,
            num_frames,
            width,
        );
        self.samples_per_pixel = new_spp;
        self.scroll_offset_frames = new_scroll;
    }
}

/// Action emitted by [`show_waveform_panel`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaveformAction {
    None,
    Seek(usize),
    SelectionChanged,
    BufferEdited,
    EditCommitted,
}

/// Computes the `samples_per_pixel` needed to fit `num_frames` across `canvas_width`.
pub fn fit_samples_per_pixel(num_frames: usize, canvas_width: f32) -> f64 {
    if num_frames == 0 || canvas_width <= 1.0 {
        return 1.0;
    }
    (num_frames as f64 / canvas_width as f64).max(MIN_SAMPLES_PER_PIXEL)
}

/// Clamps `scroll_offset_frames` so the visible window stays within `[0, num_frames]`.
pub fn clamp_scroll_offset(
    scroll_offset_frames: f64,
    num_frames: usize,
    samples_per_pixel: f64,
    canvas_width: f32,
) -> f64 {
    if num_frames == 0 || canvas_width <= 1.0 || samples_per_pixel <= 0.0 {
        return 0.0;
    }
    let visible_frames = (canvas_width as f64) * samples_per_pixel;
    let max_scroll = (num_frames as f64 - visible_frames).max(0.0);
    scroll_offset_frames.clamp(0.0, max_scroll)
}

/// Converts a frame position to an X pixel coordinate.
pub fn frame_to_x(
    frame: f64,
    scroll_offset_frames: f64,
    samples_per_pixel: f64,
    rect_left: f32,
) -> f32 {
    if samples_per_pixel <= 0.0 {
        return rect_left;
    }
    rect_left + ((frame - scroll_offset_frames) / samples_per_pixel) as f32
}

/// Converts an X pixel coordinate to a fractional frame position.
pub fn x_to_frame(
    x: f32,
    scroll_offset_frames: f64,
    samples_per_pixel: f64,
    rect_left: f32,
) -> f64 {
    scroll_offset_frames + ((x - rect_left) as f64) * samples_per_pixel
}

/// Adjusts `samples_per_pixel` and `scroll_offset_frames` so the frame under `cursor_x`
/// stays anchored at `cursor_x`.
pub fn zoom_around_x(
    scroll_offset_frames: f64,
    current_spp: f64,
    target_spp: f64,
    cursor_x: f32,
    rect_left: f32,
    num_frames: usize,
    canvas_width: f32,
) -> (f64, f64) {
    let max_spp = fit_samples_per_pixel(num_frames, canvas_width).max(MIN_SAMPLES_PER_PIXEL);
    let new_spp = target_spp.clamp(MIN_SAMPLES_PER_PIXEL, max_spp);
    let anchor_frame = x_to_frame(cursor_x, scroll_offset_frames, current_spp, rect_left);
    let raw_scroll = anchor_frame - ((cursor_x - rect_left) as f64) * new_spp;
    let clamped_scroll = clamp_scroll_offset(raw_scroll, num_frames, new_spp, canvas_width);
    (new_spp, clamped_scroll)
}

/// Computes a clamped single-cycle frame range centered around `center_frame`.
pub fn single_cycle_selection(
    center_frame: usize,
    wavelength_samples: f64,
    num_frames: usize,
) -> Option<Range<usize>> {
    if num_frames == 0 {
        return None;
    }
    let cycle_len = if wavelength_samples.is_finite() && wavelength_samples >= 1.0 {
        (wavelength_samples.round() as usize).max(1)
    } else {
        1
    };
    if cycle_len >= num_frames {
        return Some(0..num_frames);
    }
    let half = cycle_len / 2;
    let mut start = center_frame.saturating_sub(half);
    let mut end = start.saturating_add(cycle_len);
    if end > num_frames {
        end = num_frames;
        start = end.saturating_sub(cycle_len);
    }
    Some(start..end)
}

fn channel_at_y(y: f32, wave_rect: Rect, num_channels: usize) -> usize {
    if num_channels <= 1 {
        return 0;
    }
    let lane_h = wave_rect.height().max(1.0) / num_channels as f32;
    let idx = ((y - wave_rect.top()) / lane_h).floor() as isize;
    idx.clamp(0, num_channels as isize - 1) as usize
}

fn channel_lane_rect(ch: usize, wave_rect: Rect, num_channels: usize) -> Rect {
    let lane_h = wave_rect.height() / num_channels.max(1) as f32;
    let top = wave_rect.top() + ch as f32 * lane_h;
    Rect::from_min_size(
        Pos2::new(wave_rect.left(), top),
        Vec2::new(wave_rect.width(), lane_h),
    )
}

fn y_to_amp(y: f32, lane_rect: Rect) -> f32 {
    let half_h = (lane_rect.height() * 0.44).max(1.0);
    let mid_y = lane_rect.center().y;
    ((mid_y - y) / half_h).clamp(-1.0, 1.0)
}

fn amp_to_y(amp: f32, lane_rect: Rect) -> f32 {
    let half_h = (lane_rect.height() * 0.44).max(1.0);
    let mid_y = lane_rect.center().y;
    mid_y - amp.clamp(-1.0, 1.0) * half_h
}

fn float_to_i16(sample: f32) -> i16 {
    let clamped = if sample.is_nan() {
        0.0
    } else {
        sample.clamp(-1.0, 1.0)
    };
    if clamped < 0.0 {
        (clamped * 32768.0).round().clamp(-32768.0, 0.0) as i16
    } else {
        (clamped * 32767.0).round().clamp(0.0, 32767.0) as i16
    }
}

fn format_ruler_time(secs: f64) -> String {
    if secs < 0.001 {
        format!("{:.0}µs", secs * 1_000_000.0)
    } else if secs < 1.0 {
        format!("{:.1}ms", secs * 1_000.0)
    } else {
        format!("{secs:.2}s")
    }
}

fn nice_step(min_step: f64) -> f64 {
    if min_step <= 0.0 || !min_step.is_finite() {
        return 1.0;
    }
    let exp = min_step.log10().floor();
    let base = 10.0_f64.powf(exp);
    let frac = min_step / base;
    let mult = if frac <= 1.0 {
        1.0
    } else if frac <= 2.0 {
        2.0
    } else if frac <= 5.0 {
        5.0
    } else {
        10.0
    };
    mult * base
}

/// Renders the central interactive granular waveform editor panel.
#[allow(clippy::too_many_arguments)]
pub fn show_waveform_panel(
    ui: &mut egui::Ui,
    editable: &mut EditableAudio,
    selection: &mut Option<Range<usize>>,
    active_channel: &mut Option<usize>,
    playhead_frame: usize,
    wavelength_samples: f64,
    loop_enabled: bool,
    state: &mut WaveformViewState,
) -> WaveformAction {
    let mut action = WaveformAction::None;
    let num_channels = editable.buffer().num_channels();
    let num_frames = editable.buffer().num_frames();
    let sample_rate = editable.buffer().sample_rate.max(1);

    // Clamp active_channel if buffer channel count shrank
    if let Some(ch) = *active_channel
        && ch >= num_channels
    {
        *active_channel = None;
    }

    // Toolbar row
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

        if state.tool == WaveformTool::Smooth {
            ui.separator();
            ui.label("Radius");
            ui.add(egui::Slider::new(&mut state.smooth_radius, 1..=32));
        }

        ui.separator();
        if ui
            .selectable_label(active_channel.is_none(), "All")
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
        ui.checkbox(&mut state.show_wavelength_grid, "Wavelength Grid");

        ui.separator();
        if ui.button("−").on_hover_text("Zoom Out (-)").clicked() {
            state.zoom_out(num_frames);
        }
        if ui.button("+").on_hover_text("Zoom In (= / +)").clicked() {
            state.zoom_in(num_frames);
        }
        if ui
            .button("Fit")
            .on_hover_text("Fit entire buffer")
            .clicked()
        {
            state.zoom_fit(num_frames);
        }
        if ui
            .button("1:1 Samples")
            .on_hover_text("1 sample per pixel")
            .clicked()
        {
            state.zoom_one_to_one(num_frames);
        }

        let zoom_readout = if state.samples_per_pixel < 1.0 {
            format!("{:.1} px/sample", 1.0 / state.samples_per_pixel)
        } else {
            format!("{:.1} samples/px", state.samples_per_pixel)
        };
        ui.label(
            egui::RichText::new(zoom_readout)
                .monospace()
                .size(11.5)
                .color(Color32::from_rgb(148, 163, 184)),
        );
    });

    ui.separator();

    // Reserve room at bottom for the HUD strip
    let hud_reserve = 26.0;
    let canvas_w = ui.available_width().max(120.0);
    let canvas_h = (ui.available_height() - hud_reserve).max(140.0);
    state.last_canvas_width = canvas_w;

    if state.fit_to_view {
        state.samples_per_pixel = fit_samples_per_pixel(num_frames, canvas_w);
        state.scroll_offset_frames = 0.0;
        state.fit_to_view = false;
    } else {
        let max_spp = fit_samples_per_pixel(num_frames, canvas_w).max(MIN_SAMPLES_PER_PIXEL);
        state.samples_per_pixel = state
            .samples_per_pixel
            .clamp(MIN_SAMPLES_PER_PIXEL, max_spp);
        state.scroll_offset_frames = clamp_scroll_offset(
            state.scroll_offset_frames,
            num_frames,
            state.samples_per_pixel,
            canvas_w,
        );
    }

    let (canvas_rect, _) = ui.allocate_exact_size(Vec2::new(canvas_w, canvas_h), Sense::hover());

    // Background & border
    let bg_painter = ui.painter_at(canvas_rect);
    bg_painter.rect_filled(
        canvas_rect,
        CornerRadius::same(4),
        Color32::from_rgb(15, 19, 26),
    );
    bg_painter.rect_stroke(
        canvas_rect,
        CornerRadius::same(4),
        Stroke::new(1.0, Color32::from_rgb(38, 45, 56)),
        egui::StrokeKind::Inside,
    );

    let ruler_h = 22.0;
    let visible_frames = (canvas_w as f64) * state.samples_per_pixel;
    let is_zoomed_in = num_frames > 0 && visible_frames < (num_frames as f64 - 0.5);
    let scrollbar_h = if is_zoomed_in { 12.0 } else { 0.0 };

    let ruler_rect = Rect::from_min_max(
        canvas_rect.min,
        Pos2::new(
            canvas_rect.right(),
            (canvas_rect.top() + ruler_h).min(canvas_rect.bottom()),
        ),
    );
    let wave_rect = Rect::from_min_max(
        Pos2::new(canvas_rect.left(), ruler_rect.bottom()),
        Pos2::new(
            canvas_rect.right(),
            (canvas_rect.bottom() - scrollbar_h).max(ruler_rect.bottom() + 1.0),
        ),
    );
    let scrollbar_rect = Rect::from_min_max(
        Pos2::new(canvas_rect.left(), wave_rect.bottom()),
        canvas_rect.max,
    );

    if num_channels == 0 || num_frames == 0 {
        bg_painter.text(
            canvas_rect.center(),
            Align2::CENTER_CENTER,
            "Empty AudioBuffer",
            FontId::proportional(14.0),
            Color32::GRAY,
        );
        render_status_hud(ui, None, selection, sample_rate, &mut action);
        return action;
    }

    // Allocate interactive responses for waveform area & optional scrollbar
    let wave_id = ui.id().with("waveform_canvas_wave");
    let wave_resp = ui.interact(wave_rect, wave_id, Sense::click_and_drag());

    // Handle mouse wheel zoom & horizontal pan (Shift+Scroll or trackpad X or middle-mouse drag)
    let pointer_in_wave = wave_resp.hovered()
        || ui.input(|i| i.pointer.hover_pos().is_some_and(|p| wave_rect.contains(p)));
    if pointer_in_wave {
        let (scroll_delta, shift_down, pointer_pos) = ui.input(|i| {
            (
                i.smooth_scroll_delta,
                i.modifiers.shift,
                i.pointer.hover_pos(),
            )
        });
        if scroll_delta != Vec2::ZERO {
            if shift_down || scroll_delta.x.abs() > scroll_delta.y.abs() {
                let pan_px = if shift_down && scroll_delta.x.abs() < 0.01 {
                    scroll_delta.y
                } else {
                    scroll_delta.x
                };
                state.scroll_offset_frames = clamp_scroll_offset(
                    state.scroll_offset_frames - (pan_px as f64) * state.samples_per_pixel,
                    num_frames,
                    state.samples_per_pixel,
                    wave_rect.width(),
                );
            } else if scroll_delta.y.abs() > 0.0 {
                let factor = (-scroll_delta.y as f64 * 0.0025).exp();
                let cursor_x = pointer_pos.map_or(wave_rect.center().x, |p| p.x);
                let (new_spp, new_scroll) = zoom_around_x(
                    state.scroll_offset_frames,
                    state.samples_per_pixel,
                    state.samples_per_pixel * factor,
                    cursor_x,
                    wave_rect.left(),
                    num_frames,
                    wave_rect.width(),
                );
                state.samples_per_pixel = new_spp;
                state.scroll_offset_frames = new_scroll;
            }
        }
    }

    // Middle-mouse drag panning
    let (mid_pressed, mid_down, ptr_delta, ptr_pos) = ui.input(|i| {
        (
            i.pointer.button_pressed(egui::PointerButton::Middle),
            i.pointer.button_down(egui::PointerButton::Middle),
            i.pointer.delta(),
            i.pointer.interact_pos(),
        )
    });
    if mid_pressed && ptr_pos.is_some_and(|p| wave_rect.contains(p)) {
        state.middle_panning = true;
    }
    if !mid_down {
        state.middle_panning = false;
    }
    if state.middle_panning && ptr_delta.x.abs() > 0.0 {
        state.scroll_offset_frames = clamp_scroll_offset(
            state.scroll_offset_frames - (ptr_delta.x as f64) * state.samples_per_pixel,
            num_frames,
            state.samples_per_pixel,
            wave_rect.width(),
        );
    }

    // Scrollbar / overview minimap interaction
    if is_zoomed_in && scrollbar_rect.height() >= 4.0 {
        let sb_id = ui.id().with("waveform_minimap_scrollbar");
        let sb_resp = ui.interact(scrollbar_rect, sb_id, Sense::click_and_drag());
        if sb_resp.drag_started() || sb_resp.clicked() {
            state.scrollbar_dragging = true;
        }
        if !ui.input(|i| i.pointer.primary_down()) {
            state.scrollbar_dragging = false;
        }
        if (sb_resp.dragged() || sb_resp.clicked())
            && let Some(pos) = sb_resp.interact_pointer_pos()
        {
            let frac = ((pos.x - scrollbar_rect.left()) / scrollbar_rect.width().max(1.0))
                .clamp(0.0, 1.0) as f64;
            let center_frame = frac * num_frames as f64;
            let vis = (wave_rect.width() as f64) * state.samples_per_pixel;
            state.scroll_offset_frames = clamp_scroll_offset(
                center_frame - vis * 0.5,
                num_frames,
                state.samples_per_pixel,
                wave_rect.width(),
            );
        }
    } else {
        state.scrollbar_dragging = false;
    }

    // Determine hovered sample for HUD & granular dot highlighting
    let hover_info = ui.input(|i| i.pointer.hover_pos()).and_then(|pos| {
        if !wave_rect.contains(pos) {
            return None;
        }
        let raw_f = x_to_frame(
            pos.x,
            state.scroll_offset_frames,
            state.samples_per_pixel,
            wave_rect.left(),
        );
        let frame = (raw_f.round() as isize).clamp(0, num_frames as isize - 1) as usize;
        let ch = active_channel.unwrap_or_else(|| channel_at_y(pos.y, wave_rect, num_channels));
        let amp = editable.buffer().get_sample(ch, frame).unwrap_or(0.0);
        Some((frame, ch, amp))
    });

    // Interactive tools on wave_resp (only when not middle panning)
    if !state.middle_panning {
        match state.tool {
            WaveformTool::Select => {
                if wave_resp.double_clicked()
                    && let Some(pos) = wave_resp.interact_pointer_pos()
                {
                    let raw_f = x_to_frame(
                        pos.x,
                        state.scroll_offset_frames,
                        state.samples_per_pixel,
                        wave_rect.left(),
                    );
                    let frame = (raw_f.round() as isize).clamp(0, num_frames as isize - 1) as usize;
                    *selection = single_cycle_selection(frame, wavelength_samples, num_frames);
                    action = WaveformAction::SelectionChanged;
                } else if wave_resp.drag_started()
                    && let Some(pos) = wave_resp.interact_pointer_pos()
                {
                    let raw_f = x_to_frame(
                        pos.x,
                        state.scroll_offset_frames,
                        state.samples_per_pixel,
                        wave_rect.left(),
                    );
                    let frame = (raw_f.round() as isize).clamp(0, num_frames as isize) as usize;
                    state.drag_select_anchor = Some(frame);
                    *selection = Some(frame..frame);
                } else if wave_resp.dragged()
                    && let Some(anchor) = state.drag_select_anchor
                    && let Some(pos) = wave_resp.interact_pointer_pos()
                {
                    let raw_f = x_to_frame(
                        pos.x,
                        state.scroll_offset_frames,
                        state.samples_per_pixel,
                        wave_rect.left(),
                    );
                    let frame = (raw_f.round() as isize).clamp(0, num_frames as isize) as usize;
                    let lo = anchor.min(frame);
                    let hi = anchor.max(frame).min(num_frames);
                    *selection = Some(lo..hi);
                    action = WaveformAction::SelectionChanged;
                } else if wave_resp.drag_stopped() {
                    if let Some(sel) = selection.as_ref()
                        && sel.start >= sel.end
                    {
                        let seek_f = sel.start.min(num_frames);
                        *selection = None;
                        action = WaveformAction::Seek(seek_f);
                    } else {
                        action = WaveformAction::SelectionChanged;
                    }
                    state.drag_select_anchor = None;
                } else if wave_resp.clicked()
                    && let Some(pos) = wave_resp.interact_pointer_pos()
                {
                    let raw_f = x_to_frame(
                        pos.x,
                        state.scroll_offset_frames,
                        state.samples_per_pixel,
                        wave_rect.left(),
                    );
                    let frame = (raw_f.round() as isize).clamp(0, num_frames as isize) as usize;
                    action = WaveformAction::Seek(frame);
                }
            }
            WaveformTool::Pencil => {
                if wave_resp.drag_started()
                    && let Some(pos) = wave_resp.interact_pointer_pos()
                {
                    let ch = active_channel
                        .unwrap_or_else(|| channel_at_y(pos.y, wave_rect, num_channels));
                    let lane = channel_lane_rect(ch, wave_rect, num_channels);
                    let raw_f = x_to_frame(
                        pos.x,
                        state.scroll_offset_frames,
                        state.samples_per_pixel,
                        wave_rect.left(),
                    );
                    let frame = (raw_f.round() as isize).clamp(0, num_frames as isize - 1) as usize;
                    let amp = y_to_amp(pos.y, lane);
                    state.active_channel_target = Some(ch);
                    state.active_stroke.clear();
                    state.active_stroke.push((frame, amp));
                    editable.push_undo_snapshot();
                    editable.draw_stroke_raw(ch, &[(frame, amp)]);
                    action = WaveformAction::BufferEdited;
                } else if wave_resp.dragged()
                    && let Some(pos) = wave_resp.interact_pointer_pos()
                {
                    let ch = state.active_channel_target.unwrap_or_else(|| {
                        active_channel
                            .unwrap_or_else(|| channel_at_y(pos.y, wave_rect, num_channels))
                    });
                    let lane = channel_lane_rect(ch, wave_rect, num_channels);
                    let raw_f = x_to_frame(
                        pos.x,
                        state.scroll_offset_frames,
                        state.samples_per_pixel,
                        wave_rect.left(),
                    );
                    let frame = (raw_f.round() as isize).clamp(0, num_frames as isize - 1) as usize;
                    let amp = y_to_amp(pos.y, lane);
                    if let Some(&prev) = state.active_stroke.last() {
                        editable.draw_stroke_raw(ch, &[prev, (frame, amp)]);
                    } else {
                        editable.draw_stroke_raw(ch, &[(frame, amp)]);
                    }
                    state.active_stroke.push((frame, amp));
                    action = WaveformAction::BufferEdited;
                } else if wave_resp.drag_stopped() {
                    state.active_stroke.clear();
                    state.active_channel_target = None;
                    action = WaveformAction::EditCommitted;
                } else if wave_resp.clicked()
                    && let Some(pos) = wave_resp.interact_pointer_pos()
                {
                    let ch = active_channel
                        .unwrap_or_else(|| channel_at_y(pos.y, wave_rect, num_channels));
                    let lane = channel_lane_rect(ch, wave_rect, num_channels);
                    let raw_f = x_to_frame(
                        pos.x,
                        state.scroll_offset_frames,
                        state.samples_per_pixel,
                        wave_rect.left(),
                    );
                    let frame = (raw_f.round() as isize).clamp(0, num_frames as isize - 1) as usize;
                    let amp = y_to_amp(pos.y, lane);
                    editable.push_undo_snapshot();
                    editable.draw_stroke_raw(ch, &[(frame, amp)]);
                    action = WaveformAction::EditCommitted;
                }
            }
            WaveformTool::Smooth => {
                let rad = state.smooth_radius.max(1);
                if wave_resp.drag_started()
                    && let Some(pos) = wave_resp.interact_pointer_pos()
                {
                    let ch = active_channel
                        .unwrap_or_else(|| channel_at_y(pos.y, wave_rect, num_channels));
                    let raw_f = x_to_frame(
                        pos.x,
                        state.scroll_offset_frames,
                        state.samples_per_pixel,
                        wave_rect.left(),
                    );
                    let frame = (raw_f.round() as isize).clamp(0, num_frames as isize - 1) as usize;
                    state.active_channel_target = Some(ch);
                    state.last_smooth_frame = Some(frame);
                    editable.push_undo_snapshot();
                    let lo = frame.saturating_sub(rad);
                    let hi = (frame + rad).min(num_frames - 1);
                    editable.smooth_range_raw(ch, lo..(hi + 1), rad);
                    action = WaveformAction::BufferEdited;
                } else if wave_resp.dragged()
                    && let Some(pos) = wave_resp.interact_pointer_pos()
                {
                    let ch = state.active_channel_target.unwrap_or_else(|| {
                        active_channel
                            .unwrap_or_else(|| channel_at_y(pos.y, wave_rect, num_channels))
                    });
                    let raw_f = x_to_frame(
                        pos.x,
                        state.scroll_offset_frames,
                        state.samples_per_pixel,
                        wave_rect.left(),
                    );
                    let frame = (raw_f.round() as isize).clamp(0, num_frames as isize - 1) as usize;
                    let prev = state.last_smooth_frame.unwrap_or(frame);
                    let lo = prev.min(frame).saturating_sub(rad);
                    let hi = (prev.max(frame) + rad).min(num_frames - 1);
                    editable.smooth_range_raw(ch, lo..(hi + 1), rad);
                    state.last_smooth_frame = Some(frame);
                    action = WaveformAction::BufferEdited;
                } else if wave_resp.drag_stopped() {
                    state.last_smooth_frame = None;
                    state.active_channel_target = None;
                    action = WaveformAction::EditCommitted;
                } else if wave_resp.clicked()
                    && let Some(pos) = wave_resp.interact_pointer_pos()
                {
                    let ch = active_channel
                        .unwrap_or_else(|| channel_at_y(pos.y, wave_rect, num_channels));
                    let raw_f = x_to_frame(
                        pos.x,
                        state.scroll_offset_frames,
                        state.samples_per_pixel,
                        wave_rect.left(),
                    );
                    let frame = (raw_f.round() as isize).clamp(0, num_frames as isize - 1) as usize;
                    let lo = frame.saturating_sub(rad);
                    let hi = (frame + rad).min(num_frames - 1);
                    editable.push_undo_snapshot();
                    editable.smooth_range_raw(ch, lo..(hi + 1), rad);
                    action = WaveformAction::EditCommitted;
                }
            }
        }
    }

    // --- Painting ---
    let buf = editable.buffer();
    let pixels_per_sample = 1.0 / state.samples_per_pixel;
    let vis_start_f = state.scroll_offset_frames.max(0.0);
    let vis_end_f = (state.scroll_offset_frames + visible_frames).min(num_frames as f64);

    // 1. Top sample/time ruler
    let ruler_painter = ui.painter_at(ruler_rect);
    ruler_painter.rect_filled(
        ruler_rect,
        CornerRadius::ZERO,
        Color32::from_rgb(20, 25, 34),
    );
    ruler_painter.line_segment(
        [
            Pos2::new(ruler_rect.left(), ruler_rect.bottom()),
            Pos2::new(ruler_rect.right(), ruler_rect.bottom()),
        ],
        Stroke::new(1.0, Color32::from_rgb(42, 50, 64)),
    );

    if pixels_per_sample >= 4.0 {
        let min_frame_step = (64.0 * state.samples_per_pixel).max(1.0);
        let step = nice_step(min_frame_step).max(1.0).round() as usize;
        let first_tick = ((vis_start_f.floor() as usize) / step) * step;
        let last_tick = (vis_end_f.ceil() as usize).min(num_frames);
        let mut f = first_tick;
        while f <= last_tick {
            let x = frame_to_x(
                f as f64,
                state.scroll_offset_frames,
                state.samples_per_pixel,
                wave_rect.left(),
            );
            if x >= ruler_rect.left() - 2.0 && x <= ruler_rect.right() + 2.0 {
                ruler_painter.line_segment(
                    [
                        Pos2::new(x, ruler_rect.bottom() - 5.0),
                        Pos2::new(x, ruler_rect.bottom()),
                    ],
                    Stroke::new(1.0, Color32::from_rgb(100, 116, 139)),
                );
                ruler_painter.text(
                    Pos2::new(x + 3.0, ruler_rect.center().y),
                    Align2::LEFT_CENTER,
                    format!("{f}"),
                    FontId::monospace(10.0),
                    Color32::from_rgb(148, 163, 184),
                );
            }
            f = f.saturating_add(step);
        }
    } else {
        let min_time_step = (72.0 * state.samples_per_pixel) / sample_rate as f64;
        let step_s = nice_step(min_time_step).max(1e-6);
        let start_s = vis_start_f / sample_rate as f64;
        let end_s = vis_end_f / sample_rate as f64;
        let first_k = (start_s / step_s).floor() as i64;
        let last_k = (end_s / step_s).ceil() as i64;
        for k in first_k..=last_k {
            let t = k as f64 * step_s;
            if t < 0.0 {
                continue;
            }
            let x = frame_to_x(
                t * sample_rate as f64,
                state.scroll_offset_frames,
                state.samples_per_pixel,
                wave_rect.left(),
            );
            if x >= ruler_rect.left() - 2.0 && x <= ruler_rect.right() + 2.0 {
                ruler_painter.line_segment(
                    [
                        Pos2::new(x, ruler_rect.bottom() - 5.0),
                        Pos2::new(x, ruler_rect.bottom()),
                    ],
                    Stroke::new(1.0, Color32::from_rgb(100, 116, 139)),
                );
                ruler_painter.text(
                    Pos2::new(x + 3.0, ruler_rect.center().y),
                    Align2::LEFT_CENTER,
                    format_ruler_time(t),
                    FontId::monospace(10.0),
                    Color32::from_rgb(148, 163, 184),
                );
            }
        }
    }

    // 2. Waveform painter
    let painter = ui.painter_at(wave_rect);

    // Wavelength Grid Overlay
    if state.show_wavelength_grid && wavelength_samples.is_finite() && wavelength_samples > 1.0 {
        let px_per_cycle = wavelength_samples / state.samples_per_pixel;
        if px_per_cycle >= 4.0 {
            let anchor = selection.as_ref().map_or(0.0, |s| s.start as f64);
            let k_min = ((vis_start_f - anchor) / wavelength_samples).floor() as i64;
            let k_max = ((vis_end_f - anchor) / wavelength_samples).ceil() as i64;
            let grid_color = Color32::from_rgba_unmultiplied(167, 139, 250, 75);
            for k in k_min..=k_max {
                let frame_pos = anchor + k as f64 * wavelength_samples;
                if frame_pos < 0.0 || frame_pos > num_frames as f64 {
                    continue;
                }
                let x = frame_to_x(
                    frame_pos,
                    state.scroll_offset_frames,
                    state.samples_per_pixel,
                    wave_rect.left(),
                );
                painter.add(Shape::dashed_line(
                    &[
                        Pos2::new(x, wave_rect.top()),
                        Pos2::new(x, wave_rect.bottom()),
                    ],
                    Stroke::new(1.0, grid_color),
                    4.0,
                    4.0,
                ));
            }
        }
    }

    // Channel lanes & waveforms
    let vis_lo = (vis_start_f.floor() as usize).min(num_frames.saturating_sub(1));
    let vis_hi = (vis_end_f.ceil() as usize).min(num_frames);

    for ch in 0..num_channels {
        let lane_rect = channel_lane_rect(ch, wave_rect, num_channels);
        let mid_y = lane_rect.center().y;

        if ch > 0 {
            painter.line_segment(
                [
                    Pos2::new(wave_rect.left(), lane_rect.top()),
                    Pos2::new(wave_rect.right(), lane_rect.top()),
                ],
                Stroke::new(1.0, Color32::from_rgb(34, 41, 52)),
            );
        }

        // Zero amplitude baseline
        painter.line_segment(
            [
                Pos2::new(wave_rect.left(), mid_y),
                Pos2::new(wave_rect.right(), mid_y),
            ],
            Stroke::new(1.0, Color32::from_rgb(45, 55, 72)),
        );

        let is_active = active_channel.is_none_or(|c| c == ch);
        let wave_color = if is_active {
            Color32::from_rgb(45, 212, 191)
        } else {
            Color32::from_rgb(71, 85, 105)
        };
        let fill_color = if is_active {
            Color32::from_rgba_unmultiplied(45, 212, 191, 48)
        } else {
            Color32::from_rgba_unmultiplied(71, 85, 105, 32)
        };

        if pixels_per_sample < 4.0 {
            // Peak envelope mode using AudioBuffer::compute_peaks
            let range_len = vis_hi.saturating_sub(vis_lo);
            if range_len > 0 {
                let x_start = frame_to_x(
                    vis_lo as f64,
                    state.scroll_offset_frames,
                    state.samples_per_pixel,
                    wave_rect.left(),
                );
                let x_end = frame_to_x(
                    vis_hi as f64,
                    state.scroll_offset_frames,
                    state.samples_per_pixel,
                    wave_rect.left(),
                );
                let span_px = (x_end - x_start).max(1.0);
                let bucket_count = (span_px.round() as usize).clamp(1, 4096);
                let peaks = buf.compute_peaks(ch, vis_lo..vis_hi, bucket_count);
                let n = peaks.len().max(1);
                let mut top_pts = Vec::with_capacity(n);
                let mut bot_pts = Vec::with_capacity(n);
                for (i, bucket) in peaks.iter().enumerate() {
                    let x = x_start + ((i as f32 + 0.5) / n as f32) * span_px;
                    let y_top = amp_to_y(bucket.max, lane_rect);
                    let y_bot = amp_to_y(bucket.min, lane_rect);
                    painter.line_segment(
                        [Pos2::new(x, y_top), Pos2::new(x, y_bot)],
                        Stroke::new(1.2, fill_color),
                    );
                    top_pts.push(Pos2::new(x, y_top));
                    bot_pts.push(Pos2::new(x, y_bot));
                }
                if top_pts.len() >= 2 {
                    painter.add(Shape::line(top_pts, Stroke::new(1.2, wave_color)));
                    painter.add(Shape::line(bot_pts, Stroke::new(1.2, wave_color)));
                }
            }
        } else {
            // Granular sample mode: polyline + stems + circular sample dots
            let start_f = vis_lo.saturating_sub(1);
            let end_f = (vis_hi + 1).min(num_frames);
            let mut poly_pts = Vec::with_capacity(end_f.saturating_sub(start_f));
            let stem_color = if is_active {
                Color32::from_rgba_unmultiplied(45, 212, 191, 80)
            } else {
                Color32::from_rgba_unmultiplied(71, 85, 105, 50)
            };
            let base_dot_r = (pixels_per_sample as f32 * 0.14).clamp(2.2, 4.2);

            for f in start_f..end_f {
                let val = buf.channels[ch][f];
                let x = frame_to_x(
                    f as f64,
                    state.scroll_offset_frames,
                    state.samples_per_pixel,
                    wave_rect.left(),
                );
                let y = amp_to_y(val, lane_rect);
                let pt = Pos2::new(x, y);
                poly_pts.push((f, pt));
            }

            if poly_pts.len() >= 2 {
                let pts: Vec<Pos2> = poly_pts.iter().map(|&(_, p)| p).collect();
                painter.add(Shape::line(pts, Stroke::new(1.5, wave_color)));
            }

            for &(f, pt) in &poly_pts {
                painter.line_segment([Pos2::new(pt.x, mid_y), pt], Stroke::new(1.0, stem_color));
                let is_hovered = hover_info
                    .is_some_and(|(hf, hch, _)| hf == f && (hch == ch || active_channel.is_none()));
                if is_hovered {
                    painter.circle_filled(pt, base_dot_r + 2.2, Color32::from_rgb(251, 191, 36));
                    painter.circle_stroke(pt, base_dot_r + 3.2, Stroke::new(1.2, Color32::WHITE));
                } else {
                    painter.circle_filled(pt, base_dot_r, wave_color);
                }
            }
        }

        // Channel badge for multi-channel files
        if num_channels > 1 {
            painter.text(
                Pos2::new(lane_rect.left() + 6.0, lane_rect.top() + 5.0),
                Align2::LEFT_TOP,
                format!("Ch {}", ch + 1),
                FontId::monospace(10.5),
                if is_active {
                    Color32::from_rgb(148, 163, 184)
                } else {
                    Color32::from_rgb(71, 85, 105)
                },
            );
        }
    }

    // Selection overlay
    if let Some(sel) = selection.as_ref()
        && sel.start < sel.end
    {
        let x0 = frame_to_x(
            sel.start.min(num_frames) as f64,
            state.scroll_offset_frames,
            state.samples_per_pixel,
            wave_rect.left(),
        );
        let x1 = frame_to_x(
            sel.end.min(num_frames) as f64,
            state.scroll_offset_frames,
            state.samples_per_pixel,
            wave_rect.left(),
        );
        let clamped_x0 = x0.clamp(wave_rect.left(), wave_rect.right());
        let clamped_x1 = x1.clamp(wave_rect.left(), wave_rect.right());
        if clamped_x1 > clamped_x0 {
            let sel_rect = Rect::from_min_max(
                Pos2::new(clamped_x0, wave_rect.top()),
                Pos2::new(clamped_x1, wave_rect.bottom()),
            );
            painter.rect_filled(
                sel_rect,
                CornerRadius::ZERO,
                Color32::from_rgba_unmultiplied(56, 189, 248, 42),
            );
        }
        let edge_stroke = Stroke::new(1.2, Color32::from_rgb(56, 189, 248));
        if x0 >= wave_rect.left() && x0 <= wave_rect.right() {
            painter.line_segment(
                [
                    Pos2::new(x0, wave_rect.top()),
                    Pos2::new(x0, wave_rect.bottom()),
                ],
                edge_stroke,
            );
        }
        if x1 >= wave_rect.left() && x1 <= wave_rect.right() {
            painter.line_segment(
                [
                    Pos2::new(x1, wave_rect.top()),
                    Pos2::new(x1, wave_rect.bottom()),
                ],
                edge_stroke,
            );
        }
    }

    // Loop region markers when loop enabled
    if loop_enabled {
        let loop_range = match selection.as_ref() {
            Some(sel) if sel.start < sel.end && sel.end <= num_frames => sel.clone(),
            _ => 0..num_frames,
        };
        let lx0 = frame_to_x(
            loop_range.start as f64,
            state.scroll_offset_frames,
            state.samples_per_pixel,
            wave_rect.left(),
        );
        let lx1 = frame_to_x(
            loop_range.end as f64,
            state.scroll_offset_frames,
            state.samples_per_pixel,
            wave_rect.left(),
        );
        let loop_color = Color32::from_rgb(52, 211, 153);
        let loop_stroke = Stroke::new(1.5, loop_color);
        if lx0 >= wave_rect.left() && lx0 <= wave_rect.right() {
            painter.line_segment(
                [
                    Pos2::new(lx0, wave_rect.top()),
                    Pos2::new(lx0, wave_rect.bottom()),
                ],
                loop_stroke,
            );
            painter.text(
                Pos2::new(lx0 + 3.0, wave_rect.top() + 3.0),
                Align2::LEFT_TOP,
                "LOOP [",
                FontId::monospace(9.5),
                loop_color,
            );
        }
        if lx1 >= wave_rect.left() && lx1 <= wave_rect.right() {
            painter.line_segment(
                [
                    Pos2::new(lx1, wave_rect.top()),
                    Pos2::new(lx1, wave_rect.bottom()),
                ],
                loop_stroke,
            );
            painter.text(
                Pos2::new(lx1 - 3.0, wave_rect.top() + 3.0),
                Align2::RIGHT_TOP,
                "] LOOP",
                FontId::monospace(9.5),
                loop_color,
            );
        }
    }

    // Playhead cursor + top triangle handle in ruler
    let ph_x = frame_to_x(
        playhead_frame.min(num_frames) as f64,
        state.scroll_offset_frames,
        state.samples_per_pixel,
        wave_rect.left(),
    );
    if ph_x >= wave_rect.left() && ph_x <= wave_rect.right() {
        let ph_color = Color32::from_rgb(245, 158, 11);
        painter.line_segment(
            [
                Pos2::new(ph_x, wave_rect.top()),
                Pos2::new(ph_x, wave_rect.bottom()),
            ],
            Stroke::new(1.6, ph_color),
        );
        ruler_painter.add(Shape::convex_polygon(
            vec![
                Pos2::new(ph_x - 5.0, ruler_rect.bottom() - 8.0),
                Pos2::new(ph_x + 5.0, ruler_rect.bottom() - 8.0),
                Pos2::new(ph_x, ruler_rect.bottom()),
            ],
            ph_color,
            Stroke::NONE,
        ));
    }

    // Scrollbar / overview minimap bar at bottom of canvas when zoomed in
    if is_zoomed_in && scrollbar_rect.height() >= 4.0 {
        let sb_painter = ui.painter_at(scrollbar_rect);
        sb_painter.rect_filled(
            scrollbar_rect,
            CornerRadius::ZERO,
            Color32::from_rgb(18, 22, 30),
        );
        let start_frac = (state.scroll_offset_frames / num_frames as f64).clamp(0.0, 1.0) as f32;
        let width_frac = (visible_frames / num_frames as f64).clamp(0.02, 1.0) as f32;
        let thumb_x0 = scrollbar_rect.left() + start_frac * scrollbar_rect.width();
        let thumb_w = (width_frac * scrollbar_rect.width()).max(14.0);
        let thumb_x1 = (thumb_x0 + thumb_w).min(scrollbar_rect.right());
        let thumb_rect = Rect::from_min_max(
            Pos2::new(thumb_x0, scrollbar_rect.top() + 2.0),
            Pos2::new(thumb_x1, scrollbar_rect.bottom() - 2.0),
        );
        sb_painter.rect_filled(
            thumb_rect,
            CornerRadius::same(3),
            Color32::from_rgb(56, 189, 248),
        );
    }

    // Bottom Status HUD
    render_status_hud(ui, hover_info, selection, sample_rate, &mut action);

    action
}

fn render_status_hud(
    ui: &mut egui::Ui,
    hover_info: Option<(usize, usize, f32)>,
    selection: &mut Option<Range<usize>>,
    sample_rate: u32,
    action: &mut WaveformAction,
) {
    ui.horizontal(|ui| {
        if let Some((frame, ch, amp)) = hover_info {
            let time_ms = (frame as f64 / sample_rate.max(1) as f64) * 1000.0;
            let amp_i16 = float_to_i16(amp);
            ui.label(
                egui::RichText::new(format!(
                    "Sample #{frame} (Ch {})  |  Time: {time_ms:.3} ms  |  Amp: {amp:+.4} (i16: {amp_i16:+})",
                    ch + 1
                ))
                .monospace()
                .size(11.5)
                .color(Color32::from_rgb(203, 213, 225)),
            );
        } else {
            ui.label(
                egui::RichText::new("Hover waveform for sample readout  •  Scroll: Zoom  •  Shift+Scroll / Middle-drag: Pan")
                    .size(11.5)
                    .color(Color32::from_rgb(100, 116, 139)),
            );
        }

        if let Some(sel) = selection.as_ref()
            && sel.start < sel.end
        {
            let count = sel.end - sel.start;
            let ms = (count as f64 / sample_rate.max(1) as f64) * 1000.0;
            let lambda_hz = sample_rate as f64 / count as f64;
            ui.separator();
            ui.label(
                egui::RichText::new(format!(
                    "Selection: [{}..{}] ({count} samples | {ms:.2} ms | λ freq: {lambda_hz:.2} Hz)",
                    sel.start, sel.end
                ))
                .monospace()
                .size(11.5)
                .color(Color32::from_rgb(56, 189, 248)),
            );
            if ui.small_button("Clear").clicked() {
                *selection = None;
                *action = WaveformAction::SelectionChanged;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_frame_coordinate_roundtrip() {
        let scroll = 125.5;
        let spp = 0.25;
        let left = 40.0;
        let x = frame_to_x(200.0, scroll, spp, left);
        assert!((x - (40.0 + (74.5 / 0.25) as f32)).abs() < 1e-4);
        let frame = x_to_frame(x, scroll, spp, left);
        assert!((frame - 200.0).abs() < 1e-4);
    }

    #[test]
    fn cursor_anchored_zoom_preserves_frame_under_cursor() {
        let num_frames = 44_100;
        let canvas_w = 1000.0_f32;
        let left = 20.0_f32;
        let cursor_x = 320.0_f32; // 300px from left
        let initial_scroll = 1000.0;
        let initial_spp = 10.0;
        let anchor_before = x_to_frame(cursor_x, initial_scroll, initial_spp, left);

        let (new_spp, new_scroll) = zoom_around_x(
            initial_scroll,
            initial_spp,
            2.0,
            cursor_x,
            left,
            num_frames,
            canvas_w,
        );
        assert!((new_spp - 2.0).abs() < 1e-9);
        let anchor_after = x_to_frame(cursor_x, new_scroll, new_spp, left);
        assert!(
            (anchor_after - anchor_before).abs() < 1e-6,
            "anchor shifted: before={anchor_before}, after={anchor_after}"
        );

        // Zooming beyond MIN_SAMPLES_PER_PIXEL clamps cleanly
        let (clamped_spp, _) = zoom_around_x(
            new_scroll, new_spp, 0.001, cursor_x, left, num_frames, canvas_w,
        );
        assert!((clamped_spp - MIN_SAMPLES_PER_PIXEL).abs() < 1e-9);
    }

    #[test]
    fn single_cycle_selection_clamps_to_buffer_bounds() {
        // Centered in middle
        assert_eq!(single_cycle_selection(500, 100.2, 44_100), Some(450..550));
        // Near start -> shifts right to preserve cycle length
        assert_eq!(single_cycle_selection(10, 100.0, 44_100), Some(0..100));
        // Near end -> shifts left to preserve cycle length
        assert_eq!(
            single_cycle_selection(44_090, 100.0, 44_100),
            Some(44_000..44_100)
        );
        // Wavelength longer than buffer -> full buffer
        assert_eq!(single_cycle_selection(20, 100.0, 50), Some(0..50));
        // Empty buffer -> None
        assert_eq!(single_cycle_selection(0, 100.0, 0), None);
    }
}
