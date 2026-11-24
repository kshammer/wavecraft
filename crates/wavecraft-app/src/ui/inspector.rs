use std::fmt::Write as _;
use std::ops::Range;

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use wavecraft_core::{FormatNode, FormatNodeKind, InspectedFile};

use super::explorer::{format_badge_colors, format_human_size};

/// Persistent state for the right-side Deep Format Tree and Hex/ASCII Inspector.
pub struct InspectorState {
    pub selected_range: Option<Range<usize>>,
    pub scroll_to_row: Option<usize>,
    pub jump_offset_hex: String,
    pub bytes_per_row: usize,
    pub show_ascii: bool,
}

impl Default for InspectorState {
    fn default() -> Self {
        Self {
            selected_range: None,
            scroll_to_row: None,
            jump_offset_hex: String::new(),
            bytes_per_row: 16,
            show_ascii: true,
        }
    }
}

/// Returns the total number of hex rows required for `total_bytes` at `bytes_per_row`.
pub fn row_count(total_bytes: usize, bytes_per_row: usize) -> usize {
    if total_bytes == 0 || bytes_per_row == 0 {
        0
    } else {
        total_bytes.div_ceil(bytes_per_row)
    }
}

/// Returns the byte range covered by `row` within `total_bytes`.
pub fn row_byte_range(row: usize, total_bytes: usize, bytes_per_row: usize) -> Range<usize> {
    if bytes_per_row == 0 || total_bytes == 0 {
        return 0..0;
    }
    let start = row.saturating_mul(bytes_per_row).min(total_bytes);
    let end = start.saturating_add(bytes_per_row).min(total_bytes);
    start..end
}

/// Maps a raw byte to printable ASCII (`0x20..=0x7E`) or `'.'` otherwise.
pub fn byte_to_ascii_char(b: u8) -> char {
    if (0x20..=0x7E).contains(&b) {
        b as char
    } else {
        '.'
    }
}

/// Computes the byte-range intersection between a row's byte range and `selected`.
pub fn row_selection_overlap(
    row_range: &Range<usize>,
    selected: &Option<Range<usize>>,
) -> Option<Range<usize>> {
    let sel = selected.as_ref()?;
    if sel.start >= sel.end {
        return None;
    }
    let start = row_range.start.max(sel.start);
    let end = row_range.end.min(sel.end);
    if start < end { Some(start..end) } else { None }
}

/// Formats a canonical `offset  xx xx ...  |ascii|` row string for diagnostics and tests.
#[cfg(test)]
pub fn format_hex_row(bytes: &[u8], row_start: usize, bytes_per_row: usize) -> String {
    let mut out = String::new();
    let _ = write!(out, "{row_start:08X}  ");
    let end = row_start.saturating_add(bytes_per_row).min(bytes.len());
    let row_slice = if row_start < bytes.len() {
        &bytes[row_start..end]
    } else {
        &[]
    };

    for col in 0..bytes_per_row {
        if col == 8 && bytes_per_row >= 16 {
            out.push(' ');
        }
        if let Some(&b) = row_slice.get(col) {
            let _ = write!(out, "{b:02X} ");
        } else {
            out.push_str("   ");
        }
    }

    out.push(' ');
    out.push('|');
    for &b in row_slice {
        out.push(byte_to_ascii_char(b));
    }
    out.push('|');
    out
}

/// Parses a hexadecimal offset string (with optional `0x` / `0X` prefix and whitespace).
pub fn parse_hex_offset(input: &str) -> Option<usize> {
    let trimmed = input.trim();
    let hex = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);
    if hex.is_empty() {
        return None;
    }
    usize::from_str_radix(hex, 16).ok()
}

/// Formats a decoded integer preview when 2 or 4 bytes are selected.
pub fn decode_selected_preview(bytes: &[u8], range: &Range<usize>) -> Option<String> {
    if range.start > range.end || range.end > bytes.len() {
        return None;
    }
    let slice = &bytes[range.clone()];
    match slice.len() {
        1 => Some(format!("u8: {} (i8: {})", slice[0], slice[0] as i8)),
        2 => {
            let le_u16 = u16::from_le_bytes([slice[0], slice[1]]);
            let le_i16 = i16::from_le_bytes([slice[0], slice[1]]);
            Some(format!("u16 LE: {le_u16} (i16: {le_i16})"))
        }
        4 => {
            let le_u32 = u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]);
            let le_i32 = i32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]);
            Some(format!("u32 LE: {le_u32} (i32: {le_i32})"))
        }
        _ => None,
    }
}

fn node_kind_color(kind: FormatNodeKind) -> Color32 {
    match kind {
        FormatNodeKind::Container => Color32::from_rgb(94, 234, 212),
        FormatNodeKind::Header => Color32::from_rgb(125, 211, 252),
        FormatNodeKind::Field => Color32::from_rgb(226, 232, 240),
        FormatNodeKind::Frame => Color32::from_rgb(196, 181, 253),
        FormatNodeKind::Payload => Color32::from_rgb(252, 211, 77),
        FormatNodeKind::Metadata => Color32::from_rgb(167, 243, 208),
    }
}

fn format_range_tag(range: &Range<usize>) -> String {
    format!("[0x{:04X}..0x{:04X}]", range.start, range.end)
}

fn select_node_range(state: &mut InspectorState, range: &Range<usize>, bytes_per_row: usize) {
    state.selected_range = Some(range.clone());
    state.scroll_to_row = range.start.checked_div(bytes_per_row);
}

fn render_format_node(
    ui: &mut egui::Ui,
    node: &FormatNode,
    depth: usize,
    bytes_per_row: usize,
    state: &mut InspectorState,
) {
    let is_selected = state.selected_range.as_ref() == Some(&node.byte_range);
    let range_text = format_range_tag(&node.byte_range);
    let label_color = node_kind_color(node.kind);

    if node.children.is_empty() {
        let rich = egui::RichText::new(format!(
            "{}  —  {}  {}",
            node.name, node.summary, range_text
        ))
        .size(12.0)
        .color(if is_selected {
            Color32::from_rgb(254, 240, 138)
        } else {
            label_color
        });
        if ui.selectable_label(is_selected, rich).clicked() {
            select_node_range(state, &node.byte_range, bytes_per_row);
        }
    } else {
        let header_text =
            egui::RichText::new(format!("{}  ({})  {}", node.name, node.summary, range_text))
                .size(12.0)
                .strong()
                .color(if is_selected {
                    Color32::from_rgb(254, 240, 138)
                } else {
                    label_color
                });

        let id = ui.make_persistent_id(("fmt_node", depth, &node.name, node.byte_range.start));
        let resp = egui::CollapsingHeader::new(header_text)
            .id_salt(id)
            .default_open(depth <= 1)
            .show(ui, |ui| {
                for child in &node.children {
                    render_format_node(ui, child, depth + 1, bytes_per_row, state);
                }
            });

        if resp.header_response.clicked() {
            select_node_range(state, &node.byte_range, bytes_per_row);
        }
    }
}

/// Renders the Deep Format Tree and Virtualized Hex/ASCII Byte Inspector panel.
pub fn show_inspector(
    ui: &mut egui::Ui,
    inspected: Option<&InspectedFile>,
    is_dirty: bool,
    state: &mut InspectorState,
) {
    ui.heading("Byte & Format Inspector");
    ui.add_space(4.0);

    let Some(file) = inspected else {
        ui.separator();
        ui.label(
            egui::RichText::new(
                "No on-disk file inspected.\nOpen a file from the Explorer or Save WAV/PCM to inspect its binary structure.",
            )
            .color(Color32::GRAY),
        );
        return;
    };

    // Format header badge + file size + encoding summary
    ui.horizontal_wrapped(|ui| {
        let (bg, fg) = format_badge_colors(file.format);
        egui::Frame::NONE
            .fill(bg)
            .corner_radius(CornerRadius::same(4))
            .inner_margin(egui::Margin::symmetric(6, 2))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(file.format.label())
                        .size(11.0)
                        .strong()
                        .color(fg),
                );
            });
        ui.label(
            egui::RichText::new(format!(
                "{} ({} bytes)",
                format_human_size(file.raw_bytes.len() as u64),
                file.raw_bytes.len()
            ))
            .size(12.0),
        );
        ui.label(
            egui::RichText::new(&file.metadata.encoding_label)
                .size(12.0)
                .color(Color32::from_rgb(148, 163, 184)),
        );
    });

    if is_dirty {
        ui.add_space(4.0);
        egui::Frame::NONE
            .fill(Color32::from_rgb(78, 54, 14))
            .stroke(Stroke::new(1.0, Color32::from_rgb(245, 158, 11)))
            .corner_radius(CornerRadius::same(4))
            .inner_margin(egui::Margin::symmetric(8, 4))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Buffer modified — Save WAV/PCM to update on-disk bytes")
                        .size(11.5)
                        .strong()
                        .color(Color32::from_rgb(253, 224, 71)),
                );
            });
    }

    ui.separator();

    let total_avail = ui.available_height().max(180.0);
    let tree_height = (total_avail * 0.42).clamp(100.0, 340.0);
    let bytes_per_row = state.bytes_per_row.max(1);

    // Top half: Structured Format Tree
    ui.label(
        egui::RichText::new("Format Structure Tree")
            .size(12.0)
            .strong()
            .color(Color32::from_rgb(148, 163, 184)),
    );
    egui::ScrollArea::vertical()
        .id_salt("inspector_format_tree")
        .max_height(tree_height)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            render_format_node(ui, &file.tree, 0, bytes_per_row, state);
        });

    ui.separator();

    // Bottom half: Virtualized Hex + ASCII Viewer
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Hex Offset:").size(12.0));
        let jump_edit = ui.add(
            egui::TextEdit::singleline(&mut state.jump_offset_hex)
                .hint_text("0x0000")
                .desired_width(84.0),
        );
        let enter_jump = jump_edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if (ui.button("Jump").clicked() || enter_jump)
            && let Some(offset) = parse_hex_offset(&state.jump_offset_hex)
        {
            let total_bytes = file.raw_bytes.len();
            if total_bytes > 0 {
                let clamped = offset.min(total_bytes - 1);
                state.selected_range = Some(clamped..clamped + 1);
                state.scroll_to_row = Some(clamped / bytes_per_row);
            }
        }
        ui.checkbox(&mut state.show_ascii, "ASCII");
    });

    // Selected range readout + decoded u16/u32 LE preview
    if let Some(sel) = &state.selected_range {
        let len = sel.end.saturating_sub(sel.start);
        let mut readout = format!(
            "Offset 0x{:04X}..0x{:04X} ({len} bytes)",
            sel.start, sel.end
        );
        if let Some(decoded) = decode_selected_preview(&file.raw_bytes, sel) {
            let _ = write!(readout, "  •  {decoded}");
        }
        ui.label(
            egui::RichText::new(readout)
                .size(11.5)
                .monospace()
                .color(Color32::from_rgb(253, 224, 71)),
        );
    } else {
        ui.label(
            egui::RichText::new("Click a tree node or hex byte to highlight bytes.")
                .size(11.5)
                .color(Color32::GRAY),
        );
    }

    ui.add_space(2.0);

    let total_bytes = file.raw_bytes.len();
    let total_rows = row_count(total_bytes, bytes_per_row);
    let font_id = FontId::monospace(12.0);
    let char_w = ui.fonts(|f| f.glyph_width(&font_id, '0')).max(6.0);
    let row_height = 18.0_f32;

    let mut scroll = egui::ScrollArea::vertical()
        .id_salt("inspector_hex_scroll")
        .auto_shrink([false, false]);
    if let Some(target_row) = state.scroll_to_row.take() {
        scroll = scroll.vertical_scroll_offset(target_row as f32 * row_height);
    }

    scroll.show_rows(ui, row_height, total_rows, |ui, row_range| {
        ui.spacing_mut().item_spacing.y = 0.0;

        let offset_chars = 10.0; // "00000000  "
        let gap_width = if bytes_per_row >= 16 { char_w } else { 0.0 };
        let hex_start_off = offset_chars * char_w;
        let hex_total_w = bytes_per_row as f32 * 3.0 * char_w + gap_width;
        let ascii_start_off = hex_start_off + hex_total_w + char_w;
        let total_line_w = if state.show_ascii {
            ascii_start_off + (bytes_per_row as f32 + 2.0) * char_w
        } else {
            hex_start_off + hex_total_w
        };
        let row_width = ui.available_width().max(total_line_w);

        for row in row_range {
            let r_range = row_byte_range(row, total_bytes, bytes_per_row);
            let row_slice = &file.raw_bytes[r_range.clone()];
            let overlap = row_selection_overlap(&r_range, &state.selected_range);

            let (rect, resp) =
                ui.allocate_exact_size(Vec2::new(row_width, row_height), Sense::click());
            let painter = ui.painter_at(rect);

            if row % 2 == 1 {
                painter.rect_filled(rect, CornerRadius::ZERO, Color32::from_rgb(20, 24, 31));
            }

            let text_y = rect.center().y;
            let base_x = rect.left() + 4.0;

            // Offset column
            painter.text(
                Pos2::new(base_x, text_y),
                Align2::LEFT_CENTER,
                format!("{:08X}", r_range.start),
                font_id.clone(),
                Color32::from_rgb(100, 116, 139),
            );

            // Hex & ASCII bytes
            let ascii_pipe_x = base_x + ascii_start_off;
            if state.show_ascii {
                painter.text(
                    Pos2::new(ascii_pipe_x, text_y),
                    Align2::LEFT_CENTER,
                    "|",
                    font_id.clone(),
                    Color32::from_rgb(100, 116, 139),
                );
            }

            for (col, &b) in row_slice.iter().enumerate() {
                let byte_idx = r_range.start + col;
                let is_sel = overlap
                    .as_ref()
                    .is_some_and(|ov| byte_idx >= ov.start && byte_idx < ov.end);

                let group_gap = if col >= 8 && bytes_per_row >= 16 {
                    gap_width
                } else {
                    0.0
                };
                let hex_x = base_x + hex_start_off + col as f32 * 3.0 * char_w + group_gap;

                if is_sel {
                    let hl_rect = Rect::from_min_size(
                        Pos2::new(hex_x - 1.5, rect.top() + 1.0),
                        Vec2::new(char_w * 2.0 + 3.0, row_height - 2.0),
                    );
                    painter.rect_filled(
                        hl_rect,
                        CornerRadius::same(2),
                        Color32::from_rgb(180, 83, 9),
                    );
                }

                let hex_color = if is_sel {
                    Color32::from_rgb(254, 249, 195)
                } else if b == 0 {
                    Color32::from_rgb(100, 116, 139)
                } else {
                    Color32::from_rgb(226, 232, 240)
                };
                painter.text(
                    Pos2::new(hex_x, text_y),
                    Align2::LEFT_CENTER,
                    format!("{b:02X}"),
                    font_id.clone(),
                    hex_color,
                );

                if state.show_ascii {
                    let ascii_x = ascii_pipe_x + (1.0 + col as f32) * char_w;
                    if is_sel {
                        let asc_hl = Rect::from_min_size(
                            Pos2::new(ascii_x, rect.top() + 1.0),
                            Vec2::new(char_w, row_height - 2.0),
                        );
                        painter.rect_filled(
                            asc_hl,
                            CornerRadius::same(2),
                            Color32::from_rgb(180, 83, 9),
                        );
                    }
                    let ch = byte_to_ascii_char(b);
                    let asc_color = if is_sel {
                        Color32::from_rgb(254, 249, 195)
                    } else if ch == '.' {
                        Color32::from_rgb(100, 116, 139)
                    } else {
                        Color32::from_rgb(148, 226, 213)
                    };
                    painter.text(
                        Pos2::new(ascii_x, text_y),
                        Align2::LEFT_CENTER,
                        ch.to_string(),
                        font_id.clone(),
                        asc_color,
                    );
                }
            }

            if state.show_ascii {
                let close_x = ascii_pipe_x + (1.0 + row_slice.len() as f32) * char_w;
                painter.text(
                    Pos2::new(close_x, text_y),
                    Align2::LEFT_CENTER,
                    "|",
                    font_id.clone(),
                    Color32::from_rgb(100, 116, 139),
                );
            }

            // Click-to-select byte in either hex or ASCII column
            if resp.clicked()
                && let Some(pos) = resp.interact_pointer_pos()
            {
                let mut clicked_col: Option<usize> = None;
                for col in 0..row_slice.len() {
                    let group_gap = if col >= 8 && bytes_per_row >= 16 {
                        gap_width
                    } else {
                        0.0
                    };
                    let hex_x = base_x + hex_start_off + col as f32 * 3.0 * char_w + group_gap;
                    if pos.x >= hex_x - 2.0 && pos.x <= hex_x + char_w * 2.5 {
                        clicked_col = Some(col);
                        break;
                    }
                    if state.show_ascii {
                        let ascii_x = ascii_pipe_x + (1.0 + col as f32) * char_w;
                        if pos.x >= ascii_x && pos.x < ascii_x + char_w {
                            clicked_col = Some(col);
                            break;
                        }
                    }
                }
                if let Some(col) = clicked_col {
                    let byte_idx = r_range.start + col;
                    state.selected_range = Some(byte_idx..byte_idx + 1);
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_count_and_byte_ranges_handle_partial_last_row() {
        assert_eq!(row_count(0, 16), 0);
        assert_eq!(row_count(1, 16), 1);
        assert_eq!(row_count(16, 16), 1);
        assert_eq!(row_count(17, 16), 2);
        assert_eq!(row_count(44, 16), 3);

        assert_eq!(row_byte_range(0, 20, 16), 0..16);
        assert_eq!(row_byte_range(1, 20, 16), 16..20);
        assert_eq!(row_byte_range(2, 20, 16), 20..20);
    }

    #[test]
    fn ascii_mapping_and_hex_row_formatting() {
        assert_eq!(byte_to_ascii_char(b'R'), 'R');
        assert_eq!(byte_to_ascii_char(b' '), ' ');
        assert_eq!(byte_to_ascii_char(0x00), '.');
        assert_eq!(byte_to_ascii_char(0x7F), '.');
        assert_eq!(byte_to_ascii_char(0xFF), '.');

        let bytes = b"RIFF\x24\x00\x00\x00WAVEfmt ";
        let row = format_hex_row(bytes, 0, 16);
        assert!(row.starts_with("00000000  52 49 46 46 24 00 00 00  57 41 56 45 66 6D 74 20"));
        assert!(row.ends_with("|RIFF$...WAVEfmt |"));
    }

    #[test]
    fn multi_row_selection_overlap_and_hex_offset_parsing() {
        let sel = Some(12..22);
        assert_eq!(row_selection_overlap(&(0..16), &sel), Some(12..16));
        assert_eq!(row_selection_overlap(&(16..32), &sel), Some(16..22));
        assert_eq!(row_selection_overlap(&(32..48), &sel), None);
        assert_eq!(row_selection_overlap(&(0..16), &Some(5..5)), None);
        assert_eq!(row_selection_overlap(&(0..16), &None), None);

        assert_eq!(parse_hex_offset("0x18"), Some(0x18));
        assert_eq!(parse_hex_offset(" 0X002A "), Some(0x2A));
        assert_eq!(parse_hex_offset("ff"), Some(255));
        assert_eq!(parse_hex_offset("xyz"), None);
        assert_eq!(parse_hex_offset(""), None);
    }

    #[test]
    fn decode_selected_preview_formats_u16_and_u32_le() {
        let data = [0x10, 0x00, 0x44, 0xAC, 0x00, 0x00];
        let p2 = decode_selected_preview(&data, &(0..2)).unwrap();
        assert!(p2.contains("u16 LE: 16"));

        let p4 = decode_selected_preview(&data, &(2..6)).unwrap();
        assert!(p4.contains("u32 LE: 44100"));

        assert!(decode_selected_preview(&data, &(0..3)).is_none());
    }
}
