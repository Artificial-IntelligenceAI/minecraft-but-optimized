use std::time::{SystemTime, UNIX_EPOCH};

use super::quad::UiQuad;
use super::text::UiTextLine;
use crate::chat::{Chat, MessageKind, commands};

const FONT_SIZE: f32 = 30.0;
const LINE_HEIGHT: f32 = 38.0;
const PANEL_MARGIN_LEFT: f32 = 12.0;
const PANEL_MARGIN_BOTTOM: f32 = 70.0;
const PANEL_MAX_WIDTH: f32 = 1600.0;
const TEXT_PADDING_X: f32 = 8.0;
/// No text-shaping pass available here to measure real glyph widths, so the
/// cursor position is approximated using a fixed per-character advance for
/// the monospace font `text.rs` requests.
const APPROX_CHAR_WIDTH: f32 = FONT_SIZE * 0.6;

pub struct ChatDrawData<'a> {
    pub quads: Vec<UiQuad>,
    pub text_lines: Vec<UiTextLine<'a>>,
    /// Command "ghost" suggestion (gray completion text shown after the
    /// cursor), kept separate from `text_lines` because its text is computed
    /// here — owned by this struct — rather than borrowed from `chat` like
    /// everything else in `text_lines`.
    pub ghost: Option<GhostSuggestion>,
}

pub struct GhostSuggestion {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub max_width: f32,
    pub font_size: f32,
}

pub fn build(chat: &Chat, screen_width: f32, screen_height: f32) -> ChatDrawData<'_> {
    let mut quads = Vec::new();
    let mut text_lines = Vec::new();
    let mut ghost = None;

    let panel_width = PANEL_MAX_WIDTH
        .min(screen_width - PANEL_MARGIN_LEFT * 2.0)
        .max(200.0);
    let visible = chat.visible_lines();
    let text_max_width = panel_width - TEXT_PADDING_X * 2.0;

    let mut y = screen_height - PANEL_MARGIN_BOTTOM;
    if chat.is_open {
        y -= LINE_HEIGHT; // reserve the bottom row for the input bar
    }

    for (message, alpha) in visible.into_iter().rev() {
        y -= LINE_HEIGHT;
        let bg_alpha = if chat.is_open { 0.5 } else { 0.5 * alpha };
        quads.push(UiQuad {
            x: PANEL_MARGIN_LEFT,
            y,
            width: panel_width,
            height: LINE_HEIGHT,
            color: [0.0, 0.0, 0.0, bg_alpha],
        });

        let [r, g, b] = kind_color(message.kind);
        text_lines.push(UiTextLine {
            text: &message.text,
            x: PANEL_MARGIN_LEFT + TEXT_PADDING_X,
            y,
            color: [r, g, b, (255.0 * alpha) as u8],
            font_size: FONT_SIZE,
            max_width: text_max_width,
        });
    }

    if chat.is_open {
        let input_y = screen_height - PANEL_MARGIN_BOTTOM;
        quads.push(UiQuad {
            x: PANEL_MARGIN_LEFT,
            y: input_y,
            width: panel_width,
            height: LINE_HEIGHT,
            color: [0.0, 0.0, 0.0, 0.7],
        });
        text_lines.push(UiTextLine {
            text: &chat.input,
            x: PANEL_MARGIN_LEFT + TEXT_PADDING_X,
            y: input_y,
            color: [255, 255, 255, 255],
            font_size: FONT_SIZE,
            max_width: text_max_width,
        });

        let cursor_x = PANEL_MARGIN_LEFT
            + TEXT_PADDING_X
            + chat.input.chars().count() as f32 * APPROX_CHAR_WIDTH;

        if let Some(suggestion) = commands::suggest(&chat.input) {
            ghost = Some(GhostSuggestion {
                text: suggestion.ghost_tail,
                x: cursor_x,
                y: input_y,
                max_width: (text_max_width - chat.input.chars().count() as f32 * APPROX_CHAR_WIDTH)
                    .max(0.0),
                font_size: FONT_SIZE,
            });
        }

        if cursor_blink_on() {
            quads.push(UiQuad {
                x: cursor_x,
                y: input_y + 2.0,
                width: 2.0,
                height: LINE_HEIGHT - 4.0,
                color: [1.0, 1.0, 1.0, 0.9],
            });
        }
    }

    ChatDrawData {
        quads,
        text_lines,
        ghost,
    }
}

fn kind_color(kind: MessageKind) -> [u8; 3] {
    match kind {
        MessageKind::Chat => [255, 255, 255],
        MessageKind::CommandEcho => [170, 170, 170],
        MessageKind::CommandOk => [255, 255, 85],
        MessageKind::CommandError => [255, 85, 85],
    }
}

fn cursor_blink_on() -> bool {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    (millis / 500).is_multiple_of(2)
}
