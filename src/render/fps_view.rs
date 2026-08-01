use super::quad::UiQuad;
use super::text::UiTextLine;

const FONT_SIZE: f32 = 44.0;
const MARGIN: f32 = 14.0;
const HEIGHT: f32 = 58.0;
const PADDING_X: f32 = 12.0;
/// Matches the monospace-advance approximation used elsewhere for UI text
/// (see `chat_view::APPROX_CHAR_WIDTH`) — sizes the backdrop to the text.
const APPROX_CHAR_WIDTH: f32 = FONT_SIZE * 0.6;

pub struct FpsDrawData {
    pub text: String,
    pub quad: UiQuad,
}

pub fn build(fps: u32) -> FpsDrawData {
    let text = format!("FPS: {fps}");
    let width = text.chars().count() as f32 * APPROX_CHAR_WIDTH + PADDING_X * 2.0;
    FpsDrawData {
        text,
        quad: UiQuad {
            x: MARGIN,
            y: MARGIN,
            width,
            height: HEIGHT,
            color: [0.0, 0.0, 0.0, 0.5],
        },
    }
}

pub fn text_line(draw_data: &FpsDrawData) -> UiTextLine<'_> {
    UiTextLine {
        text: &draw_data.text,
        x: MARGIN + PADDING_X,
        y: MARGIN,
        color: [255, 255, 0, 255],
        font_size: FONT_SIZE,
        max_width: draw_data.quad.width - PADDING_X,
    }
}
