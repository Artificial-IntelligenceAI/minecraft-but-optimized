use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

pub struct UiTextLine<'a> {
    pub text: &'a str,
    pub x: f32,
    pub y: f32,
    pub color: [u8; 4],
    pub font_size: f32,
}

/// Wraps glyphon (cosmic-text shaping + glyph atlas) for drawing 2D screen-space
/// text — the chat log and input line — as a companion to [`super::quad::QuadRenderer`],
/// which handles the flat-colored backdrop those glyphs sit on top of.
pub struct UiText {
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    renderer: TextRenderer,
    viewport: Viewport,
    buffers: Vec<Buffer>,
}

impl UiText {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let renderer =
            TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);

        Self {
            font_system,
            swash_cache,
            atlas,
            renderer,
            viewport,
            buffers: Vec::new(),
        }
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        lines: &[UiTextLine],
    ) {
        self.viewport.update(queue, Resolution { width, height });

        self.buffers.clear();
        for line in lines {
            let metrics = Metrics::new(line.font_size, line.font_size * 1.2);
            let mut buffer = Buffer::new(&mut self.font_system, metrics);
            buffer.set_size(Some(width as f32), Some(height as f32));
            buffer.set_text(
                line.text,
                &Attrs::new().family(Family::Monospace),
                Shaping::Advanced,
                None,
            );
            buffer.shape_until_scroll(&mut self.font_system, false);
            self.buffers.push(buffer);
        }

        let text_areas = self
            .buffers
            .iter()
            .zip(lines.iter())
            .map(|(buffer, line)| TextArea {
                buffer,
                left: line.x,
                top: line.y,
                scale: 1.0,
                bounds: TextBounds::default(),
                default_color: Color::rgba(
                    line.color[0],
                    line.color[1],
                    line.color[2],
                    line.color[3],
                ),
                custom_glyphs: &[],
            });

        self.renderer
            .prepare(
                device,
                queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                text_areas,
                &mut self.swash_cache,
            )
            .expect("glyphon text prepare failed");
    }

    pub fn render<'pass>(&'pass self, pass: &mut wgpu::RenderPass<'pass>) {
        self.renderer
            .render(&self.atlas, &self.viewport, pass)
            .expect("glyphon text render failed");
    }

    /// Evicts atlas glyphs that weren't used in the most recent `prepare` call,
    /// keeping GPU atlas memory bounded to the text actually on screen.
    pub fn trim(&mut self) {
        self.atlas.trim();
    }
}
