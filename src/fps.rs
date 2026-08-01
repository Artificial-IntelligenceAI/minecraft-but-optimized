use std::time::Instant;

/// Frames-per-second, averaged over ~1 second windows (recomputed once that
/// window elapses, rather than every frame — a per-frame instantaneous
/// value jitters too much to be readable).
pub struct FpsCounter {
    frames_this_window: u32,
    window_start: Instant,
    pub current: u32,
    /// Software frame-rate cap (see `/settings fps max`). `None` (the
    /// default) means uncapped — render as fast as the GPU/event loop allow.
    pub cap: Option<u32>,
    /// Whether the on-screen counter is drawn (see `/settings fps counter`).
    pub show: bool,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            frames_this_window: 0,
            window_start: Instant::now(),
            current: 0,
            cap: None,
            show: true,
        }
    }

    pub fn tick(&mut self) {
        self.frames_this_window += 1;
        let elapsed = self.window_start.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.current = (self.frames_this_window as f32 / elapsed).round() as u32;
            self.frames_this_window = 0;
            self.window_start = Instant::now();
        }
    }
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self::new()
    }
}
