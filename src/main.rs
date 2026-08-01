mod chat;
mod fps;
mod render;
mod world;

use std::sync::Arc;
use std::time::Instant;

use glam::Vec3;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Fullscreen, Window, WindowId},
};

use chat::{Chat, MessageKind, commands};
use fps::FpsCounter;
use render::{
    RenderOutcome, Renderer,
    camera::{Camera, FlyCameraController},
};
use world::World;
use world::streaming::{ChunkStreamer, StreamingUpdate};

const WORLD_SEED: u32 = 1;
/// Chunk columns are always kept loaded within this radius of the camera.
const LOAD_RADIUS_CHUNKS: i32 = 6;
/// Columns are only unloaded once they exceed this (wider) radius, to avoid
/// load/unload thrashing when hovering near the load boundary.
const UNLOAD_RADIUS_CHUNKS: i32 = 8;
/// New columns loaded per frame once already playing; the initial load
/// around spawn uses an unbounded budget so there's no visible void at start.
const STREAM_COLUMNS_PER_FRAME: usize = 4;

struct AppState {
    renderer: Renderer,
    world: World,
    streamer: ChunkStreamer,
    camera: Camera,
    controller: FlyCameraController,
    chat: Chat,
    /// Whether the cursor was grabbed right before chat was opened, so
    /// closing chat can restore it instead of always re-grabbing.
    pre_chat_grabbed: bool,
    fps: FpsCounter,
    last_frame: Instant,
}

impl AppState {
    async fn new(window: Arc<Window>) -> Self {
        let mut renderer = Renderer::new(window).await;
        let mut world = World::new();
        let mut streamer = ChunkStreamer::new(WORLD_SEED, LOAD_RADIUS_CHUNKS, UNLOAD_RADIUS_CHUNKS);

        let camera = Camera::new(Vec3::new(0.0, 110.0, 0.0), renderer.aspect_ratio());

        log::info!("priming world around spawn...");
        let prime_start = Instant::now();
        let update = streamer.update(&mut world, camera.position, usize::MAX);
        let loaded_chunks = update.meshes.len();
        apply_streaming_update(&mut renderer, update);
        log::info!(
            "primed {loaded_chunks} chunks in {:?}",
            prime_start.elapsed()
        );

        let mut chat = Chat::new();
        chat.push_message(
            "Welcome! Press T or / to chat. Try /settings rd <chunks>.",
            MessageKind::CommandOk,
        );

        Self {
            renderer,
            world,
            streamer,
            camera,
            controller: FlyCameraController::default(),
            chat,
            pre_chat_grabbed: false,
            fps: FpsCounter::new(),
            last_frame: Instant::now(),
        }
    }

    fn set_cursor_grabbed(&mut self, grabbed: bool) {
        let window = &self.renderer.window;
        if grabbed {
            let locked = window.set_cursor_grab(CursorGrabMode::Locked).is_ok();
            if !locked {
                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
            }
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
        }
        window.set_cursor_visible(!grabbed);
        self.controller.cursor_grabbed = grabbed;
    }

    fn open_chat(&mut self, prefill: &str) {
        self.pre_chat_grabbed = self.controller.cursor_grabbed;
        self.set_cursor_grabbed(false);
        self.chat.open(prefill);
    }

    fn close_chat(&mut self) {
        self.chat.close();
        if self.pre_chat_grabbed {
            self.set_cursor_grabbed(true);
        }
    }

    /// Handles a submitted chat line: runs it as a command if it starts with
    /// `/` (echoing the command itself, then its response), otherwise logs it
    /// as a plain chat message (there's no multiplayer to send it to).
    fn handle_submitted_line(&mut self, line: String) {
        if let Some(command) = line.strip_prefix('/') {
            self.chat
                .push_message(format!("/{command}"), MessageKind::CommandEcho);
            let response = commands::execute(command, &mut self.streamer, &mut self.fps);
            if let Some(enabled) = response.set_vsync {
                self.renderer.set_vsync(enabled);
            }
            // Some responses (e.g. /help) are multiple logical lines joined
            // by '\n' rather than one long line, since the chat box clips
            // text to its width instead of wrapping it.
            for line in response.text.split('\n') {
                self.chat.push_message(line.to_string(), response.kind);
            }
        } else {
            self.chat.push_message(line, MessageKind::Chat);
        }
    }
}

fn apply_streaming_update(renderer: &mut Renderer, update: StreamingUpdate) {
    for (chunk_pos, mesh) in update.meshes {
        renderer.upsert_chunk_mesh(chunk_pos, &mesh);
    }
    for chunk_pos in update.removed {
        renderer.remove_chunk_mesh(chunk_pos);
    }
}

#[derive(Default)]
struct App {
    state: Option<AppState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attributes = Window::default_attributes()
            .with_title("minecraft-but-optimized")
            .with_fullscreen(Some(Fullscreen::Borderless(None)));
        let window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .expect("failed to create window"),
        );

        self.state = Some(pollster::block_on(AppState::new(window)));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        if state.renderer.window.id() != id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.renderer.resize(size.width, size.height);
                state.camera.aspect = state.renderer.aspect_ratio();
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.chat.is_open && !state.controller.cursor_grabbed {
                    state.set_cursor_grabbed(true);
                }
            }
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                let pressed = key_event.state == ElementState::Pressed;

                if state.chat.is_open {
                    if pressed {
                        match key_event.physical_key {
                            PhysicalKey::Code(KeyCode::Escape) => state.close_chat(),
                            PhysicalKey::Code(KeyCode::Enter | KeyCode::NumpadEnter) => {
                                // `submit()` always closes the chat itself (even on blank
                                // input); route through `close_chat()` too so the
                                // pre-chat cursor grab gets restored either way.
                                if let Some(line) = state.chat.submit() {
                                    state.handle_submitted_line(line);
                                }
                                state.close_chat();
                            }
                            PhysicalKey::Code(KeyCode::Backspace) => state.chat.backspace(),
                            PhysicalKey::Code(KeyCode::ArrowUp) => state.chat.history_prev(),
                            PhysicalKey::Code(KeyCode::ArrowDown) => state.chat.history_next(),
                            PhysicalKey::Code(KeyCode::PageUp) => state.chat.scroll_up(),
                            PhysicalKey::Code(KeyCode::PageDown) => state.chat.scroll_down(),
                            PhysicalKey::Code(KeyCode::Tab) => {
                                if let Some(insert) = commands::suggest(&state.chat.input)
                                    .and_then(|suggestion| suggestion.tab_insert)
                                {
                                    state.chat.apply_completion(&insert);
                                }
                            }
                            _ => {
                                if let Some(text) = key_event.text.as_ref() {
                                    for c in text.chars().filter(|c| !c.is_control()) {
                                        state.chat.push_char(c);
                                    }
                                }
                            }
                        }
                    }
                    return;
                }

                if let PhysicalKey::Code(code) = key_event.physical_key {
                    if pressed {
                        match code {
                            KeyCode::KeyT | KeyCode::Enter | KeyCode::NumpadEnter => {
                                state.open_chat("");
                            }
                            KeyCode::Slash => state.open_chat("/"),
                            KeyCode::Escape => state.set_cursor_grabbed(false),
                            _ => state.controller.key_changed(code, true),
                        }
                    } else {
                        state.controller.key_changed(code, false);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;

                if !state.chat.is_open {
                    state.controller.update(&mut state.camera, dt);
                }

                let update = state.streamer.update(
                    &mut state.world,
                    state.camera.position,
                    STREAM_COLUMNS_PER_FRAME,
                );
                apply_streaming_update(&mut state.renderer, update);

                state.fps.tick();
                let fps_display = state.fps.show.then_some(state.fps.current);

                match state
                    .renderer
                    .render(&state.camera, &state.chat, fps_display)
                {
                    RenderOutcome::Ok | RenderOutcome::Skip => {}
                    RenderOutcome::Reconfigure => {
                        let (w, h) = (
                            state.renderer.window.inner_size().width,
                            state.renderer.window.inner_size().height,
                        );
                        state.renderer.resize(w, h);
                    }
                    RenderOutcome::Fatal => {
                        log::error!("fatal surface validation error");
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            state
                .controller
                .apply_mouse_delta(&mut state.camera, dx, dy);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = &self.state else {
            return;
        };

        match state.fps.cap {
            Some(cap) if cap > 0 => {
                // Software frame pacing: uncapped rendering (the default) just
                // polls flat-out, but hitting an arbitrary cap like 144 needs
                // actual pacing since present modes only give you "uncapped"
                // or "synced to the display's refresh rate", not arbitrary
                // values. `WaitUntil` parks the loop instead of busy-spinning.
                let next = state.last_frame + std::time::Duration::from_secs_f64(1.0 / cap as f64);
                if Instant::now() >= next {
                    state.renderer.window.request_redraw();
                } else {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(next));
                }
            }
            _ => {
                event_loop.set_control_flow(ControlFlow::Poll);
                state.renderer.window.request_redraw();
            }
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).expect("event loop error");
}
