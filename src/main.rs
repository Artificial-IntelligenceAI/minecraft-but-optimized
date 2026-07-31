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
    window::{CursorGrabMode, Window, WindowId},
};

use render::{
    RenderOutcome, Renderer,
    camera::{Camera, FlyCameraController},
};
use world::{generation, meshing};

/// Render distance in chunks (radius), in x/z, around the origin.
const WORLD_RADIUS_CHUNKS: i32 = 4;
const WORLD_SEED: u32 = 1;

struct AppState {
    renderer: Renderer,
    camera: Camera,
    controller: FlyCameraController,
    last_frame: Instant,
}

impl AppState {
    async fn new(window: Arc<Window>) -> Self {
        let renderer = Renderer::new(window).await;

        log::info!("generating world (radius {WORLD_RADIUS_CHUNKS} chunks)...");
        let gen_start = Instant::now();
        let world = generation::generate_world(WORLD_RADIUS_CHUNKS, WORLD_SEED);
        log::info!("world generated in {:?}", gen_start.elapsed());

        let mesh_start = Instant::now();
        let meshes = meshing::mesh_world(&world);
        log::info!(
            "meshed {} chunks in {:?}",
            meshes.len(),
            mesh_start.elapsed()
        );

        let mut renderer = renderer;
        renderer.set_chunk_meshes(meshes);

        let camera = Camera::new(Vec3::new(0.0, 110.0, 0.0), renderer.aspect_ratio());

        Self {
            renderer,
            camera,
            controller: FlyCameraController::default(),
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

        let window_attributes = Window::default_attributes().with_title("minecraft-but-optimized");
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
                if !state.controller.cursor_grabbed {
                    state.set_cursor_grabbed(true);
                }
            }
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(code) = key_event.physical_key {
                    if code == KeyCode::Escape && key_event.state == ElementState::Pressed {
                        state.set_cursor_grabbed(false);
                    } else {
                        state
                            .controller
                            .key_changed(code, key_event.state == ElementState::Pressed);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;

                state.controller.update(&mut state.camera, dt);

                match state.renderer.render(&state.camera) {
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

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.renderer.window.request_redraw();
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
