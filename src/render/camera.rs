use std::collections::HashSet;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use winit::keyboard::KeyCode;

const MOVE_SPEED: f32 = 24.0;
const SPRINT_MULTIPLIER: f32 = 3.0;
const MOUSE_SENSITIVITY: f32 = 0.0025;
const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_radians: f32,
    pub aspect: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Camera {
    pub fn new(position: Vec3, aspect: f32) -> Self {
        Self {
            position,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.35,
            fov_y_radians: 70f32.to_radians(),
            aspect,
            z_near: 0.1,
            z_far: 1000.0,
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    pub fn view_proj(&self) -> Mat4 {
        let proj = Mat4::perspective_rh(self.fov_y_radians, self.aspect, self.z_near, self.z_far);
        let view = Mat4::look_to_rh(self.position, self.forward(), Vec3::Y);
        proj * view
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera) -> Self {
        Self {
            view_proj: camera.view_proj().to_cols_array_2d(),
        }
    }
}

#[derive(Default)]
pub struct FlyCameraController {
    pressed: HashSet<KeyCode>,
    pub cursor_grabbed: bool,
}

impl FlyCameraController {
    pub fn key_changed(&mut self, key: KeyCode, pressed: bool) {
        if pressed {
            self.pressed.insert(key);
        } else {
            self.pressed.remove(&key);
        }
    }

    pub fn apply_mouse_delta(&self, camera: &mut Camera, delta_x: f64, delta_y: f64) {
        if !self.cursor_grabbed {
            return;
        }
        camera.yaw += delta_x as f32 * MOUSE_SENSITIVITY;
        camera.pitch =
            (camera.pitch - delta_y as f32 * MOUSE_SENSITIVITY).clamp(-MAX_PITCH, MAX_PITCH);
    }

    pub fn update(&self, camera: &mut Camera, dt_secs: f32) {
        let mut speed = MOVE_SPEED;
        if self.pressed.contains(&KeyCode::ShiftLeft) {
            speed *= SPRINT_MULTIPLIER;
        }

        let forward = camera.forward();
        let right = camera.right();
        let mut motion = Vec3::ZERO;

        if self.pressed.contains(&KeyCode::KeyW) {
            motion += forward;
        }
        if self.pressed.contains(&KeyCode::KeyS) {
            motion -= forward;
        }
        if self.pressed.contains(&KeyCode::KeyD) {
            motion += right;
        }
        if self.pressed.contains(&KeyCode::KeyA) {
            motion -= right;
        }
        if self.pressed.contains(&KeyCode::Space) {
            motion += Vec3::Y;
        }
        if self.pressed.contains(&KeyCode::ControlLeft) {
            motion -= Vec3::Y;
        }

        if motion != Vec3::ZERO {
            camera.position += motion.normalize() * speed * dt_secs;
        }
    }
}
