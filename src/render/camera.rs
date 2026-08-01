use std::collections::HashSet;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
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

/// The 6 planes bounding the camera's view volume, derived from its
/// view-projection matrix (Gribb/Hartmann extraction). Each plane is stored
/// as `(a, b, c, d)` with `(a, b, c)` a unit normal pointing *into* the
/// frustum, so a point's signed distance is `dot(normal, point) + d`.
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    pub fn from_view_proj(m: Mat4) -> Self {
        let rows = [m.row(0), m.row(1), m.row(2), m.row(3)];
        // wgpu/Metal clip space: -w <= x,y <= w, 0 <= z <= w. Each plane below
        // is the corresponding "half-space" constraint rewritten as row
        // combinations, e.g. `x + w >= 0` (left) is `row0 + row3`.
        let mut planes = [
            rows[3] + rows[0], // left
            rows[3] - rows[0], // right
            rows[3] + rows[1], // bottom
            rows[3] - rows[1], // top
            rows[2],           // near
            rows[3] - rows[2], // far
        ];
        for p in &mut planes {
            let len = Vec3::new(p.x, p.y, p.z).length();
            *p /= len;
        }
        Self { planes }
    }

    /// Conservative test: `false` only if the AABB is entirely outside at
    /// least one plane (definitely not visible). May return `true` for a box
    /// that's actually just outside the frustum's corners — cheap to check
    /// and never wrongly culls something that should be drawn.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        let center = (min + max) * 0.5;
        let half_extent = (max - min) * 0.5;
        for p in &self.planes {
            let normal = Vec3::new(p.x, p.y, p.z);
            let r = half_extent.x * normal.x.abs()
                + half_extent.y * normal.y.abs()
                + half_extent.z * normal.z.abs();
            let s = normal.dot(center) + p.w;
            if s + r < 0.0 {
                return false;
            }
        }
        true
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Camera at the origin looking down -Z (yaw = -90deg, matching `Camera::new`'s
    /// default), 90 degree vertical FOV, aspect 1.0, near 0.1, far 100.
    fn test_camera() -> Camera {
        Camera {
            position: Vec3::ZERO,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            fov_y_radians: 90f32.to_radians(),
            aspect: 1.0,
            z_near: 0.1,
            z_far: 100.0,
        }
    }

    #[test]
    fn point_directly_ahead_is_inside() {
        let frustum = Frustum::from_view_proj(test_camera().view_proj());
        let p = Vec3::new(0.0, 0.0, -10.0);
        assert!(frustum.intersects_aabb(p, p));
    }

    #[test]
    fn point_behind_camera_is_outside() {
        let frustum = Frustum::from_view_proj(test_camera().view_proj());
        let p = Vec3::new(0.0, 0.0, 10.0);
        assert!(!frustum.intersects_aabb(p, p));
    }

    #[test]
    fn point_beyond_far_plane_is_outside() {
        let frustum = Frustum::from_view_proj(test_camera().view_proj());
        let p = Vec3::new(0.0, 0.0, -1000.0);
        assert!(!frustum.intersects_aabb(p, p));
    }

    #[test]
    fn point_before_near_plane_is_outside() {
        let frustum = Frustum::from_view_proj(test_camera().view_proj());
        let p = Vec3::new(0.0, 0.0, -0.01);
        assert!(!frustum.intersects_aabb(p, p));
    }

    #[test]
    fn point_far_to_the_side_is_outside() {
        let frustum = Frustum::from_view_proj(test_camera().view_proj());
        // Far enough sideways at this depth to fall outside a 90-degree FOV.
        let p = Vec3::new(50.0, 0.0, -10.0);
        assert!(!frustum.intersects_aabb(p, p));
    }

    #[test]
    fn large_aabb_straddling_the_frustum_boundary_is_not_culled() {
        let frustum = Frustum::from_view_proj(test_camera().view_proj());
        // Centered far outside to one side, but big enough to still overlap
        // the frustum — must not be culled even though its center isn't visible.
        let min = Vec3::new(-60.0, -5.0, -15.0);
        let max = Vec3::new(-8.0, 5.0, -5.0);
        assert!(frustum.intersects_aabb(min, max));
    }

    #[test]
    fn aabb_entirely_outside_is_culled() {
        let frustum = Frustum::from_view_proj(test_camera().view_proj());
        let min = Vec3::new(100.0, -5.0, -15.0);
        let max = Vec3::new(120.0, 5.0, -5.0);
        assert!(!frustum.intersects_aabb(min, max));
    }
}
