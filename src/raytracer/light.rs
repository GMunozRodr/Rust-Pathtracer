use glam::Vec3;
use std::f32::consts::PI;

#[derive(Clone, Copy)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub cos_angle: f32,
    pub pdf: f32,
}

impl DirectionalLight {
    pub fn new(direction: Vec3, color: Vec3) -> Self {
        Self::with_angle(direction, color, 0.25_f32.to_radians())
    }

    pub fn with_angle(direction: Vec3, color: Vec3, angle_radians: f32) -> Self {
        let cos_angle = angle_radians.cos();
        let solid_angle = 2.0 * PI * (1.0 - cos_angle);
        Self {
            direction: direction.normalize(),
            color,
            cos_angle,
            pdf: 1.0 / solid_angle,
        }
    }

    pub fn is_in_cone(&self, dir: Vec3) -> bool {
        dir.dot(self.direction) >= self.cos_angle
    }
}
