mod texture;

use glam::Vec3;

pub use texture::TextureSky;

pub struct SkySample {
    pub direction: Vec3,
    pub radiance: Vec3,
    pub pdf: f32,
}

pub trait Sky {
    fn sample(&self, direction: Vec3) -> Vec3;
    fn sample_direction(&self, u: f32, v: f32) -> SkySample;
    fn pdf(&self, direction: Vec3) -> f32;
}
