mod brdf;
mod pathtracer;
pub mod sampling;
mod subsurface;
mod unshaded;

use crate::raytracer::light::DirectionalLight;
use crate::raytracer::material::Material;
use crate::raytracer::ray::{HitData, Ray};
use crate::raytracer::sky::SkySample;
use glam::{Vec2, Vec3};

pub use pathtracer::PathTracer;
pub use unshaded::UnshadedRenderer;

#[derive(Clone, Copy, Default)]
pub struct AovData {
    pub albedo: Vec3,
    pub normal: Vec3,
}

pub struct RenderResult {
    pub color: Vec3,
    pub aov: AovData,
}

pub trait SceneAccess {
    fn hit(&self, ray: &Ray) -> Option<HitData>;
    fn hit_any(&self, ray: &Ray) -> bool;
    fn get_material(&self, material_id: u32) -> &Material;
    fn sample_texture(&self, texture_id: u32, uv: Vec2) -> Vec3;
    fn sample_sky(&self, direction: Vec3) -> Vec3;
    fn sample_sky_direction(&self, u1: f32, u2: f32) -> SkySample;
    fn sky_pdf(&self, direction: Vec3) -> f32;
    fn get_sun(&self) -> Option<&DirectionalLight>;
}

pub trait Renderer {
    fn render<S: SceneAccess>(&self, ray: &Ray, scene: &S, sample_index: u32) -> Vec3;

    fn render_with_aov<S: SceneAccess>(&self, ray: &Ray, scene: &S, sample_index: u32) -> RenderResult {
        RenderResult {
            color: self.render(ray, scene, sample_index),
            aov: AovData::default(),
        }
    }
}
