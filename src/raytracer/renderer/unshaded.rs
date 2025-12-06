use super::{Renderer, SceneAccess};
use crate::raytracer::ray::Ray;
use glam::Vec3;

pub struct UnshadedRenderer;

impl Renderer for UnshadedRenderer {
    fn render<S: SceneAccess>(&self, ray: &Ray, scene: &S, _sample_index: u32) -> Vec3 {
        if let Some(hit) = scene.hit(ray) {
            let material = scene.get_material(hit.material_id);

            let base_color = match material.base_color_texture {
                Some(tex) => scene.sample_texture(tex, hit.uv) * material.base_color_factor.truncate(),
                None => material.base_color_factor.truncate(),
            };

            let n_dot_l = hit.normal.dot(-ray.direction).max(0.0);
            base_color * (0.2 + 0.8 * n_dot_l)
        } else {
            scene.sample_sky(ray.direction) * 0.5
        }
    }
}
