mod triangle_mesh;

use crate::raytracer::accel::Bounded;
use crate::raytracer::ray::{HitData, Ray};

pub use triangle_mesh::TriangleMesh;

pub trait Shape: Bounded + Clone {
    fn hit(&self, ray: &Ray) -> Option<HitData>;

    fn hit_any(&self, ray: &Ray) -> bool {
        self.hit(ray).is_some()
    }
}
