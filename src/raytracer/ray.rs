pub const RAY_EPSILON: f32 = 0.001;

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: glam::Vec3,
    pub direction: glam::Vec3,
    pub inv_direction: glam::Vec3,
    pub t_min: f32,
    pub t_max: f32,
}

impl Ray {
    pub fn new(origin: glam::Vec3, direction: glam::Vec3) -> Self {
        Ray {
            origin,
            direction,
            inv_direction: glam::Vec3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z),
            t_min: RAY_EPSILON,
            t_max: f32::INFINITY,
        }
    }

    pub fn with_t_max(mut self, t_max: f32) -> Self {
        self.t_max = t_max;
        self
    }
}

pub struct HitData {
    pub t: f32,
    pub material_id: u32,
    pub normal: glam::Vec3,
    pub geo_normal: glam::Vec3,
    pub tangent: glam::Vec3,
    pub bitangent: glam::Vec3,
    pub uv: glam::Vec2,
}

impl crate::raytracer::accel::Hit for HitData {
    fn t(&self) -> f32 {
        self.t
    }
}
