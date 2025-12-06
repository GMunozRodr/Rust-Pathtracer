use crate::raytracer::ray::Ray;

#[derive(Clone)]
pub struct Camera {
    pub position: glam::Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub aspect_ratio: f32,
    pub aperture: f32,
    pub focus_distance: f32,
    inv_vp_matrix: glam::Mat4,
}

impl Camera {
    pub fn new(position: glam::Vec3, look_at: glam::Vec3, fov: f32, aspect_ratio: f32) -> Self {
        let dir = (look_at - position).normalize();
        let yaw = dir.x.atan2(-dir.z);
        let pitch = dir.y.asin();

        let mut camera = Camera {
            position,
            yaw,
            pitch,
            fov,
            aspect_ratio,
            aperture: 0.0,
            focus_distance: 10.0,
            inv_vp_matrix: glam::Mat4::IDENTITY,
        };
        camera.update_matrix();
        camera
    }

    pub fn forward(&self) -> glam::Vec3 {
        glam::Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
    }

    pub fn right(&self) -> glam::Vec3 {
        glam::Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin())
    }

    pub fn has_dof(&self) -> bool {
        self.aperture > 0.0
    }

    pub fn update_matrix(&mut self) {
        let forward = self.forward();
        let look_at = self.position + forward;
        let up = glam::Vec3::Y;

        let fov_rad = self.fov.to_radians();
        let vp_matrix = glam::Mat4::perspective_rh(fov_rad, self.aspect_ratio, 0.1, 1000.0)
            * glam::Mat4::look_at_rh(self.position, look_at, up);
        self.inv_vp_matrix = vp_matrix.inverse();
    }

    pub fn generate_ray(&self, u: f32, v: f32) -> Ray {
        self.generate_ray_dof(u, v, 0.0, 0.0)
    }

    pub fn generate_ray_dof(&self, u: f32, v: f32, lens_u: f32, lens_v: f32) -> Ray {
        let ndc_x = 2.0 * u - 1.0;
        let ndc_y = 1.0 - 2.0 * v;
        let near_point = self.inv_vp_matrix * glam::Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far_point = self.inv_vp_matrix * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let near_point = near_point.truncate() / near_point.w;
        let far_point = far_point.truncate() / far_point.w;

        let direction = (far_point - near_point).normalize();

        if self.aperture <= 0.0 {
            return Ray::new(near_point, direction);
        }

        let focus_point = near_point + direction * self.focus_distance;

        let r = lens_u.sqrt();
        let theta = lens_v * std::f32::consts::TAU;
        let lens_offset_x = r * theta.cos() * self.aperture;
        let lens_offset_y = r * theta.sin() * self.aperture;

        let right = self.right();
        let up = right.cross(self.forward());
        let new_origin = near_point + right * lens_offset_x + up * lens_offset_y;

        let new_direction = (focus_point - new_origin).normalize();

        Ray::new(new_origin, new_direction)
    }
}
