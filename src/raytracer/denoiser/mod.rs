mod oidn;

pub use oidn::OidnDenoiser;

use glam::Vec3;

pub struct AovBuffers {
    pub albedo: Vec<Vec3>,
    pub normal: Vec<Vec3>,
    pub width: usize,
    pub height: usize,
}

impl AovBuffers {
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            albedo: vec![Vec3::ZERO; size],
            normal: vec![Vec3::ZERO; size],
            width,
            height,
        }
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, albedo: Vec3, normal: Vec3) {
        let idx = y * self.width + x;
        self.albedo[idx] = albedo;
        self.normal[idx] = normal;
    }
}

pub struct DenoiseInput<'a> {
    pub color: &'a [Vec3],
    pub aovs: Option<&'a AovBuffers>,
    pub width: usize,
    pub height: usize,
}

pub trait Denoiser {
    fn denoise(&self, input: DenoiseInput) -> Vec<Vec3>;
}
