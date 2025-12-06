use super::{DenoiseInput, Denoiser};
use glam::Vec3;
use oidn::{Device, RayTracing};

pub struct OidnDenoiser {
    device: Device,
}

impl OidnDenoiser {
    pub fn new() -> Self {
        let device = Device::new();
        Self { device }
    }
}

impl Default for OidnDenoiser {
    fn default() -> Self {
        Self::new()
    }
}

impl Denoiser for OidnDenoiser {
    fn denoise(&self, input: DenoiseInput) -> Vec<Vec3> {
        let width = input.width;
        let height = input.height;

        let color_f32: Vec<f32> = input.color.iter().flat_map(|c| [c.x, c.y, c.z]).collect();

        let mut output_f32 = vec![0.0f32; width * height * 3];

        if let Some(aovs) = input.aovs {
            let albedo_f32: Vec<f32> = aovs.albedo.iter().flat_map(|c| [c.x, c.y, c.z]).collect();
            let normal_f32: Vec<f32> = aovs.normal.iter().flat_map(|n| [n.x, n.y, n.z]).collect();

            RayTracing::new(&self.device)
                .image_dimensions(width, height)
                .albedo_normal(&albedo_f32, &normal_f32)
                .hdr(true)
                .filter(&color_f32, &mut output_f32)
                .expect("OIDN denoising failed");
        } else {
            RayTracing::new(&self.device)
                .image_dimensions(width, height)
                .hdr(true)
                .filter(&color_f32, &mut output_f32)
                .expect("OIDN denoising failed");
        }

        output_f32
            .chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect()
    }
}
