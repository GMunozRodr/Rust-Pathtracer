use super::{linear_to_srgb_u8, Exporter, Framebuffer, ToneMap};
use crate::raytracer::denoiser::{AovBuffers, DenoiseInput, Denoiser};
use crate::raytracer::framebuffer::AdaptiveFramebuffer;
use glam::Vec3;
use image::{ImageBuffer, Rgb};

pub struct PngExporter<D = ()> {
    tonemap: ToneMap,
    exposure: f32,
    denoiser: Option<D>,
}

impl PngExporter<()> {
    pub fn srgb() -> Self {
        Self {
            tonemap: ToneMap::None,
            exposure: 1.0,
            denoiser: None,
        }
    }

    pub fn with_tonemap(tonemap: ToneMap) -> Self {
        Self {
            tonemap,
            exposure: 1.0,
            denoiser: None,
        }
    }
}

impl<D> PngExporter<D> {
    pub fn with_exposure(mut self, exposure: f32) -> Self {
        self.exposure = exposure;
        self
    }

    pub fn with_denoiser<D2: Denoiser>(self, denoiser: D2) -> PngExporter<D2> {
        PngExporter {
            tonemap: self.tonemap,
            exposure: self.exposure,
            denoiser: Some(denoiser),
        }
    }

    fn vec3_to_rgb(&self, color: Vec3) -> Rgb<u8> {
        let mapped = self.tonemap.apply_with_exposure(color, self.exposure);
        Rgb(linear_to_srgb_u8(mapped))
    }

    fn export_buffer(&self, colors: &[Vec3], width: usize, height: usize, path: &str) {
        let img = ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
            let idx = y as usize * width + x as usize;
            self.vec3_to_rgb(colors[idx])
        });
        img.save(path).expect("Failed to write PNG file");
    }
}

impl<D: Denoiser> PngExporter<D> {
    pub fn export_adaptive(&self, framebuffer: &AdaptiveFramebuffer, path: &str) {
        self.export_adaptive_with_aov(framebuffer, None, path);
    }

    pub fn export_adaptive_with_aov(
        &self,
        framebuffer: &AdaptiveFramebuffer,
        aovs: Option<&AovBuffers>,
        path: &str,
    ) {
        let width = framebuffer.width();
        let height = framebuffer.height();
        let colors = framebuffer.to_color_buffer();

        let final_colors = if let Some(ref denoiser) = self.denoiser {
            println!(
                "Denoising{}...",
                if aovs.is_some() { " with AOVs" } else { "" }
            );
            denoiser.denoise(DenoiseInput {
                color: &colors,
                aovs,
                width,
                height,
            })
        } else {
            colors
        };

        self.export_buffer(&final_colors, width, height, path);
    }
}

impl PngExporter<()> {
    pub fn export_adaptive(&self, framebuffer: &AdaptiveFramebuffer, path: &str) {
        let width = framebuffer.width();
        let height = framebuffer.height();
        let colors = framebuffer.to_color_buffer();
        self.export_buffer(&colors, width, height, path);
    }
}

impl<D> Exporter for PngExporter<D>
where
    D: Denoiser,
{
    fn export(&self, framebuffer: &Framebuffer, path: &str) {
        let width = framebuffer.width();
        let height = framebuffer.height();

        let colors: Vec<Vec3> = (0..width * height)
            .map(|idx| {
                let x = idx % width;
                let y = idx / width;
                framebuffer.get_averaged(x, y)
            })
            .collect();

        let final_colors = if let Some(ref denoiser) = self.denoiser {
            println!("Denoising...");
            denoiser.denoise(DenoiseInput {
                color: &colors,
                aovs: None,
                width,
                height,
            })
        } else {
            colors
        };

        self.export_buffer(&final_colors, width, height, path);
    }
}

impl Exporter for PngExporter<()> {
    fn export(&self, framebuffer: &Framebuffer, path: &str) {
        let width = framebuffer.width();
        let height = framebuffer.height();

        let colors: Vec<Vec3> = (0..width * height)
            .map(|idx| {
                let x = idx % width;
                let y = idx / width;
                framebuffer.get_averaged(x, y)
            })
            .collect();

        self.export_buffer(&colors, width, height, path);
    }
}
