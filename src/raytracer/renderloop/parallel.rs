use super::RenderLoop;
use crate::raytracer::bluenoise::BlueNoiseTexture;
use crate::raytracer::camera::Camera;
use crate::raytracer::denoiser::AovBuffers;
use crate::raytracer::framebuffer::AdaptiveFramebuffer;
use crate::raytracer::renderer::{AovData, Renderer, SceneAccess};
use glam::Vec3;
use rayon::prelude::*;
use std::sync::OnceLock;

pub struct ParallelRenderLoop;

pub struct AdaptiveSamplingConfig {
    pub min_samples: u32,
    pub variance_threshold: f32,
    pub max_samples_per_pass: usize,
}

static BLUE_NOISE_PIXEL: OnceLock<BlueNoiseTexture> = OnceLock::new();
static BLUE_NOISE_LENS: OnceLock<BlueNoiseTexture> = OnceLock::new();

fn get_blue_noise_pixel() -> &'static BlueNoiseTexture {
    BLUE_NOISE_PIXEL.get_or_init(|| BlueNoiseTexture::generate(64, 12345))
}

fn get_blue_noise_lens() -> &'static BlueNoiseTexture {
    BLUE_NOISE_LENS.get_or_init(|| BlueNoiseTexture::generate(64, 67890))
}

impl ParallelRenderLoop {
    fn render_pixel<S, R>(
        scene: &S,
        camera: &Camera,
        renderer: &R,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        sample_index: u32,
    ) -> Vec3
    where
        S: SceneAccess,
        R: Renderer,
    {
        let bn_pixel = get_blue_noise_pixel();
        let bn_lens = get_blue_noise_lens();

        let pixel_sample = bn_pixel.sample(x, y, sample_index);
        let lens_sample = bn_lens.sample(x, y, sample_index);

        let u = (x as f32 + pixel_sample.x) / (width as f32);
        let v = (y as f32 + pixel_sample.y) / (height as f32);

        let ray = camera.generate_ray_dof(u, v, lens_sample.x, lens_sample.y);
        renderer.render(&ray, scene, sample_index)
    }

    fn render_pixels<S, R>(
        &self,
        scene: &S,
        camera: &Camera,
        renderer: &R,
        width: usize,
        height: usize,
        pixels: &[(usize, usize, u32)],
    ) -> Vec<(usize, usize, Vec3)>
    where
        S: SceneAccess + Sync,
        R: Renderer + Sync,
    {
        pixels
            .par_iter()
            .map(|&(x, y, sample_index)| {
                let color = Self::render_pixel(scene, camera, renderer, x, y, width, height, sample_index);
                (x, y, color)
            })
            .collect()
    }

    pub fn render_adaptive_pass<S, R>(
        &self,
        scene: &S,
        camera: &Camera,
        renderer: &R,
        framebuffer: &mut AdaptiveFramebuffer,
        config: &AdaptiveSamplingConfig,
        work_buffer: &mut Vec<(usize, usize, u32)>,
    ) -> usize
    where
        S: SceneAccess + Sync,
        R: Renderer + Sync,
    {
        let width = framebuffer.width();
        let height = framebuffer.height();

        let mut noisy_pixels = framebuffer.get_noisy_pixels(config.min_samples, config.variance_threshold);

        if noisy_pixels.is_empty() {
            return 0;
        }

        if noisy_pixels.len() > 10000 {
            noisy_pixels.par_sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            noisy_pixels.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        }
        noisy_pixels.truncate(config.max_samples_per_pass);

        work_buffer.clear();
        work_buffer.extend(
            noisy_pixels
                .iter()
                .map(|&(x, y, _)| (x, y, framebuffer.sample_count(x, y))),
        );

        let results = self.render_pixels(scene, camera, renderer, width, height, work_buffer);

        for (x, y, color) in results {
            framebuffer.add_sample(x, y, color);
        }

        work_buffer.len()
    }

    pub fn render_first_pass_with_aov<S, R>(
        &self,
        scene: &S,
        camera: &Camera,
        renderer: &R,
        width: usize,
        height: usize,
    ) -> (Vec<Vec3>, AovBuffers)
    where
        S: SceneAccess + Sync,
        R: Renderer + Sync,
    {
        let bn_pixel = get_blue_noise_pixel();
        let bn_lens = get_blue_noise_lens();

        let results: Vec<(Vec3, AovData)> = (0..width * height)
            .into_par_iter()
            .map(|idx| {
                let x = idx % width;
                let y = idx / width;

                let pixel_sample = bn_pixel.sample(x, y, 0);
                let lens_sample = bn_lens.sample(x, y, 0);

                let u = (x as f32 + pixel_sample.x) / (width as f32);
                let v = (y as f32 + pixel_sample.y) / (height as f32);

                let ray = camera.generate_ray_dof(u, v, lens_sample.x, lens_sample.y);
                let result = renderer.render_with_aov(&ray, scene, 0);
                (result.color, result.aov)
            })
            .collect();

        let colors: Vec<Vec3> = results.iter().map(|(c, _)| *c).collect();

        let mut aovs = AovBuffers::new(width, height);
        for (idx, (_, aov)) in results.iter().enumerate() {
            let x = idx % width;
            let y = idx / width;
            aovs.set_pixel(x, y, aov.albedo, aov.normal);
        }

        (colors, aovs)
    }
}

impl RenderLoop for ParallelRenderLoop {
    fn render_pass<S, R>(
        &self,
        scene: &S,
        camera: &Camera,
        renderer: &R,
        width: usize,
        height: usize,
        sample_index: u32,
    ) -> Vec<Vec3>
    where
        S: SceneAccess + Sync,
        R: Renderer + Sync,
    {
        (0..width * height)
            .into_par_iter()
            .map(|idx| {
                let x = idx % width;
                let y = idx / width;
                Self::render_pixel(scene, camera, renderer, x, y, width, height, sample_index)
            })
            .collect()
    }
}
