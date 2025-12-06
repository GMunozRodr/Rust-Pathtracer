use glam::Vec3;
use rayon::prelude::*;

pub trait FramebufferView {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn get_pixel(&self, x: usize, y: usize) -> Vec3;
}

pub struct Framebuffer {
    pixels: Vec<Vec3>,
    width: usize,
    height: usize,
    sample_count: u32,
}

impl Framebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![Vec3::ZERO; width * height],
            width,
            height,
            sample_count: 0,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn reset(&mut self) {
        self.pixels.fill(Vec3::ZERO);
        self.sample_count = 0;
    }

    pub fn accumulate(&mut self, pixels: &[Vec3]) {
        debug_assert_eq!(pixels.len(), self.pixels.len());
        for (acc, &new) in self.pixels.iter_mut().zip(pixels.iter()) {
            *acc += new;
        }
        self.sample_count += 1;
    }

    pub fn get_averaged(&self, x: usize, y: usize) -> Vec3 {
        let idx = y * self.width + x;
        if self.sample_count == 0 {
            Vec3::ZERO
        } else {
            self.pixels[idx] / self.sample_count as f32
        }
    }
}

impl FramebufferView for Framebuffer {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn get_pixel(&self, x: usize, y: usize) -> Vec3 {
        self.get_averaged(x, y)
    }
}

pub struct AdaptiveFramebuffer {
    sum: Vec<Vec3>,
    sum_sq: Vec<Vec3>,
    sample_counts: Vec<u32>,
    width: usize,
    height: usize,
}

impl AdaptiveFramebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            sum: vec![Vec3::ZERO; size],
            sum_sq: vec![Vec3::ZERO; size],
            sample_counts: vec![0; size],
            width,
            height,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn reset(&mut self) {
        self.sum.fill(Vec3::ZERO);
        self.sum_sq.fill(Vec3::ZERO);
        self.sample_counts.fill(0);
    }

    #[inline]
    pub fn add_sample(&mut self, x: usize, y: usize, color: Vec3) {
        let idx = y * self.width + x;
        self.sum[idx] += color;
        self.sum_sq[idx] += color * color;
        self.sample_counts[idx] += 1;
    }

    pub fn accumulate(&mut self, pixels: &[Vec3]) {
        debug_assert_eq!(pixels.len(), self.sum.len());
        for (idx, &color) in pixels.iter().enumerate() {
            self.sum[idx] += color;
            self.sum_sq[idx] += color * color;
            self.sample_counts[idx] += 1;
        }
    }

    #[inline]
    pub fn sample_count(&self, x: usize, y: usize) -> u32 {
        self.sample_counts[y * self.width + x]
    }

    pub fn min_sample_count(&self) -> u32 {
        self.sample_counts.iter().copied().min().unwrap_or(0)
    }

    pub fn max_sample_count(&self) -> u32 {
        self.sample_counts.iter().copied().max().unwrap_or(0)
    }

    pub fn total_samples(&self) -> u64 {
        self.sample_counts.iter().map(|&c| c as u64).sum()
    }

    pub fn get_averaged(&self, x: usize, y: usize) -> Vec3 {
        let idx = y * self.width + x;
        let count = self.sample_counts[idx];
        if count == 0 {
            Vec3::ZERO
        } else {
            self.sum[idx] / count as f32
        }
    }

    pub fn get_variance(&self, x: usize, y: usize) -> f32 {
        let idx = y * self.width + x;
        let n = self.sample_counts[idx];
        if n < 2 {
            return f32::INFINITY;
        }
        let mean = self.sum[idx] / n as f32;
        let mean_sq = self.sum_sq[idx] / n as f32;
        let variance = mean_sq - mean * mean;
        (variance.x.max(0.0) + variance.y.max(0.0) + variance.z.max(0.0)) / 3.0
    }

    #[inline]
    pub fn get_standard_error(&self, x: usize, y: usize) -> f32 {
        let idx = y * self.width + x;
        let n = self.sample_counts[idx];
        if n < 2 {
            return f32::INFINITY;
        }
        let nf = n as f32;
        let mean = self.sum[idx] / nf;
        let mean_sq = self.sum_sq[idx] / nf;
        let variance = mean_sq - mean * mean;
        let var_scalar = (variance.x.max(0.0) + variance.y.max(0.0) + variance.z.max(0.0)) / 3.0;
        (var_scalar / nf).sqrt()
    }

    pub fn get_noisy_pixels(
        &self,
        min_samples: u32,
        error_threshold: f32,
    ) -> Vec<(usize, usize, f32)> {
        let width = self.width;
        let height = self.height;

        (0..width * height)
            .into_par_iter()
            .filter_map(|idx| {
                let x = idx % width;
                let y = idx / width;
                let count = self.sample_counts[idx];

                if count < min_samples {
                    Some((x, y, f32::INFINITY))
                } else {
                    let n = count as f32;
                    let mean = self.sum[idx] / n;
                    let mean_sq = self.sum_sq[idx] / n;
                    let variance = mean_sq - mean * mean;
                    let var_scalar = (variance.x.max(0.0) + variance.y.max(0.0) + variance.z.max(0.0)) / 3.0;
                    let err = (var_scalar / n).sqrt();

                    if err > error_threshold {
                        Some((x, y, err))
                    } else {
                        None
                    }
                }
            })
            .collect()
    }

    pub fn to_color_buffer(&self) -> Vec<Vec3> {
        (0..self.width * self.height)
            .into_par_iter()
            .map(|idx| {
                let count = self.sample_counts[idx];
                if count == 0 {
                    Vec3::ZERO
                } else {
                    self.sum[idx] / count as f32
                }
            })
            .collect()
    }
}

impl FramebufferView for AdaptiveFramebuffer {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn get_pixel(&self, x: usize, y: usize) -> Vec3 {
        self.get_averaged(x, y)
    }
}
