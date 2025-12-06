use super::{Sky, SkySample};
use crate::raytracer::texture::Texture;
use glam::{Vec2, Vec3};
use std::f32::consts::PI;

pub struct TextureSky {
    texture: Texture,
    marginal_cdf: Vec<f32>,
    conditional_cdfs: Vec<f32>,
    width: u32,
    height: u32,
    total_weight: f32,
    rotation: f32,
}

impl TextureSky {
    pub fn new(texture: Texture) -> Self {
        Self::with_rotation(texture, 0.0)
    }

    pub fn with_rotation(texture: Texture, rotation: f32) -> Self {
        let (width, height) = match &texture {
            Texture::SolidColor(_) => (1, 1),
            Texture::Image { width, height, .. } => (*width, *height),
        };

        let (marginal_cdf, conditional_cdfs, total_weight) =
            Self::build_cdfs(&texture, width, height);

        Self {
            texture,
            marginal_cdf,
            conditional_cdfs,
            width,
            height,
            total_weight,
            rotation,
        }
    }

    fn rotate_direction(&self, direction: Vec3) -> Vec3 {
        if self.rotation == 0.0 {
            return direction;
        }
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        Vec3::new(
            direction.x * cos_r - direction.z * sin_r,
            direction.y,
            direction.x * sin_r + direction.z * cos_r,
        )
    }

    fn unrotate_direction(&self, direction: Vec3) -> Vec3 {
        if self.rotation == 0.0 {
            return direction;
        }
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        Vec3::new(
            direction.x * cos_r + direction.z * sin_r,
            direction.y,
            -direction.x * sin_r + direction.z * cos_r,
        )
    }

    fn build_cdfs(texture: &Texture, width: u32, height: u32) -> (Vec<f32>, Vec<f32>, f32) {
        let mut conditional_cdfs = vec![0.0; (height * (width + 1)) as usize];
        let mut row_weights = vec![0.0; height as usize];

        for y in 0..height {
            let v = (y as f32 + 0.5) / height as f32;
            let sin_theta = (PI * v).sin();

            let row_offset = (y * (width + 1)) as usize;
            conditional_cdfs[row_offset] = 0.0;

            for x in 0..width {
                let uv = Vec2::new((x as f32 + 0.5) / width as f32, v);
                let radiance = texture.sample(uv);
                let lum = 0.2126 * radiance.x + 0.7152 * radiance.y + 0.0722 * radiance.z;
                let weight = lum * sin_theta;

                conditional_cdfs[row_offset + x as usize + 1] =
                    conditional_cdfs[row_offset + x as usize] + weight;
            }

            row_weights[y as usize] = conditional_cdfs[row_offset + width as usize];

            if row_weights[y as usize] > 0.0 {
                let inv = 1.0 / row_weights[y as usize];
                for x in 0..=width {
                    conditional_cdfs[row_offset + x as usize] *= inv;
                }
            }
        }

        let mut marginal_cdf = vec![0.0; (height + 1) as usize];
        for y in 0..height {
            marginal_cdf[y as usize + 1] = marginal_cdf[y as usize] + row_weights[y as usize];
        }
        let total_weight = marginal_cdf[height as usize];

        if total_weight > 0.0 {
            let inv = 1.0 / total_weight;
            for y in 0..=height {
                marginal_cdf[y as usize] *= inv;
            }
        }

        (marginal_cdf, conditional_cdfs, total_weight)
    }

    fn direction_to_uv(direction: Vec3) -> Vec2 {
        let u = 0.5 + direction.z.atan2(direction.x) / (2.0 * PI);
        let v = 0.5 - direction.y.asin() / PI;
        Vec2::new(u, v)
    }

    fn uv_to_direction(uv: Vec2) -> Vec3 {
        let phi = (uv.x - 0.5) * 2.0 * PI;
        let theta = (0.5 - uv.y) * PI;
        let cos_theta = theta.cos();
        Vec3::new(cos_theta * phi.sin(), theta.sin(), cos_theta * phi.cos())
    }

    fn binary_search(cdf: &[f32], u: f32) -> usize {
        let mut lo = 0;
        let mut hi = cdf.len() - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if cdf[mid] <= u {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.saturating_sub(1)
    }
}

impl Sky for TextureSky {
    fn sample(&self, direction: Vec3) -> Vec3 {
        let rotated = self.rotate_direction(direction);
        let uv = Self::direction_to_uv(rotated);
        self.texture.sample(uv)
    }

    fn sample_direction(&self, u1: f32, u2: f32) -> SkySample {
        if self.total_weight <= 0.0 {
            let phi = u1 * 2.0 * PI;
            let cos_theta = 1.0 - 2.0 * u2;
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
            let direction = Vec3::new(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin());
            let radiance = self.sample(direction);
            return SkySample {
                direction,
                radiance,
                pdf: 1.0 / (4.0 * PI),
            };
        }

        let y = Self::binary_search(&self.marginal_cdf, u1);
        let v_cdf_lo = self.marginal_cdf[y];
        let v_cdf_hi = self.marginal_cdf[y + 1];
        let dv = (u1 - v_cdf_lo) / (v_cdf_hi - v_cdf_lo + 1e-10);
        let v = (y as f32 + dv) / self.height as f32;

        let row_offset = y * (self.width + 1) as usize;
        let row_cdf = &self.conditional_cdfs[row_offset..row_offset + (self.width + 1) as usize];
        let x = Self::binary_search(row_cdf, u2);
        let u_cdf_lo = row_cdf[x];
        let u_cdf_hi = row_cdf[x + 1];
        let du = (u2 - u_cdf_lo) / (u_cdf_hi - u_cdf_lo + 1e-10);
        let u = (x as f32 + du) / self.width as f32;

        let uv = Vec2::new(u, v);
        let tex_direction = Self::uv_to_direction(uv);
        let direction = self.unrotate_direction(tex_direction);
        let radiance = self.texture.sample(uv);
        let pdf = self.pdf(direction);

        SkySample {
            direction,
            radiance,
            pdf,
        }
    }

    fn pdf(&self, direction: Vec3) -> f32 {
        if self.total_weight <= 0.0 {
            return 1.0 / (4.0 * PI);
        }

        let rotated = self.rotate_direction(direction);
        let uv = Self::direction_to_uv(rotated);

        let x = ((uv.x * self.width as f32) as u32).min(self.width - 1);
        let y = ((uv.y * self.height as f32) as u32).min(self.height - 1);

        let marginal_pdf = self.marginal_cdf[y as usize + 1] - self.marginal_cdf[y as usize];
        let row_offset = (y * (self.width + 1)) as usize;
        let conditional_pdf = self.conditional_cdfs[row_offset + x as usize + 1]
            - self.conditional_cdfs[row_offset + x as usize];

        let pdf_uv = marginal_pdf * conditional_pdf * (self.width * self.height) as f32;

        let sin_theta = (PI * uv.y).sin().max(1e-10);
        pdf_uv / (2.0 * PI * PI * sin_theta)
    }
}
