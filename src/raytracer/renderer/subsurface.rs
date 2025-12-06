use super::sampling::{build_basis, to_world};
use super::SceneAccess;
use crate::raytracer::ray::{RAY_EPSILON, Ray};
use glam::Vec3;
use std::f32::consts::PI;

pub struct SubsurfaceResult {
    pub exit_point: Vec3,
    pub exit_normal: Vec3,
    pub throughput: Vec3,
}

const SSS_MAX_STEPS: u32 = 64;

pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u32) -> Self {
        let mut rng = Self(seed as u64);
        rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.0;
        self.0 = old.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    pub fn next(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

fn sample_scatter_distance(rng: &mut Rng, sigma_t: f32) -> f32 {
    -rng.next().max(1e-10).ln() / sigma_t
}

fn sample_hg_phase(rng: &mut Rng, wo: Vec3, g: f32) -> Vec3 {
    let cos_theta = if g.abs() < 0.001 {
        1.0 - 2.0 * rng.next()
    } else {
        let g2 = g * g;
        let sqr = (1.0 - g2) / (1.0 - g + 2.0 * g * rng.next());
        ((1.0 + g2 - sqr * sqr) / (2.0 * g)).clamp(-1.0, 1.0)
    };

    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * rng.next();

    let (t, b, n) = build_basis(-wo);
    let local = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
    to_world(local, t, b, n)
}

pub fn random_walk_subsurface<S: SceneAccess>(
    rng: &mut Rng,
    scene: &S,
    entry_point: Vec3,
    entry_dir: Vec3,
    radius: Vec3,
    albedo: Vec3,
    anisotropy: f32,
) -> Option<SubsurfaceResult> {
    let sigma_t = Vec3::new(
        1.0 / radius.x.max(0.0001),
        1.0 / radius.y.max(0.0001),
        1.0 / radius.z.max(0.0001),
    );
    let sigma_s = sigma_t * albedo;

    let mut pos = entry_point;
    let mut dir = entry_dir;
    let mut throughput = Vec3::ONE;

    for _ in 0..SSS_MAX_STEPS {
        let weights = throughput * sigma_s;
        let sum_weights = weights.x + weights.y + weights.z;
        if sum_weights < 1e-10 {
            return None;
        }

        let r = rng.next() * sum_weights;
        let channel = if r < weights.x {
            0usize
        } else if r < weights.x + weights.y {
            1usize
        } else {
            2usize
        };

        let sigma_t_c = [sigma_t.x, sigma_t.y, sigma_t.z][channel];

        let dist = sample_scatter_distance(rng, sigma_t_c);

        let internal_ray = Ray::new(pos, dir).with_t_max(dist);

        if let Some(hit) = scene.hit(&internal_ray) {
            let hit_point = pos + dir * hit.t;
            let exiting = dir.dot(hit.normal) > 0.0;

            if exiting {
                let t = hit.t;
                let tr = Vec3::new(
                    (-sigma_t.x * t).exp(),
                    (-sigma_t.y * t).exp(),
                    (-sigma_t.z * t).exp(),
                );

                let pdf_sum = (-sigma_t.x * t).exp() * (weights.x / sum_weights)
                    + (-sigma_t.y * t).exp() * (weights.y / sum_weights)
                    + (-sigma_t.z * t).exp() * (weights.z / sum_weights);

                throughput *= tr / pdf_sum.max(1e-10);

                return Some(SubsurfaceResult {
                    exit_point: hit_point,
                    exit_normal: hit.normal,
                    throughput,
                });
            } else {
                pos = hit_point + dir * RAY_EPSILON;
                continue;
            }
        }

        pos += dir * dist;

        let tr = Vec3::new(
            (-sigma_t.x * dist).exp(),
            (-sigma_t.y * dist).exp(),
            (-sigma_t.z * dist).exp(),
        );

        let pdf_sum = sigma_t.x * (-sigma_t.x * dist).exp() * (weights.x / sum_weights)
            + sigma_t.y * (-sigma_t.y * dist).exp() * (weights.y / sum_weights)
            + sigma_t.z * (-sigma_t.z * dist).exp() * (weights.z / sum_weights);

        throughput *= tr * sigma_s / pdf_sum.max(1e-10);

        let max_tp = throughput.max_element();
        if max_tp < 0.1 {
            let survive = (max_tp * 10.0).min(1.0);
            if rng.next() > survive {
                return None;
            }
            throughput /= survive;
        }

        if !throughput.is_finite() || max_tp > 1000.0 {
            return None;
        }

        dir = sample_hg_phase(rng, dir, anisotropy);
    }

    None
}
