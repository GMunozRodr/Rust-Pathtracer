use glam::Vec2;

pub struct BlueNoiseTexture {
    pub size: usize,
    pub samples: Vec<Vec2>,
}

impl BlueNoiseTexture {
    pub fn generate(size: usize, seed: u64) -> Self {
        let values_x = Self::generate_channel(size, seed);
        let values_y = Self::generate_channel(size, seed.wrapping_add(99999));

        let samples: Vec<Vec2> = values_x
            .into_iter()
            .zip(values_y.into_iter())
            .map(|(x, y)| Vec2::new(x, y))
            .collect();

        Self { size, samples }
    }

    fn generate_channel(size: usize, seed: u64) -> Vec<f32> {
        let num_pixels = size * size;
        let mut rng = Rng::new(seed);

        let mut energy = vec![0.0f32; num_pixels];
        let mut is_set = vec![false; num_pixels];
        let mut rank = vec![0usize; num_pixels];

        let sigma = 1.5f32;
        let kernel_radius = (sigma * 3.0).ceil() as i32;

        let kernel_size = (kernel_radius * 2 + 1) as usize;
        let mut kernel = vec![0.0f32; kernel_size * kernel_size];
        for dy in -kernel_radius..=kernel_radius {
            for dx in -kernel_radius..=kernel_radius {
                let d2 = (dx * dx + dy * dy) as f32;
                let w = (-d2 / (2.0 * sigma * sigma)).exp();
                let kx = (dx + kernel_radius) as usize;
                let ky = (dy + kernel_radius) as usize;
                kernel[ky * kernel_size + kx] = w;
            }
        }

        let initial_count = num_pixels / 10;
        let grid_size = (initial_count as f32).sqrt().ceil() as usize;
        let cell_size = size / grid_size;

        for gy in 0..grid_size {
            for gx in 0..grid_size {
                if gy * grid_size + gx >= initial_count {
                    break;
                }
                let px = gx * cell_size + (rng.next() * cell_size as f32) as usize;
                let py = gy * cell_size + (rng.next() * cell_size as f32) as usize;
                let px = px.min(size - 1);
                let py = py.min(size - 1);
                let idx = py * size + px;
                if !is_set[idx] {
                    is_set[idx] = true;
                    Self::add_energy(&mut energy, size, px, py, &kernel, kernel_radius);
                }
            }
        }

        let mut set_count: usize = is_set.iter().filter(|&&x| x).count();
        let target_initial = initial_count / 2;

        while set_count > target_initial {
            let mut max_energy = f32::MIN;
            let mut max_idx = 0;
            for (idx, (&e, &s)) in energy.iter().zip(is_set.iter()).enumerate() {
                if s && e > max_energy {
                    max_energy = e;
                    max_idx = idx;
                }
            }

            let px = max_idx % size;
            let py = max_idx / size;
            is_set[max_idx] = false;
            Self::add_energy(&mut energy, size, px, py, &kernel, -kernel_radius);
            set_count -= 1;
        }

        let mut current_rank = set_count;
        while set_count > 0 {
            let mut max_energy = f32::MIN;
            let mut max_idx = 0;
            for (idx, (&e, &s)) in energy.iter().zip(is_set.iter()).enumerate() {
                if s && e > max_energy {
                    max_energy = e;
                    max_idx = idx;
                }
            }

            current_rank -= 1;
            rank[max_idx] = current_rank;

            let px = max_idx % size;
            let py = max_idx / size;
            is_set[max_idx] = false;
            Self::add_energy(&mut energy, size, px, py, &kernel, -kernel_radius);
            set_count -= 1;
        }

        current_rank = target_initial;
        while current_rank < num_pixels {
            let mut min_energy = f32::MAX;
            let mut min_idx = 0;
            for (idx, (&e, &s)) in energy.iter().zip(is_set.iter()).enumerate() {
                if !s && e < min_energy {
                    min_energy = e;
                    min_idx = idx;
                }
            }

            rank[min_idx] = current_rank;
            current_rank += 1;

            let px = min_idx % size;
            let py = min_idx / size;
            is_set[min_idx] = true;
            Self::add_energy(&mut energy, size, px, py, &kernel, kernel_radius);
        }

        rank.into_iter()
            .map(|r| r as f32 / (num_pixels - 1) as f32)
            .collect()
    }

    fn add_energy(
        energy: &mut [f32],
        size: usize,
        px: usize,
        py: usize,
        kernel: &[f32],
        kernel_radius: i32,
    ) {
        let sign = if kernel_radius >= 0 { 1.0 } else { -1.0 };
        let kr = kernel_radius.abs();
        let kernel_size = (kr * 2 + 1) as usize;

        for dy in -kr..=kr {
            for dx in -kr..=kr {
                let nx = (px as i32 + dx).rem_euclid(size as i32) as usize;
                let ny = (py as i32 + dy).rem_euclid(size as i32) as usize;
                let kx = (dx + kr) as usize;
                let ky = (dy + kr) as usize;
                let w = kernel[ky * kernel_size + kx];
                energy[ny * size + nx] += sign * w;
            }
        }
    }

    #[inline]
    pub fn sample(&self, x: usize, y: usize, sample_index: u32) -> Vec2 {
        let tx = x % self.size;
        let ty = y % self.size;
        let idx = ty * self.size + tx;

        let base = self.samples[idx];

        const R2_A1: f32 = 0.7548777;
        const R2_A2: f32 = 0.5698403;

        let n = sample_index as f32;
        let offset_x = (n * R2_A1).fract();
        let offset_y = (n * R2_A2).fract();

        Vec2::new((base.x + offset_x).fract(), (base.y + offset_y).fract())
    }

    #[cfg(test)]
    fn save_png(&self, path: &str) {
        use image::{ImageBuffer, Rgb};

        let img = ImageBuffer::from_fn(self.size as u32, self.size as u32, |x, y| {
            let idx = (y as usize) * self.size + (x as usize);
            let sample = self.samples[idx];
            Rgb([
                (sample.x * 255.0) as u8,
                (sample.y * 255.0) as u8,
                128u8,
            ])
        });

        img.save(path).expect("Failed to save blue noise texture");
    }
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    fn next(&mut self) -> f32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let result = self.state.wrapping_mul(0x2545F4914F6CDD1D);
        (result >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct BlueNoiseGenerator {
        num_points: usize,
        iterations: usize,
    }

    impl BlueNoiseGenerator {
        fn new(num_points: usize) -> Self {
            Self {
                num_points,
                iterations: 200,
            }
        }

        fn with_iterations(mut self, iterations: usize) -> Self {
            self.iterations = iterations;
            self
        }

        fn generate(&self, seed: u64) -> Vec<Vec2> {
            let mut rng = Rng::new(seed);

            let grid_size = (self.num_points as f32).sqrt().ceil() as usize;
            let cell_size = 1.0 / grid_size as f32;

            let mut points: Vec<Vec2> = (0..self.num_points)
                .map(|i| {
                    let gx = i % grid_size;
                    let gy = i / grid_size;
                    Vec2::new(
                        (gx as f32 + rng.next()) * cell_size,
                        (gy as f32 + rng.next()) * cell_size,
                    )
                })
                .collect();

            let ideal_dist = 1.0 / (self.num_points as f32).sqrt();

            for iter in 0..self.iterations {
                let step = 0.5 * ideal_dist * (1.0 - iter as f32 / self.iterations as f32);
                self.repulsion_step(&mut points, ideal_dist, step);
                for p in &mut points {
                    p.x = p.x.rem_euclid(1.0);
                    p.y = p.y.rem_euclid(1.0);
                }
            }

            points
        }

        fn repulsion_step(&self, points: &mut [Vec2], ideal_dist: f32, step: f32) {
            let mut forces = vec![Vec2::ZERO; points.len()];
            let cutoff = ideal_dist * 2.0;
            let cutoff_sq = cutoff * cutoff;

            for i in 0..points.len() {
                for j in (i + 1)..points.len() {
                    let diff = self.toroidal_diff(points[i], points[j]);
                    let dist_sq = diff.length_squared();

                    if dist_sq < cutoff_sq && dist_sq > 1e-10 {
                        let dist = dist_sq.sqrt();
                        let strength = (ideal_dist - dist).max(0.0) / ideal_dist;
                        let force = diff.normalize() * strength * step;
                        forces[i] += force;
                        forces[j] -= force;
                    }
                }
            }

            for (p, f) in points.iter_mut().zip(forces.iter()) {
                *p += *f;
            }
        }

        fn toroidal_diff(&self, a: Vec2, b: Vec2) -> Vec2 {
            let mut d = a - b;
            if d.x > 0.5 { d.x -= 1.0; } else if d.x < -0.5 { d.x += 1.0; }
            if d.y > 0.5 { d.y -= 1.0; } else if d.y < -0.5 { d.y += 1.0; }
            d
        }
    }

    #[test]
    fn test_generation() {
        let generator = BlueNoiseGenerator::new(256).with_iterations(50);
        let points = generator.generate(12345);

        assert_eq!(points.len(), 256);

        for p in &points {
            assert!(p.x >= 0.0 && p.x <= 1.0);
            assert!(p.y >= 0.0 && p.y <= 1.0);
        }
    }

    #[test]
    fn test_min_distance() {
        let generator = BlueNoiseGenerator::new(64).with_iterations(100);
        let points = generator.generate(42);

        let expected_min = 0.5 / (64.0f32).sqrt();

        let mut actual_min = f32::MAX;
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let d = (points[i] - points[j]).length();
                actual_min = actual_min.min(d);
            }
        }

        println!("Expected min distance: {}", expected_min);
        println!("Actual min distance: {}", actual_min);
        assert!(actual_min > expected_min * 0.5, "Points are too clustered");
    }

    #[test]
    fn generate_visualization() {
        let generator = BlueNoiseGenerator::new(4096).with_iterations(200);
        let points = generator.generate(42);

        {
            use image::{ImageBuffer, Luma};
            let img_size = 512u32;
            let mut img = ImageBuffer::from_pixel(img_size, img_size, Luma([0u8]));
            for p in &points {
                let x = (p.x * img_size as f32) as u32 % img_size;
                let y = (p.y * img_size as f32) as u32 % img_size;
                img.put_pixel(x, y, Luma([255u8]));
            }
            img.save("blue_noise_points.png").unwrap();
        }

        let texture = BlueNoiseTexture::generate(64, 42);

        texture.save_png("blue_noise_texture.png");

        println!("Saved blue_noise_texture.png and blue_noise_points.png");
    }
}
