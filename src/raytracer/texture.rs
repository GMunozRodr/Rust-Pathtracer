use glam::{Vec2, Vec3};
use std::path::Path;

#[derive(Clone)]
pub enum Texture {
    SolidColor(Vec3),
    Image {
        data: Vec<Vec3>,
        width: u32,
        height: u32,
    },
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

impl Texture {
    pub fn solid(color: Vec3) -> Self {
        Texture::SolidColor(color)
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, image::ImageError> {
        let path = path.as_ref();
        let is_hdr = path
            .extension()
            .map(|ext| {
                let ext = ext.to_string_lossy().to_lowercase();
                ext == "hdr" || ext == "exr"
            })
            .unwrap_or(false);

        let img = image::open(path)?.into_rgb32f();
        let width = img.width();
        let height = img.height();

        let data: Vec<Vec3> = if is_hdr {
            img.pixels()
                .map(|p| Vec3::new(p.0[0], p.0[1], p.0[2]))
                .collect()
        } else {
            img.pixels()
                .map(|p| {
                    Vec3::new(
                        srgb_to_linear(p.0[0]),
                        srgb_to_linear(p.0[1]),
                        srgb_to_linear(p.0[2]),
                    )
                })
                .collect()
        };

        Ok(Texture::Image {
            data,
            width,
            height,
        })
    }

    pub fn sample(&self, uv: Vec2) -> Vec3 {
        match self {
            Texture::SolidColor(color) => *color,
            Texture::Image {
                data,
                width,
                height,
            } => {
                let u = uv.x.rem_euclid(1.0);
                let v = uv.y.rem_euclid(1.0);

                let px = u * *width as f32 - 0.5;
                let py = v * *height as f32 - 0.5;

                let x0 = px.floor() as i32;
                let y0 = py.floor() as i32;

                let w = *width as i32;
                let h = *height as i32;
                let x0w = x0.rem_euclid(w) as usize;
                let y0h = y0.rem_euclid(h) as usize;
                let x1w = (x0 + 1).rem_euclid(w) as usize;
                let y1h = (y0 + 1).rem_euclid(h) as usize;

                let fx = px - px.floor();
                let fy = py - py.floor();

                let c00 = data[y0h * *width as usize + x0w];
                let c10 = data[y0h * *width as usize + x1w];
                let c01 = data[y1h * *width as usize + x0w];
                let c11 = data[y1h * *width as usize + x1w];

                let c0 = c00.lerp(c10, fx);
                let c1 = c01.lerp(c11, fx);
                c0.lerp(c1, fy)
            }
        }
    }
}
