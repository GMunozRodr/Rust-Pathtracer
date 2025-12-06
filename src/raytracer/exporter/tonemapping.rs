use glam::{Mat3, Vec3};

#[derive(Clone, Copy, Default)]
pub enum ToneMap {
    #[default]
    None,
    Aces,
    Reinhard,
    Agx,
}

impl ToneMap {
    pub fn apply(self, color: Vec3) -> Vec3 {
        match self {
            ToneMap::None => color,
            ToneMap::Aces => aces_tonemap(color),
            ToneMap::Reinhard => reinhard_tonemap(color),
            ToneMap::Agx => agx_tonemap(color),
        }
    }

    pub fn apply_with_exposure(self, color: Vec3, exposure: f32) -> Vec3 {
        self.apply(color * exposure)
    }
}

fn aces_tonemap(color: Vec3) -> Vec3 {
    let color = Vec3::new(
        color.x * 0.59719 + color.y * 0.35458 + color.z * 0.04823,
        color.x * 0.07600 + color.y * 0.90834 + color.z * 0.01566,
        color.x * 0.02840 + color.y * 0.13383 + color.z * 0.83777,
    );

    let a = color * (color + Vec3::splat(0.0245786)) - Vec3::splat(0.000090537);
    let b = color * (color * 0.983729 + Vec3::splat(0.432951)) + Vec3::splat(0.238081);
    let color = a / b;

    Vec3::new(
        color.x * 1.60475 + color.y * -0.53108 + color.z * -0.07367,
        color.x * -0.10208 + color.y * 1.10813 + color.z * -0.00605,
        color.x * -0.00327 + color.y * -0.07276 + color.z * 1.07602,
    )
    .clamp(Vec3::ZERO, Vec3::ONE)
}

fn reinhard_tonemap(color: Vec3) -> Vec3 {
    color / (color + Vec3::ONE)
}

fn agx_default_contrast_approx(x: Vec3) -> Vec3 {
    let x2 = x * x;
    let x4 = x2 * x2;

    x4 * x2 * 15.5
        - x4 * x * 40.14
        + x4 * 31.96
        - x2 * x * 6.868
        + x2 * 0.4298
        + x * 0.1191
        - Vec3::splat(0.00232)
}

fn agx(val: Vec3) -> Vec3 {
    let agx_mat = Mat3::from_cols(
        Vec3::new(0.842479062253094, 0.0784335999999992, 0.0792237451477643),
        Vec3::new(0.0423282422610123, 0.878468636469772, 0.0791661274605434),
        Vec3::new(0.0423756549057051, 0.0784336, 0.879142973793104),
    );

    const MIN_EV: f32 = -12.47393;
    const MAX_EV: f32 = 4.026069;

    let val = agx_mat * val;

    let val = Vec3::new(
        val.x.max(1e-10).log2().clamp(MIN_EV, MAX_EV),
        val.y.max(1e-10).log2().clamp(MIN_EV, MAX_EV),
        val.z.max(1e-10).log2().clamp(MIN_EV, MAX_EV),
    );
    let val = (val - Vec3::splat(MIN_EV)) / (MAX_EV - MIN_EV);

    agx_default_contrast_approx(val)
}

fn agx_eotf(val: Vec3) -> Vec3 {
    let agx_mat_inv = Mat3::from_cols(
        Vec3::new(1.19687900512017, -0.0980208811401368, -0.0990297440797205),
        Vec3::new(-0.0528968517574562, 1.15190312990417, -0.0989611768448433),
        Vec3::new(-0.0529716355144438, -0.0980434501171241, 1.15107367264116),
    );

    let val = agx_mat_inv * val;

    Vec3::new(val.x.powf(2.2), val.y.powf(2.2), val.z.powf(2.2))
}

fn agx_tonemap(color: Vec3) -> Vec3 {
    agx_eotf(agx(color))
}

pub fn linear_to_srgb_u8(color: Vec3) -> [u8; 3] {
    fn linear_to_srgb(c: f32) -> f32 {
        if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    }

    [
        (linear_to_srgb(color.x.clamp(0.0, 1.0)) * 255.0) as u8,
        (linear_to_srgb(color.y.clamp(0.0, 1.0)) * 255.0) as u8,
        (linear_to_srgb(color.z.clamp(0.0, 1.0)) * 255.0) as u8,
    ]
}
