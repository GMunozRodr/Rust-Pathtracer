use glam::Vec3;
use std::f32::consts::PI;

pub fn sample_ggx(u1: f32, u2: f32, alpha: f32) -> Vec3 {
    let phi = 2.0 * PI * u1;
    let cos_theta = ((1.0 - u2) / (u2 * (alpha * alpha - 1.0) + 1.0)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

pub fn sample_cosine_hemisphere(u1: f32, u2: f32) -> Vec3 {
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    Vec3::new(r * phi.cos(), r * phi.sin(), (1.0 - u1).max(0.0).sqrt())
}

pub fn build_basis(n: Vec3) -> (Vec3, Vec3, Vec3) {
    let up = if n.y.abs() < 0.999 { Vec3::Y } else { Vec3::X };
    let t = up.cross(n).normalize();
    let b = n.cross(t);
    (t, b, n)
}

pub fn to_world(local: Vec3, t: Vec3, b: Vec3, n: Vec3) -> Vec3 {
    t * local.x + b * local.y + n * local.z
}

#[inline]
pub fn half_vector_dots(v: Vec3, l: Vec3, n: Vec3) -> (f32, f32) {
    let h = (v + l).normalize();
    (n.dot(h).max(0.001), v.dot(h).max(0.001))
}

pub fn mis_weight(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    a2 / (a2 + b2 + 1e-10)
}

pub const MAX_INV_PDF: f32 = 100.0;

pub const MAX_SAMPLE_VALUE: f32 = 10.0;

pub fn safe_inv_pdf(pdf: f32) -> f32 {
    (1.0 / pdf).min(MAX_INV_PDF)
}

pub fn clamp_radiance(radiance: Vec3) -> Vec3 {
    if !radiance.is_finite() {
        return Vec3::ZERO;
    }

    let max_component = radiance.max_element();
    if max_component > MAX_SAMPLE_VALUE {
        radiance * (MAX_SAMPLE_VALUE / max_component)
    } else {
        radiance
    }
}

pub fn fix_shading_normal(shading_n: Vec3, geo_n: Vec3, v: Vec3) -> Vec3 {
    let shading_n = if shading_n.dot(geo_n) < 0.0 {
        -shading_n
    } else {
        shading_n
    };

    let n_dot_v = shading_n.dot(v);
    if n_dot_v > 0.0 {
        return shading_n;
    }

    let projected = shading_n - v * n_dot_v;
    let len = projected.length();
    if len > 1e-6 {
        (projected / len * 0.5 + geo_n * 0.5).normalize()
    } else {
        geo_n
    }
}
