use glam::Vec3;
use std::f32::consts::PI;

pub fn ggx_d(n_dot_h: f32, alpha: f32) -> f32 {
    let n_dot_h = n_dot_h.max(0.0);
    let a2 = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

pub fn ggx_g1(n_dot_v: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = n_dot_v + (a2 + (1.0 - a2) * n_dot_v * n_dot_v).sqrt();
    (2.0 * n_dot_v) / denom
}

pub fn ggx_g(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    ggx_g1(n_dot_l, alpha) * ggx_g1(n_dot_v, alpha)
}

pub fn pdf_ggx(n_dot_h: f32, v_dot_h: f32, alpha: f32) -> f32 {
    if n_dot_h <= 0.0 || v_dot_h <= 0.0 {
        return 0.0;
    }
    let d = ggx_d(n_dot_h, alpha);
    d * n_dot_h / (4.0 * v_dot_h)
}

pub fn pdf_cosine(n_dot_l: f32) -> f32 {
    n_dot_l / PI
}

pub fn pdf_brdf(n_dot_l: f32, n_dot_h: f32, v_dot_h: f32, alpha: f32, spec_prob: f32) -> f32 {
    let pdf_spec = pdf_ggx(n_dot_h, v_dot_h, alpha);
    let pdf_diff = pdf_cosine(n_dot_l);
    spec_prob * pdf_spec + (1.0 - spec_prob) * pdf_diff
}

pub fn fresnel_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
    f0 + (Vec3::ONE - f0) * (1.0 - cos_theta).clamp(0.0, 1.0).powi(5)
}

pub fn fresnel_dielectric(cos_theta: f32, ior: f32) -> f32 {
    let r0 = ((1.0 - ior) / (1.0 + ior)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cos_theta).clamp(0.0, 1.0).powi(5)
}

pub fn refract(incident: Vec3, normal: Vec3, eta: f32) -> Option<Vec3> {
    let cos_i = -incident.dot(normal);
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);

    if sin2_t > 1.0 {
        return None;
    }

    let cos_t = (1.0 - sin2_t).sqrt();
    Some(eta * incident + (eta * cos_i - cos_t) * normal)
}

pub fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    (incident - 2.0 * incident.dot(normal) * normal).normalize()
}

fn ggx_directional_albedo(n_dot_v: f32, alpha: f32) -> f32 {
    let a = alpha;
    let mu = n_dot_v;
    1.0 - (1.0 - mu).powi(5) * (1.0 - a).powi(2) * 0.65
}

pub fn multiscatter_compensation(n_dot_v: f32, n_dot_l: f32, alpha: f32, f0: Vec3) -> Vec3 {
    let e_v = ggx_directional_albedo(n_dot_v, alpha);
    let e_l = ggx_directional_albedo(n_dot_l, alpha);

    let f_avg = f0 + (Vec3::ONE - f0) / 21.0;

    let f_ms = f_avg * f_avg * (1.0 - e_v) * (1.0 - e_l) / (PI * (1.0 - f_avg * (1.0 - e_v)));

    Vec3::ONE + f_ms / (e_v * e_l + 0.001)
}

pub fn evaluate_brdf(v: Vec3, l: Vec3, n: Vec3, alpha: f32, f0: Vec3, diffuse_color: Vec3) -> Vec3 {
    let n_dot_l = n.dot(l);
    let n_dot_v = n.dot(v);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return Vec3::ZERO;
    }

    let h = (v + l).normalize();
    let n_dot_h = n.dot(h).max(0.001);
    let v_dot_h = v.dot(h).max(0.001);

    let f = fresnel_schlick(v_dot_h, f0);
    let d = ggx_d(n_dot_h, alpha);
    let g = ggx_g(n_dot_l, n_dot_v, alpha);

    let ms = multiscatter_compensation(n_dot_v, n_dot_l, alpha, f0);

    let specular = f * d * g / (4.0 * n_dot_l * n_dot_v) * ms;
    let diffuse = diffuse_color * (Vec3::ONE - f) / PI;

    (specular + diffuse) * n_dot_l
}
