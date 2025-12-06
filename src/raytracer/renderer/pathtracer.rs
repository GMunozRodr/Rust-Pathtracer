use super::brdf::{
    evaluate_brdf, fresnel_dielectric, fresnel_schlick, ggx_g, pdf_brdf, pdf_cosine, refract,
    reflect,
};
use super::sampling::{
    build_basis, clamp_radiance, fix_shading_normal, half_vector_dots, mis_weight, safe_inv_pdf,
    sample_cosine_hemisphere, sample_ggx, to_world,
};
use super::subsurface::{random_walk_subsurface, Rng};
use super::{AovData, RenderResult, Renderer, SceneAccess};
use crate::raytracer::material::Material;
use crate::raytracer::ray::{HitData, RAY_EPSILON, Ray};
use glam::Vec3;
use std::f32::consts::PI;

pub struct PathTracer<const MAX_BOUNCES: u32>;

struct BrdfContext {
    alpha: f32,
    f0: Vec3,
    diffuse_color: Vec3,
    spec_prob: f32,
}

struct SampledMaterial {
    base_color: Vec3,
    roughness: f32,
    metallic: f32,
    emissive: Vec3,
    normal: Vec3,
    transmission: f32,
    ior: f32,
}

fn sample_material<S: SceneAccess>(
    material: &Material,
    hit: &HitData,
    scene: &S,
) -> SampledMaterial {
    let base_color = match material.base_color_texture {
        Some(tex) => scene.sample_texture(tex, hit.uv) * material.base_color_factor.truncate(),
        None => material.base_color_factor.truncate(),
    };

    let (roughness, metallic) = match material.metallic_roughness_texture {
        Some(tex) => {
            let sampled = scene.sample_texture(tex, hit.uv);
            (
                sampled.y * material.roughness_factor,
                sampled.z * material.metallic_factor,
            )
        }
        None => (material.roughness_factor, material.metallic_factor),
    };

    let emissive = match material.emissive_texture {
        Some(tex) => scene.sample_texture(tex, hit.uv) * material.emissive_factor,
        None => material.emissive_factor,
    };

    let normal = match material.normal_texture {
        Some(tex) => {
            let sampled = scene.sample_texture(tex, hit.uv);
            let tn = (sampled * 2.0 - Vec3::ONE).normalize();
            let tn = Vec3::new(
                tn.x * material.normal_scale,
                tn.y * material.normal_scale,
                tn.z,
            )
            .normalize();
            (hit.tangent * tn.x + hit.bitangent * tn.y + hit.normal * tn.z).normalize()
        }
        None => hit.normal,
    };

    let roughness = roughness.clamp(0.04, 1.0);

    SampledMaterial {
        base_color,
        roughness,
        metallic,
        emissive,
        normal,
        transmission: material.transmission,
        ior: material.ior,
    }
}

fn ray_seed(ray: &Ray) -> u32 {
    ray.origin.x.to_bits()
        .wrapping_add(ray.origin.y.to_bits().wrapping_mul(73856093))
        .wrapping_add(ray.origin.z.to_bits().wrapping_mul(19349663))
        .wrapping_add(ray.direction.x.to_bits().wrapping_mul(83492791))
}

impl<const MAX_BOUNCES: u32> PathTracer<MAX_BOUNCES> {
    pub fn new() -> Self {
        Self
    }

    fn trace_inner<S: SceneAccess>(
        &self,
        initial_ray: &Ray,
        scene: &S,
        seed: u32,
        mut aov_out: Option<&mut AovData>,
    ) -> Vec3 {
        let mut rng = Rng::new(seed);
        let mut radiance = Vec3::ZERO;
        let mut throughput = Vec3::ONE;
        let mut ray = *initial_ray;
        let mut first_hit = true;

        let mut last_brdf_pdf = 0.0f32;
        let mut last_was_specular = false;

        for bounce in 0..MAX_BOUNCES {
            let Some(hit) = scene.hit(&ray) else {
                let sky_radiance = scene.sample_sky(ray.direction);
                if bounce == 0 {
                    radiance += throughput * sky_radiance;
                    if let Some(aov) = aov_out.as_deref_mut() && first_hit {
                        aov.albedo = sky_radiance.clamp(Vec3::ZERO, Vec3::ONE);
                        aov.normal = -ray.direction;
                    }
                } else if last_was_specular {
                    radiance += throughput * sky_radiance;
                } else {
                    let sky_pdf = scene.sky_pdf(ray.direction);
                    let mis = mis_weight(last_brdf_pdf, sky_pdf);
                    radiance += throughput * sky_radiance * mis;
                }
                break;
            };

            let material = scene.get_material(hit.material_id);
            let mat = sample_material(material, &hit, scene);

            if let Some(aov) = aov_out.as_deref_mut() && first_hit {
                aov.albedo = mat.base_color;
                aov.normal = mat.normal * 0.5 + Vec3::splat(0.5);
                first_hit = false;
            }

            radiance += throughput * mat.emissive;

            let front_face = ray.direction.dot(hit.geo_normal) < 0.0;
            let geo_n = if front_face { hit.geo_normal } else { -hit.geo_normal };
            let v = -ray.direction.normalize();

            let shading_n = if mat.normal.dot(geo_n) > 0.0 { mat.normal } else { -mat.normal };
            let n = fix_shading_normal(shading_n, geo_n, v);
            let hit_point = ray.origin + ray.direction * hit.t;

            if mat.transmission > 0.0 && rng.next() < mat.transmission {
                let (new_dir, offset_normal) =
                    Self::sample_transmission(&mut rng, ray.direction, n, front_face, mat.ior);

                throughput *= mat.base_color;

                if !Self::russian_roulette(&mut rng, &mut throughput, bounce) {
                    break;
                }

                ray = Ray::new(hit_point + offset_normal * RAY_EPSILON, new_dir);
                last_was_specular = true;
                continue;
            }

            let sss_prob = if material.subsurface > 0.0 && front_face {
                material.subsurface
            } else {
                0.0
            };

            if sss_prob > 0.0 && rng.next() < sss_prob {
                throughput /= sss_prob;

                let radius = material.subsurface_radius;
                let albedo = material.subsurface_albedo;

                let cos_theta = v.dot(n).min(1.0);
                let fresnel = fresnel_dielectric(cos_theta, material.ior);

                if rng.next() < fresnel {
                    let reflect_dir = reflect(-v, n);
                    ray = Ray::new(hit_point + n * RAY_EPSILON, reflect_dir);
                    last_was_specular = true;
                    continue;
                }

                let eta = 1.0 / material.ior;
                let entry_dir = match refract(-v, n, eta) {
                    Some(refracted) => refracted.normalize(),
                    None => {
                        let reflect_dir = reflect(-v, n);
                        ray = Ray::new(hit_point + n * RAY_EPSILON, reflect_dir);
                        last_was_specular = true;
                        continue;
                    }
                };

                throughput *= material.subsurface_entry_tint;

                let entry_point = hit_point - n * RAY_EPSILON;

                let thickness_ray = Ray::new(entry_point, entry_dir);
                let thickness = if let Some(back_hit) = scene.hit(&thickness_ray) {
                    if entry_dir.dot(back_hit.normal) > 0.0 {
                        back_hit.t
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

                let thickness_scale = (thickness / 1.0).clamp(0.1, 2.0);
                let scaled_radius = radius * thickness_scale;

                if let Some(sss_result) = random_walk_subsurface(
                    &mut rng,
                    scene,
                    entry_point,
                    entry_dir,
                    scaled_radius,
                    albedo,
                    material.subsurface_anisotropy,
                ) {
                    throughput *= mat.base_color * sss_result.throughput * material.subsurface_exit_tint;

                    if !Self::russian_roulette(&mut rng, &mut throughput, bounce) {
                        break;
                    }

                    let exit_n = sss_result.exit_normal;
                    let exit_point = sss_result.exit_point;

                    let diffuse_brdf = mat.base_color / PI;

                    if let Some(sun) = scene.get_sun() {
                        let n_dot_l = exit_n.dot(sun.direction);
                        if n_dot_l > 0.0 {
                            let shadow_ray = Ray::new(exit_point + exit_n * RAY_EPSILON, sun.direction);
                            if !scene.hit_any(&shadow_ray) {
                                radiance += throughput * diffuse_brdf * sun.color * n_dot_l;
                            }
                        }
                    }

                    let sky_sample = scene.sample_sky_direction(rng.next(), rng.next());
                    let n_dot_l = exit_n.dot(sky_sample.direction);
                    if n_dot_l > 0.0 && sky_sample.pdf > 0.0 {
                        let shadow_ray = Ray::new(exit_point + exit_n * RAY_EPSILON, sky_sample.direction);
                        if !scene.hit_any(&shadow_ray) {
                            let sky_contribution = clamp_radiance(sky_sample.radiance * safe_inv_pdf(sky_sample.pdf));
                            radiance += throughput * diffuse_brdf * sky_contribution * n_dot_l;
                        }
                    }

                    let (exit_t, exit_b, _) = build_basis(exit_n);
                    let exit_local = sample_cosine_hemisphere(rng.next(), rng.next());
                    let exit_dir = to_world(exit_local, exit_t, exit_b, exit_n);

                    ray = Ray::new(
                        exit_point + exit_n * RAY_EPSILON,
                        exit_dir,
                    );
                    last_was_specular = false;
                    last_brdf_pdf = pdf_cosine(exit_n.dot(exit_dir).max(0.001));
                    continue;
                } else {
                    break;
                }
            }

            if sss_prob > 0.0 {
                throughput /= 1.0 - sss_prob;
            }

            last_was_specular = false;

            let alpha = mat.roughness * mat.roughness;
            let f0 = Vec3::splat(0.04).lerp(mat.base_color, mat.metallic);
            let diffuse_color = mat.base_color * (1.0 - mat.metallic);

            let (t, b, _) = build_basis(n);
            let n_dot_v = v.dot(n).max(0.001);
            let spec_prob = Self::compute_spec_prob(n_dot_v, f0, diffuse_color);

            let brdf_ctx = BrdfContext { alpha, f0, diffuse_color, spec_prob };

            if let Some(sun) = scene.get_sun() {
                radiance += Self::sample_direct_light(
                    scene, &hit_point, n, geo_n, v, &brdf_ctx, throughput,
                    sun.direction, sun.color, sun.pdf,
                );
            }

            radiance += Self::sample_sky_light(
                &mut rng, scene, &hit_point, n, geo_n, v, &brdf_ctx, throughput,
            );

            let Some((new_dir, brdf_weight, sample_pdf)) = Self::sample_brdf(
                &mut rng, v, n, t, b, n_dot_v, &brdf_ctx,
            ) else {
                break;
            };

            throughput *= brdf_weight;

            if !Self::russian_roulette(&mut rng, &mut throughput, bounce) {
                break;
            }

            ray = Ray::new(hit_point + geo_n * RAY_EPSILON, new_dir);
            last_brdf_pdf = sample_pdf;

            if let Some(sun) = scene.get_sun() && sun.is_in_cone(new_dir) {
                let mis = mis_weight(sample_pdf, sun.pdf);
                radiance += throughput * sun.color * mis;
            }
        }

        clamp_radiance(radiance)
    }

    #[inline]
    fn sample_transmission(
        rng: &mut Rng,
        incident: Vec3,
        n: Vec3,
        front_face: bool,
        ior: f32,
    ) -> (Vec3, Vec3) {
        let eta = if front_face { 1.0 / ior } else { ior };
        let cos_theta = (-incident).dot(n).min(1.0);
        let fresnel = fresnel_dielectric(cos_theta, ior);

        if rng.next() < fresnel {
            (reflect(incident, n), n)
        } else {
            match refract(incident, n, eta) {
                Some(refracted) => (refracted.normalize(), -n),
                None => (reflect(incident, n), n),
            }
        }
    }

    #[inline]
    fn russian_roulette(rng: &mut Rng, throughput: &mut Vec3, bounce: u32) -> bool {
        if bounce > 2 {
            let p = throughput.max_element().clamp(0.05, 0.95);
            if rng.next() > p {
                return false;
            }
            *throughput /= p;
        }
        true
    }

    #[inline]
    fn compute_spec_prob(n_dot_v: f32, f0: Vec3, diffuse_color: Vec3) -> f32 {
        let spec_albedo = fresnel_schlick(n_dot_v, f0);
        let diff_albedo = diffuse_color * (Vec3::ONE - spec_albedo);
        let lum = Vec3::new(0.2126, 0.7152, 0.0722);
        let spec_lum = spec_albedo.dot(lum);
        let diff_lum = diff_albedo.dot(lum);
        (spec_lum / (spec_lum + diff_lum + 0.001)).clamp(0.1, 0.9)
    }

    #[inline]
    fn sample_direct_light<S: SceneAccess>(
        scene: &S,
        hit_point: &Vec3,
        n: Vec3,
        geo_n: Vec3,
        v: Vec3,
        brdf_ctx: &BrdfContext,
        throughput: Vec3,
        light_dir: Vec3,
        light_color: Vec3,
        light_pdf: f32,
    ) -> Vec3 {
        let n_dot_l = n.dot(light_dir);
        if n_dot_l <= 0.0 {
            return Vec3::ZERO;
        }

        let shadow_ray = Ray::new(*hit_point + geo_n * RAY_EPSILON, light_dir);
        if scene.hit_any(&shadow_ray) {
            return Vec3::ZERO;
        }

        let brdf = evaluate_brdf(v, light_dir, n, brdf_ctx.alpha, brdf_ctx.f0, brdf_ctx.diffuse_color);
        let (n_dot_h, v_dot_h) = half_vector_dots(v, light_dir, n);
        let brdf_pdf = pdf_brdf(n_dot_l, n_dot_h, v_dot_h, brdf_ctx.alpha, brdf_ctx.spec_prob);
        let mis = mis_weight(light_pdf, brdf_pdf);
        throughput * brdf * light_color * mis
    }

    #[inline]
    fn sample_sky_light<S: SceneAccess>(
        rng: &mut Rng,
        scene: &S,
        hit_point: &Vec3,
        n: Vec3,
        geo_n: Vec3,
        v: Vec3,
        brdf_ctx: &BrdfContext,
        throughput: Vec3,
    ) -> Vec3 {
        let sky_sample = scene.sample_sky_direction(rng.next(), rng.next());
        let l = sky_sample.direction;
        let n_dot_l = n.dot(l);

        if n_dot_l <= 0.0 || sky_sample.pdf <= 0.0 {
            return Vec3::ZERO;
        }

        let shadow_ray = Ray::new(*hit_point + geo_n * RAY_EPSILON, l);
        if scene.hit_any(&shadow_ray) {
            return Vec3::ZERO;
        }

        let brdf = evaluate_brdf(v, l, n, brdf_ctx.alpha, brdf_ctx.f0, brdf_ctx.diffuse_color);
        let (n_dot_h, v_dot_h) = half_vector_dots(v, l, n);
        let brdf_pdf = pdf_brdf(n_dot_l, n_dot_h, v_dot_h, brdf_ctx.alpha, brdf_ctx.spec_prob);
        let mis = mis_weight(sky_sample.pdf, brdf_pdf);

        let sky_contribution = sky_sample.radiance * safe_inv_pdf(sky_sample.pdf);
        let sky_contribution = clamp_radiance(sky_contribution);

        throughput * brdf * sky_contribution * mis
    }

    #[inline]
    fn sample_brdf(
        rng: &mut Rng,
        v: Vec3,
        n: Vec3,
        t: Vec3,
        b: Vec3,
        n_dot_v: f32,
        brdf_ctx: &BrdfContext,
    ) -> Option<(Vec3, Vec3, f32)> {
        if rng.next() < brdf_ctx.spec_prob {
            let h_local = sample_ggx(rng.next(), rng.next(), brdf_ctx.alpha);
            let h = to_world(h_local, t, b, n);
            let l = (2.0 * v.dot(h) * h - v).normalize();

            let n_dot_l = n.dot(l);
            if n_dot_l <= 0.0 {
                return None;
            }

            let n_dot_h = n.dot(h).max(0.001);
            let v_dot_h = v.dot(h).max(0.001);

            let f = fresnel_schlick(v_dot_h, brdf_ctx.f0);
            let g = ggx_g(n_dot_l, n_dot_v, brdf_ctx.alpha);
            let weight = f * g * v_dot_h / (n_dot_v * n_dot_h * brdf_ctx.spec_prob);
            let pdf = pdf_brdf(n_dot_l, n_dot_h, v_dot_h, brdf_ctx.alpha, brdf_ctx.spec_prob);

            Some((l, weight, pdf))
        } else {
            let l_local = sample_cosine_hemisphere(rng.next(), rng.next());
            let l = to_world(l_local, t, b, n);

            let n_dot_l = n.dot(l).max(0.001);
            let (n_dot_h, v_dot_h) = half_vector_dots(v, l, n);
            let f = fresnel_schlick(v_dot_h, brdf_ctx.f0);

            let weight = brdf_ctx.diffuse_color * (Vec3::ONE - f) / (1.0 - brdf_ctx.spec_prob);
            let pdf = pdf_brdf(n_dot_l, n_dot_h, v_dot_h, brdf_ctx.alpha, brdf_ctx.spec_prob);

            Some((l, weight, pdf))
        }
    }
}

impl<const MAX_BOUNCES: u32> Renderer for PathTracer<MAX_BOUNCES> {
    fn render<S: SceneAccess>(&self, ray: &Ray, scene: &S, sample_index: u32) -> Vec3 {
        let seed = ray_seed(ray).wrapping_add(sample_index.wrapping_mul(2654435761));
        self.trace_inner(ray, scene, seed, None)
    }

    fn render_with_aov<S: SceneAccess>(&self, ray: &Ray, scene: &S, sample_index: u32) -> RenderResult {
        let seed = ray_seed(ray).wrapping_add(sample_index.wrapping_mul(2654435761));
        let mut aov = AovData::default();
        let color = self.trace_inner(ray, scene, seed, Some(&mut aov));
        RenderResult { color, aov }
    }
}

impl<const MAX_BOUNCES: u32> Default for PathTracer<MAX_BOUNCES> {
    fn default() -> Self {
        Self::new()
    }
}
