use glam::{Vec3, Vec4};

#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

#[derive(Clone, Copy)]
pub struct Material {
    pub base_color_texture: Option<u32>,
    pub base_color_factor: Vec4,
    pub metallic_roughness_texture: Option<u32>,
    pub roughness_factor: f32,
    pub metallic_factor: f32,
    pub occlusion_texture: Option<u32>,
    pub occlusion_strength: f32,
    pub normal_texture: Option<u32>,
    pub normal_scale: f32,
    pub emissive_texture: Option<u32>,
    pub emissive_factor: Vec3,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub ior: f32,
    pub transmission: f32,
    pub subsurface: f32,
    pub subsurface_radius: Vec3,
    pub subsurface_albedo: Vec3,
    pub subsurface_anisotropy: f32,
    pub subsurface_entry_tint: Vec3,
    pub subsurface_exit_tint: Vec3,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color_texture: None,
            base_color_factor: Vec4::ONE,
            metallic_roughness_texture: None,
            roughness_factor: 1.0,
            metallic_factor: 0.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            normal_texture: None,
            normal_scale: 1.0,
            emissive_texture: None,
            emissive_factor: Vec3::ZERO,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            ior: 1.5,
            transmission: 0.0,
            subsurface: 0.0,
            subsurface_radius: Vec3::new(1.0, 0.2, 0.1),
            subsurface_albedo: Vec3::new(0.98, 0.80, 0.78),
            subsurface_anisotropy: 0.0,
            subsurface_entry_tint: Vec3::new(0.95, 0.85, 0.80),
            subsurface_exit_tint: Vec3::new(1.0, 0.85, 0.75),
        }
    }
}

impl Material {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn diffuse(color: Vec3) -> Self {
        Self {
            base_color_factor: color.extend(1.0),
            ..Default::default()
        }
    }

    pub fn metal(color: Vec3, roughness: f32) -> Self {
        Self {
            base_color_factor: color.extend(1.0),
            roughness_factor: roughness,
            metallic_factor: 1.0,
            ..Default::default()
        }
    }

    pub fn glass(color: Vec3, ior: f32, roughness: f32) -> Self {
        Self {
            base_color_factor: color.extend(1.0),
            roughness_factor: roughness,
            ior,
            transmission: 1.0,
            ..Default::default()
        }
    }

    pub fn subsurface(color: Vec3, radius: Vec3, albedo: Vec3) -> Self {
        Self {
            base_color_factor: color.extend(1.0),
            subsurface: 1.0,
            subsurface_radius: radius,
            subsurface_albedo: albedo,
            ..Default::default()
        }
    }
}
