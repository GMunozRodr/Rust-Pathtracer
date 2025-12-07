use crate::raytracer::accel::{Blas, Tlas};
use crate::raytracer::camera::Camera;
use crate::raytracer::light::DirectionalLight;
use crate::raytracer::material::Material;
use crate::raytracer::ray::{HitData, Ray};
use crate::raytracer::renderer::SceneAccess;
use crate::raytracer::shape::Shape;
use crate::raytracer::sky::{Sky, SkySample};
use crate::raytracer::texture::Texture;
use glam::{Mat4, Vec2, Vec3};

pub struct Scene<P: Shape, S: Sky> {
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub tlas: Tlas<P>,
    pub camera: Camera,
    pub sky: S,
    pub sun: Option<DirectionalLight>,
}

impl<P: Shape, S: Sky> Scene<P, S> {
    pub fn new(
        blases: Vec<Blas<P>>,
        instances: Vec<(u32, Mat4)>,
        materials: Vec<Material>,
        textures: Vec<Texture>,
        camera: Camera,
        sky: S,
        sun: Option<DirectionalLight>,
    ) -> Self {
        let tlas = Tlas::build(blases, instances);
        Scene {
            materials,
            textures,
            tlas,
            camera,
            sky,
            sun,
        }
    }
}

impl<P: Shape, S: Sky> SceneAccess for Scene<P, S> {
    fn hit(&self, ray: &Ray) -> Option<HitData> {
        self.tlas.hit(ray)
    }

    fn hit_any(&self, ray: &Ray) -> bool {
        self.tlas.hit_any(ray)
    }

    fn get_material(&self, material_id: u32) -> &Material {
        &self.materials[material_id as usize]
    }

    fn sample_texture(&self, texture_id: u32, uv: Vec2) -> Vec3 {
        self.textures[texture_id as usize].sample(uv)
    }

    fn sample_sky(&self, direction: Vec3) -> Vec3 {
        self.sky.sample(direction)
    }

    fn sample_sky_direction(&self, u1: f32, u2: f32) -> SkySample {
        self.sky.sample_direction(u1, u2)
    }

    fn sky_pdf(&self, direction: Vec3) -> f32 {
        self.sky.pdf(direction)
    }

    fn get_sun(&self) -> Option<&DirectionalLight> {
        self.sun.as_ref()
    }
}
