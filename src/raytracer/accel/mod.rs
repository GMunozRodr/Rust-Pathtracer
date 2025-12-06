mod bvh;

pub use bvh::{Aabb, Bvh8};

use crate::raytracer::ray::{HitData, Ray};
use crate::raytracer::shape::Shape;
use glam::{Mat4, Vec3};
use std::sync::Arc;

pub trait Bounded {
    fn bounds(&self) -> Aabb;
    fn center(&self) -> Vec3;
}

pub trait Hit {
    fn t(&self) -> f32;
}

const MIN_PRIMS_PER_LEAF: u32 = 4;

pub struct Blas<T: Shape> {
    bvh8: Bvh8,
    items: Vec<T>,
    bounds: Aabb,
}

impl<T: Shape> Blas<T> {
    pub fn build(primitives: impl Into<Vec<T>>) -> Blas<T> {
        let items: Vec<T> = primitives.into();

        if items.is_empty() {
            return Blas {
                bvh8: Bvh8::empty(),
                items,
                bounds: Aabb::EMPTY,
            };
        }

        let mut bounds = Aabb::EMPTY;
        let prim_bounds: Vec<_> = items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let b = item.bounds();
                bounds.grow(&b);
                (i as u32, b)
            })
            .collect();

        let bvh8 = Bvh8::build(prim_bounds, MIN_PRIMS_PER_LEAF);

        Blas { bvh8, items, bounds }
    }

    #[inline]
    pub fn bounds(&self) -> Aabb {
        self.bounds
    }

    #[inline]
    pub fn hit(&self, ray: &Ray) -> Option<HitData> {
        self.bvh8.traverse_closest(ray, |prim_idx, ray| {
            let item = &self.items[prim_idx as usize];
            item.hit(ray).map(|hit| (hit.t, hit))
        })
    }

    #[inline]
    pub fn hit_any(&self, ray: &Ray) -> bool {
        self.bvh8.traverse_any(ray, |prim_idx, ray| {
            self.items[prim_idx as usize].hit_any(ray)
        })
    }
}

pub struct Instance<T: Shape> {
    pub blas: Arc<Blas<T>>,
    pub transform: Mat4,
    pub inverse_transform: Mat4,
    normal_matrix: Mat4,
    world_bounds: Aabb,
}

impl<T: Shape> Instance<T> {
    pub fn new(blas: Arc<Blas<T>>, transform: Mat4) -> Self {
        let inverse_transform = transform.inverse();
        let normal_matrix = inverse_transform.transpose();
        let world_bounds = Self::compute_world_bounds(&blas, &transform);
        Instance {
            blas,
            transform,
            inverse_transform,
            normal_matrix,
            world_bounds,
        }
    }

    fn compute_world_bounds(blas: &Blas<T>, transform: &Mat4) -> Aabb {
        let local_bounds = blas.bounds();
        let corners = [
            Vec3::new(local_bounds.min.x, local_bounds.min.y, local_bounds.min.z),
            Vec3::new(local_bounds.max.x, local_bounds.min.y, local_bounds.min.z),
            Vec3::new(local_bounds.min.x, local_bounds.max.y, local_bounds.min.z),
            Vec3::new(local_bounds.max.x, local_bounds.max.y, local_bounds.min.z),
            Vec3::new(local_bounds.min.x, local_bounds.min.y, local_bounds.max.z),
            Vec3::new(local_bounds.max.x, local_bounds.min.y, local_bounds.max.z),
            Vec3::new(local_bounds.min.x, local_bounds.max.y, local_bounds.max.z),
            Vec3::new(local_bounds.max.x, local_bounds.max.y, local_bounds.max.z),
        ];

        let first = transform.transform_point3(corners[0]);
        let mut min = first;
        let mut max = first;

        for corner in corners.iter().skip(1) {
            let world = transform.transform_point3(*corner);
            min = min.min(world);
            max = max.max(world);
        }

        Aabb { min, max }
    }

    #[inline]
    pub fn hit(&self, ray: &Ray) -> Option<HitData> {
        let local_origin = self.inverse_transform.transform_point3(ray.origin);
        let local_direction = self.inverse_transform.transform_vector3(ray.direction);
        let local_ray = Ray::new(local_origin, local_direction);

        let mut hit = self.blas.hit(&local_ray)?;

        let local_hit_point = local_origin + hit.t * local_direction;
        let world_hit_point = self.transform.transform_point3(local_hit_point);
        hit.t = (world_hit_point - ray.origin).dot(ray.direction) / ray.direction.length_squared();

        hit.normal = self.normal_matrix.transform_vector3(hit.normal).normalize();
        hit.geo_normal = self.normal_matrix.transform_vector3(hit.geo_normal).normalize();

        Some(hit)
    }

    #[inline]
    pub fn hit_any(&self, ray: &Ray) -> bool {
        let local_origin = self.inverse_transform.transform_point3(ray.origin);
        let local_direction = self.inverse_transform.transform_vector3(ray.direction);
        let local_ray = Ray::new(local_origin, local_direction);
        self.blas.hit_any(&local_ray)
    }
}

impl<T: Shape> Bounded for Instance<T> {
    fn bounds(&self) -> Aabb {
        self.world_bounds
    }

    fn center(&self) -> Vec3 {
        self.world_bounds.center()
    }
}

pub struct Tlas<T: Shape> {
    bvh8: Bvh8,
    instances: Vec<Instance<T>>,
}

impl<T: Shape> Tlas<T> {
    pub fn build(instances: Vec<Instance<T>>) -> Tlas<T> {
        if instances.is_empty() {
            return Tlas {
                bvh8: Bvh8::empty(),
                instances,
            };
        }

        let instance_bounds: Vec<_> = instances
            .iter()
            .enumerate()
            .map(|(i, inst)| (i as u32, inst.world_bounds))
            .collect();

        let bvh8 = Bvh8::build(instance_bounds, 1);

        Tlas { bvh8, instances }
    }

    #[inline]
    pub fn hit(&self, ray: &Ray) -> Option<HitData> {
        self.bvh8.traverse_closest(ray, |inst_idx, ray| {
            let instance = &self.instances[inst_idx as usize];
            instance.hit(ray).map(|hit| (hit.t, hit))
        })
    }

    #[inline]
    pub fn hit_any(&self, ray: &Ray) -> bool {
        self.bvh8.traverse_any(ray, |inst_idx, ray| {
            self.instances[inst_idx as usize].hit_any(ray)
        })
    }
}
