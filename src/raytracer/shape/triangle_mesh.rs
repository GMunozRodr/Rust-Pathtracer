use crate::raytracer::accel::{Aabb, Bounded, Bvh8};
use crate::raytracer::ray::{HitData, Ray};
use crate::raytracer::shape::Shape;
use glam::{Vec2, Vec3};

pub struct TriangleMesh {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    indices: Vec<[u32; 3]>,
    material_id: u32,
    bvh: Bvh8,
    bounds: Aabb,
}

impl TriangleMesh {
    pub fn new(
        positions: Vec<Vec3>,
        normals: Vec<Vec3>,
        uvs: Vec<Vec2>,
        indices: Vec<[u32; 3]>,
        material_id: u32,
    ) -> Self {
        let bounds = Self::compute_bounds(&positions);

        const AABB_EPSILON: f32 = 1e-5;
        let triangle_bounds: Vec<_> = indices
            .iter()
            .enumerate()
            .map(|(i, tri)| {
                let v0 = positions[tri[0] as usize];
                let v1 = positions[tri[1] as usize];
                let v2 = positions[tri[2] as usize];
                let mut aabb = Aabb::from_point(v0);
                aabb.grow_point(v1);
                aabb.grow_point(v2);
                aabb.min -= Vec3::splat(AABB_EPSILON);
                aabb.max += Vec3::splat(AABB_EPSILON);
                (i as u32, aabb)
            })
            .collect();

        let bvh = Bvh8::build(triangle_bounds, 4);

        Self {
            positions,
            normals,
            uvs,
            indices,
            material_id,
            bvh,
            bounds,
        }
    }

    fn compute_bounds(positions: &[Vec3]) -> Aabb {
        positions.iter().fold(Aabb::EMPTY, |mut acc, &p| {
            acc.grow_point(p);
            acc
        })
    }

    #[inline]
    fn intersect_triangle(&self, tri_idx: usize, ray: &Ray) -> Option<(f32, Vec2)> {
        let tri = self.indices[tri_idx];
        let v0 = self.positions[tri[0] as usize];
        let v1 = self.positions[tri[1] as usize];
        let v2 = self.positions[tri[2] as usize];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let h = ray.direction.cross(edge2);
        let a = edge1.dot(h);

        if a.abs() < 1e-8 {
            return None;
        }

        let f = 1.0 / a;
        let s = ray.origin - v0;
        let u = f * s.dot(h);

        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let q = s.cross(edge1);
        let v = f * ray.direction.dot(q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * edge2.dot(q);
        if t >= ray.t_min && t <= ray.t_max {
            Some((t, Vec2::new(u, v)))
        } else {
            None
        }
    }

    fn compute_hit_data(&self, tri_idx: usize, t: f32, bary: Vec2) -> HitData {
        let tri = self.indices[tri_idx];
        let u = bary.x;
        let v = bary.y;
        let w = 1.0 - u - v;

        let v0 = self.positions[tri[0] as usize];
        let v1 = self.positions[tri[1] as usize];
        let v2 = self.positions[tri[2] as usize];
        let face_normal = (v1 - v0).cross(v2 - v0).normalize();

        let normal = if !self.normals.is_empty() {
            let n0 = self.normals[tri[0] as usize];
            let n1 = self.normals[tri[1] as usize];
            let n2 = self.normals[tri[2] as usize];
            (n0 * w + n1 * u + n2 * v).normalize()
        } else {
            face_normal
        };

        let geo_normal = if face_normal.dot(normal) >= 0.0 {
            face_normal
        } else {
            -face_normal
        };

        let uv = if !self.uvs.is_empty() {
            let uv0 = self.uvs[tri[0] as usize];
            let uv1 = self.uvs[tri[1] as usize];
            let uv2 = self.uvs[tri[2] as usize];
            uv0 * w + uv1 * u + uv2 * v
        } else {
            Vec2::new(u, v)
        };

        let (tangent, bitangent) = Self::compute_tangent_space(&normal);

        HitData {
            t,
            material_id: self.material_id,
            normal,
            geo_normal,
            tangent,
            bitangent,
            uv,
        }
    }

    fn compute_tangent_space(normal: &Vec3) -> (Vec3, Vec3) {
        let up = if normal.y.abs() < 0.999 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let tangent = up.cross(*normal).normalize();
        let bitangent = normal.cross(tangent);
        (tangent, bitangent)
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }
}

impl Clone for TriangleMesh {
    fn clone(&self) -> Self {
        Self::new(
            self.positions.clone(),
            self.normals.clone(),
            self.uvs.clone(),
            self.indices.clone(),
            self.material_id,
        )
    }
}

impl Bounded for TriangleMesh {
    fn bounds(&self) -> Aabb {
        self.bounds
    }

    fn center(&self) -> Vec3 {
        self.bounds.center()
    }
}

impl Shape for TriangleMesh {
    fn hit(&self, ray: &Ray) -> Option<HitData> {
        self.bvh
            .traverse_closest(ray, |tri_idx, ray| {
                self.intersect_triangle(tri_idx as usize, ray)
                    .map(|(t, bary)| (t, (tri_idx, t, bary)))
            })
            .map(|(tri_idx, t, bary)| self.compute_hit_data(tri_idx as usize, t, bary))
    }

    fn hit_any(&self, ray: &Ray) -> bool {
        self.bvh.traverse_any(ray, |tri_idx, ray| {
            self.intersect_triangle(tri_idx as usize, ray).is_some()
        })
    }
}
