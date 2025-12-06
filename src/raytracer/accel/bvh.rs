use crate::raytracer::ray::Ray;
use glam::Vec3;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(not(target_arch = "x86_64"))]
use wide::{f32x8, CmpLe};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for Aabb {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn from_point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }

    #[inline]
    pub fn grow(&mut self, other: &Aabb) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    #[inline]
    pub fn grow_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    #[inline]
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    #[inline]
    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }

    #[inline]
    pub fn intersect(&self, ray: &Ray, t_max: f32) -> Option<f32> {
        let t1 = (self.min - ray.origin) * ray.inv_direction;
        let t2 = (self.max - ray.origin) * ray.inv_direction;

        let t_near = t1.min(t2);
        let t_far = t1.max(t2);

        let t_enter = ray.t_min.max(t_near.x).max(t_near.y).max(t_near.z);
        let t_exit = t_max.min(t_far.x).min(t_far.y).min(t_far.z);

        if t_enter <= t_exit {
            Some(t_enter)
        } else {
            None
        }
    }
}

#[derive(Clone)]
#[repr(C, align(64))]
pub struct Bvh8Node {
    pub min_x: [f32; 8],
    pub max_x: [f32; 8],
    pub min_y: [f32; 8],
    pub max_y: [f32; 8],
    pub min_z: [f32; 8],
    pub max_z: [f32; 8],
    pub children: [u32; 8],
    pub valid_mask: u8,
    pub _padding: [u8; 7],
}

impl Default for Bvh8Node {
    fn default() -> Self {
        Self {
            min_x: [f32::INFINITY; 8],
            max_x: [f32::NEG_INFINITY; 8],
            min_y: [f32::INFINITY; 8],
            max_y: [f32::NEG_INFINITY; 8],
            min_z: [f32::INFINITY; 8],
            max_z: [f32::NEG_INFINITY; 8],
            children: [0; 8],
            valid_mask: 0,
            _padding: [0; 7],
        }
    }
}

impl Bvh8Node {
    const LEAF_FLAG: u32 = 1 << 31;
    const PRIM_START_MASK: u32 = 0x00FFFFFF;
    const PRIM_COUNT_SHIFT: u32 = 24;
    const PRIM_COUNT_MASK: u32 = 0x7F;

    #[inline(always)]
    pub fn set_child_aabb(&mut self, slot: usize, aabb: &Aabb) {
        self.min_x[slot] = aabb.min.x;
        self.max_x[slot] = aabb.max.x;
        self.min_y[slot] = aabb.min.y;
        self.max_y[slot] = aabb.max.y;
        self.min_z[slot] = aabb.min.z;
        self.max_z[slot] = aabb.max.z;
    }

    #[inline(always)]
    pub fn set_child_internal(&mut self, slot: usize, node_idx: u32) {
        self.children[slot] = node_idx;
        self.valid_mask |= 1 << slot;
    }

    #[inline(always)]
    pub fn set_child_leaf(&mut self, slot: usize, prim_start: u32, prim_count: u32) {
        debug_assert!(prim_count > 0 && prim_count <= 128);
        debug_assert!(prim_start <= Self::PRIM_START_MASK);
        self.children[slot] = Self::LEAF_FLAG
            | ((prim_count - 1) << Self::PRIM_COUNT_SHIFT)
            | prim_start;
        self.valid_mask |= 1 << slot;
    }

    #[inline(always)]
    pub fn is_child_leaf(&self, slot: usize) -> bool {
        (self.children[slot] & Self::LEAF_FLAG) != 0
    }

    #[inline(always)]
    pub fn child_node_index(&self, slot: usize) -> u32 {
        debug_assert!(!self.is_child_leaf(slot));
        self.children[slot]
    }

    #[inline(always)]
    pub fn child_prim_start(&self, slot: usize) -> u32 {
        debug_assert!(self.is_child_leaf(slot));
        self.children[slot] & Self::PRIM_START_MASK
    }

    #[inline(always)]
    pub fn child_prim_count(&self, slot: usize) -> u32 {
        debug_assert!(self.is_child_leaf(slot));
        ((self.children[slot] >> Self::PRIM_COUNT_SHIFT) & Self::PRIM_COUNT_MASK) + 1
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn intersect(&self, ray: &Ray, t_max: f32) -> (u8, [f32; 8]) {
        unsafe { self.intersect_avx2(ray, t_max) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn intersect_avx2(&self, ray: &Ray, t_max: f32) -> (u8, [f32; 8]) {
        unsafe {
            let origin_x = _mm256_set1_ps(ray.origin.x);
            let origin_y = _mm256_set1_ps(ray.origin.y);
            let origin_z = _mm256_set1_ps(ray.origin.z);
            let inv_dir_x = _mm256_set1_ps(ray.inv_direction.x);
            let inv_dir_y = _mm256_set1_ps(ray.inv_direction.y);
            let inv_dir_z = _mm256_set1_ps(ray.inv_direction.z);
            let t_min_v = _mm256_set1_ps(ray.t_min);
            let t_max_v = _mm256_set1_ps(t_max);

            let min_x = _mm256_loadu_ps(self.min_x.as_ptr());
            let max_x = _mm256_loadu_ps(self.max_x.as_ptr());
            let min_y = _mm256_loadu_ps(self.min_y.as_ptr());
            let max_y = _mm256_loadu_ps(self.max_y.as_ptr());
            let min_z = _mm256_loadu_ps(self.min_z.as_ptr());
            let max_z = _mm256_loadu_ps(self.max_z.as_ptr());

            let t1_x = _mm256_mul_ps(_mm256_sub_ps(min_x, origin_x), inv_dir_x);
            let t2_x = _mm256_mul_ps(_mm256_sub_ps(max_x, origin_x), inv_dir_x);
            let t1_y = _mm256_mul_ps(_mm256_sub_ps(min_y, origin_y), inv_dir_y);
            let t2_y = _mm256_mul_ps(_mm256_sub_ps(max_y, origin_y), inv_dir_y);
            let t1_z = _mm256_mul_ps(_mm256_sub_ps(min_z, origin_z), inv_dir_z);
            let t2_z = _mm256_mul_ps(_mm256_sub_ps(max_z, origin_z), inv_dir_z);

            let t_near_x = _mm256_min_ps(t1_x, t2_x);
            let t_far_x = _mm256_max_ps(t1_x, t2_x);
            let t_near_y = _mm256_min_ps(t1_y, t2_y);
            let t_far_y = _mm256_max_ps(t1_y, t2_y);
            let t_near_z = _mm256_min_ps(t1_z, t2_z);
            let t_far_z = _mm256_max_ps(t1_z, t2_z);

            let t_enter = _mm256_max_ps(t_min_v, _mm256_max_ps(t_near_x, _mm256_max_ps(t_near_y, t_near_z)));
            let t_exit = _mm256_min_ps(t_max_v, _mm256_min_ps(t_far_x, _mm256_min_ps(t_far_y, t_far_z)));

            let hit_cmp = _mm256_cmp_ps(t_enter, t_exit, _CMP_LE_OQ);
            let hit_mask = _mm256_movemask_ps(hit_cmp) as u8;

            let mut distances = [0.0f32; 8];
            _mm256_storeu_ps(distances.as_mut_ptr(), t_enter);

            (hit_mask, distances)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[inline(always)]
    pub fn intersect(&self, ray: &Ray, t_max: f32) -> (u8, [f32; 8]) {
        let origin_x = f32x8::splat(ray.origin.x);
        let origin_y = f32x8::splat(ray.origin.y);
        let origin_z = f32x8::splat(ray.origin.z);
        let inv_dir_x = f32x8::splat(ray.inv_direction.x);
        let inv_dir_y = f32x8::splat(ray.inv_direction.y);
        let inv_dir_z = f32x8::splat(ray.inv_direction.z);
        let t_min_v = f32x8::splat(ray.t_min);
        let t_max_v = f32x8::splat(t_max);

        let min_x = f32x8::from(self.min_x);
        let max_x = f32x8::from(self.max_x);
        let min_y = f32x8::from(self.min_y);
        let max_y = f32x8::from(self.max_y);
        let min_z = f32x8::from(self.min_z);
        let max_z = f32x8::from(self.max_z);

        let t1_x = (min_x - origin_x) * inv_dir_x;
        let t2_x = (max_x - origin_x) * inv_dir_x;
        let t1_y = (min_y - origin_y) * inv_dir_y;
        let t2_y = (max_y - origin_y) * inv_dir_y;
        let t1_z = (min_z - origin_z) * inv_dir_z;
        let t2_z = (max_z - origin_z) * inv_dir_z;

        let t_near_x = t1_x.min(t2_x);
        let t_far_x = t1_x.max(t2_x);
        let t_near_y = t1_y.min(t2_y);
        let t_far_y = t1_y.max(t2_y);
        let t_near_z = t1_z.min(t2_z);
        let t_far_z = t1_z.max(t2_z);

        let t_enter = t_min_v.max(t_near_x).max(t_near_y).max(t_near_z);
        let t_exit = t_max_v.min(t_far_x).min(t_far_y).min(t_far_z);

        let hit_mask = t_enter.cmp_le(t_exit);
        (hit_mask.move_mask() as u8, t_enter.to_array())
    }
}

pub struct Bvh8 {
    pub nodes: Vec<Bvh8Node>,
    pub primitive_indices: Vec<u32>,
}

impl Bvh8 {
    pub fn empty() -> Self {
        Self {
            nodes: vec![],
            primitive_indices: vec![],
        }
    }

    pub fn build(primitive_bounds: Vec<(u32, Aabb)>, max_prims_per_leaf: u32) -> Self {
        if primitive_bounds.is_empty() {
            return Self::empty();
        }

        let bvh2 = Bvh2::build(primitive_bounds.into_iter(), max_prims_per_leaf);

        Self::from_bvh2(&bvh2)
    }

    pub fn from_bvh2(bvh2: &Bvh2) -> Self {
        if bvh2.nodes.is_empty() || bvh2.primitive_indices.is_empty() {
            return Self::empty();
        }

        let mut builder = Bvh8Builder {
            bvh2,
            nodes: Vec::with_capacity(bvh2.nodes.len() / 4),
            primitive_indices: Vec::with_capacity(bvh2.primitive_indices.len()),
        };

        builder.build_recursive(0);

        Self {
            nodes: builder.nodes,
            primitive_indices: builder.primitive_indices,
        }
    }

    #[inline(always)]
    pub fn traverse_closest<T, F>(&self, ray: &Ray, mut test_primitive: F) -> Option<T>
    where
        F: FnMut(u32, &Ray) -> Option<(f32, T)>,
    {
        if self.nodes.is_empty() {
            return None;
        }

        let mut closest: Option<T> = None;
        let mut closest_t = ray.t_max;

        let mut stack: [(u32, f32); 64] = [(0, 0.0); 64];
        let mut stack_ptr = 1usize;
        stack[0] = (0, 0.0);

        while stack_ptr > 0 {
            stack_ptr -= 1;

            if stack[stack_ptr].1 >= closest_t {
                continue;
            }

            let node = &self.nodes[stack[stack_ptr].0 as usize];
            let (hit_mask, distances) = node.intersect(ray, closest_t);
            let active = hit_mask & node.valid_mask;

            if active == 0 {
                continue;
            }

            let mut mask = active;
            while mask != 0 {
                let slot = mask.trailing_zeros() as usize;
                mask &= mask - 1;

                let dist = distances[slot];
                if dist >= closest_t {
                    continue;
                }

                if node.is_child_leaf(slot) {
                    let prim_start = node.child_prim_start(slot) as usize;
                    let prim_count = node.child_prim_count(slot) as usize;
                    for &prim_idx in &self.primitive_indices[prim_start..prim_start + prim_count] {
                        if let Some((t, data)) = test_primitive(prim_idx, ray) {
                            if t < closest_t {
                                closest_t = t;
                                closest = Some(data);
                            }
                        }
                    }
                } else {
                    stack[stack_ptr] = (node.child_node_index(slot), dist);
                    stack_ptr += 1;
                }
            }
        }

        closest
    }

    #[inline(always)]
    pub fn traverse_any<F>(&self, ray: &Ray, mut test_primitive: F) -> bool
    where
        F: FnMut(u32, &Ray) -> bool,
    {
        if self.nodes.is_empty() {
            return false;
        }

        let mut stack = [0u32; 64];
        let mut stack_ptr = 1;
        stack[0] = 0;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node = &self.nodes[stack[stack_ptr] as usize];

            let (hit_mask, _) = node.intersect(ray, ray.t_max);
            let active = hit_mask & node.valid_mask;

            if active == 0 {
                continue;
            }

            let mut mask = active;
            while mask != 0 {
                let slot = mask.trailing_zeros() as usize;
                mask &= mask - 1;

                if node.is_child_leaf(slot) {
                    let prim_start = node.child_prim_start(slot) as usize;
                    let prim_count = node.child_prim_count(slot) as usize;
                    for &prim_idx in &self.primitive_indices[prim_start..prim_start + prim_count] {
                        if test_primitive(prim_idx, ray) {
                            return true;
                        }
                    }
                } else {
                    stack[stack_ptr] = node.child_node_index(slot);
                    stack_ptr += 1;
                }
            }
        }

        false
    }
}

const SAH_BINS: usize = 16;
const TRAVERSAL_COST: f32 = 1.0;
const INTERSECTION_COST: f32 = 1.0;

#[derive(Clone, Debug)]
pub(crate) struct Bvh2Node {
    pub aabb: Aabb,
    right_or_prim: u32,
    count: u32,
}

impl Bvh2Node {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.count > 0
    }

    #[inline]
    pub fn left_child(&self, self_idx: u32) -> u32 {
        debug_assert!(!self.is_leaf());
        self_idx + 1
    }

    #[inline]
    pub fn right_child(&self) -> u32 {
        debug_assert!(!self.is_leaf());
        self.right_or_prim
    }

    #[inline]
    pub fn first_primitive(&self) -> u32 {
        debug_assert!(self.is_leaf());
        self.right_or_prim
    }

    #[inline]
    pub fn primitive_count(&self) -> u32 {
        debug_assert!(self.is_leaf());
        self.count
    }
}

pub(crate) struct Bvh2 {
    pub nodes: Vec<Bvh2Node>,
    pub primitive_indices: Vec<u32>,
}

impl Bvh2 {
    pub fn build<I>(primitive_bounds: I, max_prims_per_leaf: u32) -> Self
    where
        I: IntoIterator<Item = (u32, Aabb)>,
    {
        let mut primitives: Vec<_> = primitive_bounds
            .into_iter()
            .map(|(index, bounds)| PrimitiveInfo {
                index,
                centroid: bounds.center(),
                bounds,
            })
            .collect();

        if primitives.is_empty() {
            return Self {
                nodes: vec![Bvh2Node {
                    aabb: Aabb::EMPTY,
                    right_or_prim: 0,
                    count: 0,
                }],
                primitive_indices: vec![],
            };
        }

        let count = primitives.len();
        let mut nodes = Vec::with_capacity(2 * count);
        let mut ordered_prims = Vec::with_capacity(count);

        build_bvh2_recursive(
            &mut primitives,
            0,
            count,
            &mut nodes,
            &mut ordered_prims,
            max_prims_per_leaf,
        );

        Self {
            nodes,
            primitive_indices: ordered_prims,
        }
    }
}

struct PrimitiveInfo {
    index: u32,
    centroid: Vec3,
    bounds: Aabb,
}

#[derive(Clone, Copy, Default)]
struct Bin {
    bounds: Aabb,
    count: u32,
}

fn build_bvh2_recursive(
    primitives: &mut [PrimitiveInfo],
    start: usize,
    end: usize,
    nodes: &mut Vec<Bvh2Node>,
    ordered_prims: &mut Vec<u32>,
    max_prims_per_leaf: u32,
) -> u32 {
    let node_idx = nodes.len() as u32;
    let count = end - start;
    let prims = &primitives[start..end];

    let bounds = prims.iter().fold(Aabb::EMPTY, |mut acc, p| {
        acc.grow(&p.bounds);
        acc
    });

    if count as u32 <= max_prims_per_leaf {
        return create_leaf(nodes, ordered_prims, prims, bounds);
    }

    let centroid_bounds = prims.iter().fold(Aabb::EMPTY, |mut acc, p| {
        acc.grow_point(p.centroid);
        acc
    });

    let extent = centroid_bounds.extent();
    if extent.x.max(extent.y).max(extent.z) < 1e-7 {
        return create_leaf(nodes, ordered_prims, prims, bounds);
    }

    let (axis, split_pos, split_cost) = find_best_split(prims, &centroid_bounds, &bounds);

    let leaf_cost = INTERSECTION_COST * count as f32;
    if split_cost >= leaf_cost {
        return create_leaf(nodes, ordered_prims, prims, bounds);
    }

    let mid = partition(&mut primitives[start..end], axis, split_pos);
    let mid = start + mid;

    let mid = if mid == start || mid == end {
        primitives[start..end].sort_by(|a, b| {
            a.centroid[axis]
                .partial_cmp(&b.centroid[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        start + count / 2
    } else {
        mid
    };

    nodes.push(Bvh2Node {
        aabb: bounds,
        right_or_prim: 0,
        count: 0,
    });

    build_bvh2_recursive(primitives, start, mid, nodes, ordered_prims, max_prims_per_leaf);

    let right_idx = nodes.len() as u32;
    nodes[node_idx as usize].right_or_prim = right_idx;
    build_bvh2_recursive(primitives, mid, end, nodes, ordered_prims, max_prims_per_leaf);

    node_idx
}

fn create_leaf(
    nodes: &mut Vec<Bvh2Node>,
    ordered_prims: &mut Vec<u32>,
    prims: &[PrimitiveInfo],
    bounds: Aabb,
) -> u32 {
    let node_idx = nodes.len() as u32;
    let first_prim = ordered_prims.len() as u32;

    ordered_prims.extend(prims.iter().map(|p| p.index));

    nodes.push(Bvh2Node {
        aabb: bounds,
        right_or_prim: first_prim,
        count: prims.len() as u32,
    });

    node_idx
}

fn find_best_split(
    primitives: &[PrimitiveInfo],
    centroid_bounds: &Aabb,
    node_bounds: &Aabb,
) -> (usize, f32, f32) {
    let node_area = node_bounds.surface_area();
    if node_area <= 0.0 {
        return (0, 0.0, f32::INFINITY);
    }

    let extent = centroid_bounds.extent();
    let mut best = (0, 0.0f32, f32::INFINITY);

    for axis in 0..3 {
        if extent[axis] < 1e-7 {
            continue;
        }

        let scale = SAH_BINS as f32 / extent[axis];
        let base = centroid_bounds.min[axis];
        let mut bins = [Bin::default(); SAH_BINS];

        for prim in primitives {
            let bin = ((prim.centroid[axis] - base) * scale)
                .clamp(0.0, (SAH_BINS - 1) as f32) as usize;
            bins[bin].bounds.grow(&prim.bounds);
            bins[bin].count += 1;
        }

        let mut left_bounds = [Aabb::EMPTY; SAH_BINS - 1];
        let mut left_counts = [0u32; SAH_BINS - 1];
        let mut running = (Aabb::EMPTY, 0u32);

        for i in 0..(SAH_BINS - 1) {
            running.0.grow(&bins[i].bounds);
            running.1 += bins[i].count;
            left_bounds[i] = running.0;
            left_counts[i] = running.1;
        }

        running = (Aabb::EMPTY, 0);
        for i in (0..(SAH_BINS - 1)).rev() {
            running.0.grow(&bins[i + 1].bounds);
            running.1 += bins[i + 1].count;

            if left_counts[i] == 0 || running.1 == 0 {
                continue;
            }

            let cost = TRAVERSAL_COST
                + INTERSECTION_COST
                    * (left_counts[i] as f32 * left_bounds[i].surface_area()
                        + running.1 as f32 * running.0.surface_area())
                    / node_area;

            if cost < best.2 {
                best = (
                    axis,
                    centroid_bounds.min[axis] + (i + 1) as f32 * extent[axis] / SAH_BINS as f32,
                    cost,
                );
            }
        }
    }

    best
}

fn partition(primitives: &mut [PrimitiveInfo], axis: usize, split_pos: f32) -> usize {
    let mut left = 0;
    let mut right = primitives.len();

    while left < right {
        if primitives[left].centroid[axis] < split_pos {
            left += 1;
        } else {
            right -= 1;
            primitives.swap(left, right);
        }
    }

    left
}

struct Bvh8Builder<'a> {
    bvh2: &'a Bvh2,
    nodes: Vec<Bvh8Node>,
    primitive_indices: Vec<u32>,
}

impl<'a> Bvh8Builder<'a> {
    fn build_recursive(&mut self, bvh2_root: usize) -> u32 {
        let bvh2_node = &self.bvh2.nodes[bvh2_root];

        let mut children: Vec<usize> = Vec::with_capacity(8);
        self.collect_children(bvh2_root, &mut children, 8);

        if children.is_empty() {
            if bvh2_node.is_leaf() {
                let node_idx = self.nodes.len() as u32;
                let mut node = Bvh8Node::default();

                let prim_start = self.primitive_indices.len() as u32;
                let bvh2_first = bvh2_node.first_primitive() as usize;
                let count = bvh2_node.primitive_count() as usize;
                self.primitive_indices.extend_from_slice(
                    &self.bvh2.primitive_indices[bvh2_first..bvh2_first + count]
                );

                node.set_child_aabb(0, &bvh2_node.aabb);
                node.set_child_leaf(0, prim_start, count as u32);

                self.nodes.push(node);
                return node_idx;
            }
            panic!("Empty children for non-leaf node");
        }

        let node_idx = self.nodes.len();
        self.nodes.push(Bvh8Node::default());

        let mut child_info: Vec<(usize, bool, Aabb)> = Vec::with_capacity(children.len());
        for &bvh2_idx in &children {
            let child_node = &self.bvh2.nodes[bvh2_idx];
            child_info.push((bvh2_idx, child_node.is_leaf(), child_node.aabb));
        }

        for (slot, (bvh2_idx, is_leaf, aabb)) in child_info.into_iter().enumerate() {
            self.nodes[node_idx].set_child_aabb(slot, &aabb);

            if is_leaf {
                let child_node = &self.bvh2.nodes[bvh2_idx];
                let prim_start = self.primitive_indices.len() as u32;
                let bvh2_first = child_node.first_primitive() as usize;
                let count = child_node.primitive_count() as usize;
                self.primitive_indices.extend_from_slice(
                    &self.bvh2.primitive_indices[bvh2_first..bvh2_first + count]
                );
                self.nodes[node_idx].set_child_leaf(slot, prim_start, count as u32);
            } else {
                let child_idx = self.build_recursive(bvh2_idx);
                self.nodes[node_idx].set_child_internal(slot, child_idx);
            }
        }

        node_idx as u32
    }

    fn collect_children(&self, bvh2_root: usize, children: &mut Vec<usize>, max_children: usize) {
        let root_node = &self.bvh2.nodes[bvh2_root];

        if root_node.is_leaf() {
            return;
        }

        let mut frontier: Vec<usize> = vec![
            root_node.left_child(bvh2_root as u32) as usize,
            root_node.right_child() as usize,
        ];

        while !frontier.is_empty() && children.len() + frontier.len() <= max_children {
            children.extend(frontier.iter().cloned());

            if children.len() >= max_children {
                break;
            }

            let mut best_idx = None;
            let mut best_area = 0.0f32;

            for (i, &child_idx) in children.iter().enumerate() {
                let child_node = &self.bvh2.nodes[child_idx];
                if !child_node.is_leaf() {
                    let area = child_node.aabb.surface_area();
                    if area > best_area {
                        best_area = area;
                        best_idx = Some(i);
                    }
                }
            }

            if let Some(idx) = best_idx {
                if children.len() < max_children {
                    let child_idx = children[idx];
                    let child_node = &self.bvh2.nodes[child_idx];

                    let left = child_node.left_child(child_idx as u32) as usize;
                    let right = child_node.right_child() as usize;

                    children.remove(idx);
                    children.push(left);
                    if children.len() < max_children {
                        children.push(right);
                    } else {
                        frontier = vec![right];
                        continue;
                    }
                }
            } else {
                break;
            }

            frontier = vec![];
        }
    }
}
