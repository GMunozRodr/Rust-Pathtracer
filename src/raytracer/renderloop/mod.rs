mod parallel;

use crate::raytracer::camera::Camera;
use crate::raytracer::renderer::{Renderer, SceneAccess};
use glam::Vec3;

pub use parallel::{AdaptiveSamplingConfig, ParallelRenderLoop};

pub trait RenderLoop {
    fn render_pass<S, R>(
        &self,
        scene: &S,
        camera: &Camera,
        renderer: &R,
        width: usize,
        height: usize,
        sample_index: u32,
    ) -> Vec<Vec3>
    where
        S: SceneAccess + Sync,
        R: Renderer + Sync;
}
