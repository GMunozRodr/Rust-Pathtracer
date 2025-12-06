use crate::raytracer::camera::Camera;
use crate::raytracer::denoiser::{AovBuffers, OidnDenoiser};
use crate::raytracer::exporter::{PngExporter, ToneMap, WindowExporter};
use crate::raytracer::input::CameraInput;
use crate::raytracer::framebuffer::{AdaptiveFramebuffer, Framebuffer};
use crate::raytracer::light::DirectionalLight;
use crate::raytracer::loader::gltf::load_gltf;
use crate::raytracer::material::Material;
use crate::raytracer::renderloop::{AdaptiveSamplingConfig, ParallelRenderLoop, RenderLoop};
use crate::raytracer::renderer::{PathTracer, UnshadedRenderer};
use crate::raytracer::scene::Scene;
use crate::raytracer::shape::TriangleMesh;
use crate::raytracer::sky::TextureSky;
use crate::raytracer::texture::Texture;
use glam::Vec3;
use std::time::Instant;

pub struct RenderConfig {
    pub width: usize,
    pub height: usize,
    pub window_width: usize,
    pub window_height: usize,
    pub lowres_factor: usize,
    pub move_speed: f32,
    pub pitch_limit: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            window_width: 1920,
            window_height: 1080,
            lowres_factor: 4,
            move_speed: 30.0,
            pitch_limit: std::f32::consts::FRAC_PI_2 - 0.01,
        }
    }
}

pub struct RenderState {
    pub framebuffer: AdaptiveFramebuffer,
    pub framebuffer_lowres: Framebuffer,
    pub work_buffer: Vec<(usize, usize, u32)>,
    pub aov_buffers: Option<AovBuffers>,
    pub last_frame: Instant,
    pub render_start: Instant,
    pub is_moving: bool,
    pub adaptive_mode: bool,
}

impl RenderState {
    pub fn new(config: &RenderConfig) -> Self {
        let width = config.width;
        let height = config.height;
        Self {
            framebuffer: AdaptiveFramebuffer::new(width, height),
            framebuffer_lowres: Framebuffer::new(
                width / config.lowres_factor,
                height / config.lowres_factor,
            ),
            work_buffer: Vec::with_capacity(width * height),
            aov_buffers: None,
            last_frame: Instant::now(),
            render_start: Instant::now(),
            is_moving: false,
            adaptive_mode: true,
        }
    }

    pub fn on_camera_stopped(&mut self) {
        self.framebuffer.reset();
        self.aov_buffers = None;
        self.render_start = Instant::now();
    }
}

pub struct App {
    pub config: RenderConfig,
    pub adaptive_config: AdaptiveSamplingConfig,
    pub scene: Scene<TriangleMesh, TextureSky>,
    pub renderer: PathTracer<16>,
    pub preview_renderer: UnshadedRenderer,
    pub render_loop: ParallelRenderLoop,
    pub state: RenderState,
    pub window: WindowExporter,
}

impl App {
    pub fn new(gltf_path: &str) -> Result<Self, String> {
        let config = RenderConfig::default();

        println!("Loading: {}", gltf_path);
        let start = Instant::now();
        let gltf_scene = load_gltf(gltf_path).map_err(|e| format!("Failed to load glTF: {}", e))?;
        println!("Loaded in {:.2}s:", start.elapsed().as_secs_f32());
        println!("  {} meshes", gltf_scene.meshes.len());
        println!("  {} materials", gltf_scene.materials.len());
        println!("  {} textures", gltf_scene.textures.len());

        let aspect = config.width as f32 / config.height as f32;
        let mut camera = gltf_scene.camera.unwrap_or_else(|| {
            Camera::new(Vec3::new(0.0, 1.0, 3.0), Vec3::new(0.0, 0.5, 0.0), 60.0, aspect)
        });
        camera.aspect_ratio = aspect;
        camera.update_matrix();

        let mut materials = gltf_scene.materials;
        if materials.is_empty() {
            materials.push(Material::default());
        }

        let sky_texture = Texture::from_file("assets/textures/default.exr")
            .unwrap_or_else(|_| {
                println!("Warning: Could not load default skybox, using solid color");
                Texture::solid(Vec3::new(0.8, 0.9, 1.0))
            });
        let sky_rotation = 0.0;
        let sky = TextureSky::with_rotation(sky_texture, sky_rotation);

        let sun = DirectionalLight::new(
            Vec3::new(-0.5, -1.0, -0.3).normalize(),
            Vec3::new(1.0, 0.95, 0.85) * 8.0,
        );

        println!("Building scene...");
        let build_start = Instant::now();
        let scene = Scene::new(gltf_scene.meshes, materials, gltf_scene.textures, camera, sky, Some(sun));
        println!("Scene built in {:.2}s", build_start.elapsed().as_secs_f32());

        let adaptive_config = AdaptiveSamplingConfig {
            min_samples: 64,
            variance_threshold: 0.001,
            max_samples_per_pass: config.width * config.height,
        };

        let state = RenderState::new(&config);
        let window = WindowExporter::new(config.window_width, config.window_height);

        Ok(Self {
            config,
            adaptive_config,
            scene,
            renderer: PathTracer::<16>::new(),
            preview_renderer: UnshadedRenderer,
            render_loop: ParallelRenderLoop,
            state,
            window,
        })
    }

    pub fn update_camera(&mut self, input: &CameraInput, delta_time: f32) {
        self.scene.camera.yaw += input.yaw_delta;
        self.scene.camera.pitch = (self.scene.camera.pitch - input.pitch_delta)
            .clamp(-self.config.pitch_limit, self.config.pitch_limit);

        let forward = self.scene.camera.forward();
        let right = self.scene.camera.right();
        let mut move_delta =
            forward * input.move_forward + right * input.move_right + Vec3::Y * input.move_up;
        let len = move_delta.length();
        if len > 0.0 {
            move_delta /= len;
        }
        self.scene.camera.position += move_delta * self.config.move_speed * delta_time;
        self.scene.camera.update_matrix();
        self.state.framebuffer_lowres.reset();
    }

    pub fn render_lowres_frame(&mut self) {
        let lowres_width = self.config.width / self.config.lowres_factor;
        let lowres_height = self.config.height / self.config.lowres_factor;
        let pixels = self.render_loop.render_pass(
            &self.scene,
            &self.scene.camera,
            &self.preview_renderer,
            lowres_width,
            lowres_height,
            0,
        );
        self.state.framebuffer_lowres.reset();
        self.state.framebuffer_lowres.accumulate(&pixels);
        self.window.update(&self.state.framebuffer_lowres);
        self.window.set_title("Raytracer - MOVING");
    }

    pub fn render_fullres_frame(&mut self) {
        let min_samples = self.state.framebuffer.min_sample_count();
        let width = self.config.width;
        let height = self.config.height;

        if self.state.adaptive_mode && min_samples >= self.adaptive_config.min_samples {
            let samples_rendered = self.render_loop.render_adaptive_pass(
                &self.scene,
                &self.scene.camera,
                &self.renderer,
                &mut self.state.framebuffer,
                &self.adaptive_config,
                &mut self.state.work_buffer,
            );
            self.window.update(&self.state.framebuffer);
            self.update_adaptive_title(samples_rendered);
        } else {
            if min_samples == 0 && self.state.aov_buffers.is_none() {
                let (pixels, aovs) = self.render_loop.render_first_pass_with_aov(
                    &self.scene,
                    &self.scene.camera,
                    &self.renderer,
                    width,
                    height,
                );
                self.state.framebuffer.accumulate(&pixels);
                self.state.aov_buffers = Some(aovs);
            } else {
                let pixels = self.render_loop.render_pass(
                    &self.scene,
                    &self.scene.camera,
                    &self.renderer,
                    width,
                    height,
                    min_samples,
                );
                self.state.framebuffer.accumulate(&pixels);
            }
            self.window.update(&self.state.framebuffer);
            self.update_standard_title();
        }
    }

    fn update_adaptive_title(&mut self, samples_rendered: usize) {
        let total = self.state.framebuffer.total_samples();
        let pixels = (self.config.width * self.config.height) as u64;
        let avg_samples = total as f64 / pixels as f64;
        let elapsed = self.state.render_start.elapsed().as_secs_f32();
        self.window.set_title(&format!(
            "Raytracer - {:.1}s - ADAPTIVE {:.1} avg samples ({} this pass, min {} max {})",
            elapsed,
            avg_samples,
            samples_rendered,
            self.state.framebuffer.min_sample_count(),
            self.state.framebuffer.max_sample_count()
        ));
    }

    fn update_standard_title(&mut self) {
        let elapsed = self.state.render_start.elapsed().as_secs_f32();
        self.window.set_title(&format!(
            "Raytracer - {:.1}s - {} samples",
            elapsed,
            self.state.framebuffer.min_sample_count()
        ));
    }

    pub fn export_results(&self) {
        let min_samples = self.state.framebuffer.min_sample_count();
        if min_samples == 0 {
            return;
        }

        let total = self.state.framebuffer.total_samples();
        let pixels = (self.config.width * self.config.height) as u64;
        let avg_samples = total as f64 / pixels as f64;

        PngExporter::with_tonemap(ToneMap::Agx)
            .with_exposure(self.window.exposure())
            .export_adaptive(&self.state.framebuffer, "output.png");
        println!(
            "Saved output.png with {:.1} avg samples (min {}, max {}, exposure: {:.2})",
            avg_samples,
            min_samples,
            self.state.framebuffer.max_sample_count(),
            self.window.exposure()
        );

        let aovs_for_denoise = if self.scene.camera.has_dof() {
            None
        } else {
            self.state.aov_buffers.as_ref()
        };
        PngExporter::with_tonemap(ToneMap::Agx)
            .with_exposure(self.window.exposure())
            .with_denoiser(OidnDenoiser::new())
            .export_adaptive_with_aov(&self.state.framebuffer, aovs_for_denoise, "output_denoised.png");
        println!(
            "Saved output_denoised.png (OIDN denoised{})",
            if aovs_for_denoise.is_some() {
                " with AOVs"
            } else if self.scene.camera.has_dof() {
                ", AOVs disabled due to DoF"
            } else {
                ""
            }
        );
    }
}
