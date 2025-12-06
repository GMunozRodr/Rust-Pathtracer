use super::{linear_to_srgb_u8, ToneMap};
use crate::raytracer::input::{CameraInput, InputHandler};
use crate::raytracer::framebuffer::FramebufferView;
use glam::Vec3;
use minifb::{Key, Window, WindowOptions};

pub struct WindowExporter {
    window: Window,
    buffer: Vec<u32>,
    buffer_width: usize,
    buffer_height: usize,
    input_handler: InputHandler,
    tonemap: ToneMap,
    t_key_was_down: bool,
    exposure: f32,
}

impl WindowExporter {
    pub fn new(width: usize, height: usize) -> Self {
        let window = Window::new(
            "Raytracer",
            width,
            height,
            WindowOptions {
                resize: true,
                ..WindowOptions::default()
            },
        )
        .expect("Failed to create window");

        Self {
            window,
            buffer: Vec::new(),
            buffer_width: 0,
            buffer_height: 0,
            input_handler: InputHandler::new(),
            tonemap: ToneMap::Aces,
            t_key_was_down: false,
            exposure: 1.5,
        }
    }

    fn ensure_buffer_size(&mut self, width: usize, height: usize) {
        if self.buffer_width != width || self.buffer_height != height {
            self.buffer.resize(width * height, 0);
            self.buffer_width = width;
            self.buffer_height = height;
        }
    }

    pub fn get_camera_input(&mut self) -> CameraInput {
        self.input_handler.get_camera_input(&self.window)
    }

    pub fn get_camera_input_with_keys(&mut self) -> (CameraInput, bool) {
        self.input_handler.get_camera_input_with_keys(&self.window)
    }

    fn vec3_to_u32(&self, color: Vec3) -> u32 {
        let mapped = self.tonemap.apply_with_exposure(color, self.exposure);
        let [r, g, b] = linear_to_srgb_u8(mapped);
        ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
    }

    pub fn is_open(&self) -> bool {
        self.window.is_open() && !self.window.is_key_down(Key::Escape)
    }

    fn handle_tonemap_toggle(&mut self) {
        let t_down = self.window.is_key_down(Key::T);
        if self.t_key_was_down && !t_down {
            self.tonemap = match self.tonemap {
                ToneMap::Agx => ToneMap::Aces,
                ToneMap::Aces => ToneMap::Reinhard,
                ToneMap::Reinhard => ToneMap::None,
                ToneMap::None => ToneMap::Agx,
            };
            println!(
                "Tonemapping: {}",
                match self.tonemap {
                    ToneMap::None => "OFF",
                    ToneMap::Aces => "ACES",
                    ToneMap::Reinhard => "Reinhard",
                    ToneMap::Agx => "AGX",
                }
            );
        }
        self.t_key_was_down = t_down;
    }

    fn handle_exposure(&mut self) {
        if self.window.is_key_down(Key::LeftBracket) {
            self.exposure *= 0.98;
        }
        if self.window.is_key_down(Key::RightBracket) {
            self.exposure *= 1.02;
        }
        self.exposure = self.exposure.clamp(0.1, 10.0);
    }

    pub fn update<F: FramebufferView>(&mut self, framebuffer: &F) {
        self.handle_tonemap_toggle();
        self.handle_exposure();

        let width = framebuffer.width();
        let height = framebuffer.height();
        self.ensure_buffer_size(width, height);

        for y in 0..height {
            for x in 0..width {
                let color = framebuffer.get_pixel(x, y);
                let idx = y * width + x;
                self.buffer[idx] = self.vec3_to_u32(color);
            }
        }

        self.window
            .update_with_buffer(&self.buffer, width, height)
            .expect("Failed to update window");
    }

    pub fn tonemap(&self) -> ToneMap {
        self.tonemap
    }

    pub fn exposure(&self) -> f32 {
        self.exposure
    }

    pub fn set_title(&mut self, title: &str) {
        self.window.set_title(title);
    }
}
