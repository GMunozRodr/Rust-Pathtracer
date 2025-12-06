mod png;
mod tonemapping;
mod window;

pub use png::PngExporter;
pub use tonemapping::{linear_to_srgb_u8, ToneMap};
pub use window::WindowExporter;

use super::framebuffer::Framebuffer;

pub trait Exporter {
    fn export(&self, framebuffer: &Framebuffer, path: &str);
}
