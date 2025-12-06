use minifb::{Key, MouseButton, MouseMode, Window};

#[derive(Default)]
pub struct CameraInput {
    pub move_forward: f32,
    pub move_right: f32,
    pub move_up: f32,
    pub yaw_delta: f32,
    pub pitch_delta: f32,
    pub right_click_held: bool,
    pub scroll_delta: f32,
}

impl CameraInput {
    pub fn has_movement(&self) -> bool {
        self.right_click_held
    }
}

pub struct InputHandler {
    last_mouse: Option<(f32, f32)>,
    mouse_captured: bool,
    a_key_was_down: bool,
}

impl InputHandler {
    pub fn new() -> Self {
        Self {
            last_mouse: None,
            mouse_captured: false,
            a_key_was_down: false,
        }
    }

    pub fn get_camera_input(&mut self, window: &Window) -> CameraInput {
        let mut input = CameraInput::default();

        if window.is_key_down(Key::W) {
            input.move_forward += 1.0;
        }
        if window.is_key_down(Key::S) {
            input.move_forward -= 1.0;
        }
        if window.is_key_down(Key::D) {
            input.move_right += 1.0;
        }
        if window.is_key_down(Key::A) {
            input.move_right -= 1.0;
        }
        if window.is_key_down(Key::Space) {
            input.move_up += 1.0;
        }
        if window.is_key_down(Key::LeftShift) {
            input.move_up -= 1.0;
        }

        if window.get_mouse_down(MouseButton::Right) {
            input.right_click_held = true;
            if !self.mouse_captured {
                self.mouse_captured = true;
                self.last_mouse = window.get_mouse_pos(MouseMode::Pass);
            }

            if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Pass) {
                if let Some((last_x, last_y)) = self.last_mouse {
                    let sensitivity = 0.003;
                    input.yaw_delta = (mx - last_x) * sensitivity;
                    input.pitch_delta = (my - last_y) * sensitivity;
                }
                self.last_mouse = Some((mx, my));
            }
        } else {
            self.mouse_captured = false;
            self.last_mouse = None;
        }

        if let Some((_scroll_x, scroll_y)) = window.get_scroll_wheel() {
            input.scroll_delta = scroll_y;
        }

        input
    }

    pub fn get_camera_input_with_keys(&mut self, window: &Window) -> (CameraInput, bool) {
        let input = self.get_camera_input(window);

        let a_down = window.is_key_down(Key::A);
        let a_pressed = self.a_key_was_down && !a_down && !input.right_click_held;
        self.a_key_was_down = a_down;

        (input, a_pressed)
    }
}

impl Default for InputHandler {
    fn default() -> Self {
        Self::new()
    }
}
