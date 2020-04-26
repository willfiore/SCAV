use winit::event::{VirtualKeyCode, MouseButton};

#[derive(Default, Copy, Clone)]
pub struct InputState {
    pub is_down: bool,

    pub pressed: bool,
    pub released: bool,
}

pub struct Input {
    key_states: [InputState; 1024],
    mouse_states: [InputState; 32],
}

impl Input {

    pub fn new() -> Input {
        Input {
            key_states: [InputState::default(); 1024],
            mouse_states: [InputState::default(); 32],
        }
    }

    pub fn reset(&mut self) {
        for x in self.key_states.iter_mut() {
            x.pressed = false;
            x.released = false;
        }
    }

    pub fn is_key_down(&self, key: VirtualKeyCode) -> bool {
        self.key_states[key as usize].is_down
    }

    pub fn was_key_pressed(&self, key: VirtualKeyCode) -> bool {
        self.key_states[key as usize].pressed
    }

    pub fn was_key_released(&self, key: VirtualKeyCode) -> bool {
        self.key_states[key as usize].released
    }

    pub fn press_key(&mut self, key: VirtualKeyCode) {
        let state = &mut self.key_states[key as usize];
        state.pressed = true;
        state.is_down = true;
    }

    pub fn release_key(&mut self, key: VirtualKeyCode) {
        let state = &mut self.key_states[key as usize];
        state.released = true;
        state.is_down = false;
    }
}