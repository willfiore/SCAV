#![windows_subsystem = "windows"]

extern crate nalgebra as na;

mod input;
mod physics;
mod renderer;
mod static_resources;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use noise::{NoiseFn, *};

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use crate::input::Input;
use crate::renderer::{Camera, Renderer};
use crate::static_resources::model_cube;
use crate::physics::PhysicsWorld;
use na::{Matrix4, Point3, Rotation3};
use nalgebra::Vector3;
use std::f32::consts::PI;
use std::time::{Duration, Instant};
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, ElementState, VirtualKeyCode};
use winit::window::Fullscreen;

const LOGIC_TICK_DURATION: Duration = Duration::from_millis(20);
const PLAYER_SPEED: f32 = 5.0;
const CAMERA_MOUSE_SENSITIVITY: f32 = 0.001;

struct GameState {
    player_position: Point3<f32>,
    previous_player_position: Point3<f32>,
    physics_world: PhysicsWorld,
}

impl GameState {
    fn new() -> Self {
        Self {
            player_position: Point3::new(0.0, 0.0, 0.0),
            previous_player_position: Point3::new(0.0, 0.0, 0.0),
            physics_world: PhysicsWorld::new()
        }
    }
}

fn fixed_update(state: &mut GameState, input: &Input, camera: &Camera, tick: u64) {
    let mut movement_direction = Vector3::<f32>::zeros();

    if input.is_key_down(VirtualKeyCode::W) {
        movement_direction.z += 1.0;
    }

    if input.is_key_down(VirtualKeyCode::S) {
        movement_direction.z -= 1.0;
    }

    if input.is_key_down(VirtualKeyCode::A) {
        movement_direction.x -= 1.0;
    }

    if input.is_key_down(VirtualKeyCode::D) {
        movement_direction.x += 1.0;
    }

    state.previous_player_position = state.player_position.clone();

    if movement_direction != Vector3::zeros() {
        let rotation_yaw = Rotation3::from_euler_angles(0.0, camera.yaw, 0.0);
        movement_direction = rotation_yaw * movement_direction;

        let movement_distance = PLAYER_SPEED * LOGIC_TICK_DURATION.as_secs_f32();
        let player_position_delta = movement_distance * movement_direction.normalize();

        state.player_position += player_position_delta;
    }

    state.physics_world.step();
}

fn main() {
    simple_logger::init().expect("Failed to initialise logger");

    let event_loop = EventLoop::new();

    let monitor = event_loop.available_monitors().next().unwrap();
    let fullscreen = Fullscreen::Borderless(monitor);

    let window = WindowBuilder::new()
        .with_title("SCAV")
        .with_inner_size(LogicalSize {
            width: 1280,
            height: 720,
        })
        // .with_fullscreen(Some(fullscreen))
        .build(&event_loop)
        .expect("Failed to create window");

    let mut renderer = Renderer::new(&window).unwrap();

    let cube_model_id = renderer.upload_model(&model_cube()).unwrap();

    // Timers
    let start_time = Instant::now();
    let mut last_time = start_time;
    let mut accumulator = Duration::new(0, 0);
    let mut current_tick = 0u64;

    // States
    let mut game_state = GameState::new();
    let mut fixed_update_input = Input::new();

    game_state.physics_world.set_timestep(LOGIC_TICK_DURATION.as_secs_f32());

    // Track which keys are down at the winit event level,
    // so we can ignore virtual repeat key presses
    let mut key_down = [false; 1024];

    let mut camera = Camera::new();

    // window.set_cursor_visible(false);
    // window.set_cursor_grab(true).ok();

    let perlin = Perlin::new();

    event_loop.run(move |event, _, control_flow| {
        // Continuously poll instead of waiting for events
        *control_flow = ControlFlow::Poll;

        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                camera.yaw += CAMERA_MOUSE_SENSITIVITY * delta.0 as f32;
                camera.pitch += CAMERA_MOUSE_SENSITIVITY * delta.1 as f32;

                camera.yaw = na::wrap(camera.yaw, -PI, PI);
                camera.pitch = na::clamp(camera.pitch, -PI / 2.0, PI / 2.0);
            }

            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                renderer.recreate_swapchain(new_size);
            }

            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        device_id,
                        input,
                        is_synthetic,
                    },
                ..
            } => {
                if let Some(virtual_keycode) = input.virtual_keycode {
                    match input.state {
                        ElementState::Pressed => {
                            let is_down = &mut key_down[virtual_keycode as usize];

                            if !(*is_down) {
                                fixed_update_input.press_key(virtual_keycode);
                            }

                            *is_down = true;
                        }
                        ElementState::Released => {
                            key_down[virtual_keycode as usize] = false;
                            fixed_update_input.release_key(virtual_keycode)
                        }
                    }
                }
            }

            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }

            Event::MainEventsCleared => {
                let current_time = Instant::now();
                accumulator += current_time.duration_since(last_time);
                last_time = current_time;

                while accumulator >= LOGIC_TICK_DURATION {
                    accumulator -= LOGIC_TICK_DURATION;

                    fixed_update(&mut game_state, &fixed_update_input, &camera, current_tick);

                    current_tick += 1;
                    fixed_update_input.reset();
                }

                window.request_redraw();
            }

            Event::RedrawRequested(_) => {
                // Redraw the application
                // Redrawing here instead of MainEventsCleared allows redraws
                // requested by the OS

                let alpha = accumulator.as_secs_f32() / LOGIC_TICK_DURATION.as_secs_f32();

                let current_time_seconds = start_time.elapsed().as_secs_f64();

                // Lerp player position
                camera.position = (1.0 - alpha) * game_state.previous_player_position
                    + alpha * game_state.player_position.coords;

                // Hard-coded head height for now
                camera.position.y = 2.0;

                renderer.begin_frame();

                // Ground of cubes
                {
                    let num_cubes = 32768;
                    let dim = (num_cubes as f32).sqrt() as i32;

                    for i in 0..num_cubes {

                        let x = (i % dim) as f32 - 100.0;
                        let z = (i / dim) as f32 - 100.0;
                        let y = -0.5;

                        let mat_local = Matrix4::identity()
                            .append_scaling(1.0)
                            .append_translation(&Vector3::new(x, y, z));

                        renderer.add_model(cube_model_id, mat_local);
                    }
                }

                // Rigid bodies
                for (idx, rb) in game_state.physics_world.bodies.iter() {
                    if rb.is_static() { continue; }

                    let mut mat_local = Matrix4::identity();

                    mat_local.append_scaling_mut(1.0);
                    mat_local *= rb.position.to_homogeneous();

                    renderer.add_model(cube_model_id, mat_local);
                }

                renderer.end_frame(&camera);
            }
            _ => {}
        }
    });
}
