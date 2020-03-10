use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use gfx_hal::{Instance, Backend};

#[allow(unused_imports)]
use log::{error, warn, info, debug, trace};

use gfx_backend_vulkan as gfx_backend;
use gfx_hal::window::{Surface, SwapchainConfig};
use gfx_hal::queue::QueueFamily;
use gfx_hal::adapter::{Adapter, PhysicalDevice, Gpu};
use gfx_hal::Features;

struct Renderer<B: gfx_hal::Backend> {
    instance: gfx_backend::Instance,
    surface: <gfx_backend::Backend as Backend>::Surface,
    adapter: Adapter<gfx_backend::Backend>,
    device: gfx_backend::Device,
}

impl Renderer {
    fn new(window: &Window) -> Result<Self, &'static str> {
        let instance = gfx_backend::Instance::create("Window", 1)
            .map_err(|_| "Failed to create instance.")?;

        let mut surface = unsafe { instance.create_surface(window) }
            .map_err(|_| "Failed to create surface.")?;

        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .find(|a| {
                a.queue_families
                    .iter()
                    .any(|qf| qf.queue_type().supports_graphics() && surface.supports_queue_family(qf))
            }).ok_or("Failed to find a suitable graphical adapter.")?;

        let queue_family = adapter
            .queue_families
            .iter()
            .find(|qf| qf.queue_type().supports_graphics() && surface.supports_queue_family(qf))
            .ok_or("Failed to find a queue family that supports graphics.")?;

        let Gpu { device, mut queue_groups } = unsafe {
            adapter
                .physical_device
                .open(&[(&queue_family, &[1.0; 1])], Features::empty())
                .map_err(|_| "Failed to open physical device.")?
        };

        Ok(Self {
            instance,
            surface,
            adapter,
        })
    }

    fn draw_clear_frame(&self) -> Result<(), ()> {
        Ok(())
    }
}

fn main() {
    simple_logger::init().expect("Failed to initialise logger");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Game")
        .build(&event_loop)
        .expect("Failed to create window");

    let instance = gfx_backend::Instance::create("InstanceName", 1).unwrap();
    let surface = unsafe { instance.create_surface(&window).unwrap(); };
    let adapters = instance.enumerate_adapters();

    let mut renderer = Renderer::new(&window).expect("AA");

    event_loop.run(move |event, _, control_flow| {
        // Continuously poll instead of waiting for events
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit
            },
            Event::MainEventsCleared => {
                // Application update code
                window.request_redraw();
            },
            Event::RedrawRequested(_) => {
                // Redraw the application
                // Redrawing here instead of MainEventsCleared allows redraws
                // requested by the OS
                if let Err(e) = render(&mut renderer) {
                }
            },
            _ => {}
        }
    });
}

fn render(renderer: &mut Renderer) -> Result<(), ()> {
    renderer.draw_clear_frame()
}
