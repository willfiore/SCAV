#![windows_subsystem = "windows"]

mod renderer;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[allow(unused_imports)]
use log::{error, warn, info, debug, trace};

use winit::dpi::LogicalSize;
use crate::renderer::Renderer;

fn main() {
    simple_logger::init().expect("Failed to initialise logger");

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Game")
        .with_inner_size(LogicalSize { width: 1280, height: 720 })
        .with_resizable(false)
        .build(&event_loop)
        .expect("Failed to create window");

    let mut renderer = Renderer::new(&window);

    event_loop.run(move |event, _, control_flow| {

        // Continuously poll instead of waiting for events
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {

                // TODO: Why is this hanging?
                // Vulkan cleanup
                // unsafe {
                //     device.device_wait_idle();

                // image_available_semaphores.iter().for_each(|&s| {
                //     device.destroy_semaphore(s, None);
                // });
                //
                // render_finished_semaphores.iter().for_each(|&s| {
                //     device.destroy_semaphore(s, None);
                // });

                // TODO: Destroy fences

                //     device.destroy_command_pool(command_pool, None);
                //     device.destroy_pipeline(pipeline, None);
                //     device.destroy_pipeline_layout(pipeline_layout, None);

                //     framebuffers.iter().for_each(|&fb| {
                //         device.destroy_framebuffer(fb, None);
                //     });

                //     device.destroy_render_pass(render_pass, None);

                //     swapchain_image_views.iter().for_each(|&v| {
                //         device.destroy_image_view(v, None);
                //     });

                //     swapchain_loader.destroy_swapchain(swapchain, None);
                //     surface_loader.destroy_surface(surface, None);
                //     device.destroy_device(None);
                //     instance.destroy_instance(None);
                // }

                *control_flow = ControlFlow::Exit;
            },
            Event::MainEventsCleared => {
                // Application update code
                window.request_redraw();
            },
            Event::RedrawRequested(_) => {

                // Redraw the application
                // Redrawing here instead of MainEventsCleared allows redraws
                // requested by the OS
                renderer.draw_frame();

            },
            _ => {}
        }
    });

}


