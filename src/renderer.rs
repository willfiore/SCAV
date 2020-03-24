extern crate nalgebra as na;

use ash::vk;
use ash::extensions::khr;
use ash::util::Align;
use std::ffi::{CString};
use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use std::io::Cursor;

use winit::{
    window::Window,
    platform::windows::WindowExtWindows,
};
use ash::prelude::VkResult;

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex {
    position: na::Vector2<f32>,
    color: na::Vector3<f32>
}

impl Vertex {
    fn binding_description() -> vk::VertexInputBindingDescription {
        let binding_description = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);

        *binding_description
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0)
                .build(),

            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::size_of::<na::Vector2<f32>>() as u32)
                .build(),
        ]
    }
}

fn extension_names() -> Vec<*const i8> {
    use ash::extensions::*;

    vec![
        khr::Surface::name().as_ptr(),
        khr::Win32Surface::name().as_ptr(),
        ext::DebugReport::name().as_ptr()
    ]
}

static MAX_FRAMES_IN_FLIGHT: usize = 3;

/// Vulkan renderer
pub struct Renderer {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    surface: vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    current_frame: usize,
    in_flight_frame_fences: Vec<vk::Fence>,
    in_flight_image_fences: Vec<vk::Fence>,
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
}

fn find_memory_type_index(type_filter: u32, properties: vk::MemoryPropertyFlags,
                    instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Result<u32, ()> {

    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    for (i, memory_type) in memory_properties.memory_types.iter().enumerate() {

        let memory_type_suitable = (type_filter & (1 << (i as u32))) != 0;
        let memory_type_supports_properties = (memory_type.property_flags & properties) == properties;

        if memory_type_suitable && memory_type_supports_properties {
            return Ok(i as u32);
        }
    }

    return Err(());
}

impl Renderer {

    pub fn new(window: &Window) -> Self {
        let entry = ash::Entry::new().unwrap();

        let application_name = CString::new("Game").unwrap();

        let application_info = vk::ApplicationInfo::builder()
            .application_name(&application_name)
            .application_version(0)
            .engine_name(&application_name)
            .engine_version(0)
            .api_version(ash::vk_make_version!(1, 0, 0));

        let instance_layer_names = [
            CString::new("VK_LAYER_KHRONOS_validation").unwrap(),
        ];

        let instance_layer_names_raw: Vec<*const i8> = instance_layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let instance_extension_names_raw = extension_names();
        let device_extension_names_raw = [ash::extensions::khr::Swapchain::name().as_ptr()];

        // INSTANCE

        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_extension_names(&instance_extension_names_raw)
            .enabled_layer_names(&instance_layer_names_raw);

        let instance = unsafe { entry.create_instance(&instance_create_info, None) }
            .expect("Failed to create Vulkan instance");

        // PHYSICAL DEVICE

        let physical_device = {
            let mut physical_devices = unsafe { instance.enumerate_physical_devices() }
                .expect("Failed to enumerate physical devices");

            // TODO: Ensure physical device supports extensions

            physical_devices.pop()
        }.expect("Failed to find a physical device with Vulkan support");

        // SURFACE

        let surface = {

            let surface_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                .hwnd(window.hwnd())
                .hinstance(window.hinstance());

            let win32_surface = ash::extensions::khr::Win32Surface::new(&entry, &instance);

            unsafe { win32_surface.create_win32_surface(&surface_create_info, None) }
                .expect("Failed to create Vulkan surface")
        };

        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

        // QUEUES, LOGICAL DEVICE

        let queue_family_properties = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let graphics_queue_family_index = queue_family_properties.iter().enumerate().find(|(_, qf)| {
            qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        }).expect("Failed to find suitable graphics queue family").0 as u32;

        let present_queue_family_index = queue_family_properties.iter().enumerate().find(|(i, _)| {
            unsafe { surface_loader.get_physical_device_surface_support(physical_device, *i as u32, surface) }
        }).expect("Failed to find suitable present queue family for surface").0 as u32;

        let unique_queue_families = {
            let mut queue_families = vec![graphics_queue_family_index, present_queue_family_index];
            queue_families.sort();
            queue_families.dedup();
            queue_families
        };

        let physical_device_features = vk::PhysicalDeviceFeatures::builder();

        // Create a DeviceQueueCreateInfo per unique queue family
        let device_queue_create_infos = unique_queue_families.iter().map(|qf| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*qf)
                .queue_priorities(&[1.0])
                .build()
        }).collect::<Vec<_>>();

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&device_queue_create_infos)
            .enabled_features(&physical_device_features)
            .enabled_extension_names(&device_extension_names_raw);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .expect("Failed to create logical device");

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_queue_family_index, 0) };

        // SWAPCHAIN

        let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);

        let surface_format = {
            let supported_surface_formats = unsafe {
                surface_loader.get_physical_device_surface_formats(physical_device, surface)
                    .expect("Failed to fetch physical device surface formats")
            };

            supported_surface_formats.iter().find(|sf| {
                sf.format == vk::Format::B8G8R8A8_SRGB && sf.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            }).unwrap_or_else(||
                supported_surface_formats.first()
                    .expect("Failed to find any surface formats for physical device")
            ).clone()
        };

        let present_mode = {
            let supported_surface_present_modes = unsafe {
                surface_loader.get_physical_device_surface_present_modes(physical_device, surface) }
                .expect("Failed to fetch physical device surface formats");

            // Prefer MAILBOX (3-buffering) otherwise FIFO is always available
            supported_surface_present_modes.iter().find(|&&spm| {
                spm == vk::PresentModeKHR::MAILBOX
            }).unwrap_or(&vk::PresentModeKHR::FIFO).clone()
        };

        let capabilities = unsafe { surface_loader.get_physical_device_surface_capabilities(physical_device, surface) }
            .expect("Failed to get physical device surface capabilities");

        let window_extent = {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width,
                height: size.height,
            }
        };

        let extent =
            if capabilities.current_extent.width != u32::max_value() {
                capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: window_extent.width.max(capabilities.min_image_extent.width).min(capabilities.max_image_extent.width),
                    height: window_extent.height.max(capabilities.min_image_extent.height).min(capabilities.max_image_extent.height)
                }
            };

        let image_count = (capabilities.min_image_count + 1).min(capabilities.max_image_count);

        let (image_sharing_mode, queue_family_indices) =
            if graphics_queue_family_index != present_queue_family_index {
                (vk::SharingMode::CONCURRENT,
                 vec![graphics_queue_family_index, present_queue_family_index])
            } else {
                (vk::SharingMode::EXCLUSIVE, vec![])
            };

        let swapchain_create_info =
            vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(image_count as u32)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_sharing_mode(image_sharing_mode)
                .queue_family_indices(&queue_family_indices);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }
            .expect("Failed to create swapchain");

        // IMAGES

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
            .expect("Failed to get swapchain images");

        // IMAGE VIEWS

        let swapchain_image_views = swapchain_images.iter().map(|&image| {
            let create_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .components(vk::ComponentMapping::default())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1
                })
                .image(image);

            unsafe { device.create_image_view(&create_info, None) }
                .expect("Failed to create image view")

        }).collect::<Vec<_>>();

        // RENDER PIPELINE

        // Shaders
        let shader_function_entry_point = CString::new("main").unwrap();

        let vertex_shader_module = {
            let mut spv_file = Cursor::new(&include_bytes!("../generated/shaders/basic.vert.spv")[..]);
            let code = ash::util::read_spv(&mut spv_file)
                .expect("Failed to read vertex shader spv file");

            let create_info = vk::ShaderModuleCreateInfo::builder()
                .code(&code);

            unsafe { device.create_shader_module(&create_info, None) }
                .expect("Failed to create vertex shader module")
        };

        let fragment_shader_module = {
            let mut spv_file = Cursor::new(&include_bytes!("../generated/shaders/basic.frag.spv")[..]);
            let code = ash::util::read_spv(&mut spv_file)
                .expect("Failed to read fragment shader spv file");

            let create_info = vk::ShaderModuleCreateInfo::builder()
                .code(&code);

            unsafe { device.create_shader_module(&create_info, None) }
                .expect("Failed to create fragment shader module")
        };

        let vertex_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&shader_function_entry_point);

        let fragment_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&shader_function_entry_point);

        let shader_stage_create_infos = [*vertex_shader_stage_create_info, *fragment_shader_stage_create_info];

        // Vertex bindings
        let vertex_binding_descriptions = [Vertex::binding_description()];
        let vertex_attribute_descriptions = Vertex::attribute_descriptions();

        // VertexInputState - Format of vertex data passed to vertex shader
        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_descriptions)
            .vertex_attribute_descriptions(&vertex_attribute_descriptions);

        // InputAssemblyState - What kind of geometry will be drawn, and allow primitive restart?
        let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Viewport - region of the framebuffer that the output will be rendered to (scales output)
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0
        };

        // Scissor - region of the framebuffer where output is not discarded (crops output)
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: 0,
                y: 0
            },
            extent
        };

        let viewports = [viewport];
        let scissors = [scissor];

        // ViewportState - combines viewports and scissors
        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        // RasterizationState - defines rasterizer fixed function stage, which turns vertices
        // into fragments for input into the fragment shader
        let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0);

        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            // .sample_mask(&[])
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        // let depth_stencil_state_create_info = vk::PipelineDepthStencilStateCreateInfo::builder();

        // ColorBlendAttachmentState - after a fragment shader has returned a color, how to
        // blend it with the color already in the framebuffer
        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                vk::ColorComponentFlags::R |
                    vk::ColorComponentFlags::G |
                    vk::ColorComponentFlags::B |
                    vk::ColorComponentFlags::A
            )
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ZERO)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blend_attachment_states = [*color_blend_attachment_state];

        // ColorBlendState - Holds color blend attachments for each framebuffer
        let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachment_states)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        // PipelineLayout - for defining uniform values, and push constants
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder();
        // .set_layouts(&[])
        // .push_constant_ranges(&[]);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }
            .expect("Failed to create pipeline layout");

        // RENDER PASS

        // Specify the framebuffer attachments that will be used while rendering
        let color_attachment = vk::AttachmentDescription::builder()
            .format(surface_format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        // Can have multiple subpasses - subsequent render operations that depend on the
        // content of framebuffers in previous passes, e.g. for post-processing.
        // Subpasses reference one or more of the above attachments, using AttachmentReference objects
        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let attachments = [*color_attachment];
        let color_attachment_refs = [*color_attachment_ref];

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs);

        let subpasses = [*subpass];

        let subpass_dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)

        .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let subpass_dependencies = [*subpass_dependency];

        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&subpass_dependencies);

        let render_pass = unsafe { device.create_render_pass(&render_pass_create_info, None) }
            .expect("Failed to create render pass");

        let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state_create_info)
            .input_assembly_state(&input_assembly_state_create_info)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&rasterization_state_create_info)
            .multisample_state(&multisample_state_create_info)
            .color_blend_state(&color_blend_state_create_info)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1);

        let pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[*graphics_pipeline_create_info], None)
        }.expect("Failed to create graphics pipeline").pop().unwrap();

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        }

        // FRAMEBUFFERS

        // Create a framebuffer per image view
        let framebuffers = swapchain_image_views.iter().map(|&v| {
            let attachments = [v];

            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);

            unsafe { device.create_framebuffer(&create_info, None) }
                .expect("Failed to create framebuffer")

        }).collect::<Vec<_>>();

        // VERTEX BUFFER

        // Vertices
        let vertices = vec![
            Vertex { position: na::Vector2::new( 0.5,  0.5), color: na::Vector3::new(0.0, 1.0, 0.0) },
            Vertex { position: na::Vector2::new( 0.0, -0.5), color: na::Vector3::new(1.0, 0.0, 0.0) },
            Vertex { position: na::Vector2::new(-0.5,  0.5), color: na::Vector3::new(0.0, 0.0, 1.0) },
        ];

        let vertex_buffer_size = (std::mem::size_of::<Vertex>() * vertices.len()) as u64;

        // Create buffer
        let vertex_buffer_create_info = vk::BufferCreateInfo::builder()
            .size(vertex_buffer_size)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let vertex_buffer = unsafe { device.create_buffer(&vertex_buffer_create_info, None)
            .expect("Failed to create vertex buffer")
        };

        // Allocate memory
        let vertex_buffer_memory_requirements = unsafe { device.get_buffer_memory_requirements(vertex_buffer) };

        let memory_type_index = find_memory_type_index(
            vertex_buffer_memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &instance,
            physical_device
        ).expect("Failed to find suitable memory type for vertex buffer");

        let vertex_buffer_memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(vertex_buffer_memory_requirements.size)
            .memory_type_index(memory_type_index);

        let vertex_buffer_memory = unsafe {
            device.allocate_memory(&vertex_buffer_memory_allocate_info, None)
                .expect("Failed to allocate vertex buffer memory")
        };

        // Bind memory to buffer
        unsafe {
            device.bind_buffer_memory(vertex_buffer, vertex_buffer_memory, 0)
                .expect("Failed to bind vertex buffer memory")
        };

        // Fill memory
        let vertex_buffer_memory_ptr = unsafe {
            device.map_memory(vertex_buffer_memory, 0, vertex_buffer_size, vk::MemoryMapFlags::empty())
                .expect("Failed to map memory")
        };

        let mut vert_align = unsafe { Align::new(
            vertex_buffer_memory_ptr,
            std::mem::align_of::<Vertex>() as u64,
            vertex_buffer_memory_requirements.size
        ) };

        vert_align.copy_from_slice(&vertices);

        unsafe { device.unmap_memory(vertex_buffer_memory) };

        // COMMAND POOL

        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_queue_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);

        let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None) }
            .expect("Failed to create command pool");

        // COMMAND BUFFERS

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as u32);

        let command_buffers = unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }
            .expect("Failed to allocate command buffers");

        command_buffers.iter().enumerate().for_each(|(i, &command_buffer)| {
            let begin_info = vk::CommandBufferBeginInfo::builder();

            unsafe { device.begin_command_buffer(command_buffer, &begin_info) }
                .expect("Failed to begin comomand buffer");

            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue { float32: [ 0.1, 0.1, 0.1, 1.0 ] }
            };

            let clear_values = [clear_value];

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(render_pass)
                .framebuffer(framebuffers[i])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D {
                        x: 0,
                        y: 0
                    },
                    extent
                })
                .clear_values(&clear_values);

            // Render pass
            unsafe {
                device.cmd_begin_render_pass(command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
                device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

                let vertex_buffers = [vertex_buffer];
                let offsets = [0];

                device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                device.cmd_draw(command_buffer, vertices.len() as u32, 1, 0, 0);
                device.cmd_end_render_pass(command_buffer);
            };

            unsafe { device.end_command_buffer(command_buffer) }
                .expect("Failed to end command buffer");
        });

        // SEMAPHORES AND FENCES

        // Semaphores are sync primitives internal to the GPU, they handle synchronization
        // between stages of the GPU.

        // Fences are for CPU <-> GPU synchronization

        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();

        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED);

        // Semaphore, per frame in flight, which is signalled after the image
        // is acquired from the swapchain. Queued command buffers start
        // after this is signalled
        let image_available_semaphores = (0..MAX_FRAMES_IN_FLIGHT).map(|_| {
            unsafe { device.create_semaphore(&semaphore_create_info, None) }
                .expect("Failed to create semaphore")
        }).collect::<Vec<_>>();

        // Signalled after submitted command buffers finish rendering.
        // Surface present waits until this is signalled before presenting
        // the final framebuffer to the screen
        let render_finished_semaphores = (0..MAX_FRAMES_IN_FLIGHT).map(|_| {
            unsafe { device.create_semaphore(&semaphore_create_info, None) }
                .expect("Failed to create semaphore")
        }).collect::<Vec<_>>();

        let in_flight_frame_fences = (0..MAX_FRAMES_IN_FLIGHT).map(|_| {
            unsafe { device.create_fence(&fence_create_info, None) }
                .expect("Failed to create fence")
        }).collect::<Vec<_>>();

        let in_flight_image_fences = vec![vk::Fence::null(); swapchain_images.len()];
        let current_frame: usize = 0;

        Self {
            entry,
            instance,
            physical_device,
            device,
            surface,
            surface_loader,
            graphics_queue,
            present_queue,
            current_frame,
            in_flight_frame_fences,
            in_flight_image_fences,
            swapchain_loader,
            swapchain,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores
        }
    }

    pub fn draw_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        // Ensure that there are no more than MAX_FRAMES_IN_FLIGHT frames
        // in the pipeline at once
        let wait_frame_fences = [self.in_flight_frame_fences[self.current_frame]];
        unsafe {
            self.device.wait_for_fences(&wait_frame_fences, true, u64::max_value())
                .expect("Failed to wait for current frame fence");
        }

        let acquire_image_result = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::max_value(),
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null()
            )
        };

        let image_index = match acquire_image_result {
            Ok(result) => result.0,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return;
            }
            _ => { panic!("AAA"); }
        };

        let image_index = image_index as usize;

        // Ensure that a current frame in flight isn't using this swapchain image
        if self.in_flight_image_fences[image_index] != vk::Fence::null() {
            let wait_image_fences = [self.in_flight_image_fences[image_index]];
            unsafe {
                self.device.wait_for_fences(&wait_image_fences, true, u64::max_value())
                    .expect("Failed to wait for swapchain image fence");
            }
        }

        // Mark this swapchain image as now being used by this frame
        self.in_flight_image_fences[image_index] = self.in_flight_frame_fences[self.current_frame];

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

        let current_command_buffers = [self.command_buffers[image_index]];

        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&current_command_buffers)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores);

        let submit_infos = [*submit_info];

        unsafe {
            self.device.reset_fences(&wait_frame_fences)
                .expect("Failed to reset fences");

            self.device.queue_submit(self.graphics_queue, &submit_infos, self.in_flight_frame_fences[self.current_frame])
                .expect("Failed to submit draw command buffer");
        }

        let swapchains = [self.swapchain];
        let image_indices = [image_index as u32];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.swapchain_loader.queue_present(self.present_queue, &present_info)
                .unwrap_or_else(|err| {
                    if err == vk::Result::ERROR_OUT_OF_DATE_KHR {
                        true
                    } else {
                        panic!("Failed to present queue");
                    }
                });
        };
    }
}
