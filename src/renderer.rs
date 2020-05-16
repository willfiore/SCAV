extern crate nalgebra as na;
extern crate ash;

use ash::vk as vk;
use ash::util::Align;
use std::ffi::{CString};
use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use std::io::Cursor;

use winit::{
    window::Window,
    platform::windows::WindowExtWindows,
};

use na::{Matrix4, Point3, Vector3};
use std::collections::{HashMap, BTreeMap};
use nalgebra::Vector4;
use std::cmp::Ordering;

pub struct Camera {
    pub position: Point3<f32>,

    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            position: Point3::origin(),
            yaw: 0.0,
            pitch: 0.0
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Vector3<f32>,
    pub color: Vector3<f32>
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
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0)
                .build(),

            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::size_of::<na::Vector3<f32>>() as u32)
                .build(),
        ]
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
struct InstanceData {
    transform: Matrix4<f32>,
}

impl InstanceData {
    fn binding_description() -> vk::VertexInputBindingDescription {
        let binding_description = vk::VertexInputBindingDescription::builder()
            .binding(1)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::INSTANCE);

        *binding_description
    }

    // Matrix as a vertex attribute takes up 4 locations, one for each row
    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
        [
            vk::VertexInputAttributeDescription::builder()
                .binding(1)
                .location(2)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(0)
                .build(),

            vk::VertexInputAttributeDescription::builder()
                .binding(1)
                .location(3)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(std::mem::size_of::<Vector4<f32>>() as u32)
                .build(),

            vk::VertexInputAttributeDescription::builder()
                .binding(1)
                .location(4)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(2 * std::mem::size_of::<Vector4<f32>>() as u32)
                .build(),

            vk::VertexInputAttributeDescription::builder()
                .binding(1)
                .location(5)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(3 * std::mem::size_of::<Vector4<f32>>() as u32)
                .build(),
        ]
    }
}

// MODEL

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, PartialEq, Eq)]
pub struct ModelId(u32);

#[derive(Debug)]
struct ModelDescriptor {
    model_id: ModelId,

    num_vertices: usize,
    num_indices: usize,

    vertex_offset: usize, // offset in num vertices since beginning of geometry buffer
    index_offset: usize,  // offset in num indices since beginning of geometry buffer
}

struct ModelDrawCommand {
    model_id: ModelId,
    transform: Matrix4::<f32>,
}

enum DrawCommand {
    Model(ModelDrawCommand)
}

pub type ModelIndex = u32;

pub struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<ModelIndex>,
}

#[derive(Debug, Clone, Copy)]
struct BufferRegion {
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct UniformBufferObject {
    proj_view_model: Matrix4<f32>,
}

static MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Vulkan renderer
pub struct Renderer {

    // VULKAN STATE
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,

    surface: vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
    surface_format: vk::SurfaceFormatKHR,

    extent: vk::Extent2D,

    graphics_queue: vk::Queue,
    graphics_queue_family_index: u32,

    present_queue: vk::Queue,
    present_queue_family_index: u32,

    current_frame: usize,

    in_flight_frame_fences: Vec<vk::Fence>,
    in_flight_image_fences: Vec<vk::Fence>,

    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,

    depth_image: GpuImage,

    command_pool: vk::CommandPool,
    swapchain_command_buffers: Vec<vk::CommandBuffer>,
    upload_command_buffer: vk::CommandBuffer,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,

    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,

    descriptor_sets: Vec<vk::DescriptorSet>,

    uniform_buffer: GpuBuffer,
    staging_buffer: GpuBuffer,

    geometry_buffer: GpuBuffer,
    geometry_buffer_indices_offset:  usize,
    geometry_buffer_num_vertices: usize,
    geometry_buffer_num_indices: usize,

    model_instance_data_buffer: GpuBuffer, // Stores transform matrices per instance

    model_descriptors: HashMap<ModelId, ModelDescriptor>,
    next_model_id: ModelId,

    queued_draw_commands: Vec<DrawCommand>,

    // Frame state
    is_frame_writing: bool,
}

#[derive(Debug, Clone, Copy)]
struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

#[derive(Debug, Clone, Copy)]
struct GpuImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

fn extension_names() -> Vec<*const i8> {
    use ash::extensions::*;

    vec![
        khr::Surface::name().as_ptr(),
        khr::Win32Surface::name().as_ptr(),
        ext::DebugReport::name().as_ptr()
    ]
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

fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect_flags: vk::ImageAspectFlags,
) -> Result<vk::ImageView, ()> {

    let image_view_create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect_flags,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1
        });

    let image_view = unsafe { device.create_image_view(&image_view_create_info, None) }
        .map_err(|_|())?;

    Ok(image_view)
}

fn create_image(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    size: vk::Extent2D,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage_flags: vk::ImageUsageFlags,
    aspect_flags: vk::ImageAspectFlags,
    memory_properties: vk::MemoryPropertyFlags,
) -> Result<GpuImage, ()> {

    // IMAGE

    let image_create_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: size.width,
            height: size.height,
            depth: 1
        })
        .mip_levels(1)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage_flags)
        .samples(vk::SampleCountFlags::TYPE_1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let image = unsafe { device.create_image(&image_create_info, None) }
        .map_err(|_|())?;

    // MEMORY

    let memory_requirements = unsafe { device.get_image_memory_requirements(image) };

    let memory_type_index = find_memory_type_index(
        memory_requirements.memory_type_bits,
        memory_properties,
        instance,
        physical_device
    ).map_err(|_|())?;

    let memory_allocate_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(memory_requirements.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe { device.allocate_memory(&memory_allocate_info, None) }
        .map_err(|_|())?;

    unsafe { device.bind_image_memory(image, memory, 0) }
        .map_err(|_|())?;

    // VIEW

    let view = create_image_view(device, image, format, aspect_flags)
        .map_err(|_|())?;

    Ok( GpuImage { image, memory, view } )
}

fn create_buffer(instance: &ash::Instance,
                 device: &ash::Device,
                 physical_device: vk::PhysicalDevice,
                 size: vk::DeviceSize,
                 usage_flags: vk::BufferUsageFlags,
                 memory_property_flags: vk::MemoryPropertyFlags)
                 -> Result<GpuBuffer, ()>
{

    // Create buffer
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage_flags)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device.create_buffer(&buffer_create_info, None)
            .map_err(|_|())?
    };

    // Allocate memory
    let buffer_memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let memory_type_index = find_memory_type_index(
        buffer_memory_requirements.memory_type_bits,
        memory_property_flags,
        &instance,
        physical_device
    )?;

    let buffer_memory_allocate_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(buffer_memory_requirements.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe {
        device.allocate_memory(&buffer_memory_allocate_info, None)
            .map_err(|_|())?
    };

    // Bind memory to buffer
    unsafe {
        device.bind_buffer_memory(buffer, memory, 0)
            .map_err(|_|())?
    };

    Ok( GpuBuffer { buffer, memory } )
}

fn copy_buffer(src_buffer: GpuBuffer,
               dst_buffer: GpuBuffer,
               size: vk::DeviceSize,
               src_offset: vk::DeviceSize,
               dst_offset: vk::DeviceSize,
               device: &ash::Device,
               command_buffer: vk::CommandBuffer,
               submit_queue: vk::Queue)
    -> Result<(), ()>
{
    unsafe {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let buffer_copy = vk::BufferCopy::builder()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(size);

        let regions = [*buffer_copy];

        device.begin_command_buffer(command_buffer, &command_buffer_begin_info);
        device.cmd_copy_buffer(command_buffer, src_buffer.buffer, dst_buffer.buffer, &regions);
        device.end_command_buffer(command_buffer);

        let command_buffers = [command_buffer];

        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&command_buffers);

        let submit_infos = [*submit_info];

        device.queue_submit(submit_queue, &submit_infos, vk::Fence::null());
        device.queue_wait_idle(submit_queue);
    }

    Ok(())
}

fn upload_to_buffer<T: Copy>(device: &ash::Device, buffer: GpuBuffer, offset: vk::DeviceSize, slice: &[T])
    -> Result<(), ()>
{

    let align = std::mem::align_of::<T>() as u64;
    let size = (slice.len() * std::mem::size_of::<T>()) as u64;

    unsafe {
        let memory_ptr =
            device.map_memory(buffer.memory, offset, size, vk::MemoryMapFlags::empty())
                .map_err(|_|())?;

        let mut vk_align = Align::new(memory_ptr, align, size);
        vk_align.copy_from_slice(slice);

        device.unmap_memory(buffer.memory);
    }

    Ok(())
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
            .api_version(vk::make_version(1, 0, 0));

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
                .expect("Failed to get device surface support")
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

        // SWAPCHAIN IMAGES

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
            .expect("Failed to get swapchain images");

        // SWAPCHAIN IMAGE VIEWS

        let swapchain_image_views = swapchain_images.iter().map(|&image| {
            create_image_view(&device, image, surface_format.format, vk::ImageAspectFlags::COLOR)
                .expect("Failed to create image view")
        }).collect::<Vec<_>>();

        // DEPTH IMAGES

        let depth_image_format = vk::Format::D32_SFLOAT_S8_UINT;

        let depth_image = create_image(
            &instance, &device, physical_device,
            extent,
            depth_image_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ).expect("Failed to create depth image");

        // RENDER PIPELINE

        // Shaders
        let shader_function_entry_point = CString::new("main").unwrap();

        let vertex_shader_module = {
            let mut spv_file = Cursor::new(&include_bytes!("../assets/shaders/generated/basic.vert.spv")[..]);
            let code = ash::util::read_spv(&mut spv_file)
                .expect("Failed to read vertex shader spv file");

            let create_info = vk::ShaderModuleCreateInfo::builder()
                .code(&code);

            unsafe { device.create_shader_module(&create_info, None) }
                .expect("Failed to create vertex shader module")
        };

        let fragment_shader_module = {
            let mut spv_file = Cursor::new(&include_bytes!("../assets/shaders/generated/basic.frag.spv")[..]);
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
        let vertex_binding_descriptions = [Vertex::binding_description(), InstanceData::binding_description()];
        let vertex_attribute_descriptions = [&Vertex::attribute_descriptions()[..], &InstanceData::attribute_descriptions()[..]].concat();

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

        let depth_stencil_state_create_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false)
            .front(vk::StencilOpState::default())
            .back(vk::StencilOpState::default());

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

        // Descriptor set layouts

        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX); // in which shader stages will this be used?

        let layout_bindings = [*ubo_layout_binding];

        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&layout_bindings);

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .expect("Failed to create descriptor set layout")
        };

        let pipeline_descriptor_set_layouts = [descriptor_set_layout];

        // PipelineLayout - for defining uniform values, and push constants
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&pipeline_descriptor_set_layouts);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }
            .expect("Failed to create pipeline layout");

        // Render pass

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

        let depth_attachment = vk::AttachmentDescription::builder()
            .format(depth_image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let depth_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let attachments = [*color_attachment, *depth_attachment];
        let color_attachment_refs = [*color_attachment_ref];

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref);

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
            .depth_stencil_state(&depth_stencil_state_create_info)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1);

        let pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[*graphics_pipeline_create_info], None)
        }.expect("Failed to create graphics pipeline").pop().unwrap();

        // FRAMEBUFFERS

        // Create a framebuffer per image view
        let framebuffers = swapchain_image_views.iter().map(|&image_view| {
            let attachments = [image_view, depth_image.view];

            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);

            unsafe { device.create_framebuffer(&create_info, None) }
                .expect("Failed to create framebuffer")

        }).collect::<Vec<_>>();

        // COMMAND POOL
        // Create command pool before buffers because it's needed to allocate
        // command buffers for transferring data between the staging buffer
        // and device local buffers.
        // TODO: use a separate command pool for this

        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_queue_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None) }
            .expect("Failed to create command pool");

        // BUFFERS

        // Static geometry should live on device local memory (since we don't have to change it after
        // upload). To get it there, we need to use a host coherent intermediate staging buffer,
        // and then copy that data over using a command buffer.

        let staging_buffer_max_size: usize = 10 * 1024 * 1024;

        let staging_buffer = create_buffer(
            &instance, &device, physical_device,
            staging_buffer_max_size as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ).expect("Failed to create staging buffer");

        let geometry_buffer_vertices_max_size : usize = 10 * 1024 * 1024;
        let geometry_buffer_indices_max_size  : usize = 10 * 1024 * 1024;
        let geometry_buffer_max_size          : usize = geometry_buffer_vertices_max_size + geometry_buffer_indices_max_size;

        let geometry_buffer_indices_offset: usize = geometry_buffer_vertices_max_size;

        let geometry_buffer = create_buffer(
            &instance, &device, physical_device,
            geometry_buffer_max_size as u64,
            vk::BufferUsageFlags::TRANSFER_DST |
                vk::BufferUsageFlags::VERTEX_BUFFER |
                vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ).expect("Failed to create vertex buffer");

        let model_instance_data_buffer_max_size: usize = 10 * 1024 * 1024;

        let model_instance_data_buffer = create_buffer(
            &instance, &device, physical_device,
            model_instance_data_buffer_max_size as u64,
            vk::BufferUsageFlags::TRANSFER_DST |
                vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ).expect("Failed to create vertex buffer");

        // Uniform buffer

        let uniform_buffer_size = (std::mem::size_of::<UniformBufferObject>() * swapchain_images.len()) as u64;

        let uniform_buffer = create_buffer(
            &instance, &device, physical_device,
            uniform_buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ).expect("Failed to create uniform buffer");

        let aspect_ratio = (extent.width as f32) / (extent.height as f32);

        let projection = na::geometry::Perspective3::new(aspect_ratio, 0.78, 0.1, 10.0)
            .to_homogeneous();

        let view = Matrix4::identity()
            .append_translation(&Vector3::new(0.2, 0.2, 0.2));

        let model = Matrix4::identity()
            .append_translation(&Vector3::new(3.0, 3.0, -10.0));

        let default_ubo = UniformBufferObject {
            proj_view_model: projection * view * model
        };

        let default_ubos = vec![default_ubo; swapchain_images.len()];
        upload_to_buffer(&device, uniform_buffer, 0, &default_ubos);

        // DESCRIPTOR SETS

        let descriptor_pool_size = vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(swapchain_images.len() as u32);

        let descriptor_pool_sizes = [*descriptor_pool_size];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(swapchain_images.len() as u32);

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create descriptor pool")
        };

        let descriptor_set_layouts = vec![descriptor_set_layout; swapchain_images.len()];

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets")
        };

        descriptor_sets.iter().enumerate().for_each(|(i, &descriptor_set)| {
            let range = std::mem::size_of::<UniformBufferObject>();
            let offset = i * range;

            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(uniform_buffer.buffer)
                .offset(offset as u64)
                .range(range as u64);

            let buffer_infos = [*buffer_info];

            let write_descriptor_set = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos);

            let write_descriptor_sets = [*write_descriptor_set];

            unsafe { device.update_descriptor_sets(&write_descriptor_sets, &[]); };
        });

        // COMMAND BUFFERS

        let swapchain_command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as u32);

        let upload_command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let swapchain_command_buffers = unsafe { device.allocate_command_buffers(&swapchain_command_buffer_allocate_info) }
            .expect("Failed to allocate swapchain command buffers");

        let upload_command_buffer = unsafe { device.allocate_command_buffers(&upload_command_buffer_allocate_info) }
            .expect("Failed to allocate upload command buffer")
            .pop().unwrap();

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

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        }

        let renderer = Self {
            instance,
            physical_device,
            device,

            surface,
            surface_loader,
            surface_format,

            extent,

            graphics_queue,
            graphics_queue_family_index,

            present_queue,
            present_queue_family_index,

            current_frame,

            in_flight_frame_fences,
            in_flight_image_fences,

            swapchain_loader,
            swapchain,
            swapchain_images,

            depth_image,

            command_pool,
            swapchain_command_buffers,
            upload_command_buffer,

            image_available_semaphores,
            render_finished_semaphores,

            pipeline_layout,
            pipeline,

            render_pass,
            framebuffers,
            descriptor_sets,

            uniform_buffer,
            staging_buffer,

            geometry_buffer,
            geometry_buffer_indices_offset,
            geometry_buffer_num_vertices: 0,
            geometry_buffer_num_indices: 0,

            model_instance_data_buffer,

            model_descriptors: HashMap::new(),
            next_model_id: ModelId(0),

            queued_draw_commands: Vec::new(),

            //

            is_frame_writing: false
        };

        renderer
    }

    pub fn begin_frame(&mut self) {
        assert!(!self.is_frame_writing, "begin_frame called again before draw_frame");
        self.is_frame_writing = true;

        self.queued_draw_commands.clear();
    }

    pub fn end_frame(&mut self, camera: &Camera) {
        assert!(self.is_frame_writing, "Can't draw frame when frame hasn't begun - did you call begin_frame?");
        self.is_frame_writing = false;

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

        // Record command buffer for frame

        let command_buffer = self.swapchain_command_buffers[image_index];

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::builder();

            self.device.begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin comomand buffer");

            let clear_values = [
                vk::ClearValue { color: vk::ClearColorValue { float32: [ 0.0, 0.0, 0.0, 1.0 ] } },
                vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } }
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D {
                        x: 0,
                        y: 0
                    },
                    extent: self.extent
                })
                .clear_values(&clear_values);

            // Render pass
            self.device.cmd_begin_render_pass(command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            let mut model_transforms_map: BTreeMap<ModelId, Vec<Matrix4<f32>>> = BTreeMap::new();

            // Sort out draw commands
            // - For each model, group all transform matrices together in model_transforms.
            // [ M1-1, M1-2, .. M1-N, M2-1, .. ]
            // - Then, for each model, cmd_draw_indexed

            for draw_command in &self.queued_draw_commands {
                match draw_command {
                    DrawCommand::Model(data) => {
                        match model_transforms_map.get_mut(&data.model_id) {
                            Some(transforms) => { transforms.push(data.transform.clone()); },
                            None => { model_transforms_map.insert(data.model_id, vec![data.transform.clone()]); }
                        }
                    }
                }
            }

            let model_counts = model_transforms_map.iter().map(|(&model_id, transforms)| {
                (model_id, transforms.len())
            }).collect::<BTreeMap<_,_>>();

            let mut transforms = Vec::new();

            model_transforms_map.iter_mut().for_each(|(model_id, t)| {
                transforms.append(t);
            });

            upload_to_buffer(&self.device, self.model_instance_data_buffer, 0, &transforms);

            // Bind geometry and instance data buffers
            let vertex_buffers = [self.geometry_buffer.buffer, self.model_instance_data_buffer.buffer];
            let vertex_offsets = [0, 0];

            self.device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &vertex_offsets);
            self.device.cmd_bind_index_buffer(command_buffer, self.geometry_buffer.buffer, self.geometry_buffer_indices_offset as u64, vk::IndexType::UINT32);

            let descriptor_sets = [self.descriptor_sets[image_index]];
            self.device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &descriptor_sets, &[]);

            for (model_id, count) in model_counts {
                let model_descriptor = &self.model_descriptors[&model_id];

                let vertex_offset = model_descriptor.vertex_offset;
                let index_offset = model_descriptor.index_offset;
                let index_count = model_descriptor.num_indices;

                self.device.cmd_draw_indexed(command_buffer, index_count as u32, count as u32, index_offset as u32, vertex_offset as i32, 0);
            }

            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)
                .expect("Failed to end command buffer");
        };

        // Update uniform buffer for frame
        let aspect_ratio = (self.extent.width as f32) / (self.extent.height as f32);

        let projection = na::geometry::Perspective3::new(aspect_ratio, 1.0, 0.1, 100.0)
            .to_homogeneous();

        let view = {

            let eye = &camera.position;
            let target = &Point3::new(camera.position[0], camera.position[1], camera.position[2] + 1.0);
            let up = &-Vector3::y();

            let translation = Matrix4::look_at_rh(eye, target, up);

            let rotation_yaw   = Matrix4::from_euler_angles(0.0, camera.yaw, 0.0);
            let rotation_pitch = Matrix4::from_euler_angles(-camera.pitch, 0.0, 0.0);

            rotation_pitch * rotation_yaw * translation
        };

        let model = Matrix4::identity();

        let ubo = UniformBufferObject {
            proj_view_model: projection * view * model
        };

        let memory_offset = image_index * std::mem::size_of::<UniformBufferObject>();

        upload_to_buffer(&self.device, self.uniform_buffer, memory_offset as u64, std::slice::from_ref(&ubo));

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
        let current_command_buffers = [command_buffer];

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

    pub fn draw_model(&mut self, model_id: ModelId, transform: Matrix4<f32>) {
        assert!(self.is_frame_writing, "draw_model called before begin_frame");

        let draw_command = DrawCommand::Model(
            ModelDrawCommand { model_id, transform }
        );

        self.queued_draw_commands.push(draw_command);
    }

    pub fn upload_model(&mut self, model: &Model) -> Result<ModelId, ()> {

        let model_num_vertices = model.vertices.len();
        let model_num_indices = model.indices.len();

        let vertices_size = (std::mem::size_of::<Vertex>() * model_num_vertices) as u64;
        let indices_size = (std::mem::size_of::<ModelIndex>() * model_num_indices) as u64;

        // Upload vertices and indices to staging buffer

        upload_to_buffer(&self.device, self.staging_buffer, 0, &model.vertices)
            .map_err(|_|())?;

        upload_to_buffer(&self.device, self.staging_buffer, vertices_size, &model.indices)
            .map_err(|_|())?;

        // Copy over to geometry buffer

        // Vertices
        copy_buffer(
            self.staging_buffer,     // Source
            self.geometry_buffer,    // Destination
            vertices_size,           // Size
            0,                       // Src offset
            (self.geometry_buffer_num_vertices * std::mem::size_of::<Vertex>()) as u64, // Dst offset
            &self.device, self.upload_command_buffer, self.graphics_queue
        ).map_err(|_|())?;

        // Indices
        copy_buffer(
            self.staging_buffer,     // Source
            self.geometry_buffer,    // Destination
            indices_size,            // Size
            vertices_size,           // Src offset
            self.geometry_buffer_indices_offset as u64 + (self.geometry_buffer_num_indices * std::mem::size_of::<ModelIndex>()) as u64, // Dst offset
            &self.device, self.upload_command_buffer, self.graphics_queue
        ).map_err(|_|())?;

        let model_id = self.next_model_id;

        let model_descriptor = ModelDescriptor {
            model_id,
            num_vertices: model.vertices.len(),
            num_indices: model.indices.len(),
            vertex_offset: self.geometry_buffer_num_vertices,
            index_offset: self.geometry_buffer_num_indices,
        };

        self.model_descriptors.insert(model_id, model_descriptor);

        self.geometry_buffer_num_vertices += model_num_vertices;
        self.geometry_buffer_num_indices += model_num_indices;

        self.next_model_id = ModelId(model_id.0 + 1);

        Ok(model_id)
    }
}

