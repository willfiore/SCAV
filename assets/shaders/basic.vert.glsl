#version 450
#extension GL_ARB_separate_shader_objects : enable

/////////////
// INPUTS
/////////////

layout (binding = 0) uniform UniformBufferObject {
    mat4 proj_view;
} ubo;

// Per-vertex position and color
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_color;
layout (location = 2) in vec2 a_tex_coords;

// Per-instance model matrix
layout (location = 3) in mat4 a_model;

/////////////
// OUTPUTS
/////////////

layout (location = 0) out vec3 vs_color;
layout (location = 1) out vec2 vs_tex_coords;

void main() {
    vs_color = a_color;
    vs_tex_coords = a_tex_coords;
    gl_Position = ubo.proj_view * a_model * vec4(a_position, 1.0);
}