#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (binding = 0) uniform UniformBufferObject {
    mat4 proj_view_model;
} ubo;

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_color;
layout (location = 2) in mat4 a_model;

layout (location = 0) out vec3 vs_color;

void main() {
    vs_color = a_color;
    gl_Position = ubo.proj_view_model * a_model * vec4(a_position, 1.0);
}