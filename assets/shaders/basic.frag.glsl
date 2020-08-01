#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec3 vs_color;
layout (location = 1) in vec2 vs_tex_coords;

layout (location = 0) out vec4 color;

void main() {
    color = vec4(vs_tex_coords, 0.0, 1.0);
}
