extern crate shaderc;

use std::error::Error;
use shaderc::{ShaderKind};

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/shaders");
    println!("cargo:rerun-if-changed=generated");

    // Create destination path if necessary
    std::fs::create_dir_all("generated/shaders")?;

    let mut compiler = shaderc::Compiler::new()
        .expect("Failed to create compiler");

    let options = shaderc::CompileOptions::new()
        .expect("Failed to create options");

    for entry in std::fs::read_dir("src/shaders")? {
        let entry = entry?;

        if entry.file_type()?.is_file() {
            let path = entry.path();

            let file_name = path.file_name()
                .ok_or("Failed to read file name for source shader")?
                .to_string_lossy();

            let shader_kind =
                match file_name.splitn(2, ".").last().unwrap() {
                    "vert.glsl" => Some(ShaderKind::Vertex),
                    "frag.glsl" => Some(ShaderKind::Fragment),
                    _ => None,
                };

            if let Some(shader_kind) = shader_kind {
                let source = std::fs::read_to_string(&path)?;

                let compilation_artifact = compiler.compile_into_spirv(
                    &source, shader_kind, file_name.as_ref(), "main", Some(&options))
                    .expect("Failed to compile shader");

                let out_path = format!(
                    "generated/shaders/{}.spv",
                    file_name.rsplitn(2, ".").last().unwrap()
                );

                std::fs::write(&out_path, compilation_artifact.as_binary_u8())?;
            }
        };
    }

    Ok(())
}