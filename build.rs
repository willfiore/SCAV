extern crate shaderc;

use std::error::Error;
use shaderc::{ShaderKind};

// fn build_vr() -> Result<(), Box<dyn Error>> {
//
//     let lib_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("assets/c_src/openvr/lib/win64");
//     let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
//
//     // Link OpenVR
//     println!("cargo:rustc-link-search=native={}", lib_dir.display());
//     println!("cargo:rustc-link-lib=openvr_api");
//
//     // Copy DLL to output dir
//     std::fs::copy(lib_dir.join("openvr_api.dll"), out_dir.join("openvr_api.dll"));
//
//     // // Generate bindings
//     // let bindings = bindgen::Builder::default()
//     //     .header("assets/c_src/openvr_wrapper.hpp")
//     //     .constified_enum(".*")
//     //     .prepend_enum_name(false)
//     //     .parse_callbacks(Box::new(bindgen::CargoCallbacks))
//     //     .generate()
//     //     .expect("Failed to generate bindings");
//
//     // // Create destination path if necessary
//     // std::fs::create_dir_all("src/vr/generated")
//     //     .unwrap();
//
//     // bindings.write_to_file("src/vr/openvr_bindings.rs")?;
//
//     Ok(())
// }

fn build_shaders() -> Result<(), Box<dyn Error>> {
    let shaders_input_dir = "assets/shaders";
    let shaders_output_dir = "assets/shaders/generated";

    // println!("cargo:rerun-if-changed=build.rs");
    // println!("cargo:rerun-if-changed={}", shaders_input_dir);
    // println!("cargo:rerun-if-changed={}", shaders_output_dir);

    // Create destination path if necessary
    std::fs::create_dir_all(shaders_output_dir)
        .expect("Failed to create shader output directory");

    let mut compiler = shaderc::Compiler::new()
        .expect("Failed to create compiler");

    let options = shaderc::CompileOptions::new()
        .expect("Failed to create options");

    for entry in std::fs::read_dir(shaders_input_dir)? {
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
                    "{}/{}.spv",
                    shaders_output_dir,
                    file_name.rsplitn(2, ".").last().unwrap()
                );

                std::fs::write(&out_path, compilation_artifact.as_binary_u8())?;
            }
        };
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    build_shaders()?;
    // build_vr()?;

    Ok(())
}