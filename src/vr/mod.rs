#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

mod openvr_bindings;
use openvr_bindings as sys;

pub mod defines;
pub use defines::*;

pub mod system;
pub use system::*;

pub mod compositor;
pub use compositor::*;

use nalgebra as na;
use na::{Vector3, Vector4, Matrix3x4};

pub struct System(&'static sys::VR_IVRSystem_FnTable);
pub struct Compositor(&'static sys::VR_IVRCompositor_FnTable);

#[derive(Copy, Clone)]
pub struct Context;

impl Context {
    pub fn system(&self) -> Result<System, InitError> {
        get_fn_table(sys::IVRSystem_Version).map(|t| unsafe { System(&*t) })
    }

    pub fn compositor(&self) -> Result<Compositor, InitError> {
        get_fn_table(sys::IVRCompositor_Version).map(|t| unsafe { Compositor(&*t) })
    }
}

fn get_fn_table<T>(version: &[u8]) -> Result<*const T, InitError> {
    let mut version_str = Vec::from(b"FnTable:".as_ref());
    version_str.extend(version);
    let version_str = version_str.as_ptr() as *const i8;

    let mut error = sys::EVRInitError_VRInitError_None;
    let ptr = unsafe { sys::VR_GetGenericInterface(version_str, &mut error) };

    if error != sys::EVRInitError_VRInitError_None {
        return Err(InitError::Other(error));
    }

    Ok(ptr as *const T)
}

pub fn init() -> Result<Context, InitError> {

    let mut error: sys::EVRInitError = 0;

    unsafe {
        sys::VR_InitInternal(&mut error, sys::EVRApplicationType_VRApplication_Scene);

        if error != 0 {
            return Err(match error {
                sys::EVRInitError_VRInitError_Init_HmdNotFound => InitError::HmdNotFound,
                _ => InitError::Other(error),
            })
        }
    }

    Ok(Context{})
}
