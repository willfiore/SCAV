#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

mod openvr_bindings;
use openvr_bindings::*;

pub struct System(&'static VR_IVRSystem_FnTable);
pub struct Compositor(&'static VR_IVRCompositor_FnTable);

pub mod system;

impl Compositor {
    pub fn wait_get_poses(&self) {
        unsafe {
            let x = self.0.WaitGetPoses.unwrap();

            let y = "Hello";

            let x = Vec::from(y);

            let mut poses: [TrackedDevicePose_t; 16 as usize]
                = std::mem::MaybeUninit::uninit().assume_init();

            self.0.WaitGetPoses.unwrap()(
                poses.as_mut_ptr(), k_unMaxTrackedDeviceCount, std::ptr::null_mut(), 0
            );
        }
    }
}

pub struct Entry {}

impl Entry {
    pub fn system(&self) -> Result<System, InitError> {
        load_fn_table(IVRSystem_Version).map(|t| unsafe { System(&*t) })
    }

    pub fn compositor(&self) -> Result<Compositor, InitError> {
        load_fn_table(IVRCompositor_Version).map(|t| unsafe { Compositor(&*t) })
    }
}


#[derive(Debug)]
pub enum InitError {
    HmdNotFound,
    Other(i32),
}

fn load_fn_table<T>(version: &[u8]) -> Result<*const T, InitError> {
    let mut version_str = Vec::from(b"FnTable:".as_ref());
    version_str.extend(version);
    let version_str = version_str.as_ptr() as *const i8;

    let mut error = EVRInitError_VRInitError_None;
    let ptr = unsafe { VR_GetGenericInterface(version_str, &mut error) };

    if error != EVRInitError_VRInitError_None {
        return Err(InitError::Other(error));
    }

    Ok(ptr as *const T)
}

pub fn init() -> Result<Entry, InitError> {

    let mut error: EVRInitError = 0;

    unsafe {
        VR_InitInternal(&mut error, EVRApplicationType_VRApplication_Other);

        if error != 0 {
            return Err(match error {
                EVRInitError_VRInitError_Init_HmdNotFound => InitError::HmdNotFound,
                _ => InitError::Other(error),
            })
        }
    }

    Ok(Entry {})
}
