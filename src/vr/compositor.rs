use super::*;
use std::os::raw::c_void;

impl Compositor {
    pub fn wait_get_poses(&self) -> Vec<TrackedDevicePose> {
        unsafe {
            let x = self.0.WaitGetPoses.unwrap();

            let mut poses: [sys::TrackedDevicePose_t; sys::k_unMaxTrackedDeviceCount as usize]
                = std::mem::MaybeUninit::zeroed().assume_init();

            self.0.WaitGetPoses.unwrap()(
                poses.as_mut_ptr(), sys::k_unMaxTrackedDeviceCount, std::ptr::null_mut(), 0
            );

            poses.iter().map(|&p| {
                TrackedDevicePose::from(p)
            }).collect::<Vec<_>>()
        }
    }

    pub fn submit(&self, eye: Eye, texture: &VulkanTextureData, bounds: &TextureBounds) {
        unsafe {

            let mut raw_texture: sys::VRVulkanTextureData_t = (*texture).into();
            let mut raw_bounds: sys::VRTextureBounds_t = (*bounds).into();

            let mut raw_generic_texture = sys::Texture_t {
                handle: (&mut raw_texture) as *mut _ as *mut c_void,
                eType: sys::ETextureType_TextureType_Vulkan,
                eColorSpace: sys::EColorSpace_ColorSpace_Auto,
            };

            let x = self.0.Submit.unwrap();
            let err = x(eye.into(), &mut raw_generic_texture, &mut raw_bounds, 0);
        }
    }
}

