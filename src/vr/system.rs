use super::*;
use std::os::raw::c_char;
use std::ffi::CStr;

impl System {
    pub fn recommended_render_target_size(&self) -> (u32, u32) {
        unsafe {
            let mut result: (u32, u32) = (0, 0);
            self.0.GetRecommendedRenderTargetSize.unwrap()(&mut result.0, &mut result.1);
            result
        }
    }

    pub fn tracked_device_class(&self, index: usize) -> TrackedDeviceClass {
        let raw_class = unsafe { self.0.GetTrackedDeviceClass.unwrap()(index as u32) };
        TrackedDeviceClass::from(raw_class)
    }

    pub fn tracked_device_property_string(&self, index: usize, property: TrackedDevicePropertyString) -> String {
        unsafe {
            const buffer_size: usize = 1024;
            let mut result: [c_char; buffer_size] = std::mem::MaybeUninit::uninit().assume_init();

            self.0.GetStringTrackedDeviceProperty.unwrap()
                (index as u32, property.into(), result.as_mut_ptr(), buffer_size as u32, std::ptr::null_mut());

            CStr::from_ptr(result.as_ptr())
                .to_str().unwrap()
                .to_owned()
        }
    }

    pub fn device_to_absolute_tracking_pose(&self,
                                            origin: TrackingUniverseOrigin,
                                            predicted_seconds_to_photons_from_now: f32)
        -> Vec<TrackedDevicePose>
    {
        unsafe {
            let mut poses: [sys::TrackedDevicePose_t; sys::k_unMaxTrackedDeviceCount as usize]
                = std::mem::MaybeUninit::zeroed().assume_init();

            self.0.GetDeviceToAbsoluteTrackingPose.unwrap()
                (origin.into(), predicted_seconds_to_photons_from_now, poses.as_mut_ptr(), sys::k_unMaxTrackedDeviceCount);

            poses.iter().map(|&p| {
                TrackedDevicePose::from(p)
            }).collect::<Vec<_>>()
        }
    }
}

