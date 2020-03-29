use crate::vr::openvr_bindings::*;

use super::*;

pub type TrackedDeviceIndex = TrackedDeviceIndex_t;

#[derive(Debug, Clone, Copy)]
pub enum TrackedDeviceClass {
    Invalid,
    Hmd,
    Controller,
    GenericTracker,
    TrackingReference,
    DisplayRedirect,
}

impl System {
    pub fn recommended_render_target_size(&self) -> (u32, u32) {
        unsafe {
            let mut result: (u32, u32) = (0, 0);
            self.0.GetRecommendedRenderTargetSize.unwrap()(&mut result.0, &mut result.1);
            result
        }
    }

    pub fn tracked_device_class(&self, index: TrackedDeviceIndex) -> TrackedDeviceClass {
        return match unsafe { self.0.GetTrackedDeviceClass.unwrap()(index) } {
            ETrackedDeviceClass_TrackedDeviceClass_HMD => TrackedDeviceClass::Hmd,
            ETrackedDeviceClass_TrackedDeviceClass_Controller => TrackedDeviceClass::Controller,
            ETrackedDeviceClass_TrackedDeviceClass_GenericTracker => TrackedDeviceClass::GenericTracker,
            ETrackedDeviceClass_TrackedDeviceClass_TrackingReference => TrackedDeviceClass::TrackingReference,
            ETrackedDeviceClass_TrackedDeviceClass_DisplayRedirect => TrackedDeviceClass::DisplayRedirect,
            _ => TrackedDeviceClass::Invalid
        }
    }
}
