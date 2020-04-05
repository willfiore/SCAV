use super::*;
use ash::vk as vk;
use ash::vk::Handle;

#[derive(Debug)]
pub enum Eye {
    Left,
    Right,
}

impl From<Eye> for sys::EVREye {
    fn from(s: Eye) -> Self {
        use Eye::*;
        match s {
            Left => sys::EVREye_Eye_Left,
            Right => sys::EVREye_Eye_Right,
        }
    }
}

#[derive(Debug)]
pub enum InitError {
    HmdNotFound,
    Other(i32),
}

#[derive(Debug, Copy, Clone)]
pub struct VulkanTextureData {
    pub image: u64,
    pub device: vk::Device,
    pub physical_device: vk::PhysicalDevice,
    pub instance: vk::Instance,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub sample_count: u32,
}

impl From<VulkanTextureData> for sys::VRVulkanTextureData_t {
    fn from(s: VulkanTextureData) -> Self {
        Self {
            m_nImage: s.image,
            m_pDevice: s.device.as_raw() as _,
            m_pPhysicalDevice: s.physical_device.as_raw() as _,
            m_pInstance: s.instance.as_raw() as _,
            m_pQueue: s.queue.as_raw() as _,
            m_nQueueFamilyIndex: s.queue_family_index,
            m_nWidth: s.width,
            m_nHeight: s.height,
            m_nFormat: s.format,
            m_nSampleCount: s.sample_count
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct TextureBounds {
    pub u_min: f32,
    pub u_max: f32,
    pub v_min: f32,
    pub v_max: f32,
}

impl From<TextureBounds> for sys::VRTextureBounds_t {
    fn from(s: TextureBounds) -> Self {
        Self {
            uMin: s.u_min,
            uMax: s.u_max,
            vMin: s.v_min,
            vMax: s.v_max,
        }
    }
}

#[derive(Debug)]
pub struct TrackedDevicePose {
    pub valid: bool,
    pub device_connected: bool,
    pub tracking_result: TrackingResult,
    pub device_to_absolute: Matrix3x4<f32>,
    pub velocity: Vector3<f32>,
    pub angular_velocity: Vector3<f32>,
}

impl From<sys::TrackedDevicePose_t> for TrackedDevicePose {
    fn from(s: sys::TrackedDevicePose_t) -> Self {

        let t: &[f32; 12] = unsafe { std::mem::transmute(&s.mDeviceToAbsoluteTracking.m) };

        TrackedDevicePose {
            valid: s.bPoseIsValid,
            device_connected: s.bDeviceIsConnected,
            device_to_absolute: Matrix3x4::from_row_slice(t),
            velocity: Vector3::from_row_slice(&s.vVelocity.v),
            angular_velocity: Vector3::from_row_slice(&s.vAngularVelocity.v),
            tracking_result: TrackingResult::from(s.eTrackingResult)
        }
    }
}

#[derive(Debug)]
pub enum TrackingResult {
    Uninitialized,
    Running_Ok,
    Running_OutOfRange,
    Calibrating_InProgress,
    Calibrating_OutOfRange,
    Fallback_RotationOnly,
}

impl From<sys::ETrackingResult> for TrackingResult {
    fn from(s: sys::ETrackingResult) -> Self {
        return match s {
            sys::ETrackingResult_TrackingResult_Running_OK => TrackingResult::Running_Ok,
            sys::ETrackingResult_TrackingResult_Running_OutOfRange => TrackingResult::Running_OutOfRange,
            sys::ETrackingResult_TrackingResult_Calibrating_InProgress => TrackingResult::Calibrating_InProgress,
            sys::ETrackingResult_TrackingResult_Calibrating_OutOfRange => TrackingResult::Calibrating_OutOfRange,
            sys::ETrackingResult_TrackingResult_Fallback_RotationOnly => TrackingResult::Fallback_RotationOnly,
            sys::ETrackingResult_TrackingResult_Uninitialized => TrackingResult::Uninitialized,
            _ => TrackingResult::Uninitialized,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TrackedDeviceClass {
    Invalid,
    Hmd,
    Controller,
    GenericTracker,
    TrackingReference,
    DisplayRedirect,
}

impl From<sys::ETrackedDeviceClass> for TrackedDeviceClass {
    fn from(s: sys::ETrackedDeviceClass) -> Self {
        use TrackedDeviceClass::*;
        match s {
            sys::ETrackedDeviceClass_TrackedDeviceClass_HMD => Hmd,
            sys::ETrackedDeviceClass_TrackedDeviceClass_Controller => Controller,
            sys::ETrackedDeviceClass_TrackedDeviceClass_GenericTracker => GenericTracker,
            sys::ETrackedDeviceClass_TrackedDeviceClass_TrackingReference => TrackingReference,
            sys::ETrackedDeviceClass_TrackedDeviceClass_DisplayRedirect => DisplayRedirect,
            _ => TrackedDeviceClass::Invalid
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TrackingUniverseOrigin {
    Seated,
    Standing,
    Raw
}

impl From<TrackingUniverseOrigin> for sys::ETrackingUniverseOrigin {
    fn from(s: TrackingUniverseOrigin) -> Self {
        use TrackingUniverseOrigin::*;
        match s {
            Seated => sys::ETrackingUniverseOrigin_TrackingUniverseSeated,
            Standing => sys::ETrackingUniverseOrigin_TrackingUniverseStanding,
            Raw => sys::ETrackingUniverseOrigin_TrackingUniverseRawAndUncalibrated,
        }
    }
}

pub enum TrackedDevicePropertyString {
    TrackingSystemName,
    SerialNumber
}

impl From<TrackedDevicePropertyString> for sys::ETrackedDeviceProperty {
    fn from(s: TrackedDevicePropertyString) -> Self {
        use TrackedDevicePropertyString::*;
        match s {
            TrackingSystemName => sys::ETrackedDeviceProperty_Prop_TrackingSystemName_String,
            SerialNumber => sys::ETrackedDeviceProperty_Prop_SerialNumber_String,
        }
    }
}
