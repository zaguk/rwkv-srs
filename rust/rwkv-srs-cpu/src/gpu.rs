use std::borrow::Cow;
use std::sync::mpsc;

use anyhow::{anyhow, bail, Context, Result};
use half::f16;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use thiserror::Error;
use wgpu::util::DeviceExt;

const PREDICT_STORAGE_BUFFERS: u32 = 10;
const PROCESS_STORAGE_BUFFERS: u32 = 19;
const PREDICT_WORKGROUP_STORAGE_BYTES: u32 = 6_912;
// Keep this sum aligned with the workgroup arrays in
// model/gpu_process_fused_projections.wgsl. The default process path requires
// those fused shaders, so adapter selection must enforce their complete shared
// memory footprint rather than the smaller compatibility shader's footprint.
const PROCESS_WORKGROUP_STORAGE_FLOATS: u32 = 3 * 512 + 64 + 32 + 2_048 + 256;
const PROCESS_WORKGROUP_STORAGE_BYTES: u32 =
    PROCESS_WORKGROUP_STORAGE_FLOATS * std::mem::size_of::<f32>() as u32;
const DEFAULT_RESOURCE_MEMORY_THRESHOLD_PERCENT: u8 = 90;
const DEFAULT_DEVICE_LOSS_MEMORY_THRESHOLD_PERCENT: u8 = 95;

pyo3::create_exception!(
    _native,
    NativeGpuError,
    PyRuntimeError,
    "Internal operational GPU failure. Use rwkv_srs.GpuError from Python."
);
pyo3::create_exception!(
    _native,
    NativeGpuOutOfMemoryError,
    NativeGpuError,
    "Internal GPU allocation failure. Use rwkv_srs.GpuOutOfMemoryError from Python."
);
pyo3::create_exception!(
    _native,
    NativeGpuUnavailableError,
    NativeGpuError,
    "Internal GPU initialization failure. Use rwkv_srs.GpuUnavailableError from Python."
);

const FP16_PROBE_SHADER: &str = r#"
enable f16;

@group(0) @binding(0)
var<storage, read> input_values: array<f16>;

@group(0) @binding(1)
var<storage, read_write> output_values: array<f16>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index < 4u) {
        output_values[index] = input_values[index] * 2.0h + 1.0h;
    }
}
"#;

/// Shared GPU objects used by the predictor and its state cache.
///
/// The instance and adapter are retained for the lifetime of the device. The
/// fields intentionally remain private until the prediction backend is added;
/// this keeps adapter policy in one place.
pub(crate) struct GpuContext {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) info: wgpu::AdapterInfo,
    pub(crate) limits: wgpu::Limits,
    pub(crate) timestamp_queries: bool,
    pub(crate) subgroup_operations: bool,
    pub(crate) shader_f16: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuOperation {
    Predict,
    Process,
}

impl GpuOperation {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "predict" => Ok(Self::Predict),
            "process" => Ok(Self::Process),
            _ => bail!("GPU operation must be 'predict' or 'process'; got {value:?}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Predict => "predict",
            Self::Process => "process",
        }
    }

    fn required_workgroup_storage_bytes(self) -> u32 {
        match self {
            Self::Predict => PREDICT_WORKGROUP_STORAGE_BYTES,
            Self::Process => PROCESS_WORKGROUP_STORAGE_BYTES,
        }
    }

    fn adapter_supported(self, features: wgpu::Features, limits: &wgpu::Limits) -> bool {
        let enough_workgroup_storage =
            limits.max_compute_workgroup_storage_size >= self.required_workgroup_storage_bytes();
        match self {
            Self::Predict => {
                enough_workgroup_storage
                    && features.contains(wgpu::Features::SHADER_F16)
                    && limits.max_storage_buffers_per_shader_stage >= PREDICT_STORAGE_BUFFERS
                    && limits.max_compute_invocations_per_workgroup >= 64
                    && limits.max_compute_workgroup_size_x >= 64
            }
            Self::Process => {
                enough_workgroup_storage
                    && limits.max_storage_buffers_per_shader_stage >= PROCESS_STORAGE_BUFFERS
                    && limits.max_compute_invocations_per_workgroup >= 256
                    && limits.max_compute_workgroup_size_x >= 256
            }
        }
    }
}

#[derive(Debug, Error)]
#[error("GPU out of memory while allocating {label}: {detail}")]
pub(crate) struct GpuOutOfMemory {
    label: String,
    detail: String,
}

pub(crate) fn is_gpu_out_of_memory(error: &anyhow::Error) -> bool {
    error.downcast_ref::<GpuOutOfMemory>().is_some()
}

pub(crate) fn py_gpu_error(error: anyhow::Error) -> PyErr {
    if is_gpu_out_of_memory(&error) {
        NativeGpuOutOfMemoryError::new_err(error.to_string())
    } else {
        NativeGpuError::new_err(error.to_string())
    }
}

pub(crate) fn py_gpu_unavailable_error(error: anyhow::Error) -> PyErr {
    if is_gpu_out_of_memory(&error) {
        NativeGpuOutOfMemoryError::new_err(error.to_string())
    } else {
        NativeGpuUnavailableError::new_err(error.to_string())
    }
}

pub(crate) fn py_gpu_unavailable(message: impl std::fmt::Display) -> PyErr {
    NativeGpuUnavailableError::new_err(message.to_string())
}

pub(crate) fn register_exceptions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "NativeGpuError",
        module.py().get_type_bound::<NativeGpuError>(),
    )?;
    module.add(
        "NativeGpuOutOfMemoryError",
        module.py().get_type_bound::<NativeGpuOutOfMemoryError>(),
    )?;
    module.add(
        "NativeGpuUnavailableError",
        module.py().get_type_bound::<NativeGpuUnavailableError>(),
    )?;
    Ok(())
}

fn adapter_score(info: &wgpu::AdapterInfo) -> i32 {
    use wgpu::DeviceType;

    match info.device_type {
        DeviceType::DiscreteGpu => 400,
        DeviceType::IntegratedGpu => 300,
        DeviceType::VirtualGpu => 200,
        DeviceType::Other => 100,
        DeviceType::Cpu => 0,
    }
}

impl GpuContext {
    pub(crate) fn new(operation: GpuOperation) -> Result<Self> {
        let backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = backends;
        descriptor.memory_budget_thresholds = wgpu::MemoryBudgetThresholds {
            for_resource_creation: Some(DEFAULT_RESOURCE_MEMORY_THRESHOLD_PERCENT),
            for_device_loss: Some(DEFAULT_DEVICE_LOSS_MEMORY_THRESHOLD_PERCENT),
        };
        descriptor = descriptor.with_env();
        let instance = wgpu::Instance::new(descriptor);

        let requested_name = std::env::var("RWKV_SRS_GPU_ADAPTER")
            .ok()
            .or_else(|| std::env::var("WGPU_ADAPTER_NAME").ok())
            .map(|value| value.to_lowercase());

        let mut compatible = pollster::block_on(instance.enumerate_adapters(backends))
            .into_iter()
            .filter_map(|adapter| {
                let info = adapter.get_info();
                let features = adapter.features();
                let limits = adapter.limits();
                let name_matches = requested_name
                    .as_ref()
                    .is_none_or(|requested| info.name.to_lowercase().contains(requested));
                let usable = info.device_type != wgpu::DeviceType::Cpu
                    && operation.adapter_supported(features, &limits)
                    && name_matches;
                usable.then_some((adapter_score(&info), adapter, info, features))
            })
            .collect::<Vec<_>>();

        compatible.sort_by_key(|(score, _, _, _)| *score);
        let (_, adapter, info, features) = compatible.pop().ok_or_else(|| {
            let requested = requested_name
                .as_deref()
                .map_or(String::new(), |name| format!(" matching {name:?}"));
            anyhow!(
                "no hardware GPU adapter{requested} satisfies the {} GPU requirements, \
                 including at least {} bytes of compute workgroup storage",
                operation.name(),
                operation.required_workgroup_storage_bytes(),
            )
        })?;

        let limits = adapter.limits();
        let timestamp_queries = features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let subgroup_operations = features.contains(wgpu::Features::SUBGROUP);
        let shader_f16 = features.contains(wgpu::Features::SHADER_F16);
        let mut required_features = if operation == GpuOperation::Predict {
            wgpu::Features::SHADER_F16
        } else {
            wgpu::Features::empty()
        };
        if timestamp_queries {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if subgroup_operations {
            required_features |= wgpu::Features::SUBGROUP;
        }

        let device_descriptor = wgpu::DeviceDescriptor {
            label: Some("rwkv-srs predictor"),
            required_features,
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&device_descriptor))
            .with_context(|| format!("could not create compute device for {}", info.name))?;

        Ok(Self {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            info,
            limits,
            timestamp_queries,
            subgroup_operations,
            shader_f16,
        })
    }

    pub(crate) fn create_buffer(
        &self,
        descriptor: &wgpu::BufferDescriptor<'_>,
    ) -> Result<wgpu::Buffer> {
        self.capture_resource_creation(descriptor.label.unwrap_or("unnamed GPU buffer"), || {
            self.device.create_buffer(descriptor)
        })
    }

    pub(crate) fn create_buffer_init(
        &self,
        descriptor: &wgpu::util::BufferInitDescriptor<'_>,
    ) -> Result<wgpu::Buffer> {
        self.capture_resource_creation(
            descriptor.label.unwrap_or("unnamed initialized GPU buffer"),
            || self.device.create_buffer_init(descriptor),
        )
    }

    pub(crate) fn write_buffer(
        &self,
        label: &str,
        buffer: &wgpu::Buffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        self.capture_resource_creation(label, || self.queue.write_buffer(buffer, offset, bytes))
    }

    fn capture_resource_creation<T>(&self, label: &str, create: impl FnOnce() -> T) -> Result<T> {
        let out_of_memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let resource = create();
        let out_of_memory_error = pollster::block_on(out_of_memory.pop());
        if let Some(error) = out_of_memory_error {
            return Err(GpuOutOfMemory {
                label: label.to_owned(),
                detail: error.to_string(),
            }
            .into());
        }
        Ok(resource)
    }

    fn fp16_probe(&self) -> Result<[f32; 4]> {
        let input = [
            f16::from_f32(-2.0),
            f16::from_f32(-0.5),
            f16::from_f32(0.25),
            f16::from_f32(3.0),
        ];
        let input_buffer = self.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rwkv-srs fp16 probe input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        })?;
        let output_buffer = self.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rwkv-srs fp16 probe output"),
            size: std::mem::size_of_val(&input) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })?;

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rwkv-srs fp16 probe"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(FP16_PROBE_SHADER)),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rwkv-srs fp16 probe"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rwkv-srs fp16 probe"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rwkv-srs fp16 probe"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rwkv-srs fp16 probe"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit([encoder.finish()]);

        let (sender, receiver) = mpsc::sync_channel(1);
        wgpu::util::DownloadBuffer::read_buffer(
            &self.device,
            &self.queue,
            &output_buffer.slice(..),
            move |result| {
                let _ = sender.send(result.map(|buffer| buffer.to_vec()));
            },
        );
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("FP16 compute probe did not complete")?;
        let bytes = receiver
            .recv()
            .context("FP16 compute probe readback callback was not invoked")?
            .context("FP16 compute probe readback failed")?;
        let values = bytemuck::try_cast_slice::<u8, f16>(&bytes).map_err(|error| {
            anyhow!("FP16 compute probe returned a malformed buffer: {error:?}")
        })?;
        if values.len() != 4 {
            bail!(
                "FP16 compute probe returned {} values, expected 4",
                values.len()
            );
        }
        Ok(std::array::from_fn(|index| values[index].to_f32()))
    }
}

#[pyfunction(name = "gpu_device_info", signature = (operation="predict"))]
#[allow(clippy::useless_conversion)]
pub(crate) fn gpu_device_info_py(py: Python<'_>, operation: &str) -> PyResult<Py<PyDict>> {
    let operation =
        GpuOperation::parse(operation).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let context = GpuContext::new(operation).map_err(py_gpu_unavailable_error)?;
    let probe = if operation == GpuOperation::Predict {
        Some(context.fp16_probe().map_err(py_gpu_unavailable_error)?)
    } else {
        None
    };
    let dict = PyDict::new_bound(py);
    dict.set_item("operation", operation.name())?;
    dict.set_item("name", &context.info.name)?;
    dict.set_item("backend", format!("{:?}", context.info.backend))?;
    dict.set_item("device_type", format!("{:?}", context.info.device_type))?;
    dict.set_item("vendor_id", context.info.vendor)?;
    dict.set_item("device_id", context.info.device)?;
    dict.set_item("driver", &context.info.driver)?;
    dict.set_item("driver_info", &context.info.driver_info)?;
    dict.set_item("pci_bus_id", &context.info.device_pci_bus_id)?;
    dict.set_item("subgroup_min_size", context.info.subgroup_min_size)?;
    dict.set_item("subgroup_max_size", context.info.subgroup_max_size)?;
    dict.set_item("shader_f16", context.shader_f16)?;
    dict.set_item("timestamp_queries", context.timestamp_queries)?;
    dict.set_item("subgroup_operations", context.subgroup_operations)?;
    dict.set_item(
        "max_storage_buffer_binding_size",
        context.limits.max_storage_buffer_binding_size,
    )?;
    dict.set_item("max_buffer_size", context.limits.max_buffer_size)?;
    dict.set_item(
        "max_compute_workgroup_storage_size",
        context.limits.max_compute_workgroup_storage_size,
    )?;
    dict.set_item(
        "max_storage_buffers_per_shader_stage",
        context.limits.max_storage_buffers_per_shader_stage,
    )?;
    dict.set_item(
        "max_compute_invocations_per_workgroup",
        context.limits.max_compute_invocations_per_workgroup,
    )?;
    dict.set_item(
        "max_compute_workgroup_size_x",
        context.limits.max_compute_workgroup_size_x,
    )?;
    dict.set_item(
        "max_compute_workgroups_per_dimension",
        context.limits.max_compute_workgroups_per_dimension,
    )?;
    dict.set_item("fp16_probe", probe.map(|values| values.to_vec()))?;
    dict.set_item(
        "resource_memory_threshold_percent",
        DEFAULT_RESOURCE_MEMORY_THRESHOLD_PERCENT,
    )?;
    dict.set_item(
        "device_loss_memory_threshold_percent",
        DEFAULT_DEVICE_LOSS_MEMORY_THRESHOLD_PERCENT,
    )?;
    Ok(dict.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_preference_keeps_cpu_last() {
        let mut info = wgpu::AdapterInfo::new(wgpu::DeviceType::Cpu, wgpu::Backend::Vulkan);
        assert_eq!(adapter_score(&info), 0);
        info.device_type = wgpu::DeviceType::IntegratedGpu;
        assert_eq!(adapter_score(&info), 300);
        info.device_type = wgpu::DeviceType::DiscreteGpu;
        assert_eq!(adapter_score(&info), 400);
    }

    #[test]
    fn operation_capabilities_do_not_require_fp16_for_processing() {
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = PROCESS_STORAGE_BUFFERS;
        limits.max_compute_invocations_per_workgroup = 256;
        limits.max_compute_workgroup_size_x = 256;
        limits.max_compute_workgroup_storage_size = PROCESS_WORKGROUP_STORAGE_BYTES;

        assert!(GpuOperation::Process.adapter_supported(wgpu::Features::empty(), &limits));
        assert!(!GpuOperation::Predict.adapter_supported(wgpu::Features::empty(), &limits));
        assert!(GpuOperation::Predict.adapter_supported(wgpu::Features::SHADER_F16, &limits));
    }

    #[test]
    fn operation_capabilities_enforce_storage_binding_counts() {
        let mut limits = wgpu::Limits::default();
        limits.max_compute_invocations_per_workgroup = 256;
        limits.max_compute_workgroup_size_x = 256;
        limits.max_compute_workgroup_storage_size = PROCESS_WORKGROUP_STORAGE_BYTES;
        limits.max_storage_buffers_per_shader_stage = PREDICT_STORAGE_BUFFERS;
        assert!(GpuOperation::Predict.adapter_supported(wgpu::Features::SHADER_F16, &limits));
        assert!(!GpuOperation::Process.adapter_supported(wgpu::Features::SHADER_F16, &limits));
    }

    #[test]
    fn operation_capabilities_enforce_fused_process_workgroup_storage() {
        assert_eq!(PROCESS_WORKGROUP_STORAGE_BYTES, 15_744);
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = PROCESS_STORAGE_BUFFERS;
        limits.max_compute_invocations_per_workgroup = 256;
        limits.max_compute_workgroup_size_x = 256;
        limits.max_compute_workgroup_storage_size = PREDICT_WORKGROUP_STORAGE_BYTES;

        assert!(GpuOperation::Predict.adapter_supported(wgpu::Features::SHADER_F16, &limits));
        assert!(!GpuOperation::Process.adapter_supported(wgpu::Features::empty(), &limits));

        limits.max_compute_workgroup_storage_size = PROCESS_WORKGROUP_STORAGE_BYTES - 1;
        assert!(!GpuOperation::Process.adapter_supported(wgpu::Features::empty(), &limits));
        limits.max_compute_workgroup_storage_size = PROCESS_WORKGROUP_STORAGE_BYTES;
        assert!(GpuOperation::Process.adapter_supported(wgpu::Features::empty(), &limits));
    }

    #[test]
    fn out_of_memory_errors_remain_downcastable_for_retry() {
        let error: anyhow::Error = GpuOutOfMemory {
            label: "test".to_owned(),
            detail: "budget exhausted".to_owned(),
        }
        .into();
        assert!(is_gpu_out_of_memory(&error));
        assert!(error.to_string().contains("budget exhausted"));
    }
}
