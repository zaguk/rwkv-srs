// PyO3 0.22's exception macro probes its legacy `gil-refs` feature in the
// destination crate. Rust's check-cfg lint otherwise reports that internal
// probe as an unknown local feature.
#![allow(unexpected_cfgs)]

mod cpu_config;
mod features;
mod id_encoding;
#[macro_use]
mod profile;
mod gpu;
mod model;
pub mod model_weights;
mod ops;
mod portable_simd;
mod py_state;
mod state;
mod tensor_io;
pub mod weights;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyfunction]
fn backend_name() -> &'static str {
    "rust"
}

#[pyfunction]
fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

#[pyfunction]
fn native_api_version() -> u32 {
    33
}

#[pyfunction]
fn claim_cpu_profile(profile: &str) -> PyResult<()> {
    cpu_config::claim_cpu_profile(profile).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn physical_cpu_count() -> usize {
    num_cpus::get_physical().max(1)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    gpu::register_exceptions(module)?;
    module.add_function(wrap_pyfunction!(backend_name, module)?)?;
    module.add_function(wrap_pyfunction!(build_profile, module)?)?;
    module.add_function(wrap_pyfunction!(native_api_version, module)?)?;
    module.add_function(wrap_pyfunction!(claim_cpu_profile, module)?)?;
    module.add_function(wrap_pyfunction!(physical_cpu_count, module)?)?;
    module.add_function(wrap_pyfunction!(gpu::gpu_device_info_py, module)?)?;
    module.add_function(wrap_pyfunction!(features::scale_elapsed_days_py, module)?)?;
    module.add_function(wrap_pyfunction!(
        features::scale_elapsed_days_cumulative_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        features::scale_elapsed_seconds_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        features::scale_elapsed_seconds_cumulative_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(features::scale_duration_py, module)?)?;
    module.add_function(wrap_pyfunction!(features::scale_diff_new_cards_py, module)?)?;
    module.add_function(wrap_pyfunction!(features::scale_diff_reviews_py, module)?)?;
    module.add_function(wrap_pyfunction!(
        features::scale_cum_new_cards_today_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        features::scale_cum_reviews_today_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(features::scale_state_py, module)?)?;
    module.add_function(wrap_pyfunction!(
        features::scale_day_offset_diff_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(features::day_offset_encoding_py, module)?)?;
    module.add_function(wrap_pyfunction!(features::card_feature_columns_py, module)?)?;
    module.add_function(wrap_pyfunction!(ops::forgetting_curve_py, module)?)?;
    module.add_function(wrap_pyfunction!(ops::interp_py, module)?)?;
    module.add_function(wrap_pyfunction!(ops::predict_curve_py, module)?)?;
    module.add_function(wrap_pyfunction!(ops::curve_interval_py, module)?)?;
    module.add_function(wrap_pyfunction!(ops::simd_status_py, module)?)?;
    module.add_function(wrap_pyfunction!(ops::l2_normalize_last_dim_py, module)?)?;
    module.add_function(wrap_pyfunction!(ops::layer_norm_2d_py, module)?)?;
    module.add_function(wrap_pyfunction!(ops::group_norm_2d_py, module)?)?;
    module.add_function(wrap_pyfunction!(ops::single_timestep_py, module)?)?;
    module.add_function(wrap_pyfunction!(model::channel_mixer_forward_py, module)?)?;
    module.add_function(wrap_pyfunction!(model::time_mixer_forward_py, module)?)?;
    module.add_function(wrap_pyfunction!(model::rwkv_layer_forward_py, module)?)?;
    module.add_function(wrap_pyfunction!(model::rwkv_rnn_forward_py, module)?)?;
    module.add_function(wrap_pyfunction!(model::srs_review_py, module)?)?;
    module.add_function(wrap_pyfunction!(model::prediction_probability_py, module)?)?;
    module.add_class::<model::NativeRnn>()?;
    module.add_class::<model::NativeRuntime>()?;
    module.add_class::<model::NativePredictionBatch>()?;
    module.add_class::<model::NativeReviewBatch>()?;
    module.add_class::<py_state::DeterministicState>()?;
    Ok(())
}
