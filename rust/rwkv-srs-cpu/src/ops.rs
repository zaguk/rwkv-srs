#![allow(clippy::useless_conversion)]

use std::{
    cell::{Cell, RefCell},
    sync::atomic::{AtomicUsize, Ordering},
};

use candle_core::{bail, Device, Result, Storage, Tensor, D};
use candle_nn::ops as nn_ops;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::cpu_config::*;
use crate::portable_simd::{
    cpu_simd_kernel, cpu_simd_kernel_name, custom_avx2_fma_available, pulp_arch_name,
    pulp_f32_lanes, pulp_simd_available, pulp_timestep_output_head_algebraic,
    pulp_timestep_output_row, pulp_timestep_state_output_row, CpuSimdKernel, RecurrenceOptions,
};
use crate::profile::{ProfileTimer, SingleTimestepProfile};
use crate::tensor_io::{
    tensor_from_1d, tensor_from_2d, tensor_from_3d, tensor_from_4d, tensor_to_vec4, Tensor3List,
    Tensor4List,
};
use crate::weights::{NUM_CURVES, NUM_POINTS};

#[cfg(target_arch = "x86")]
use std::arch::x86 as x86_arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as x86_arch;

const PROBABILITY_EPS: f64 = 1e-5;
const PROBABILITY_SCALE: f64 = 1.0 - 2.0 * PROBABILITY_EPS;
const S_POINT_SPREAD: f32 = 18.5;
const S_MAX: f32 = 22.0;
const POINT_SPREAD: f32 = 18.5;
const MAX_E: f32 = 21.0;
const L2_NORM_MIN: f32 = 1e-12;
const TIME_MIXER_ZERO_STATE_STACK_ROW_CAP: usize = 64;
const ALGEBRAIC_WKV_STACK_HEAD_SIZE_CAP: usize = 64;
static LIGHTNING_RECURRENCE_APPROXIMATIONS_ACTIVE: AtomicUsize = AtomicUsize::new(0);
struct LightningRecurrenceActivityGuard;

impl LightningRecurrenceActivityGuard {
    fn enter() -> Self {
        LIGHTNING_RECURRENCE_APPROXIMATIONS_ACTIVE.fetch_add(1, Ordering::AcqRel);
        Self
    }
}

impl Drop for LightningRecurrenceActivityGuard {
    fn drop(&mut self) {
        LIGHTNING_RECURRENCE_APPROXIMATIONS_ACTIVE.fetch_sub(1, Ordering::AcqRel);
    }
}

thread_local! {
    static FAST_LAYER_NORM_ENABLED: Cell<bool> = const { Cell::new(false) };
    static TIME_MIXER_MIDDLE_SCRATCH_POOL: RefCell<TimeMixerMiddleScratchPool> =
        RefCell::new(TimeMixerMiddleScratchPool::default());
}

#[derive(Default)]
struct TimeMixerMiddleScratchPool {
    k_deformed_values: Vec<f32>,
    k_values: Vec<f32>,
    v_values: Vec<f32>,
    out_values: Vec<f32>,
}

type TimestepOutputKernel = CpuSimdKernel;

pub(crate) fn with_fast_layer_norm<T>(enabled: bool, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !enabled {
        return f();
    }
    FAST_LAYER_NORM_ENABLED.with(|flag| {
        let previous = flag.replace(true);
        let result = f();
        flag.set(previous);
        result
    })
}

pub(crate) fn with_lightning_recurrence_approximations<T>(
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let _activity = LightningRecurrenceActivityGuard::enter();
    f()
}

fn lightning_skip_deformation_enabled() -> bool {
    LIGHTNING_RECURRENCE_APPROXIMATIONS_ACTIVE.load(Ordering::Relaxed) > 0
        && lightning_skip_deformation_configured()
}

fn lightning_deformation_left_threshold() -> Option<f32> {
    if LIGHTNING_RECURRENCE_APPROXIMATIONS_ACTIVE.load(Ordering::Relaxed) == 0 {
        return None;
    }
    lightning_deformation_left_threshold_configured()
}

fn lightning_algebraic_wkv_enabled() -> bool {
    LIGHTNING_RECURRENCE_APPROXIMATIONS_ACTIVE.load(Ordering::Relaxed) > 0
        && lightning_deformation_left_threshold().is_none()
        && lightning_algebraic_wkv_configured()
}

fn timestep_output_kernel() -> TimestepOutputKernel {
    cpu_simd_kernel(!simd_disabled())
}

fn timestep_state_output_kernel() -> TimestepOutputKernel {
    if !state_simd_enabled() {
        return TimestepOutputKernel::Scalar;
    }
    timestep_output_kernel()
}

fn timestep_kernel_name(kernel: TimestepOutputKernel) -> &'static str {
    cpu_simd_kernel_name(kernel)
}

fn record_timestep_kernel_rows(
    profile: &mut SingleTimestepProfile,
    kernel: TimestepOutputKernel,
    row_count: usize,
    returns_state: bool,
) {
    match (kernel, returns_state) {
        (TimestepOutputKernel::Scalar, true) => {
            profile.state_output_scalar_rows += row_count;
        }
        (TimestepOutputKernel::Scalar, false) => profile.output_scalar_rows += row_count,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        (TimestepOutputKernel::Avx2Fma, true) => {
            profile.state_output_avx2_fma_rows += row_count;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        (TimestepOutputKernel::Avx2Fma, false) => {
            profile.output_avx2_fma_rows += row_count;
        }
        (TimestepOutputKernel::Pulp, true) => profile.state_output_pulp_rows += row_count,
        (TimestepOutputKernel::Pulp, false) => profile.output_pulp_rows += row_count,
    }
}

fn avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return std::is_x86_feature_detected!("avx2");
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

fn fma_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return std::is_x86_feature_detected!("fma");
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[allow(clippy::too_many_arguments)]
fn timestep_output_head(
    kernel: TimestepOutputKernel,
    state_head: &[f32],
    out_head: &mut [f32],
    r_vector: &[f32],
    k_vector: &[f32],
    v_vector: &[f32],
    w_vector: &[f32],
    a_vector: &[f32],
    k_deformed_vector: &[f32],
) {
    let head_size = out_head.len();
    debug_assert_eq!(state_head.len(), head_size * head_size);
    for vector in [
        r_vector,
        k_vector,
        v_vector,
        w_vector,
        a_vector,
        k_deformed_vector,
    ] {
        debug_assert_eq!(vector.len(), head_size);
    }

    if lightning_algebraic_wkv_enabled() {
        match kernel {
            TimestepOutputKernel::Scalar => timestep_output_head_algebraic_scalar(
                state_head,
                out_head,
                r_vector,
                k_vector,
                v_vector,
                w_vector,
                a_vector,
                k_deformed_vector,
            ),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            TimestepOutputKernel::Avx2Fma => unsafe {
                timestep_output_head_algebraic_avx2_fma(
                    state_head,
                    out_head,
                    r_vector,
                    k_vector,
                    v_vector,
                    w_vector,
                    a_vector,
                    k_deformed_vector,
                )
            },
            TimestepOutputKernel::Pulp => pulp_timestep_output_head_algebraic(
                state_head,
                out_head,
                r_vector,
                k_vector,
                v_vector,
                w_vector,
                a_vector,
                k_deformed_vector,
                lightning_skip_deformation_enabled(),
                recurrence_reuse_deformed_key_values_enabled(),
            ),
        }
        return;
    }

    for row_index in 0..head_size {
        let row_base = row_index * head_size;
        out_head[row_index] = timestep_output_row(
            kernel,
            &state_head[row_base..row_base + head_size],
            r_vector,
            k_vector,
            v_vector[row_index],
            w_vector,
            a_vector,
            k_deformed_vector,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn timestep_output_head_algebraic_scalar(
    state_head: &[f32],
    out_head: &mut [f32],
    r_vector: &[f32],
    k_vector: &[f32],
    v_vector: &[f32],
    w_vector: &[f32],
    a_vector: &[f32],
    k_deformed_vector: &[f32],
) {
    let head_size = out_head.len();
    let skip_deformation = lightning_skip_deformation_enabled();
    let reuse_deformed_key = recurrence_reuse_deformed_key_values_enabled();
    let mut key_readout = 0.0f32;
    let mut deformation_readout = 0.0f32;
    for index in 0..head_size {
        key_readout += k_vector[index] * r_vector[index];
        if !skip_deformation {
            let deformed_key = if reuse_deformed_key {
                k_vector[index]
            } else {
                a_vector[index] * k_deformed_vector[index]
            };
            deformation_readout += deformed_key * r_vector[index];
        }
    }

    let mut query_stack = [0.0f32; ALGEBRAIC_WKV_STACK_HEAD_SIZE_CAP];
    let mut query_heap = Vec::new();
    let query = if head_size <= query_stack.len() {
        &mut query_stack[..head_size]
    } else {
        query_heap.resize(head_size, 0.0);
        query_heap.as_mut_slice()
    };
    for index in 0..head_size {
        query[index] =
            w_vector[index] * r_vector[index] - deformation_readout * k_deformed_vector[index];
    }
    for row_index in 0..head_size {
        let row_base = row_index * head_size;
        let state_row = &state_head[row_base..row_base + head_size];
        let mut output = 0.0f32;
        for index in 0..head_size {
            output += state_row[index] * query[index];
        }
        out_head[row_index] = output + v_vector[row_index] * key_readout;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn timestep_output_head_algebraic_avx2_fma(
    state_head: &[f32],
    out_head: &mut [f32],
    r_vector: &[f32],
    k_vector: &[f32],
    v_vector: &[f32],
    w_vector: &[f32],
    a_vector: &[f32],
    k_deformed_vector: &[f32],
) {
    let head_size = out_head.len();
    let skip_deformation = lightning_skip_deformation_enabled();
    let reuse_deformed_key = recurrence_reuse_deformed_key_values_enabled();
    let mut key_readout_acc = x86_arch::_mm256_setzero_ps();
    let mut deformation_readout_acc = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= head_size {
        let r = x86_arch::_mm256_loadu_ps(r_vector.as_ptr().add(index));
        let k = x86_arch::_mm256_loadu_ps(k_vector.as_ptr().add(index));
        key_readout_acc = x86_arch::_mm256_fmadd_ps(k, r, key_readout_acc);
        if !skip_deformation {
            let deformed_key = if reuse_deformed_key {
                k
            } else {
                let a = x86_arch::_mm256_loadu_ps(a_vector.as_ptr().add(index));
                let k_deformed = x86_arch::_mm256_loadu_ps(k_deformed_vector.as_ptr().add(index));
                x86_arch::_mm256_mul_ps(a, k_deformed)
            };
            deformation_readout_acc =
                x86_arch::_mm256_fmadd_ps(deformed_key, r, deformation_readout_acc);
        }
        index += 8;
    }

    let use_direct_sum =
        recurrence_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut key_readout = horizontal_sum_avx2_recurrence(key_readout_acc, use_direct_sum);
    let mut deformation_readout =
        horizontal_sum_avx2_recurrence(deformation_readout_acc, use_direct_sum);
    while index < head_size {
        key_readout += k_vector[index] * r_vector[index];
        if !skip_deformation {
            let deformed_key = if reuse_deformed_key {
                k_vector[index]
            } else {
                a_vector[index] * k_deformed_vector[index]
            };
            deformation_readout += deformed_key * r_vector[index];
        }
        index += 1;
    }

    let mut query_stack = [0.0f32; ALGEBRAIC_WKV_STACK_HEAD_SIZE_CAP];
    let mut query_heap = Vec::new();
    let query = if head_size <= query_stack.len() {
        &mut query_stack[..head_size]
    } else {
        query_heap.resize(head_size, 0.0);
        query_heap.as_mut_slice()
    };
    let deformation = x86_arch::_mm256_set1_ps(deformation_readout);
    index = 0;
    while index + 8 <= head_size {
        let r = x86_arch::_mm256_loadu_ps(r_vector.as_ptr().add(index));
        let w = x86_arch::_mm256_loadu_ps(w_vector.as_ptr().add(index));
        let k_deformed = x86_arch::_mm256_loadu_ps(k_deformed_vector.as_ptr().add(index));
        let weighted_r = x86_arch::_mm256_mul_ps(w, r);
        let value = x86_arch::_mm256_fnmadd_ps(deformation, k_deformed, weighted_r);
        x86_arch::_mm256_storeu_ps(query.as_mut_ptr().add(index), value);
        index += 8;
    }
    while index < head_size {
        query[index] =
            w_vector[index] * r_vector[index] - deformation_readout * k_deformed_vector[index];
        index += 1;
    }

    for row_index in 0..head_size {
        let state_row = state_head.as_ptr().add(row_index * head_size);
        let mut output_acc = x86_arch::_mm256_setzero_ps();
        index = 0;
        while index + 8 <= head_size {
            let state = x86_arch::_mm256_loadu_ps(state_row.add(index));
            let query_values = x86_arch::_mm256_loadu_ps(query.as_ptr().add(index));
            output_acc = x86_arch::_mm256_fmadd_ps(state, query_values, output_acc);
            index += 8;
        }
        let mut output = horizontal_sum_avx2_recurrence(output_acc, use_direct_sum);
        while index < head_size {
            output += *state_row.add(index) * query[index];
            index += 1;
        }
        out_head[row_index] = output + v_vector[row_index] * key_readout;
    }
}

#[allow(clippy::too_many_arguments)]
fn timestep_output_row(
    kernel: TimestepOutputKernel,
    state_row: &[f32],
    r_vector: &[f32],
    k_vector: &[f32],
    value: f32,
    w_vector: &[f32],
    a_vector: &[f32],
    k_deformed_vector: &[f32],
) -> f32 {
    if matches!(kernel, TimestepOutputKernel::Pulp) {
        return pulp_timestep_output_row(
            state_row,
            r_vector,
            k_vector,
            value,
            w_vector,
            a_vector,
            k_deformed_vector,
            RecurrenceOptions {
                force_skip_deformation: lightning_skip_deformation_enabled(),
                deformation_threshold: lightning_deformation_left_threshold(),
                reuse_deformed_key: recurrence_reuse_deformed_key_values_enabled(),
            },
        );
    }
    if lightning_skip_deformation_enabled() {
        return match kernel {
            TimestepOutputKernel::Scalar => timestep_output_row_skip_deformation_scalar(
                state_row, r_vector, k_vector, value, w_vector,
            ),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            TimestepOutputKernel::Avx2Fma => unsafe {
                timestep_output_row_skip_deformation_avx2_fma(
                    state_row, r_vector, k_vector, value, w_vector,
                )
            },
            TimestepOutputKernel::Pulp => unreachable!("Pulp returned above"),
        };
    }

    match kernel {
        TimestepOutputKernel::Scalar => timestep_output_row_scalar(
            state_row,
            r_vector,
            k_vector,
            value,
            w_vector,
            a_vector,
            k_deformed_vector,
        ),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        TimestepOutputKernel::Avx2Fma => unsafe {
            timestep_output_row_avx2_fma(
                state_row,
                r_vector,
                k_vector,
                value,
                w_vector,
                a_vector,
                k_deformed_vector,
            )
        },
        TimestepOutputKernel::Pulp => unreachable!("Pulp returned above"),
    }
}

fn timestep_output_row_skip_deformation_scalar(
    state_row: &[f32],
    r_vector: &[f32],
    k_vector: &[f32],
    value: f32,
    w_vector: &[f32],
) -> f32 {
    debug_assert_eq!(state_row.len(), r_vector.len());
    debug_assert_eq!(state_row.len(), k_vector.len());
    debug_assert_eq!(state_row.len(), w_vector.len());

    let mut output = 0.0f32;
    for col_index in 0..state_row.len() {
        let next = state_row[col_index] * w_vector[col_index] + value * k_vector[col_index];
        output += next * r_vector[col_index];
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn timestep_output_row_scalar(
    state_row: &[f32],
    r_vector: &[f32],
    k_vector: &[f32],
    value: f32,
    w_vector: &[f32],
    a_vector: &[f32],
    k_deformed_vector: &[f32],
) -> f32 {
    debug_assert_eq!(state_row.len(), r_vector.len());
    debug_assert_eq!(state_row.len(), k_vector.len());
    debug_assert_eq!(state_row.len(), w_vector.len());
    debug_assert_eq!(state_row.len(), a_vector.len());
    debug_assert_eq!(state_row.len(), k_deformed_vector.len());

    let mut deformation_left = 0.0f32;
    for col_index in 0..state_row.len() {
        deformation_left += state_row[col_index] * k_deformed_vector[col_index];
    }
    let skip_deformation = lightning_deformation_left_threshold()
        .map(|threshold| deformation_left.abs() <= threshold)
        .unwrap_or(false);
    let reuse_deformed_key = recurrence_reuse_deformed_key_values_enabled();

    let mut output = 0.0f32;
    for col_index in 0..state_row.len() {
        let next = if skip_deformation {
            state_row[col_index] * w_vector[col_index] + value * k_vector[col_index]
        } else {
            let deformed_key = if reuse_deformed_key {
                k_vector[col_index]
            } else {
                a_vector[col_index] * k_deformed_vector[col_index]
            };
            state_row[col_index] * w_vector[col_index] - deformation_left * deformed_key
                + value * k_vector[col_index]
        };
        output += next * r_vector[col_index];
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn timestep_state_output_row(
    kernel: TimestepOutputKernel,
    state_row: &[f32],
    next_state_row: &mut [f32],
    r_vector: &[f32],
    k_vector: &[f32],
    value: f32,
    w_vector: &[f32],
    a_vector: &[f32],
    k_deformed_vector: &[f32],
) -> f32 {
    if matches!(kernel, TimestepOutputKernel::Pulp) {
        return pulp_timestep_state_output_row(
            state_row,
            next_state_row,
            r_vector,
            k_vector,
            value,
            w_vector,
            a_vector,
            k_deformed_vector,
            RecurrenceOptions {
                force_skip_deformation: lightning_skip_deformation_enabled(),
                deformation_threshold: lightning_deformation_left_threshold(),
                reuse_deformed_key: recurrence_reuse_deformed_key_values_enabled(),
            },
        );
    }
    if lightning_skip_deformation_enabled() {
        return match kernel {
            TimestepOutputKernel::Scalar => timestep_state_output_row_skip_deformation_scalar(
                state_row,
                next_state_row,
                r_vector,
                k_vector,
                value,
                w_vector,
            ),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            TimestepOutputKernel::Avx2Fma => unsafe {
                timestep_state_output_row_skip_deformation_avx2_fma(
                    state_row,
                    next_state_row,
                    r_vector,
                    k_vector,
                    value,
                    w_vector,
                )
            },
            TimestepOutputKernel::Pulp => unreachable!("Pulp returned above"),
        };
    }

    match kernel {
        TimestepOutputKernel::Scalar => timestep_state_output_row_scalar(
            state_row,
            next_state_row,
            r_vector,
            k_vector,
            value,
            w_vector,
            a_vector,
            k_deformed_vector,
        ),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        TimestepOutputKernel::Avx2Fma => unsafe {
            timestep_state_output_row_avx2_fma(
                state_row,
                next_state_row,
                r_vector,
                k_vector,
                value,
                w_vector,
                a_vector,
                k_deformed_vector,
            )
        },
        TimestepOutputKernel::Pulp => unreachable!("Pulp returned above"),
    }
}

fn timestep_state_output_row_skip_deformation_scalar(
    state_row: &[f32],
    next_state_row: &mut [f32],
    r_vector: &[f32],
    k_vector: &[f32],
    value: f32,
    w_vector: &[f32],
) -> f32 {
    debug_assert_eq!(state_row.len(), next_state_row.len());
    debug_assert_eq!(state_row.len(), r_vector.len());
    debug_assert_eq!(state_row.len(), k_vector.len());
    debug_assert_eq!(state_row.len(), w_vector.len());

    let mut output = 0.0f32;
    for col_index in 0..state_row.len() {
        let next = state_row[col_index] * w_vector[col_index] + value * k_vector[col_index];
        next_state_row[col_index] = next;
        output += next * r_vector[col_index];
    }
    output
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn timestep_output_row_skip_deformation_avx2_fma(
    state_row: &[f32],
    r_vector: &[f32],
    k_vector: &[f32],
    value: f32,
    w_vector: &[f32],
) -> f32 {
    debug_assert_eq!(state_row.len(), r_vector.len());
    debug_assert_eq!(state_row.len(), k_vector.len());
    debug_assert_eq!(state_row.len(), w_vector.len());

    let value_scalar = value;
    let value = x86_arch::_mm256_set1_ps(value_scalar);
    let mut output_acc = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= state_row.len() {
        let state = x86_arch::_mm256_loadu_ps(state_row.as_ptr().add(index));
        let r = x86_arch::_mm256_loadu_ps(r_vector.as_ptr().add(index));
        let k = x86_arch::_mm256_loadu_ps(k_vector.as_ptr().add(index));
        let w = x86_arch::_mm256_loadu_ps(w_vector.as_ptr().add(index));

        let state_decay = x86_arch::_mm256_mul_ps(state, w);
        let next = x86_arch::_mm256_fmadd_ps(value, k, state_decay);
        output_acc = x86_arch::_mm256_fmadd_ps(next, r, output_acc);
        index += 8;
    }

    let use_direct_sum =
        recurrence_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut output = horizontal_sum_avx2_recurrence(output_acc, use_direct_sum);
    while index < state_row.len() {
        let next = state_row[index] * w_vector[index] + value_scalar * k_vector[index];
        output += next * r_vector[index];
        index += 1;
    }
    output
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn timestep_state_output_row_skip_deformation_avx2_fma(
    state_row: &[f32],
    next_state_row: &mut [f32],
    r_vector: &[f32],
    k_vector: &[f32],
    value: f32,
    w_vector: &[f32],
) -> f32 {
    debug_assert_eq!(state_row.len(), next_state_row.len());
    debug_assert_eq!(state_row.len(), r_vector.len());
    debug_assert_eq!(state_row.len(), k_vector.len());
    debug_assert_eq!(state_row.len(), w_vector.len());

    let value_scalar = value;
    let value = x86_arch::_mm256_set1_ps(value_scalar);
    let mut output_acc = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= state_row.len() {
        let state = x86_arch::_mm256_loadu_ps(state_row.as_ptr().add(index));
        let r = x86_arch::_mm256_loadu_ps(r_vector.as_ptr().add(index));
        let k = x86_arch::_mm256_loadu_ps(k_vector.as_ptr().add(index));
        let w = x86_arch::_mm256_loadu_ps(w_vector.as_ptr().add(index));

        let state_decay = x86_arch::_mm256_mul_ps(state, w);
        let next = x86_arch::_mm256_fmadd_ps(value, k, state_decay);
        x86_arch::_mm256_storeu_ps(next_state_row.as_mut_ptr().add(index), next);
        output_acc = x86_arch::_mm256_fmadd_ps(next, r, output_acc);
        index += 8;
    }

    let use_direct_sum =
        recurrence_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut output = horizontal_sum_avx2_recurrence(output_acc, use_direct_sum);
    while index < state_row.len() {
        let next = state_row[index] * w_vector[index] + value_scalar * k_vector[index];
        next_state_row[index] = next;
        output += next * r_vector[index];
        index += 1;
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn timestep_state_output_row_scalar(
    state_row: &[f32],
    next_state_row: &mut [f32],
    r_vector: &[f32],
    k_vector: &[f32],
    value: f32,
    w_vector: &[f32],
    a_vector: &[f32],
    k_deformed_vector: &[f32],
) -> f32 {
    debug_assert_eq!(state_row.len(), next_state_row.len());
    debug_assert_eq!(state_row.len(), r_vector.len());
    debug_assert_eq!(state_row.len(), k_vector.len());
    debug_assert_eq!(state_row.len(), w_vector.len());
    debug_assert_eq!(state_row.len(), a_vector.len());
    debug_assert_eq!(state_row.len(), k_deformed_vector.len());

    let mut deformation_left = 0.0f32;
    for col_index in 0..state_row.len() {
        deformation_left += state_row[col_index] * k_deformed_vector[col_index];
    }
    let skip_deformation = lightning_deformation_left_threshold()
        .map(|threshold| deformation_left.abs() <= threshold)
        .unwrap_or(false);
    let reuse_deformed_key = recurrence_reuse_deformed_key_values_enabled();

    let mut output = 0.0f32;
    for col_index in 0..state_row.len() {
        let next = if skip_deformation {
            state_row[col_index] * w_vector[col_index] + value * k_vector[col_index]
        } else {
            let deformed_key = if reuse_deformed_key {
                k_vector[col_index]
            } else {
                a_vector[col_index] * k_deformed_vector[col_index]
            };
            state_row[col_index] * w_vector[col_index] - deformation_left * deformed_key
                + value * k_vector[col_index]
        };
        next_state_row[col_index] = next;
        output += next * r_vector[col_index];
    }
    output
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn timestep_output_row_avx2_fma(
    state_row: &[f32],
    r_vector: &[f32],
    k_vector: &[f32],
    value: f32,
    w_vector: &[f32],
    a_vector: &[f32],
    k_deformed_vector: &[f32],
) -> f32 {
    debug_assert_eq!(state_row.len(), r_vector.len());
    debug_assert_eq!(state_row.len(), k_vector.len());
    debug_assert_eq!(state_row.len(), w_vector.len());
    debug_assert_eq!(state_row.len(), a_vector.len());
    debug_assert_eq!(state_row.len(), k_deformed_vector.len());

    let mut deformation_acc = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= state_row.len() {
        let state = x86_arch::_mm256_loadu_ps(state_row.as_ptr().add(index));
        let k_deformed = x86_arch::_mm256_loadu_ps(k_deformed_vector.as_ptr().add(index));
        deformation_acc = x86_arch::_mm256_fmadd_ps(state, k_deformed, deformation_acc);
        index += 8;
    }

    let use_direct_sum =
        recurrence_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut deformation_left = horizontal_sum_avx2_recurrence(deformation_acc, use_direct_sum);
    while index < state_row.len() {
        deformation_left += state_row[index] * k_deformed_vector[index];
        index += 1;
    }

    let deformation_left_scalar = deformation_left;
    let skip_deformation = lightning_deformation_left_threshold()
        .map(|threshold| deformation_left_scalar.abs() <= threshold)
        .unwrap_or(false);
    let reuse_deformed_key = recurrence_reuse_deformed_key_values_enabled();
    let deformation_left = x86_arch::_mm256_set1_ps(deformation_left_scalar);
    let value_scalar = value;
    let value = x86_arch::_mm256_set1_ps(value_scalar);
    let mut output_acc = x86_arch::_mm256_setzero_ps();
    index = 0;
    while index + 8 <= state_row.len() {
        let state = x86_arch::_mm256_loadu_ps(state_row.as_ptr().add(index));
        let r = x86_arch::_mm256_loadu_ps(r_vector.as_ptr().add(index));
        let k = x86_arch::_mm256_loadu_ps(k_vector.as_ptr().add(index));
        let w = x86_arch::_mm256_loadu_ps(w_vector.as_ptr().add(index));

        let state_decay = x86_arch::_mm256_mul_ps(state, w);
        let next = if skip_deformation {
            x86_arch::_mm256_fmadd_ps(value, k, state_decay)
        } else {
            let deformed_key = if reuse_deformed_key {
                k
            } else {
                let a = x86_arch::_mm256_loadu_ps(a_vector.as_ptr().add(index));
                let k_deformed = x86_arch::_mm256_loadu_ps(k_deformed_vector.as_ptr().add(index));
                x86_arch::_mm256_mul_ps(a, k_deformed)
            };
            let deformation = x86_arch::_mm256_mul_ps(deformation_left, deformed_key);
            x86_arch::_mm256_fmadd_ps(value, k, x86_arch::_mm256_sub_ps(state_decay, deformation))
        };
        output_acc = x86_arch::_mm256_fmadd_ps(next, r, output_acc);
        index += 8;
    }

    let mut output = horizontal_sum_avx2_recurrence(output_acc, use_direct_sum);
    while index < state_row.len() {
        let next = if skip_deformation {
            state_row[index] * w_vector[index] + value_scalar * k_vector[index]
        } else {
            let deformed_key = if reuse_deformed_key {
                k_vector[index]
            } else {
                a_vector[index] * k_deformed_vector[index]
            };
            state_row[index] * w_vector[index] - deformation_left_scalar * deformed_key
                + value_scalar * k_vector[index]
        };
        output += next * r_vector[index];
        index += 1;
    }
    output
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn timestep_state_output_row_avx2_fma(
    state_row: &[f32],
    next_state_row: &mut [f32],
    r_vector: &[f32],
    k_vector: &[f32],
    value: f32,
    w_vector: &[f32],
    a_vector: &[f32],
    k_deformed_vector: &[f32],
) -> f32 {
    debug_assert_eq!(state_row.len(), next_state_row.len());
    debug_assert_eq!(state_row.len(), r_vector.len());
    debug_assert_eq!(state_row.len(), k_vector.len());
    debug_assert_eq!(state_row.len(), w_vector.len());
    debug_assert_eq!(state_row.len(), a_vector.len());
    debug_assert_eq!(state_row.len(), k_deformed_vector.len());

    let mut deformation_acc = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= state_row.len() {
        let state = x86_arch::_mm256_loadu_ps(state_row.as_ptr().add(index));
        let k_deformed = x86_arch::_mm256_loadu_ps(k_deformed_vector.as_ptr().add(index));
        deformation_acc = x86_arch::_mm256_fmadd_ps(state, k_deformed, deformation_acc);
        index += 8;
    }

    let use_direct_sum =
        recurrence_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut deformation_left = horizontal_sum_avx2_recurrence(deformation_acc, use_direct_sum);
    while index < state_row.len() {
        deformation_left += state_row[index] * k_deformed_vector[index];
        index += 1;
    }

    let deformation_left_scalar = deformation_left;
    let skip_deformation = lightning_deformation_left_threshold()
        .map(|threshold| deformation_left_scalar.abs() <= threshold)
        .unwrap_or(false);
    let reuse_deformed_key = recurrence_reuse_deformed_key_values_enabled();
    let deformation_left = x86_arch::_mm256_set1_ps(deformation_left_scalar);
    let value_scalar = value;
    let value = x86_arch::_mm256_set1_ps(value_scalar);
    let mut output_acc = x86_arch::_mm256_setzero_ps();
    index = 0;
    while index + 8 <= state_row.len() {
        let state = x86_arch::_mm256_loadu_ps(state_row.as_ptr().add(index));
        let r = x86_arch::_mm256_loadu_ps(r_vector.as_ptr().add(index));
        let k = x86_arch::_mm256_loadu_ps(k_vector.as_ptr().add(index));
        let w = x86_arch::_mm256_loadu_ps(w_vector.as_ptr().add(index));

        let state_decay = x86_arch::_mm256_mul_ps(state, w);
        let next = if skip_deformation {
            x86_arch::_mm256_fmadd_ps(value, k, state_decay)
        } else {
            let deformed_key = if reuse_deformed_key {
                k
            } else {
                let a = x86_arch::_mm256_loadu_ps(a_vector.as_ptr().add(index));
                let k_deformed = x86_arch::_mm256_loadu_ps(k_deformed_vector.as_ptr().add(index));
                x86_arch::_mm256_mul_ps(a, k_deformed)
            };
            let deformation = x86_arch::_mm256_mul_ps(deformation_left, deformed_key);
            x86_arch::_mm256_fmadd_ps(value, k, x86_arch::_mm256_sub_ps(state_decay, deformation))
        };
        x86_arch::_mm256_storeu_ps(next_state_row.as_mut_ptr().add(index), next);
        output_acc = x86_arch::_mm256_fmadd_ps(next, r, output_acc);
        index += 8;
    }

    let mut output = horizontal_sum_avx2_recurrence(output_acc, use_direct_sum);
    while index < state_row.len() {
        let next = if skip_deformation {
            state_row[index] * w_vector[index] + value_scalar * k_vector[index]
        } else {
            let deformed_key = if reuse_deformed_key {
                k_vector[index]
            } else {
                a_vector[index] * k_deformed_vector[index]
            };
            state_row[index] * w_vector[index] - deformation_left_scalar * deformed_key
                + value_scalar * k_vector[index]
        };
        next_state_row[index] = next;
        output += next * r_vector[index];
        index += 1;
    }
    output
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(values: x86_arch::__m256) -> f32 {
    if fast_avx2_horizontal_sum_enabled() {
        return horizontal_sum_avx2_register(values);
    }
    let mut lanes = [0.0f32; 8];
    x86_arch::_mm256_storeu_ps(lanes.as_mut_ptr(), values);
    lanes.iter().copied().sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2_recurrence(values: x86_arch::__m256, use_direct_sum: bool) -> f32 {
    if use_direct_sum {
        horizontal_sum_avx2_register(values)
    } else {
        horizontal_sum_avx2(values)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2_register(values: x86_arch::__m256) -> f32 {
    let low = x86_arch::_mm256_castps256_ps128(values);
    let high = x86_arch::_mm256_extractf128_ps(values, 1);
    let sum128 = x86_arch::_mm_add_ps(low, high);
    let high64 = x86_arch::_mm_movehl_ps(sum128, sum128);
    let sum64 = x86_arch::_mm_add_ps(sum128, high64);
    let high32 = x86_arch::_mm_shuffle_ps(sum64, sum64, 0b01_01_01_01);
    x86_arch::_mm_cvtss_f32(x86_arch::_mm_add_ss(sum64, high32))
}

pub(crate) enum F32TensorData<'a> {
    Borrowed {
        storage: std::sync::RwLockReadGuard<'a, Storage>,
        start: usize,
        end: usize,
    },
    Owned(Vec<f32>),
}

impl F32TensorData<'_> {
    pub(crate) fn as_slice(&self) -> Result<&[f32]> {
        match self {
            Self::Borrowed {
                storage,
                start,
                end,
            } => match &**storage {
                Storage::Cpu(storage) => Ok(&storage.as_slice::<f32>()?[*start..*end]),
                _ => bail!("expected CPU tensor storage"),
            },
            Self::Owned(values) => Ok(values.as_slice()),
        }
    }
}

pub(crate) fn f32_tensor_data<'a>(tensor: &'a Tensor) -> Result<F32TensorData<'a>> {
    if let Some((start, end)) = tensor.layout().contiguous_offsets() {
        let (storage_guard, _) = tensor.storage_and_layout();
        match &*storage_guard {
            Storage::Cpu(storage) => {
                storage.as_slice::<f32>()?;
                Ok(F32TensorData::Borrowed {
                    storage: storage_guard,
                    start,
                    end,
                })
            }
            _ => Ok(F32TensorData::Owned(
                tensor.flatten_all()?.to_vec1::<f32>()?,
            )),
        }
    } else {
        Ok(F32TensorData::Owned(
            tensor.flatten_all()?.to_vec1::<f32>()?,
        ))
    }
}

pub fn forgetting_curve(weights: &Tensor, elapsed_seconds: &Tensor) -> Result<Tensor> {
    let (batch_size, curve_count) = weights.dims2()?;
    if curve_count != NUM_CURVES {
        bail!("forgetting_curve expected {NUM_CURVES} curves, got {curve_count}");
    }

    let elapsed = clamped_elapsed_vec(elapsed_seconds, batch_size)?;
    let elapsed = Tensor::from_vec(elapsed, (batch_size, 1), weights.device())?;
    let s_space = Tensor::from_vec(s_space(), (NUM_CURVES,), weights.device())?;
    let retention = (elapsed.broadcast_div(&s_space)? * (0.9f64).ln())?.exp()?;
    let weighted = weights.broadcast_mul(&retention)?;
    weighted
        .sum(D::Minus1)?
        .affine(PROBABILITY_SCALE, PROBABILITY_EPS)
}

pub fn interp(ahead_logits: &Tensor, elapsed_seconds: &Tensor) -> Result<Tensor> {
    let (batch_size, point_count) = ahead_logits.dims2()?;
    if point_count != NUM_POINTS {
        bail!("interp expected {NUM_POINTS} points, got {point_count}");
    }

    let point_space = point_space();
    let logits = ahead_logits.to_vec2::<f32>()?;
    let elapsed = clamped_elapsed_vec(elapsed_seconds, batch_size)?;
    let mut out = Vec::with_capacity(batch_size);

    for (batch_index, elapsed_seconds) in elapsed.into_iter().enumerate() {
        let right_idx = searchsorted_left(&point_space, elapsed_seconds);
        if right_idx >= point_space.len() {
            bail!(
                "interp elapsed_seconds {elapsed_seconds} exceeds point-space max {}",
                point_space[point_space.len() - 1]
            );
        }
        let left_idx = right_idx.saturating_sub(1);
        let xl = point_space[left_idx];
        let xr = point_space[right_idx];
        let yl = logits[batch_index][left_idx];
        let yr = logits[batch_index][right_idx];
        let interpolated = yl + (yr - yl) * (elapsed_seconds - xl) / (xr - xl);
        out.push(PROBABILITY_EPS as f32 + PROBABILITY_SCALE as f32 * interpolated);
    }

    Tensor::from_vec(out, (batch_size,), ahead_logits.device())
}

pub fn predict_curve(
    ahead_logits: &Tensor,
    weights: &Tensor,
    elapsed_seconds: &Tensor,
) -> Result<Tensor> {
    let curve_probs_raw = forgetting_curve(weights, elapsed_seconds)?;
    let ahead_logit_residual = interp(ahead_logits, elapsed_seconds)?;
    let curve_logits_raw = curve_probs_raw
        .broadcast_div(&curve_probs_raw.affine(-1.0, 1.0)?)?
        .log()?;
    nn_ops::sigmoid(&(curve_logits_raw + ahead_logit_residual)?)
}

pub fn curve_interval(
    ahead_logits: &[f32],
    weights: &[f32],
    retention_probability: f32,
) -> Result<Option<f32>> {
    if ahead_logits.len() != NUM_POINTS {
        bail!(
            "curve_interval expected {NUM_POINTS} ahead-logit values, got {}",
            ahead_logits.len()
        );
    }
    if weights.len() != NUM_CURVES {
        bail!(
            "curve_interval expected {NUM_CURVES} weight values, got {}",
            weights.len()
        );
    }
    if !retention_probability.is_finite()
        || retention_probability <= 0.0
        || retention_probability >= 1.0
    {
        bail!("retention_probability must be finite, greater than 0, and less than 1");
    }
    if ahead_logits.iter().any(|value| !value.is_finite()) {
        bail!("curve_interval ahead logits must be finite");
    }
    if weights.iter().any(|value| !value.is_finite()) {
        bail!("curve_interval weights must be finite");
    }

    let s_space = s_space();
    let point_space = point_space();
    let mut scan_times = Vec::with_capacity(point_space.len() + 1);
    scan_times.push(1.0);
    scan_times.extend(point_space.iter().copied().filter(|elapsed| *elapsed > 1.0));

    let threshold = retention_probability as f64;
    if curve_probability_scalar(ahead_logits, weights, 1.0, &s_space, &point_space)? < threshold {
        return Ok(Some(1.0));
    }

    let mut previous_time = 1.0;
    for right_time in scan_times.iter().copied().skip(1) {
        let right_probability =
            curve_probability_scalar(ahead_logits, weights, right_time, &s_space, &point_space)?;
        if right_probability >= threshold {
            previous_time = right_time;
            continue;
        }

        let mut lo = previous_time;
        let mut hi = right_time;
        for _ in 0..32 {
            let mid = (lo + hi) * 0.5;
            let probability =
                curve_probability_scalar(ahead_logits, weights, mid, &s_space, &point_space)?;
            if probability < threshold {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return Ok(Some(hi));
    }

    Ok(None)
}

pub fn l2_normalize_last_dim(xs: &Tensor) -> Result<Tensor> {
    let norm = xs
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .maximum(L2_NORM_MIN)?;
    xs.broadcast_div(&norm)
}

pub(crate) fn l2_normalize_scaled_b1hk(xs: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let (batch_size, one, heads, head_size) = xs.dims4()?;
    if one != 1 {
        bail!(
            "l2_normalize_scaled_b1hk expected xs shape [B, 1, H, K], got {:?}",
            xs.dims()
        );
    }
    if scale.dims() != [batch_size, 1, heads] {
        bail!(
            "l2_normalize_scaled_b1hk expected scale shape [{batch_size}, 1, {heads}], got {:?}",
            scale.dims()
        );
    }

    let xs_data = f32_tensor_data(xs)?;
    let scale_data = f32_tensor_data(scale)?;
    let xs_values = xs_data.as_slice()?;
    let scale_values = scale_data.as_slice()?;
    let mut out = vec![0.0f32; xs_values.len()];

    for batch_index in 0..batch_size {
        for head_index in 0..heads {
            let vector_base = (batch_index * heads + head_index) * head_size;
            let mut squared_sum = 0.0f32;
            for value in &xs_values[vector_base..vector_base + head_size] {
                squared_sum += value * value;
            }
            let norm = squared_sum.sqrt().max(L2_NORM_MIN);
            let scale_value = scale_values[batch_index * heads + head_index];
            for offset in 0..head_size {
                let value_index = vector_base + offset;
                out[value_index] = xs_values[value_index] / norm * scale_value;
            }
        }
    }

    Tensor::from_vec(out, (batch_size, 1usize, heads, head_size), xs.device())
}

pub fn layer_norm_last_dim(
    xs: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if native_centered_layer_norm_enabled() {
        return layer_norm_last_dim_native_centered(xs, weight, bias, eps);
    }
    // Candle's fast CPU layer norm uses E[x^2] - mean^2 variance, which drifts
    // from PyTorch enough for RWKV projections to amplify. The default path
    // keeps Candle's centered slow path for strict process/curve parity.
    // Batched predict_many enables the fast path because its probabilities are
    // used for ranking and are not fed back into recurrent state.
    if FAST_LAYER_NORM_ENABLED.with(Cell::get) {
        return nn_ops::layer_norm(xs, weight, bias, eps);
    }
    nn_ops::layer_norm_slow(xs, weight, bias, eps)
}

fn layer_norm_last_dim_native_centered(
    xs: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dims = xs.dims();
    let Some(&width) = dims.last() else {
        bail!("layer_norm_last_dim requires at least one dimension");
    };
    if width == 0 {
        bail!("layer_norm_last_dim last dimension must not be empty");
    }
    if weight.dims1()? != width || bias.dims1()? != width {
        bail!(
            "layer_norm_last_dim expected weight/bias shape [{width}], got {:?} and {:?}",
            weight.dims(),
            bias.dims()
        );
    }

    let row_count = xs.elem_count() / width;
    let xs_data = f32_tensor_data(xs)?;
    let weight_data = f32_tensor_data(weight)?;
    let bias_data = f32_tensor_data(bias)?;
    let xs_values = xs_data.as_slice()?;
    let weight_values = weight_data.as_slice()?;
    let bias_values = bias_data.as_slice()?;
    let mut out = vec![0.0f32; xs_values.len()];

    for row_index in 0..row_count {
        let row_base = row_index * width;
        let row = &xs_values[row_base..row_base + width];
        let mean = row.iter().copied().sum::<f32>() / width as f32;
        let variance = row
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / width as f32;
        let inv_std = 1.0f32 / (variance + eps).sqrt();
        let out_row = &mut out[row_base..row_base + width];
        for channel_index in 0..width {
            out_row[channel_index] =
                (row[channel_index] - mean) * inv_std * weight_values[channel_index]
                    + bias_values[channel_index];
        }
    }

    Tensor::from_vec(out, xs.shape(), xs.device())
}

pub(crate) fn time_lerp_parts_b1c(
    xs: &Tensor,
    x_shift: &Tensor,
    lerp: &Tensor,
) -> Result<Vec<Tensor>> {
    let (batch_size, one, channels) = xs.dims3()?;
    if one != 1 {
        bail!(
            "time_lerp_parts_b1c expected xs shape [B, 1, C], got {:?}",
            xs.dims()
        );
    }
    if x_shift.dims() != [batch_size, 1, channels] {
        bail!(
            "time_lerp_parts_b1c expected x_shift shape [{batch_size}, 1, {channels}], got {:?}",
            x_shift.dims()
        );
    }
    if lerp.dims() != [8, 1, 1, channels] {
        bail!(
            "time_lerp_parts_b1c expected lerp shape [8, 1, 1, {channels}], got {:?}",
            lerp.dims()
        );
    }

    if std::ptr::eq(xs, x_shift) {
        return Ok((0..8).map(|_| xs.clone()).collect());
    }

    let xs_data = f32_tensor_data(xs)?;
    let x_shift_data = f32_tensor_data(x_shift)?;
    let lerp_data = f32_tensor_data(lerp)?;
    let xs_values = xs_data.as_slice()?;
    let x_shift_values = x_shift_data.as_slice()?;
    let lerp_values = lerp_data.as_slice()?;
    let row_len = batch_size * channels;
    let mut parts = Vec::with_capacity(8);

    for part_index in 0..8 {
        let mut values = vec![0.0f32; row_len];
        let lerp_base = part_index * channels;
        for batch_index in 0..batch_size {
            let row_base = batch_index * channels;
            for channel_index in 0..channels {
                let value_index = row_base + channel_index;
                let x = xs_values[value_index];
                values[value_index] =
                    x + (x_shift_values[value_index] - x) * lerp_values[lerp_base + channel_index];
            }
        }
        parts.push(Tensor::from_vec(
            values,
            (batch_size, 1usize, channels),
            xs.device(),
        )?);
    }

    Ok(parts)
}

pub(crate) fn time_decay_w_b1c(d: &Tensor) -> Result<Tensor> {
    let (batch_size, one, channels) = d.dims3()?;
    if one != 1 {
        bail!(
            "time_decay_w_b1c expected d shape [B, 1, C], got {:?}",
            d.dims()
        );
    }

    let d_data = f32_tensor_data(d)?;
    let d_values = d_data.as_slice()?;
    let w_values = if fast_time_decay_enabled() {
        time_decay_w_values_fast(d_values)
    } else {
        time_decay_w_values_reference(d_values)
    };

    Tensor::from_vec(w_values, (batch_size, 1usize, channels), d.device())
}

pub(crate) fn time_decay_w_scalar(d: f32) -> f32 {
    if fast_time_decay_enabled() {
        time_decay_w_scalar_fast(d)
    } else {
        time_decay_w_scalar_reference(d)
    }
}

fn time_decay_w_values_reference(d_values: &[f32]) -> Vec<f32> {
    let mut w_values = vec![0.0f32; d_values.len()];

    for index in 0..d_values.len() {
        w_values[index] = time_decay_w_scalar_reference(d_values[index]);
    }

    w_values
}

fn time_decay_w_values_fast(d_values: &[f32]) -> Vec<f32> {
    let mut w_values = vec![0.0f32; d_values.len()];

    for index in 0..d_values.len() {
        w_values[index] = time_decay_w_scalar_fast(d_values[index]);
    }

    w_values
}

fn time_decay_w_scalar_reference(d: f32) -> f32 {
    let softplus_neg = ((-d).exp() + 1.0).ln();
    let decay = -(softplus_neg + 0.5);
    (-(decay.exp())).exp()
}

fn time_decay_w_scalar_fast(d: f32) -> f32 {
    let sigmoid = if d >= 0.0 {
        1.0 / (1.0 + (-d).exp())
    } else {
        let exp_d = d.exp();
        exp_d / (1.0 + exp_d)
    };
    (-((-0.5f32).exp() * sigmoid)).exp()
}

pub fn group_norm_2d(
    xs: &Tensor,
    num_groups: usize,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let (batch_size, channels) = xs.dims2()?;
    if num_groups == 0 {
        bail!("group_norm_2d requires at least one group");
    }
    if channels % num_groups != 0 {
        bail!("group_norm_2d channels ({channels}) must be divisible by groups ({num_groups})");
    }
    if weight.dims1()? != channels || bias.dims1()? != channels {
        bail!(
            "group_norm_2d expected weight/bias shape [{channels}], got {:?} and {:?}",
            weight.dims(),
            bias.dims()
        );
    }

    let group_size = channels / num_groups;
    let xs_data = f32_tensor_data(xs)?;
    let weight_data = f32_tensor_data(weight)?;
    let bias_data = f32_tensor_data(bias)?;
    let xs_values = xs_data.as_slice()?;
    let weight_values = weight_data.as_slice()?;
    let bias_values = bias_data.as_slice()?;
    let mut out = vec![0.0f32; xs_values.len()];
    let eps = eps as f32;

    for batch_index in 0..batch_size {
        let batch_base = batch_index * channels;
        for group_index in 0..num_groups {
            let group_base = batch_base + group_index * group_size;
            let group = &xs_values[group_base..group_base + group_size];
            let mean = group.iter().copied().sum::<f32>() / group_size as f32;
            let variance = group
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / group_size as f32;
            let inv_std = 1.0f32 / (variance + eps).sqrt();
            for offset in 0..group_size {
                let value_index = group_base + offset;
                let channel_index = group_index * group_size + offset;
                out[value_index] =
                    (xs_values[value_index] - mean) * inv_std * weight_values[channel_index]
                        + bias_values[channel_index];
            }
        }
    }

    Tensor::from_vec(out, (batch_size, channels), xs.device())
}

pub fn single_timestep(
    r_bhk: &Tensor,
    k_bhk: &Tensor,
    v_bhk: &Tensor,
    w_bhk: &Tensor,
    a_bhk: &Tensor,
    k_deformed_bhk: &Tensor,
    state_bhkk: &Tensor,
) -> Result<(Tensor, Tensor)> {
    single_timestep_profiled(
        r_bhk,
        k_bhk,
        v_bhk,
        w_bhk,
        a_bhk,
        k_deformed_bhk,
        state_bhkk,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn single_timestep_profiled(
    r_bhk: &Tensor,
    k_bhk: &Tensor,
    v_bhk: &Tensor,
    w_bhk: &Tensor,
    a_bhk: &Tensor,
    k_deformed_bhk: &Tensor,
    state_bhkk: &Tensor,
    mut profile: Option<&mut SingleTimestepProfile>,
) -> Result<(Tensor, Tensor)> {
    let total_start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    let dims = r_bhk.dims3()?;
    for (name, tensor) in [
        ("k_BHK", k_bhk),
        ("v_BHK", v_bhk),
        ("w_BHK", w_bhk),
        ("a_BHK", a_bhk),
        ("k_deformed_BHK", k_deformed_bhk),
    ] {
        if tensor.dims3()? != dims {
            bail!(
                "single_timestep expected {name} shape {:?}, got {:?}",
                r_bhk.dims(),
                tensor.dims()
            );
        }
    }

    let (batch_size, heads, head_size) = dims;
    if state_bhkk.dims4()? != (batch_size, heads, head_size, head_size) {
        bail!(
            "single_timestep expected state shape [{batch_size}, {heads}, {head_size}, {head_size}], got {:?}",
            state_bhkk.dims()
        );
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.validate_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let r_data = f32_tensor_data(r_bhk)?;
    let k_data = f32_tensor_data(k_bhk)?;
    let v_data = f32_tensor_data(v_bhk)?;
    let w_data = f32_tensor_data(w_bhk)?;
    let a_data = f32_tensor_data(a_bhk)?;
    let k_deformed_data = f32_tensor_data(k_deformed_bhk)?;
    let state_data = f32_tensor_data(state_bhkk)?;
    let r_values = r_data.as_slice()?;
    let k_values = k_data.as_slice()?;
    let v_values = v_data.as_slice()?;
    let w_values = w_data.as_slice()?;
    let a_values = a_data.as_slice()?;
    let k_deformed_values = k_deformed_data.as_slice()?;
    let state_values = state_data.as_slice()?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.tensor_read_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.deformation_prepare_ns += start.elapsed_ns();
    }

    let state_batch_stride = heads * head_size * head_size;
    let vector_batch_stride = heads * head_size;
    let mut next_state_values = vec![0.0f32; state_values.len()];
    let mut out_values = vec![0.0f32; batch_size * heads * head_size];
    let kernel = timestep_state_output_kernel();
    let start = ProfileTimer::start(profile.is_some());
    if batch_size > 1 {
        next_state_values
            .par_chunks_mut(state_batch_stride)
            .zip(out_values.par_chunks_mut(vector_batch_stride))
            .enumerate()
            .for_each(|(batch_index, (next_state_batch, out_batch))| {
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let state_base = vector_base * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    for row_index in 0..head_size {
                        let row_base = state_base + row_index * head_size;
                        let local_row_base = local_state_base + row_index * head_size;
                        let value = v_values[vector_base + row_index];
                        let state_row = &state_values[row_base..row_base + head_size];
                        let next_state_row =
                            &mut next_state_batch[local_row_base..local_row_base + head_size];
                        out_batch[local_vector_base + row_index] = timestep_state_output_row(
                            kernel,
                            state_row,
                            next_state_row,
                            r_vector,
                            k_vector,
                            value,
                            w_vector,
                            a_vector,
                            k_deformed_vector,
                        );
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                let state_base = vector_base * head_size;
                let local_state_base = head_index * head_size * head_size;
                let local_vector_base = head_index * head_size;
                let next_state_batch = &mut next_state_values[batch_index * state_batch_stride..]
                    [..state_batch_stride];
                let out_batch =
                    &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
                let r_vector = &r_values[vector_base..vector_base + head_size];
                let k_vector = &k_values[vector_base..vector_base + head_size];
                let w_vector = &w_values[vector_base..vector_base + head_size];
                let a_vector = &a_values[vector_base..vector_base + head_size];
                let k_deformed_vector = &k_deformed_values[vector_base..vector_base + head_size];
                for row_index in 0..head_size {
                    let row_base = state_base + row_index * head_size;
                    let local_row_base = local_state_base + row_index * head_size;
                    let value = v_values[vector_base + row_index];
                    let state_row = &state_values[row_base..row_base + head_size];
                    let next_state_row =
                        &mut next_state_batch[local_row_base..local_row_base + head_size];
                    out_batch[local_vector_base + row_index] = timestep_state_output_row(
                        kernel,
                        state_row,
                        next_state_row,
                        r_vector,
                        k_vector,
                        value,
                        w_vector,
                        a_vector,
                        k_deformed_vector,
                    );
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.fused_state_output_ns += start.elapsed_ns();
        let row_count = batch_size * heads * head_size;
        record_timestep_kernel_rows(profile, kernel, row_count, true);
    }

    let start = ProfileTimer::start(profile.is_some());
    let out_bhk = Tensor::from_vec(out_values, (batch_size, heads, head_size), r_bhk.device())?;
    let state_bhkk = Tensor::from_vec(
        next_state_values,
        (batch_size, heads, head_size, head_size),
        state_bhkk.device(),
    )?;
    if let Some(profile) = profile {
        profile.tensor_write_ns += start.elapsed_ns();
        profile.total_ns += total_start.elapsed_ns();
    }

    Ok((out_bhk, state_bhkk))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn single_timestep_output_profiled(
    r_bhk: &Tensor,
    k_bhk: &Tensor,
    v_bhk: &Tensor,
    w_bhk: &Tensor,
    a_bhk: &Tensor,
    k_deformed_bhk: &Tensor,
    state_bhkk: &Tensor,
    mut profile: Option<&mut SingleTimestepProfile>,
) -> Result<Tensor> {
    let total_start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    let dims = r_bhk.dims3()?;
    for (name, tensor) in [
        ("k_BHK", k_bhk),
        ("v_BHK", v_bhk),
        ("w_BHK", w_bhk),
        ("a_BHK", a_bhk),
        ("k_deformed_BHK", k_deformed_bhk),
    ] {
        if tensor.dims3()? != dims {
            bail!(
                "single_timestep expected {name} shape {:?}, got {:?}",
                r_bhk.dims(),
                tensor.dims()
            );
        }
    }

    let (batch_size, heads, head_size) = dims;
    if state_bhkk.dims4()? != (batch_size, heads, head_size, head_size) {
        bail!(
            "single_timestep expected state shape [{batch_size}, {heads}, {head_size}, {head_size}], got {:?}",
            state_bhkk.dims()
        );
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.validate_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let r_data = f32_tensor_data(r_bhk)?;
    let k_data = f32_tensor_data(k_bhk)?;
    let v_data = f32_tensor_data(v_bhk)?;
    let w_data = f32_tensor_data(w_bhk)?;
    let a_data = f32_tensor_data(a_bhk)?;
    let k_deformed_data = f32_tensor_data(k_deformed_bhk)?;
    let state_data = f32_tensor_data(state_bhkk)?;
    let r_values = r_data.as_slice()?;
    let k_values = k_data.as_slice()?;
    let v_values = v_data.as_slice()?;
    let w_values = w_data.as_slice()?;
    let a_values = a_data.as_slice()?;
    let k_deformed_values = k_deformed_data.as_slice()?;
    let state_values = state_data.as_slice()?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.tensor_read_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.deformation_prepare_ns += start.elapsed_ns();
    }

    let vector_batch_stride = heads * head_size;
    let mut out_values = vec![0.0f32; batch_size * vector_batch_stride];
    let kernel = timestep_output_kernel();
    let start = ProfileTimer::start(profile.is_some());
    if batch_size > 1 {
        out_values
            .par_chunks_mut(vector_batch_stride)
            .enumerate()
            .for_each(|(batch_index, out_batch)| {
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let state_base = vector_base * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    timestep_output_head(
                        kernel,
                        &state_values[state_base..state_base + head_size * head_size],
                        &mut out_batch[local_vector_base..local_vector_base + head_size],
                        r_vector,
                        k_vector,
                        &v_values[vector_base..vector_base + head_size],
                        w_vector,
                        a_vector,
                        k_deformed_vector,
                    );
                }
            });
    } else {
        for batch_index in 0..batch_size {
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                let state_base = vector_base * head_size;
                let local_vector_base = head_index * head_size;
                let out_batch =
                    &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
                let r_vector = &r_values[vector_base..vector_base + head_size];
                let k_vector = &k_values[vector_base..vector_base + head_size];
                let w_vector = &w_values[vector_base..vector_base + head_size];
                let a_vector = &a_values[vector_base..vector_base + head_size];
                let k_deformed_vector = &k_deformed_values[vector_base..vector_base + head_size];
                timestep_output_head(
                    kernel,
                    &state_values[state_base..state_base + head_size * head_size],
                    &mut out_batch[local_vector_base..local_vector_base + head_size],
                    r_vector,
                    k_vector,
                    &v_values[vector_base..vector_base + head_size],
                    w_vector,
                    a_vector,
                    k_deformed_vector,
                );
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.fused_state_output_ns += start.elapsed_ns();
        let row_count = batch_size * heads * head_size;
        record_timestep_kernel_rows(profile, kernel, row_count, false);
    }

    let start = ProfileTimer::start(profile.is_some());
    let out_bhk = Tensor::from_vec(out_values, (batch_size, heads, head_size), r_bhk.device())?;
    if let Some(profile) = profile {
        profile.tensor_write_ns += start.elapsed_ns();
        profile.total_ns += total_start.elapsed_ns();
    }

    Ok(out_bhk)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn single_timestep_b1c_profiled(
    r_b1c: &Tensor,
    k_b1c: &Tensor,
    v_b1c: &Tensor,
    w_b1c: &Tensor,
    a_b1c: &Tensor,
    k_scale_b1h: &Tensor,
    v_scale_b1h: &Tensor,
    state_b1hkk: &Tensor,
    heads: usize,
    head_size: usize,
    mut profile: Option<&mut SingleTimestepProfile>,
) -> Result<(Tensor, Tensor)> {
    let total_start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, one, channels) = r_b1c.dims3()?;
    if one != 1 || channels != heads * head_size {
        bail!(
            "single_timestep_b1c expected r shape [B, 1, {}], got {:?}",
            heads * head_size,
            r_b1c.dims()
        );
    }
    for (name, tensor) in [
        ("k_B1C", k_b1c),
        ("v_B1C", v_b1c),
        ("w_B1C", w_b1c),
        ("a_B1C", a_b1c),
    ] {
        if tensor.dims() != [batch_size, 1, channels] {
            bail!(
                "single_timestep_b1c expected {name} shape [{batch_size}, 1, {channels}], got {:?}",
                tensor.dims()
            );
        }
    }
    if k_scale_b1h.dims() != [batch_size, 1, heads] {
        bail!(
            "single_timestep_b1c expected k_scale shape [{batch_size}, 1, {heads}], got {:?}",
            k_scale_b1h.dims()
        );
    }
    if v_scale_b1h.dims() != [batch_size, 1, heads] {
        bail!(
            "single_timestep_b1c expected v_scale shape [{batch_size}, 1, {heads}], got {:?}",
            v_scale_b1h.dims()
        );
    }
    if state_b1hkk.dims() != [batch_size, 1, heads, head_size, head_size] {
        bail!(
            "single_timestep_b1c expected state shape [{batch_size}, 1, {heads}, {head_size}, {head_size}], got {:?}",
            state_b1hkk.dims()
        );
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.validate_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let r_data = f32_tensor_data(r_b1c)?;
    let k_data = f32_tensor_data(k_b1c)?;
    let v_data = f32_tensor_data(v_b1c)?;
    let w_data = f32_tensor_data(w_b1c)?;
    let a_data = f32_tensor_data(a_b1c)?;
    let k_scale_data = f32_tensor_data(k_scale_b1h)?;
    let v_scale_data = f32_tensor_data(v_scale_b1h)?;
    let state_data = f32_tensor_data(state_b1hkk)?;
    let r_values = r_data.as_slice()?;
    let k_values = k_data.as_slice()?;
    let v_values = v_data.as_slice()?;
    let w_values = w_data.as_slice()?;
    let a_values = a_data.as_slice()?;
    let k_scale_values = k_scale_data.as_slice()?;
    let v_scale_values = v_scale_data.as_slice()?;
    let state_values = state_data.as_slice()?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.tensor_read_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.deformation_prepare_ns += start.elapsed_ns();
    }

    let state_batch_stride = heads * head_size * head_size;
    let vector_batch_stride = heads * head_size;
    let mut next_state_values = vec![0.0f32; state_values.len()];
    let mut out_values = vec![0.0f32; batch_size * vector_batch_stride];
    let kernel = timestep_state_output_kernel();
    let start = ProfileTimer::start(profile.is_some());
    if batch_size > 1 {
        next_state_values
            .par_chunks_mut(state_batch_stride)
            .zip(out_values.par_chunks_mut(vector_batch_stride))
            .enumerate()
            .for_each(|(batch_index, (next_state_batch, out_batch))| {
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let state_base = vector_base * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let (k_deformed_vector, k_vector) = normalized_k_vectors(
                        &k_values[vector_base..vector_base + head_size],
                        a_vector,
                        k_scale_values[batch_index * heads + head_index],
                    );
                    let v_vector = normalized_vector(
                        &v_values[vector_base..vector_base + head_size],
                        v_scale_values[batch_index * heads + head_index],
                    );
                    for row_index in 0..head_size {
                        let row_base = state_base + row_index * head_size;
                        let local_row_base = local_state_base + row_index * head_size;
                        let state_row = &state_values[row_base..row_base + head_size];
                        let next_state_row =
                            &mut next_state_batch[local_row_base..local_row_base + head_size];
                        out_batch[local_vector_base + row_index] = timestep_state_output_row(
                            kernel,
                            state_row,
                            next_state_row,
                            r_vector,
                            &k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            &k_deformed_vector,
                        );
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                let state_base = vector_base * head_size;
                let local_state_base = head_index * head_size * head_size;
                let local_vector_base = head_index * head_size;
                let next_state_batch = &mut next_state_values[batch_index * state_batch_stride..]
                    [..state_batch_stride];
                let out_batch =
                    &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
                let r_vector = &r_values[vector_base..vector_base + head_size];
                let w_vector = &w_values[vector_base..vector_base + head_size];
                let a_vector = &a_values[vector_base..vector_base + head_size];
                let (k_deformed_vector, k_vector) = normalized_k_vectors(
                    &k_values[vector_base..vector_base + head_size],
                    a_vector,
                    k_scale_values[batch_index * heads + head_index],
                );
                let v_vector = normalized_vector(
                    &v_values[vector_base..vector_base + head_size],
                    v_scale_values[batch_index * heads + head_index],
                );
                for row_index in 0..head_size {
                    let row_base = state_base + row_index * head_size;
                    let local_row_base = local_state_base + row_index * head_size;
                    let state_row = &state_values[row_base..row_base + head_size];
                    let next_state_row =
                        &mut next_state_batch[local_row_base..local_row_base + head_size];
                    out_batch[local_vector_base + row_index] = timestep_state_output_row(
                        kernel,
                        state_row,
                        next_state_row,
                        r_vector,
                        &k_vector,
                        v_vector[row_index],
                        w_vector,
                        a_vector,
                        &k_deformed_vector,
                    );
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.fused_state_output_ns += start.elapsed_ns();
        let row_count = batch_size * heads * head_size;
        record_timestep_kernel_rows(profile, kernel, row_count, true);
    }

    let start = ProfileTimer::start(profile.is_some());
    let out_bhk = Tensor::from_vec(out_values, (batch_size, heads, head_size), r_b1c.device())?;
    let state_bhkk = Tensor::from_vec(
        next_state_values,
        (batch_size, heads, head_size, head_size),
        state_b1hkk.device(),
    )?;
    if let Some(profile) = profile {
        profile.tensor_write_ns += start.elapsed_ns();
        profile.total_ns += total_start.elapsed_ns();
    }

    Ok((out_bhk, state_bhkk))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn single_timestep_output_b1c_profiled(
    r_b1c: &Tensor,
    k_b1c: &Tensor,
    v_b1c: &Tensor,
    w_b1c: &Tensor,
    a_b1c: &Tensor,
    k_scale_b1h: &Tensor,
    v_scale_b1h: &Tensor,
    state_b1hkk: &Tensor,
    heads: usize,
    head_size: usize,
    mut profile: Option<&mut SingleTimestepProfile>,
) -> Result<Tensor> {
    let total_start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, one, channels) = r_b1c.dims3()?;
    if one != 1 || channels != heads * head_size {
        bail!(
            "single_timestep_b1c expected r shape [B, 1, {}], got {:?}",
            heads * head_size,
            r_b1c.dims()
        );
    }
    for (name, tensor) in [
        ("k_B1C", k_b1c),
        ("v_B1C", v_b1c),
        ("w_B1C", w_b1c),
        ("a_B1C", a_b1c),
    ] {
        if tensor.dims() != [batch_size, 1, channels] {
            bail!(
                "single_timestep_b1c expected {name} shape [{batch_size}, 1, {channels}], got {:?}",
                tensor.dims()
            );
        }
    }
    if k_scale_b1h.dims() != [batch_size, 1, heads]
        || v_scale_b1h.dims() != [batch_size, 1, heads]
        || state_b1hkk.dims() != [batch_size, 1, heads, head_size, head_size]
    {
        bail!(
            "single_timestep_b1c expected scale/state shapes [{batch_size}, 1, {heads}] and [{batch_size}, 1, {heads}, {head_size}, {head_size}], got {:?}, {:?}, and {:?}",
            k_scale_b1h.dims(),
            v_scale_b1h.dims(),
            state_b1hkk.dims()
        );
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.validate_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let r_data = f32_tensor_data(r_b1c)?;
    let k_data = f32_tensor_data(k_b1c)?;
    let v_data = f32_tensor_data(v_b1c)?;
    let w_data = f32_tensor_data(w_b1c)?;
    let a_data = f32_tensor_data(a_b1c)?;
    let k_scale_data = f32_tensor_data(k_scale_b1h)?;
    let v_scale_data = f32_tensor_data(v_scale_b1h)?;
    let state_data = f32_tensor_data(state_b1hkk)?;
    let r_values = r_data.as_slice()?;
    let k_values = k_data.as_slice()?;
    let v_values = v_data.as_slice()?;
    let w_values = w_data.as_slice()?;
    let a_values = a_data.as_slice()?;
    let k_scale_values = k_scale_data.as_slice()?;
    let v_scale_values = v_scale_data.as_slice()?;
    let state_values = state_data.as_slice()?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.tensor_read_ns += start.elapsed_ns();
    }

    let state_batch_stride = heads * head_size * head_size;
    let vector_batch_stride = heads * head_size;
    let mut out_values = vec![0.0f32; batch_size * vector_batch_stride];
    let kernel = timestep_output_kernel();
    let start = ProfileTimer::start(profile.is_some());
    if batch_size > 1 {
        out_values
            .par_chunks_mut(vector_batch_stride)
            .enumerate()
            .for_each(|(batch_index, out_batch)| {
                let state_batch =
                    &state_values[batch_index * state_batch_stride..][..state_batch_stride];
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let (k_deformed_vector, k_vector) = normalized_k_vectors(
                        &k_values[vector_base..vector_base + head_size],
                        a_vector,
                        k_scale_values[batch_index * heads + head_index],
                    );
                    let v_vector = normalized_vector(
                        &v_values[vector_base..vector_base + head_size],
                        v_scale_values[batch_index * heads + head_index],
                    );
                    for row_index in 0..head_size {
                        let row_base = local_state_base + row_index * head_size;
                        let state_row = &state_batch[row_base..row_base + head_size];
                        out_batch[local_vector_base + row_index] = timestep_output_row(
                            kernel,
                            state_row,
                            r_vector,
                            &k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            &k_deformed_vector,
                        );
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            let state_batch =
                &state_values[batch_index * state_batch_stride..][..state_batch_stride];
            let out_batch =
                &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                let local_state_base = head_index * head_size * head_size;
                let local_vector_base = head_index * head_size;
                let r_vector = &r_values[vector_base..vector_base + head_size];
                let w_vector = &w_values[vector_base..vector_base + head_size];
                let a_vector = &a_values[vector_base..vector_base + head_size];
                let (k_deformed_vector, k_vector) = normalized_k_vectors(
                    &k_values[vector_base..vector_base + head_size],
                    a_vector,
                    k_scale_values[batch_index * heads + head_index],
                );
                let v_vector = normalized_vector(
                    &v_values[vector_base..vector_base + head_size],
                    v_scale_values[batch_index * heads + head_index],
                );
                for row_index in 0..head_size {
                    let row_base = local_state_base + row_index * head_size;
                    let state_row = &state_batch[row_base..row_base + head_size];
                    out_batch[local_vector_base + row_index] = timestep_output_row(
                        kernel,
                        state_row,
                        r_vector,
                        &k_vector,
                        v_vector[row_index],
                        w_vector,
                        a_vector,
                        &k_deformed_vector,
                    );
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.fused_state_output_ns += start.elapsed_ns();
        let row_count = batch_size * heads * head_size;
        record_timestep_kernel_rows(profile, kernel, row_count, false);
    }

    let start = ProfileTimer::start(profile.is_some());
    let out_bhk = Tensor::from_vec(out_values, (batch_size, heads, head_size), r_b1c.device())?;
    if let Some(profile) = profile {
        profile.tensor_write_ns += start.elapsed_ns();
        profile.total_ns += total_start.elapsed_ns();
    }

    Ok(out_bhk)
}

pub(crate) struct TimeMixerMiddleScratchOutput {
    pub(crate) out_bhk: Option<Tensor>,
    pub(crate) out_values: Vec<f32>,
    pub(crate) out_sum_values: Option<Vec<f32>>,
    pub(crate) out_variance_values: Option<Vec<f32>>,
    pub(crate) next_state_bhkk: Option<Tensor>,
    pub(crate) next_state_values: Option<Vec<f32>>,
    pub(crate) k_values: Vec<f32>,
    pub(crate) v_values: Vec<f32>,
}

fn take_time_mixer_middle_scratch_vec(values: &mut Vec<f32>, len: usize) -> Vec<f32> {
    let mut taken = std::mem::take(values);
    taken.resize(len, 0.0);
    taken
}

fn time_mixer_middle_scratch_vectors(len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    if !time_mixer_middle_buffer_reuse_enabled() {
        return (
            vec![0.0f32; len],
            vec![0.0f32; len],
            vec![0.0f32; len],
            vec![0.0f32; len],
        );
    }

    TIME_MIXER_MIDDLE_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        (
            take_time_mixer_middle_scratch_vec(&mut pool.k_deformed_values, len),
            take_time_mixer_middle_scratch_vec(&mut pool.k_values, len),
            take_time_mixer_middle_scratch_vec(&mut pool.v_values, len),
            take_time_mixer_middle_scratch_vec(&mut pool.out_values, len),
        )
    })
}

fn recycle_time_mixer_middle_temp_vec(values: Vec<f32>) {
    if !time_mixer_middle_buffer_reuse_enabled() {
        return;
    }
    TIME_MIXER_MIDDLE_SCRATCH_POOL.with(|pool| {
        pool.borrow_mut().k_deformed_values = values;
    });
}

pub(crate) fn recycle_time_mixer_middle_scratch_output(output: TimeMixerMiddleScratchOutput) {
    if !time_mixer_middle_buffer_reuse_enabled() {
        return;
    }
    TIME_MIXER_MIDDLE_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        pool.k_values = output.k_values;
        pool.v_values = output.v_values;
        pool.out_values = output.out_values;
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn time_mixer_middle_scratch_profiled(
    r_b1c: &Tensor,
    k_b1c: &Tensor,
    v_b1c: &Tensor,
    w_b1c: &Tensor,
    a_b1c: &Tensor,
    k_scale_b1h: &Tensor,
    v_scale_b1h: &Tensor,
    state_b1hkk: &Tensor,
    heads: usize,
    head_size: usize,
    return_state: bool,
    materialize_out_tensor: bool,
    mut profile: Option<&mut SingleTimestepProfile>,
) -> Result<TimeMixerMiddleScratchOutput> {
    let total_start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, one, channels) = r_b1c.dims3()?;
    if one != 1 || channels != heads * head_size {
        bail!(
            "time_mixer_middle_scratch expected r shape [B, 1, {}], got {:?}",
            heads * head_size,
            r_b1c.dims()
        );
    }
    for (name, tensor) in [
        ("k_B1C", k_b1c),
        ("v_B1C", v_b1c),
        ("w_B1C", w_b1c),
        ("a_B1C", a_b1c),
    ] {
        if tensor.dims() != [batch_size, 1, channels] {
            bail!(
                "time_mixer_middle_scratch expected {name} shape [{batch_size}, 1, {channels}], got {:?}",
                tensor.dims()
            );
        }
    }
    if k_scale_b1h.dims() != [batch_size, 1, heads]
        || v_scale_b1h.dims() != [batch_size, 1, heads]
        || state_b1hkk.dims() != [batch_size, 1, heads, head_size, head_size]
    {
        bail!(
            "time_mixer_middle_scratch expected scale/state shapes [{batch_size}, 1, {heads}] and [{batch_size}, 1, {heads}, {head_size}, {head_size}], got {:?}, {:?}, and {:?}",
            k_scale_b1h.dims(),
            v_scale_b1h.dims(),
            state_b1hkk.dims()
        );
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.validate_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let r_data = f32_tensor_data(r_b1c)?;
    let k_data = f32_tensor_data(k_b1c)?;
    let v_data = f32_tensor_data(v_b1c)?;
    let w_data = f32_tensor_data(w_b1c)?;
    let a_data = f32_tensor_data(a_b1c)?;
    let k_scale_data = f32_tensor_data(k_scale_b1h)?;
    let v_scale_data = f32_tensor_data(v_scale_b1h)?;
    let state_data = f32_tensor_data(state_b1hkk)?;
    let r_values = r_data.as_slice()?;
    let k_source_values = k_data.as_slice()?;
    let v_source_values = v_data.as_slice()?;
    let w_values = w_data.as_slice()?;
    let a_values = a_data.as_slice()?;
    let k_scale_values = k_scale_data.as_slice()?;
    let v_scale_values = v_scale_data.as_slice()?;
    let state_values = state_data.as_slice()?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.tensor_read_ns += start.elapsed_ns();
    }

    let vector_batch_stride = heads * head_size;
    let state_batch_stride = heads * head_size * head_size;
    let vector_value_count = batch_size * vector_batch_stride;
    let (mut k_deformed_values, mut k_values, mut v_values, mut out_values) =
        time_mixer_middle_scratch_vectors(vector_value_count);

    let start = ProfileTimer::start(profile.is_some());
    if batch_size > 1 {
        k_deformed_values
            .par_chunks_mut(vector_batch_stride)
            .zip(k_values.par_chunks_mut(vector_batch_stride))
            .zip(v_values.par_chunks_mut(vector_batch_stride))
            .enumerate()
            .for_each(|(batch_index, ((k_deformed_batch, k_batch), v_batch))| {
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let local_vector_base = head_index * head_size;
                    normalize_scaled_head_into(
                        &k_source_values[vector_base..vector_base + head_size],
                        k_scale_values[batch_index * heads + head_index],
                        &mut k_deformed_batch[local_vector_base..local_vector_base + head_size],
                    );
                    normalize_scaled_head_into(
                        &v_source_values[vector_base..vector_base + head_size],
                        v_scale_values[batch_index * heads + head_index],
                        &mut v_batch[local_vector_base..local_vector_base + head_size],
                    );
                    for offset in 0..head_size {
                        let value_index = local_vector_base + offset;
                        k_batch[value_index] =
                            k_deformed_batch[value_index] * a_values[vector_base + offset];
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                normalize_scaled_head_into(
                    &k_source_values[vector_base..vector_base + head_size],
                    k_scale_values[batch_index * heads + head_index],
                    &mut k_deformed_values[vector_base..vector_base + head_size],
                );
                normalize_scaled_head_into(
                    &v_source_values[vector_base..vector_base + head_size],
                    v_scale_values[batch_index * heads + head_index],
                    &mut v_values[vector_base..vector_base + head_size],
                );
                for offset in 0..head_size {
                    let value_index = vector_base + offset;
                    k_values[value_index] = k_deformed_values[value_index] * a_values[value_index];
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.deformation_prepare_ns += start.elapsed_ns();
    }

    let mut next_state_values = return_state.then(|| vec![0.0f32; state_values.len()]);
    let kernel = if return_state {
        timestep_state_output_kernel()
    } else {
        timestep_output_kernel()
    };
    let start = ProfileTimer::start(profile.is_some());
    if return_state {
        let next_state_values = next_state_values
            .as_mut()
            .expect("next state is allocated when return_state is true");
        if batch_size > 1 {
            next_state_values
                .par_chunks_mut(state_batch_stride)
                .zip(out_values.par_chunks_mut(vector_batch_stride))
                .enumerate()
                .for_each(|(batch_index, (next_state_batch, out_batch))| {
                    for head_index in 0..heads {
                        let vector_base = (batch_index * heads + head_index) * head_size;
                        let state_base = vector_base * head_size;
                        let local_state_base = head_index * head_size * head_size;
                        let local_vector_base = head_index * head_size;
                        let r_vector = &r_values[vector_base..vector_base + head_size];
                        let k_vector = &k_values[vector_base..vector_base + head_size];
                        let v_vector = &v_values[vector_base..vector_base + head_size];
                        let w_vector = &w_values[vector_base..vector_base + head_size];
                        let a_vector = &a_values[vector_base..vector_base + head_size];
                        let k_deformed_vector =
                            &k_deformed_values[vector_base..vector_base + head_size];
                        for row_index in 0..head_size {
                            let row_base = state_base + row_index * head_size;
                            let local_row_base = local_state_base + row_index * head_size;
                            let state_row = &state_values[row_base..row_base + head_size];
                            let next_state_row =
                                &mut next_state_batch[local_row_base..local_row_base + head_size];
                            out_batch[local_vector_base + row_index] = timestep_state_output_row(
                                kernel,
                                state_row,
                                next_state_row,
                                r_vector,
                                k_vector,
                                v_vector[row_index],
                                w_vector,
                                a_vector,
                                k_deformed_vector,
                            );
                        }
                    }
                });
        } else {
            for batch_index in 0..batch_size {
                let next_state_batch = &mut next_state_values[batch_index * state_batch_stride..]
                    [..state_batch_stride];
                let out_batch =
                    &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let state_base = vector_base * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let v_vector = &v_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    for row_index in 0..head_size {
                        let row_base = state_base + row_index * head_size;
                        let local_row_base = local_state_base + row_index * head_size;
                        let state_row = &state_values[row_base..row_base + head_size];
                        let next_state_row =
                            &mut next_state_batch[local_row_base..local_row_base + head_size];
                        out_batch[local_vector_base + row_index] = timestep_state_output_row(
                            kernel,
                            state_row,
                            next_state_row,
                            r_vector,
                            k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            k_deformed_vector,
                        );
                    }
                }
            }
        }
    } else if batch_size > 1 {
        out_values
            .par_chunks_mut(vector_batch_stride)
            .enumerate()
            .for_each(|(batch_index, out_batch)| {
                let state_batch =
                    &state_values[batch_index * state_batch_stride..][..state_batch_stride];
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let v_vector = &v_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    for row_index in 0..head_size {
                        let row_base = local_state_base + row_index * head_size;
                        let state_row = &state_batch[row_base..row_base + head_size];
                        out_batch[local_vector_base + row_index] = timestep_output_row(
                            kernel,
                            state_row,
                            r_vector,
                            k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            k_deformed_vector,
                        );
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            let state_batch =
                &state_values[batch_index * state_batch_stride..][..state_batch_stride];
            let out_batch =
                &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                let local_state_base = head_index * head_size * head_size;
                let local_vector_base = head_index * head_size;
                let r_vector = &r_values[vector_base..vector_base + head_size];
                let k_vector = &k_values[vector_base..vector_base + head_size];
                let v_vector = &v_values[vector_base..vector_base + head_size];
                let w_vector = &w_values[vector_base..vector_base + head_size];
                let a_vector = &a_values[vector_base..vector_base + head_size];
                let k_deformed_vector = &k_deformed_values[vector_base..vector_base + head_size];
                for row_index in 0..head_size {
                    let row_base = local_state_base + row_index * head_size;
                    let state_row = &state_batch[row_base..row_base + head_size];
                    out_batch[local_vector_base + row_index] = timestep_output_row(
                        kernel,
                        state_row,
                        r_vector,
                        k_vector,
                        v_vector[row_index],
                        w_vector,
                        a_vector,
                        k_deformed_vector,
                    );
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.fused_state_output_ns += start.elapsed_ns();
        let row_count = batch_size * heads * head_size;
        record_timestep_kernel_rows(profile, kernel, row_count, return_state);
    }
    recycle_time_mixer_middle_temp_vec(k_deformed_values);

    let start = ProfileTimer::start(profile.is_some());
    let out_bhk = if materialize_out_tensor {
        Some(Tensor::from_vec(
            out_values.clone(),
            (batch_size, heads, head_size),
            r_b1c.device(),
        )?)
    } else {
        None
    };
    let next_state_bhkk = next_state_values
        .map(|values| {
            Tensor::from_vec(
                values,
                (batch_size, heads, head_size, head_size),
                state_b1hkk.device(),
            )
        })
        .transpose()?;
    if let Some(profile) = profile {
        profile.tensor_write_ns += start.elapsed_ns();
        profile.total_ns += total_start.elapsed_ns();
    }

    Ok(TimeMixerMiddleScratchOutput {
        out_bhk,
        out_values,
        out_sum_values: None,
        out_variance_values: None,
        next_state_bhkk,
        next_state_values: None,
        k_values,
        v_values,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn time_mixer_middle_scratch_values_profiled(
    r_b1c: &Tensor,
    k_b1c: &Tensor,
    v_b1c: &Tensor,
    w_values: &[f32],
    a_values: &[f32],
    k_scale_b1h: &Tensor,
    v_scale_b1h: &Tensor,
    state_b1hkk: &Tensor,
    heads: usize,
    head_size: usize,
    return_state: bool,
    materialize_out_tensor: bool,
    mut profile: Option<&mut SingleTimestepProfile>,
) -> Result<TimeMixerMiddleScratchOutput> {
    let total_start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, one, channels) = r_b1c.dims3()?;
    if one != 1 || channels != heads * head_size {
        bail!(
            "time_mixer_middle_scratch expected r shape [B, 1, {}], got {:?}",
            heads * head_size,
            r_b1c.dims()
        );
    }
    for (name, tensor) in [("k_B1C", k_b1c), ("v_B1C", v_b1c)] {
        if tensor.dims() != [batch_size, 1, channels] {
            bail!(
                "time_mixer_middle_scratch expected {name} shape [{batch_size}, 1, {channels}], got {:?}",
                tensor.dims()
            );
        }
    }
    if w_values.len() != batch_size * channels || a_values.len() != batch_size * channels {
        bail!(
            "time_mixer_middle_scratch expected raw w/a values length {}, got {} and {}",
            batch_size * channels,
            w_values.len(),
            a_values.len()
        );
    }
    if k_scale_b1h.dims() != [batch_size, 1, heads]
        || v_scale_b1h.dims() != [batch_size, 1, heads]
        || state_b1hkk.dims() != [batch_size, 1, heads, head_size, head_size]
    {
        bail!(
            "time_mixer_middle_scratch expected scale/state shapes [{batch_size}, 1, {heads}] and [{batch_size}, 1, {heads}, {head_size}, {head_size}], got {:?}, {:?}, and {:?}",
            k_scale_b1h.dims(),
            v_scale_b1h.dims(),
            state_b1hkk.dims()
        );
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.validate_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let r_data = f32_tensor_data(r_b1c)?;
    let k_data = f32_tensor_data(k_b1c)?;
    let v_data = f32_tensor_data(v_b1c)?;
    let k_scale_data = f32_tensor_data(k_scale_b1h)?;
    let v_scale_data = f32_tensor_data(v_scale_b1h)?;
    let state_data = f32_tensor_data(state_b1hkk)?;
    let r_values = r_data.as_slice()?;
    let k_source_values = k_data.as_slice()?;
    let v_source_values = v_data.as_slice()?;
    let k_scale_values = k_scale_data.as_slice()?;
    let v_scale_values = v_scale_data.as_slice()?;
    let state_values = state_data.as_slice()?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.tensor_read_ns += start.elapsed_ns();
    }

    let vector_batch_stride = heads * head_size;
    let state_batch_stride = heads * head_size * head_size;
    let vector_value_count = batch_size * vector_batch_stride;
    let (mut k_deformed_values, mut k_values, mut v_values, mut out_values) =
        time_mixer_middle_scratch_vectors(vector_value_count);

    let start = ProfileTimer::start(profile.is_some());
    if batch_size > 1 {
        k_deformed_values
            .par_chunks_mut(vector_batch_stride)
            .zip(k_values.par_chunks_mut(vector_batch_stride))
            .zip(v_values.par_chunks_mut(vector_batch_stride))
            .enumerate()
            .for_each(|(batch_index, ((k_deformed_batch, k_batch), v_batch))| {
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let local_vector_base = head_index * head_size;
                    normalize_scaled_head_into(
                        &k_source_values[vector_base..vector_base + head_size],
                        k_scale_values[batch_index * heads + head_index],
                        &mut k_deformed_batch[local_vector_base..local_vector_base + head_size],
                    );
                    normalize_scaled_head_into(
                        &v_source_values[vector_base..vector_base + head_size],
                        v_scale_values[batch_index * heads + head_index],
                        &mut v_batch[local_vector_base..local_vector_base + head_size],
                    );
                    for offset in 0..head_size {
                        let value_index = local_vector_base + offset;
                        k_batch[value_index] =
                            k_deformed_batch[value_index] * a_values[vector_base + offset];
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                normalize_scaled_head_into(
                    &k_source_values[vector_base..vector_base + head_size],
                    k_scale_values[batch_index * heads + head_index],
                    &mut k_deformed_values[vector_base..vector_base + head_size],
                );
                normalize_scaled_head_into(
                    &v_source_values[vector_base..vector_base + head_size],
                    v_scale_values[batch_index * heads + head_index],
                    &mut v_values[vector_base..vector_base + head_size],
                );
                for offset in 0..head_size {
                    let value_index = vector_base + offset;
                    k_values[value_index] = k_deformed_values[value_index] * a_values[value_index];
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.deformation_prepare_ns += start.elapsed_ns();
    }

    let mut next_state_values = return_state.then(|| vec![0.0f32; state_values.len()]);
    let kernel = if return_state {
        timestep_state_output_kernel()
    } else {
        timestep_output_kernel()
    };
    let start = ProfileTimer::start(profile.is_some());
    if return_state {
        let next_state_values = next_state_values
            .as_mut()
            .expect("next state is allocated when return_state is true");
        if batch_size > 1 {
            next_state_values
                .par_chunks_mut(state_batch_stride)
                .zip(out_values.par_chunks_mut(vector_batch_stride))
                .enumerate()
                .for_each(|(batch_index, (next_state_batch, out_batch))| {
                    for head_index in 0..heads {
                        let vector_base = (batch_index * heads + head_index) * head_size;
                        let state_base = vector_base * head_size;
                        let local_state_base = head_index * head_size * head_size;
                        let local_vector_base = head_index * head_size;
                        let r_vector = &r_values[vector_base..vector_base + head_size];
                        let k_vector = &k_values[vector_base..vector_base + head_size];
                        let v_vector = &v_values[vector_base..vector_base + head_size];
                        let w_vector = &w_values[vector_base..vector_base + head_size];
                        let a_vector = &a_values[vector_base..vector_base + head_size];
                        let k_deformed_vector =
                            &k_deformed_values[vector_base..vector_base + head_size];
                        for row_index in 0..head_size {
                            let row_base = state_base + row_index * head_size;
                            let local_row_base = local_state_base + row_index * head_size;
                            let state_row = &state_values[row_base..row_base + head_size];
                            let next_state_row =
                                &mut next_state_batch[local_row_base..local_row_base + head_size];
                            out_batch[local_vector_base + row_index] = timestep_state_output_row(
                                kernel,
                                state_row,
                                next_state_row,
                                r_vector,
                                k_vector,
                                v_vector[row_index],
                                w_vector,
                                a_vector,
                                k_deformed_vector,
                            );
                        }
                    }
                });
        } else {
            for batch_index in 0..batch_size {
                let next_state_batch = &mut next_state_values[batch_index * state_batch_stride..]
                    [..state_batch_stride];
                let out_batch =
                    &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let state_base = vector_base * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let v_vector = &v_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    for row_index in 0..head_size {
                        let row_base = state_base + row_index * head_size;
                        let local_row_base = local_state_base + row_index * head_size;
                        let state_row = &state_values[row_base..row_base + head_size];
                        let next_state_row =
                            &mut next_state_batch[local_row_base..local_row_base + head_size];
                        out_batch[local_vector_base + row_index] = timestep_state_output_row(
                            kernel,
                            state_row,
                            next_state_row,
                            r_vector,
                            k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            k_deformed_vector,
                        );
                    }
                }
            }
        }
    } else if batch_size > 1 {
        out_values
            .par_chunks_mut(vector_batch_stride)
            .enumerate()
            .for_each(|(batch_index, out_batch)| {
                let state_batch =
                    &state_values[batch_index * state_batch_stride..][..state_batch_stride];
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let v_vector = &v_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    for row_index in 0..head_size {
                        let row_base = local_state_base + row_index * head_size;
                        let state_row = &state_batch[row_base..row_base + head_size];
                        out_batch[local_vector_base + row_index] = timestep_output_row(
                            kernel,
                            state_row,
                            r_vector,
                            k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            k_deformed_vector,
                        );
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            let state_batch =
                &state_values[batch_index * state_batch_stride..][..state_batch_stride];
            let out_batch =
                &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                let local_state_base = head_index * head_size * head_size;
                let local_vector_base = head_index * head_size;
                let r_vector = &r_values[vector_base..vector_base + head_size];
                let k_vector = &k_values[vector_base..vector_base + head_size];
                let v_vector = &v_values[vector_base..vector_base + head_size];
                let w_vector = &w_values[vector_base..vector_base + head_size];
                let a_vector = &a_values[vector_base..vector_base + head_size];
                let k_deformed_vector = &k_deformed_values[vector_base..vector_base + head_size];
                for row_index in 0..head_size {
                    let row_base = local_state_base + row_index * head_size;
                    let state_row = &state_batch[row_base..row_base + head_size];
                    out_batch[local_vector_base + row_index] = timestep_output_row(
                        kernel,
                        state_row,
                        r_vector,
                        k_vector,
                        v_vector[row_index],
                        w_vector,
                        a_vector,
                        k_deformed_vector,
                    );
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.fused_state_output_ns += start.elapsed_ns();
        let row_count = batch_size * heads * head_size;
        record_timestep_kernel_rows(profile, kernel, row_count, return_state);
    }
    recycle_time_mixer_middle_temp_vec(k_deformed_values);

    let start = ProfileTimer::start(profile.is_some());
    let out_bhk = if materialize_out_tensor {
        Some(Tensor::from_vec(
            out_values.clone(),
            (batch_size, heads, head_size),
            r_b1c.device(),
        )?)
    } else {
        None
    };
    let next_state_bhkk = next_state_values
        .map(|values| {
            Tensor::from_vec(
                values,
                (batch_size, heads, head_size, head_size),
                state_b1hkk.device(),
            )
        })
        .transpose()?;
    if let Some(profile) = profile {
        profile.tensor_write_ns += start.elapsed_ns();
        profile.total_ns += total_start.elapsed_ns();
    }

    Ok(TimeMixerMiddleScratchOutput {
        out_bhk,
        out_values,
        out_sum_values: None,
        out_variance_values: None,
        next_state_bhkk,
        next_state_values: None,
        k_values,
        v_values,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn time_mixer_middle_scratch_wav_values_profiled(
    r_b1c: &Tensor,
    k_b1c: &Tensor,
    v_source_values: &[f32],
    w_values: &[f32],
    a_values: &[f32],
    k_scale_b1h: &Tensor,
    v_scale_b1h: &Tensor,
    state_b1hkk: &Tensor,
    heads: usize,
    head_size: usize,
    return_state: bool,
    materialize_out_tensor: bool,
    mut profile: Option<&mut SingleTimestepProfile>,
) -> Result<TimeMixerMiddleScratchOutput> {
    let total_start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    let (batch_size, one, channels) = r_b1c.dims3()?;
    if one != 1 || channels != heads * head_size {
        bail!(
            "time_mixer_middle_scratch expected r shape [B, 1, {}], got {:?}",
            heads * head_size,
            r_b1c.dims()
        );
    }
    if k_b1c.dims() != [batch_size, 1, channels] {
        bail!(
            "time_mixer_middle_scratch expected k_B1C shape [{batch_size}, 1, {channels}], got {:?}",
            k_b1c.dims()
        );
    }
    let raw_len = batch_size * channels;
    if v_source_values.len() != raw_len || w_values.len() != raw_len || a_values.len() != raw_len {
        bail!(
            "time_mixer_middle_scratch expected raw v/w/a values length {raw_len}, got {}, {}, and {}",
            v_source_values.len(),
            w_values.len(),
            a_values.len()
        );
    }
    if k_scale_b1h.dims() != [batch_size, 1, heads]
        || v_scale_b1h.dims() != [batch_size, 1, heads]
        || state_b1hkk.dims() != [batch_size, 1, heads, head_size, head_size]
    {
        bail!(
            "time_mixer_middle_scratch expected scale/state shapes [{batch_size}, 1, {heads}] and [{batch_size}, 1, {heads}, {head_size}, {head_size}], got {:?}, {:?}, and {:?}",
            k_scale_b1h.dims(),
            v_scale_b1h.dims(),
            state_b1hkk.dims()
        );
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.validate_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let r_data = f32_tensor_data(r_b1c)?;
    let k_data = f32_tensor_data(k_b1c)?;
    let k_scale_data = f32_tensor_data(k_scale_b1h)?;
    let v_scale_data = f32_tensor_data(v_scale_b1h)?;
    let state_data = f32_tensor_data(state_b1hkk)?;
    let r_values = r_data.as_slice()?;
    let k_source_values = k_data.as_slice()?;
    let k_scale_values = k_scale_data.as_slice()?;
    let v_scale_values = v_scale_data.as_slice()?;
    let state_values = state_data.as_slice()?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.tensor_read_ns += start.elapsed_ns();
    }

    let vector_batch_stride = heads * head_size;
    let state_batch_stride = heads * head_size * head_size;
    let vector_value_count = batch_size * vector_batch_stride;
    let (mut k_deformed_values, mut k_values, mut v_values, mut out_values) =
        time_mixer_middle_scratch_vectors(vector_value_count);

    let start = ProfileTimer::start(profile.is_some());
    if batch_size > 1 {
        k_deformed_values
            .par_chunks_mut(vector_batch_stride)
            .zip(k_values.par_chunks_mut(vector_batch_stride))
            .zip(v_values.par_chunks_mut(vector_batch_stride))
            .enumerate()
            .for_each(|(batch_index, ((k_deformed_batch, k_batch), v_batch))| {
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let local_vector_base = head_index * head_size;
                    normalize_scaled_head_into(
                        &k_source_values[vector_base..vector_base + head_size],
                        k_scale_values[batch_index * heads + head_index],
                        &mut k_deformed_batch[local_vector_base..local_vector_base + head_size],
                    );
                    normalize_scaled_head_into(
                        &v_source_values[vector_base..vector_base + head_size],
                        v_scale_values[batch_index * heads + head_index],
                        &mut v_batch[local_vector_base..local_vector_base + head_size],
                    );
                    for offset in 0..head_size {
                        let value_index = local_vector_base + offset;
                        k_batch[value_index] =
                            k_deformed_batch[value_index] * a_values[vector_base + offset];
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                normalize_scaled_head_into(
                    &k_source_values[vector_base..vector_base + head_size],
                    k_scale_values[batch_index * heads + head_index],
                    &mut k_deformed_values[vector_base..vector_base + head_size],
                );
                normalize_scaled_head_into(
                    &v_source_values[vector_base..vector_base + head_size],
                    v_scale_values[batch_index * heads + head_index],
                    &mut v_values[vector_base..vector_base + head_size],
                );
                for offset in 0..head_size {
                    let value_index = vector_base + offset;
                    k_values[value_index] = k_deformed_values[value_index] * a_values[value_index];
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.deformation_prepare_ns += start.elapsed_ns();
    }

    let mut next_state_values = return_state.then(|| vec![0.0f32; state_values.len()]);
    let kernel = if return_state {
        timestep_state_output_kernel()
    } else {
        timestep_output_kernel()
    };
    let start = ProfileTimer::start(profile.is_some());
    if return_state {
        let next_state_values = next_state_values
            .as_mut()
            .expect("next state is allocated when return_state is true");
        if batch_size > 1 {
            next_state_values
                .par_chunks_mut(state_batch_stride)
                .zip(out_values.par_chunks_mut(vector_batch_stride))
                .enumerate()
                .for_each(|(batch_index, (next_state_batch, out_batch))| {
                    for head_index in 0..heads {
                        let vector_base = (batch_index * heads + head_index) * head_size;
                        let state_base = vector_base * head_size;
                        let local_state_base = head_index * head_size * head_size;
                        let local_vector_base = head_index * head_size;
                        let r_vector = &r_values[vector_base..vector_base + head_size];
                        let k_vector = &k_values[vector_base..vector_base + head_size];
                        let v_vector = &v_values[vector_base..vector_base + head_size];
                        let w_vector = &w_values[vector_base..vector_base + head_size];
                        let a_vector = &a_values[vector_base..vector_base + head_size];
                        let k_deformed_vector =
                            &k_deformed_values[vector_base..vector_base + head_size];
                        for row_index in 0..head_size {
                            let row_base = state_base + row_index * head_size;
                            let local_row_base = local_state_base + row_index * head_size;
                            let state_row = &state_values[row_base..row_base + head_size];
                            let next_state_row =
                                &mut next_state_batch[local_row_base..local_row_base + head_size];
                            out_batch[local_vector_base + row_index] = timestep_state_output_row(
                                kernel,
                                state_row,
                                next_state_row,
                                r_vector,
                                k_vector,
                                v_vector[row_index],
                                w_vector,
                                a_vector,
                                k_deformed_vector,
                            );
                        }
                    }
                });
        } else {
            for batch_index in 0..batch_size {
                let next_state_batch = &mut next_state_values[batch_index * state_batch_stride..]
                    [..state_batch_stride];
                let out_batch =
                    &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let state_base = vector_base * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let v_vector = &v_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    for row_index in 0..head_size {
                        let row_base = state_base + row_index * head_size;
                        let local_row_base = local_state_base + row_index * head_size;
                        let state_row = &state_values[row_base..row_base + head_size];
                        let next_state_row =
                            &mut next_state_batch[local_row_base..local_row_base + head_size];
                        out_batch[local_vector_base + row_index] = timestep_state_output_row(
                            kernel,
                            state_row,
                            next_state_row,
                            r_vector,
                            k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            k_deformed_vector,
                        );
                    }
                }
            }
        }
    } else if batch_size > 1 {
        out_values
            .par_chunks_mut(vector_batch_stride)
            .enumerate()
            .for_each(|(batch_index, out_batch)| {
                let state_batch =
                    &state_values[batch_index * state_batch_stride..][..state_batch_stride];
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let v_vector = &v_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    for row_index in 0..head_size {
                        let row_base = local_state_base + row_index * head_size;
                        let state_row = &state_batch[row_base..row_base + head_size];
                        out_batch[local_vector_base + row_index] = timestep_output_row(
                            kernel,
                            state_row,
                            r_vector,
                            k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            k_deformed_vector,
                        );
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            let state_batch =
                &state_values[batch_index * state_batch_stride..][..state_batch_stride];
            let out_batch =
                &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                let local_state_base = head_index * head_size * head_size;
                let local_vector_base = head_index * head_size;
                let r_vector = &r_values[vector_base..vector_base + head_size];
                let k_vector = &k_values[vector_base..vector_base + head_size];
                let v_vector = &v_values[vector_base..vector_base + head_size];
                let w_vector = &w_values[vector_base..vector_base + head_size];
                let a_vector = &a_values[vector_base..vector_base + head_size];
                let k_deformed_vector = &k_deformed_values[vector_base..vector_base + head_size];
                for row_index in 0..head_size {
                    let row_base = local_state_base + row_index * head_size;
                    let state_row = &state_batch[row_base..row_base + head_size];
                    out_batch[local_vector_base + row_index] = timestep_output_row(
                        kernel,
                        state_row,
                        r_vector,
                        k_vector,
                        v_vector[row_index],
                        w_vector,
                        a_vector,
                        k_deformed_vector,
                    );
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.fused_state_output_ns += start.elapsed_ns();
        let row_count = batch_size * heads * head_size;
        record_timestep_kernel_rows(profile, kernel, row_count, return_state);
    }
    recycle_time_mixer_middle_temp_vec(k_deformed_values);

    let start = ProfileTimer::start(profile.is_some());
    let out_bhk = if materialize_out_tensor {
        Some(Tensor::from_vec(
            out_values.clone(),
            (batch_size, heads, head_size),
            r_b1c.device(),
        )?)
    } else {
        None
    };
    let next_state_bhkk = next_state_values
        .map(|values| {
            Tensor::from_vec(
                values,
                (batch_size, heads, head_size, head_size),
                state_b1hkk.device(),
            )
        })
        .transpose()?;
    if let Some(profile) = profile {
        profile.tensor_write_ns += start.elapsed_ns();
        profile.total_ns += total_start.elapsed_ns();
    }

    Ok(TimeMixerMiddleScratchOutput {
        out_bhk,
        out_values,
        out_sum_values: None,
        out_variance_values: None,
        next_state_bhkk,
        next_state_values: None,
        k_values,
        v_values,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn time_mixer_middle_scratch_all_values_profiled(
    r_values: &[f32],
    k_source_values: &[f32],
    v_source_values: &[f32],
    w_values: &[f32],
    a_values: &[f32],
    k_scale_values: &[f32],
    v_scale_values: &[f32],
    state_b1hkk: &Tensor,
    batch_size: usize,
    heads: usize,
    head_size: usize,
    return_state: bool,
    materialize_out_tensor: bool,
    mut profile: Option<&mut SingleTimestepProfile>,
) -> Result<TimeMixerMiddleScratchOutput> {
    let total_start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    let channels = heads * head_size;
    let raw_len = batch_size * channels;
    if r_values.len() != raw_len
        || k_source_values.len() != raw_len
        || v_source_values.len() != raw_len
        || w_values.len() != raw_len
        || a_values.len() != raw_len
    {
        bail!(
            "time_mixer_middle_scratch expected raw r/k/v/w/a values length {raw_len}, got {}, {}, {}, {}, and {}",
            r_values.len(),
            k_source_values.len(),
            v_source_values.len(),
            w_values.len(),
            a_values.len()
        );
    }
    if k_scale_values.len() != batch_size * heads || v_scale_values.len() != batch_size * heads {
        bail!(
            "time_mixer_middle_scratch expected raw scale values length {}, got {} and {}",
            batch_size * heads,
            k_scale_values.len(),
            v_scale_values.len()
        );
    }
    if state_b1hkk.dims() != [batch_size, 1, heads, head_size, head_size] {
        bail!(
            "time_mixer_middle_scratch expected state shape [{batch_size}, 1, {heads}, {head_size}, {head_size}], got {:?}",
            state_b1hkk.dims()
        );
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.validate_ns += start.elapsed_ns();
    }

    let start = ProfileTimer::start(profile.is_some());
    let state_data = f32_tensor_data(state_b1hkk)?;
    let state_values = state_data.as_slice()?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.tensor_read_ns += start.elapsed_ns();
    }

    let vector_batch_stride = heads * head_size;
    let state_batch_stride = heads * head_size * head_size;
    let vector_value_count = batch_size * vector_batch_stride;
    let (mut k_deformed_values, mut k_values, mut v_values, mut out_values) =
        time_mixer_middle_scratch_vectors(vector_value_count);

    let start = ProfileTimer::start(profile.is_some());
    if batch_size > 1 {
        k_deformed_values
            .par_chunks_mut(vector_batch_stride)
            .zip(k_values.par_chunks_mut(vector_batch_stride))
            .zip(v_values.par_chunks_mut(vector_batch_stride))
            .enumerate()
            .for_each(|(batch_index, ((k_deformed_batch, k_batch), v_batch))| {
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let local_vector_base = head_index * head_size;
                    normalize_scaled_head_into(
                        &k_source_values[vector_base..vector_base + head_size],
                        k_scale_values[batch_index * heads + head_index],
                        &mut k_deformed_batch[local_vector_base..local_vector_base + head_size],
                    );
                    normalize_scaled_head_into(
                        &v_source_values[vector_base..vector_base + head_size],
                        v_scale_values[batch_index * heads + head_index],
                        &mut v_batch[local_vector_base..local_vector_base + head_size],
                    );
                    for offset in 0..head_size {
                        let value_index = local_vector_base + offset;
                        k_batch[value_index] =
                            k_deformed_batch[value_index] * a_values[vector_base + offset];
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                normalize_scaled_head_into(
                    &k_source_values[vector_base..vector_base + head_size],
                    k_scale_values[batch_index * heads + head_index],
                    &mut k_deformed_values[vector_base..vector_base + head_size],
                );
                normalize_scaled_head_into(
                    &v_source_values[vector_base..vector_base + head_size],
                    v_scale_values[batch_index * heads + head_index],
                    &mut v_values[vector_base..vector_base + head_size],
                );
                for offset in 0..head_size {
                    let value_index = vector_base + offset;
                    k_values[value_index] = k_deformed_values[value_index] * a_values[value_index];
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.deformation_prepare_ns += start.elapsed_ns();
    }

    let mut next_state_values = return_state.then(|| vec![0.0f32; state_values.len()]);
    let kernel = if return_state {
        timestep_state_output_kernel()
    } else {
        timestep_output_kernel()
    };
    let start = ProfileTimer::start(profile.is_some());
    if return_state {
        let next_state_values = next_state_values
            .as_mut()
            .expect("next state is allocated when return_state is true");
        if batch_size > 1 {
            next_state_values
                .par_chunks_mut(state_batch_stride)
                .zip(out_values.par_chunks_mut(vector_batch_stride))
                .enumerate()
                .for_each(|(batch_index, (next_state_batch, out_batch))| {
                    for head_index in 0..heads {
                        let vector_base = (batch_index * heads + head_index) * head_size;
                        let state_base = vector_base * head_size;
                        let local_state_base = head_index * head_size * head_size;
                        let local_vector_base = head_index * head_size;
                        let r_vector = &r_values[vector_base..vector_base + head_size];
                        let k_vector = &k_values[vector_base..vector_base + head_size];
                        let v_vector = &v_values[vector_base..vector_base + head_size];
                        let w_vector = &w_values[vector_base..vector_base + head_size];
                        let a_vector = &a_values[vector_base..vector_base + head_size];
                        let k_deformed_vector =
                            &k_deformed_values[vector_base..vector_base + head_size];
                        for row_index in 0..head_size {
                            let row_base = state_base + row_index * head_size;
                            let local_row_base = local_state_base + row_index * head_size;
                            let state_row = &state_values[row_base..row_base + head_size];
                            let next_state_row =
                                &mut next_state_batch[local_row_base..local_row_base + head_size];
                            out_batch[local_vector_base + row_index] = timestep_state_output_row(
                                kernel,
                                state_row,
                                next_state_row,
                                r_vector,
                                k_vector,
                                v_vector[row_index],
                                w_vector,
                                a_vector,
                                k_deformed_vector,
                            );
                        }
                    }
                });
        } else {
            for batch_index in 0..batch_size {
                let next_state_batch = &mut next_state_values[batch_index * state_batch_stride..]
                    [..state_batch_stride];
                let out_batch =
                    &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let state_base = vector_base * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let v_vector = &v_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    for row_index in 0..head_size {
                        let row_base = state_base + row_index * head_size;
                        let local_row_base = local_state_base + row_index * head_size;
                        let state_row = &state_values[row_base..row_base + head_size];
                        let next_state_row =
                            &mut next_state_batch[local_row_base..local_row_base + head_size];
                        out_batch[local_vector_base + row_index] = timestep_state_output_row(
                            kernel,
                            state_row,
                            next_state_row,
                            r_vector,
                            k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            k_deformed_vector,
                        );
                    }
                }
            }
        }
    } else if batch_size > 1 {
        out_values
            .par_chunks_mut(vector_batch_stride)
            .enumerate()
            .for_each(|(batch_index, out_batch)| {
                let state_batch =
                    &state_values[batch_index * state_batch_stride..][..state_batch_stride];
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let v_vector = &v_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    for row_index in 0..head_size {
                        let row_base = local_state_base + row_index * head_size;
                        let state_row = &state_batch[row_base..row_base + head_size];
                        out_batch[local_vector_base + row_index] = timestep_output_row(
                            kernel,
                            state_row,
                            r_vector,
                            k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            k_deformed_vector,
                        );
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            let state_batch =
                &state_values[batch_index * state_batch_stride..][..state_batch_stride];
            let out_batch =
                &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                let local_state_base = head_index * head_size * head_size;
                let local_vector_base = head_index * head_size;
                let r_vector = &r_values[vector_base..vector_base + head_size];
                let k_vector = &k_values[vector_base..vector_base + head_size];
                let v_vector = &v_values[vector_base..vector_base + head_size];
                let w_vector = &w_values[vector_base..vector_base + head_size];
                let a_vector = &a_values[vector_base..vector_base + head_size];
                let k_deformed_vector = &k_deformed_values[vector_base..vector_base + head_size];
                for row_index in 0..head_size {
                    let row_base = local_state_base + row_index * head_size;
                    let state_row = &state_batch[row_base..row_base + head_size];
                    out_batch[local_vector_base + row_index] = timestep_output_row(
                        kernel,
                        state_row,
                        r_vector,
                        k_vector,
                        v_vector[row_index],
                        w_vector,
                        a_vector,
                        k_deformed_vector,
                    );
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.fused_state_output_ns += start.elapsed_ns();
        let row_count = batch_size * heads * head_size;
        record_timestep_kernel_rows(profile, kernel, row_count, return_state);
    }
    recycle_time_mixer_middle_temp_vec(k_deformed_values);

    let start = ProfileTimer::start(profile.is_some());
    let out_bhk = if materialize_out_tensor {
        Some(Tensor::from_vec(
            out_values.clone(),
            (batch_size, heads, head_size),
            state_b1hkk.device(),
        )?)
    } else {
        None
    };
    let next_state_bhkk = next_state_values
        .map(|values| {
            Tensor::from_vec(
                values,
                (batch_size, heads, head_size, head_size),
                state_b1hkk.device(),
            )
        })
        .transpose()?;
    if let Some(profile) = profile {
        profile.tensor_write_ns += start.elapsed_ns();
        profile.total_ns += total_start.elapsed_ns();
    }

    Ok(TimeMixerMiddleScratchOutput {
        out_bhk,
        out_values,
        out_sum_values: None,
        out_variance_values: None,
        next_state_bhkk,
        next_state_values: None,
        k_values,
        v_values,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn time_mixer_middle_scratch_all_values_flat_state_profiled(
    r_values: &[f32],
    k_source_values: &[f32],
    v_source_values: &[f32],
    w_values: &[f32],
    a_values: &[f32],
    k_scale_values: &[f32],
    v_scale_values: &[f32],
    state_values: &[f32],
    state_is_zero: bool,
    batch_size: usize,
    heads: usize,
    head_size: usize,
    return_state: bool,
    materialize_out_tensor: bool,
    out_device: &Device,
    mut profile: Option<&mut SingleTimestepProfile>,
) -> Result<TimeMixerMiddleScratchOutput> {
    let total_start = ProfileTimer::start(profile.is_some());
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let start = ProfileTimer::start(profile.is_some());
    let channels = heads * head_size;
    let raw_len = batch_size * channels;
    if r_values.len() != raw_len
        || k_source_values.len() != raw_len
        || v_source_values.len() != raw_len
        || w_values.len() != raw_len
        || a_values.len() != raw_len
    {
        bail!(
            "time_mixer_middle_scratch expected raw r/k/v/w/a values length {raw_len}, got {}, {}, {}, {}, and {}",
            r_values.len(),
            k_source_values.len(),
            v_source_values.len(),
            w_values.len(),
            a_values.len()
        );
    }
    if k_scale_values.len() != batch_size * heads || v_scale_values.len() != batch_size * heads {
        bail!(
            "time_mixer_middle_scratch expected raw scale values length {}, got {} and {}",
            batch_size * heads,
            k_scale_values.len(),
            v_scale_values.len()
        );
    }
    let state_len = batch_size * heads * head_size * head_size;
    if state_is_zero {
        if !state_values.is_empty() {
            bail!(
                "time_mixer_middle_scratch expected empty state values for zero-state recurrence, got {}",
                state_values.len()
            );
        }
    } else if state_values.len() != state_len {
        bail!(
            "time_mixer_middle_scratch expected flat state length {state_len}, got {}",
            state_values.len()
        );
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.validate_ns += start.elapsed_ns();
    }

    let vector_batch_stride = heads * head_size;
    let state_batch_stride = heads * head_size * head_size;
    let vector_value_count = batch_size * vector_batch_stride;
    let (mut k_deformed_values, mut k_values, mut v_values, mut out_values) =
        time_mixer_middle_scratch_vectors(vector_value_count);
    let mut out_sum_values = (batch_size == 1 && time_mixer_recurrence_output_sums_enabled())
        .then(|| vec![0.0f32; batch_size * heads]);
    let mut out_variance_values = (batch_size == 1
        && time_mixer_recurrence_output_variances_enabled())
    .then(|| vec![0.0f32; batch_size * heads]);

    let start = ProfileTimer::start(profile.is_some());
    if batch_size > 1 {
        k_deformed_values
            .par_chunks_mut(vector_batch_stride)
            .zip(k_values.par_chunks_mut(vector_batch_stride))
            .zip(v_values.par_chunks_mut(vector_batch_stride))
            .enumerate()
            .for_each(|(batch_index, ((k_deformed_batch, k_batch), v_batch))| {
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let local_vector_base = head_index * head_size;
                    normalize_scaled_head_into(
                        &k_source_values[vector_base..vector_base + head_size],
                        k_scale_values[batch_index * heads + head_index],
                        &mut k_deformed_batch[local_vector_base..local_vector_base + head_size],
                    );
                    normalize_scaled_head_into(
                        &v_source_values[vector_base..vector_base + head_size],
                        v_scale_values[batch_index * heads + head_index],
                        &mut v_batch[local_vector_base..local_vector_base + head_size],
                    );
                    for offset in 0..head_size {
                        let value_index = local_vector_base + offset;
                        k_batch[value_index] =
                            k_deformed_batch[value_index] * a_values[vector_base + offset];
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                normalize_scaled_head_into(
                    &k_source_values[vector_base..vector_base + head_size],
                    k_scale_values[batch_index * heads + head_index],
                    &mut k_deformed_values[vector_base..vector_base + head_size],
                );
                normalize_scaled_head_into(
                    &v_source_values[vector_base..vector_base + head_size],
                    v_scale_values[batch_index * heads + head_index],
                    &mut v_values[vector_base..vector_base + head_size],
                );
                for offset in 0..head_size {
                    let value_index = vector_base + offset;
                    k_values[value_index] = k_deformed_values[value_index] * a_values[value_index];
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.deformation_prepare_ns += start.elapsed_ns();
    }

    let mut next_state_values = return_state.then(|| vec![0.0f32; state_len]);
    let kernel = if return_state {
        timestep_state_output_kernel()
    } else {
        timestep_output_kernel()
    };
    let start = ProfileTimer::start(profile.is_some());
    if state_is_zero {
        let zero_state_stack_row = [0.0f32; TIME_MIXER_ZERO_STATE_STACK_ROW_CAP];
        let zero_state_heap_row;
        let zero_state_row = if time_mixer_zero_state_stack_row_enabled()
            && head_size <= TIME_MIXER_ZERO_STATE_STACK_ROW_CAP
        {
            &zero_state_stack_row[..head_size]
        } else {
            zero_state_heap_row = vec![0.0f32; head_size];
            zero_state_heap_row.as_slice()
        };
        if return_state {
            let next_state_values = next_state_values
                .as_mut()
                .expect("next state is allocated when return_state is true");
            if batch_size > 1 {
                next_state_values
                    .par_chunks_mut(state_batch_stride)
                    .zip(out_values.par_chunks_mut(vector_batch_stride))
                    .enumerate()
                    .for_each(|(batch_index, (next_state_batch, out_batch))| {
                        for head_index in 0..heads {
                            let vector_base = (batch_index * heads + head_index) * head_size;
                            let local_state_base = head_index * head_size * head_size;
                            let local_vector_base = head_index * head_size;
                            let r_vector = &r_values[vector_base..vector_base + head_size];
                            let k_vector = &k_values[vector_base..vector_base + head_size];
                            let v_vector = &v_values[vector_base..vector_base + head_size];
                            let w_vector = &w_values[vector_base..vector_base + head_size];
                            let a_vector = &a_values[vector_base..vector_base + head_size];
                            let k_deformed_vector =
                                &k_deformed_values[vector_base..vector_base + head_size];
                            for row_index in 0..head_size {
                                let local_row_base = local_state_base + row_index * head_size;
                                let next_state_row = &mut next_state_batch
                                    [local_row_base..local_row_base + head_size];
                                out_batch[local_vector_base + row_index] =
                                    timestep_state_output_row(
                                        kernel,
                                        &zero_state_row,
                                        next_state_row,
                                        r_vector,
                                        k_vector,
                                        v_vector[row_index],
                                        w_vector,
                                        a_vector,
                                        k_deformed_vector,
                                    );
                            }
                        }
                    });
            } else {
                for batch_index in 0..batch_size {
                    let next_state_batch = &mut next_state_values
                        [batch_index * state_batch_stride..][..state_batch_stride];
                    let out_batch =
                        &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
                    for head_index in 0..heads {
                        let vector_base = (batch_index * heads + head_index) * head_size;
                        let local_state_base = head_index * head_size * head_size;
                        let local_vector_base = head_index * head_size;
                        let r_vector = &r_values[vector_base..vector_base + head_size];
                        let k_vector = &k_values[vector_base..vector_base + head_size];
                        let v_vector = &v_values[vector_base..vector_base + head_size];
                        let w_vector = &w_values[vector_base..vector_base + head_size];
                        let a_vector = &a_values[vector_base..vector_base + head_size];
                        let k_deformed_vector =
                            &k_deformed_values[vector_base..vector_base + head_size];
                        let mut out_sum = 0.0f32;
                        for row_index in 0..head_size {
                            let local_row_base = local_state_base + row_index * head_size;
                            let next_state_row =
                                &mut next_state_batch[local_row_base..local_row_base + head_size];
                            let out_value = timestep_state_output_row(
                                kernel,
                                &zero_state_row,
                                next_state_row,
                                r_vector,
                                k_vector,
                                v_vector[row_index],
                                w_vector,
                                a_vector,
                                k_deformed_vector,
                            );
                            out_batch[local_vector_base + row_index] = out_value;
                            out_sum += out_value;
                        }
                        if let Some(out_sum_values) = out_sum_values.as_mut() {
                            out_sum_values[batch_index * heads + head_index] = out_sum;
                        }
                        if let Some(out_variance_values) = out_variance_values.as_mut() {
                            let mean = out_sum / head_size as f32;
                            let group =
                                &out_batch[local_vector_base..local_vector_base + head_size];
                            out_variance_values[batch_index * heads + head_index] = group
                                .iter()
                                .map(|value| {
                                    let centered = *value - mean;
                                    centered * centered
                                })
                                .sum::<f32>()
                                / head_size as f32;
                        }
                    }
                }
            }
        } else if batch_size > 1 {
            out_values
                .par_chunks_mut(vector_batch_stride)
                .enumerate()
                .for_each(|(batch_index, out_batch)| {
                    for head_index in 0..heads {
                        let vector_base = (batch_index * heads + head_index) * head_size;
                        let local_vector_base = head_index * head_size;
                        let r_vector = &r_values[vector_base..vector_base + head_size];
                        let k_vector = &k_values[vector_base..vector_base + head_size];
                        let v_vector = &v_values[vector_base..vector_base + head_size];
                        let w_vector = &w_values[vector_base..vector_base + head_size];
                        let a_vector = &a_values[vector_base..vector_base + head_size];
                        let k_deformed_vector =
                            &k_deformed_values[vector_base..vector_base + head_size];
                        for row_index in 0..head_size {
                            out_batch[local_vector_base + row_index] = timestep_output_row(
                                kernel,
                                &zero_state_row,
                                r_vector,
                                k_vector,
                                v_vector[row_index],
                                w_vector,
                                a_vector,
                                k_deformed_vector,
                            );
                        }
                    }
                });
        } else {
            for batch_index in 0..batch_size {
                let out_batch =
                    &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let v_vector = &v_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    let mut out_sum = 0.0f32;
                    for row_index in 0..head_size {
                        let out_value = timestep_output_row(
                            kernel,
                            &zero_state_row,
                            r_vector,
                            k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            k_deformed_vector,
                        );
                        out_batch[local_vector_base + row_index] = out_value;
                        out_sum += out_value;
                    }
                    if let Some(out_sum_values) = out_sum_values.as_mut() {
                        out_sum_values[batch_index * heads + head_index] = out_sum;
                    }
                    if let Some(out_variance_values) = out_variance_values.as_mut() {
                        let mean = out_sum / head_size as f32;
                        let group = &out_batch[local_vector_base..local_vector_base + head_size];
                        out_variance_values[batch_index * heads + head_index] = group
                            .iter()
                            .map(|value| {
                                let centered = *value - mean;
                                centered * centered
                            })
                            .sum::<f32>()
                            / head_size as f32;
                    }
                }
            }
        }
    } else if return_state {
        let next_state_values = next_state_values
            .as_mut()
            .expect("next state is allocated when return_state is true");
        if batch_size > 1 {
            next_state_values
                .par_chunks_mut(state_batch_stride)
                .zip(out_values.par_chunks_mut(vector_batch_stride))
                .enumerate()
                .for_each(|(batch_index, (next_state_batch, out_batch))| {
                    for head_index in 0..heads {
                        let vector_base = (batch_index * heads + head_index) * head_size;
                        let state_base = vector_base * head_size;
                        let local_state_base = head_index * head_size * head_size;
                        let local_vector_base = head_index * head_size;
                        let r_vector = &r_values[vector_base..vector_base + head_size];
                        let k_vector = &k_values[vector_base..vector_base + head_size];
                        let v_vector = &v_values[vector_base..vector_base + head_size];
                        let w_vector = &w_values[vector_base..vector_base + head_size];
                        let a_vector = &a_values[vector_base..vector_base + head_size];
                        let k_deformed_vector =
                            &k_deformed_values[vector_base..vector_base + head_size];
                        for row_index in 0..head_size {
                            let row_base = state_base + row_index * head_size;
                            let local_row_base = local_state_base + row_index * head_size;
                            let state_row = &state_values[row_base..row_base + head_size];
                            let next_state_row =
                                &mut next_state_batch[local_row_base..local_row_base + head_size];
                            out_batch[local_vector_base + row_index] = timestep_state_output_row(
                                kernel,
                                state_row,
                                next_state_row,
                                r_vector,
                                k_vector,
                                v_vector[row_index],
                                w_vector,
                                a_vector,
                                k_deformed_vector,
                            );
                        }
                    }
                });
        } else {
            for batch_index in 0..batch_size {
                let next_state_batch = &mut next_state_values[batch_index * state_batch_stride..]
                    [..state_batch_stride];
                let out_batch =
                    &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let state_base = vector_base * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let v_vector = &v_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    let mut out_sum = 0.0f32;
                    for row_index in 0..head_size {
                        let row_base = state_base + row_index * head_size;
                        let local_row_base = local_state_base + row_index * head_size;
                        let state_row = &state_values[row_base..row_base + head_size];
                        let next_state_row =
                            &mut next_state_batch[local_row_base..local_row_base + head_size];
                        let out_value = timestep_state_output_row(
                            kernel,
                            state_row,
                            next_state_row,
                            r_vector,
                            k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            k_deformed_vector,
                        );
                        out_batch[local_vector_base + row_index] = out_value;
                        out_sum += out_value;
                    }
                    if let Some(out_sum_values) = out_sum_values.as_mut() {
                        out_sum_values[batch_index * heads + head_index] = out_sum;
                    }
                    if let Some(out_variance_values) = out_variance_values.as_mut() {
                        let mean = out_sum / head_size as f32;
                        let group = &out_batch[local_vector_base..local_vector_base + head_size];
                        out_variance_values[batch_index * heads + head_index] = group
                            .iter()
                            .map(|value| {
                                let centered = *value - mean;
                                centered * centered
                            })
                            .sum::<f32>()
                            / head_size as f32;
                    }
                }
            }
        }
    } else if batch_size > 1 {
        out_values
            .par_chunks_mut(vector_batch_stride)
            .enumerate()
            .for_each(|(batch_index, out_batch)| {
                let state_batch =
                    &state_values[batch_index * state_batch_stride..][..state_batch_stride];
                for head_index in 0..heads {
                    let vector_base = (batch_index * heads + head_index) * head_size;
                    let local_state_base = head_index * head_size * head_size;
                    let local_vector_base = head_index * head_size;
                    let r_vector = &r_values[vector_base..vector_base + head_size];
                    let k_vector = &k_values[vector_base..vector_base + head_size];
                    let v_vector = &v_values[vector_base..vector_base + head_size];
                    let w_vector = &w_values[vector_base..vector_base + head_size];
                    let a_vector = &a_values[vector_base..vector_base + head_size];
                    let k_deformed_vector =
                        &k_deformed_values[vector_base..vector_base + head_size];
                    for row_index in 0..head_size {
                        let row_base = local_state_base + row_index * head_size;
                        let state_row = &state_batch[row_base..row_base + head_size];
                        out_batch[local_vector_base + row_index] = timestep_output_row(
                            kernel,
                            state_row,
                            r_vector,
                            k_vector,
                            v_vector[row_index],
                            w_vector,
                            a_vector,
                            k_deformed_vector,
                        );
                    }
                }
            });
    } else {
        for batch_index in 0..batch_size {
            let state_batch =
                &state_values[batch_index * state_batch_stride..][..state_batch_stride];
            let out_batch =
                &mut out_values[batch_index * vector_batch_stride..][..vector_batch_stride];
            for head_index in 0..heads {
                let vector_base = (batch_index * heads + head_index) * head_size;
                let local_state_base = head_index * head_size * head_size;
                let local_vector_base = head_index * head_size;
                let r_vector = &r_values[vector_base..vector_base + head_size];
                let k_vector = &k_values[vector_base..vector_base + head_size];
                let v_vector = &v_values[vector_base..vector_base + head_size];
                let w_vector = &w_values[vector_base..vector_base + head_size];
                let a_vector = &a_values[vector_base..vector_base + head_size];
                let k_deformed_vector = &k_deformed_values[vector_base..vector_base + head_size];
                let mut out_sum = 0.0f32;
                for row_index in 0..head_size {
                    let row_base = local_state_base + row_index * head_size;
                    let state_row = &state_batch[row_base..row_base + head_size];
                    let out_value = timestep_output_row(
                        kernel,
                        state_row,
                        r_vector,
                        k_vector,
                        v_vector[row_index],
                        w_vector,
                        a_vector,
                        k_deformed_vector,
                    );
                    out_batch[local_vector_base + row_index] = out_value;
                    out_sum += out_value;
                }
                if let Some(out_sum_values) = out_sum_values.as_mut() {
                    out_sum_values[batch_index * heads + head_index] = out_sum;
                }
                if let Some(out_variance_values) = out_variance_values.as_mut() {
                    let mean = out_sum / head_size as f32;
                    let group = &out_batch[local_vector_base..local_vector_base + head_size];
                    out_variance_values[batch_index * heads + head_index] = group
                        .iter()
                        .map(|value| {
                            let centered = *value - mean;
                            centered * centered
                        })
                        .sum::<f32>()
                        / head_size as f32;
                }
            }
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.fused_state_output_ns += start.elapsed_ns();
        let row_count = batch_size * heads * head_size;
        record_timestep_kernel_rows(profile, kernel, row_count, return_state);
    }
    recycle_time_mixer_middle_temp_vec(k_deformed_values);

    let start = ProfileTimer::start(profile.is_some());
    let out_bhk = if materialize_out_tensor {
        Some(Tensor::from_vec(
            out_values.clone(),
            (batch_size, heads, head_size),
            out_device,
        )?)
    } else {
        None
    };
    if let Some(profile) = profile {
        profile.tensor_write_ns += start.elapsed_ns();
        profile.total_ns += total_start.elapsed_ns();
    }

    Ok(TimeMixerMiddleScratchOutput {
        out_bhk,
        out_values,
        out_sum_values,
        out_variance_values,
        next_state_bhkk: None,
        next_state_values,
        k_values,
        v_values,
    })
}

fn normalize_scaled_head_into(values: &[f32], scale: f32, out: &mut [f32]) {
    debug_assert_eq!(values.len(), out.len());
    let mut squared_sum = 0.0f32;
    for value in values {
        squared_sum += value * value;
    }
    let norm = squared_sum.sqrt().max(L2_NORM_MIN);
    for index in 0..values.len() {
        out[index] = values[index] / norm * scale;
    }
}

fn normalized_vector(values: &[f32], scale: f32) -> Vec<f32> {
    let squared_sum = values.iter().map(|value| value * value).sum::<f32>();
    let norm = squared_sum.sqrt().max(L2_NORM_MIN);
    values.iter().map(|value| value / norm * scale).collect()
}

fn normalized_k_vectors(values: &[f32], a_values: &[f32], scale: f32) -> (Vec<f32>, Vec<f32>) {
    let k_deformed = normalized_vector(values, scale);
    let k = k_deformed
        .iter()
        .zip(a_values)
        .map(|(k, a)| k * a)
        .collect();
    (k_deformed, k)
}

#[allow(clippy::useless_conversion)]
#[pyfunction(name = "forgetting_curve")]
pub fn forgetting_curve_py(
    weights: Vec<Vec<f32>>,
    elapsed_seconds: Vec<f32>,
) -> PyResult<Vec<f32>> {
    let weights = tensor_from_2d(weights, "weights").map_err(py_value_error)?;
    let elapsed = elapsed_tensor(elapsed_seconds, weights.dims()[0]).map_err(py_value_error)?;
    forgetting_curve(&weights, &elapsed)
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(py_value_error)
}

#[allow(clippy::useless_conversion)]
#[pyfunction(name = "interp")]
pub fn interp_py(ahead_logits: Vec<Vec<f32>>, elapsed_seconds: Vec<f32>) -> PyResult<Vec<f32>> {
    let ahead_logits = tensor_from_2d(ahead_logits, "ahead_logits").map_err(py_value_error)?;
    let elapsed =
        elapsed_tensor(elapsed_seconds, ahead_logits.dims()[0]).map_err(py_value_error)?;
    interp(&ahead_logits, &elapsed)
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(py_value_error)
}

#[allow(clippy::useless_conversion)]
#[pyfunction(name = "predict_curve")]
pub fn predict_curve_py(
    ahead_logits: Vec<Vec<f32>>,
    weights: Vec<Vec<f32>>,
    elapsed_seconds: Vec<f32>,
) -> PyResult<Vec<f32>> {
    let ahead_logits = tensor_from_2d(ahead_logits, "ahead_logits").map_err(py_value_error)?;
    let weights = tensor_from_2d(weights, "weights").map_err(py_value_error)?;
    let elapsed = elapsed_tensor(elapsed_seconds, weights.dims()[0]).map_err(py_value_error)?;
    predict_curve(&ahead_logits, &weights, &elapsed)
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(py_value_error)
}

#[allow(clippy::useless_conversion)]
#[pyfunction(name = "curve_interval")]
pub fn curve_interval_py(
    ahead_logits: Vec<f32>,
    weights: Vec<f32>,
    retention_probability: f32,
) -> PyResult<Option<f32>> {
    curve_interval(&ahead_logits, &weights, retention_probability).map_err(py_value_error)
}

#[allow(clippy::useless_conversion)]
#[pyfunction(name = "l2_normalize_last_dim")]
pub fn l2_normalize_last_dim_py(values: Tensor3List) -> PyResult<Tensor3List> {
    let values = tensor_from_3d(values, "values").map_err(py_value_error)?;
    l2_normalize_last_dim(&values)
        .and_then(|tensor| tensor.to_vec3::<f32>())
        .map_err(py_value_error)
}

#[allow(clippy::useless_conversion)]
#[pyfunction(name = "layer_norm_2d")]
pub fn layer_norm_2d_py(
    values: Vec<Vec<f32>>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    eps: f32,
) -> PyResult<Vec<Vec<f32>>> {
    let values = tensor_from_2d(values, "values").map_err(py_value_error)?;
    let width = values.dims()[1];
    let weight = tensor_from_1d(weight, width, "weight").map_err(py_value_error)?;
    let bias = tensor_from_1d(bias, width, "bias").map_err(py_value_error)?;
    layer_norm_last_dim(&values, &weight, &bias, eps)
        .and_then(|tensor| tensor.to_vec2::<f32>())
        .map_err(py_value_error)
}

#[allow(clippy::useless_conversion)]
#[pyfunction(name = "group_norm_2d")]
pub fn group_norm_2d_py(
    values: Vec<Vec<f32>>,
    num_groups: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
    eps: f64,
) -> PyResult<Vec<Vec<f32>>> {
    let values = tensor_from_2d(values, "values").map_err(py_value_error)?;
    let width = values.dims()[1];
    let weight = tensor_from_1d(weight, width, "weight").map_err(py_value_error)?;
    let bias = tensor_from_1d(bias, width, "bias").map_err(py_value_error)?;
    group_norm_2d(&values, num_groups, &weight, &bias, eps)
        .and_then(|tensor| tensor.to_vec2::<f32>())
        .map_err(py_value_error)
}

#[allow(clippy::useless_conversion)]
#[pyfunction(name = "single_timestep")]
pub fn single_timestep_py(
    r_bhk: Tensor3List,
    k_bhk: Tensor3List,
    v_bhk: Tensor3List,
    w_bhk: Tensor3List,
    a_bhk: Tensor3List,
    k_deformed_bhk: Tensor3List,
    state_bhkk: Tensor4List,
) -> PyResult<(Tensor3List, Tensor4List)> {
    let r_bhk = tensor_from_3d(r_bhk, "r_BHK").map_err(py_value_error)?;
    let k_bhk = tensor_from_3d(k_bhk, "k_BHK").map_err(py_value_error)?;
    let v_bhk = tensor_from_3d(v_bhk, "v_BHK").map_err(py_value_error)?;
    let w_bhk = tensor_from_3d(w_bhk, "w_BHK").map_err(py_value_error)?;
    let a_bhk = tensor_from_3d(a_bhk, "a_BHK").map_err(py_value_error)?;
    let k_deformed_bhk =
        tensor_from_3d(k_deformed_bhk, "k_deformed_BHK").map_err(py_value_error)?;
    let state_bhkk = tensor_from_4d(state_bhkk, "state_BHKK").map_err(py_value_error)?;

    let (out_bhk, state_bhkk) = single_timestep(
        &r_bhk,
        &k_bhk,
        &v_bhk,
        &w_bhk,
        &a_bhk,
        &k_deformed_bhk,
        &state_bhkk,
    )
    .map_err(py_value_error)?;

    Ok((
        out_bhk.to_vec3::<f32>().map_err(py_value_error)?,
        tensor_to_vec4(&state_bhkk).map_err(py_value_error)?,
    ))
}

fn s_space() -> Vec<f32> {
    let scale = (S_MAX - S_POINT_SPREAD).exp();
    linspace(0.0, S_POINT_SPREAD, NUM_CURVES)
        .into_iter()
        .map(|x| 0.1 + (x.exp() - 1.0) * scale)
        .collect()
}

fn curve_probability_scalar(
    ahead_logits: &[f32],
    weights: &[f32],
    elapsed_seconds: f32,
    s_space: &[f32],
    point_space: &[f32],
) -> Result<f64> {
    let elapsed = elapsed_seconds.max(1.0);
    if elapsed > point_space[point_space.len() - 1] {
        bail!(
            "curve_probability elapsed_seconds {elapsed} exceeds point-space max {}",
            point_space[point_space.len() - 1]
        );
    }

    let curve_probs_raw = weights
        .iter()
        .zip(s_space)
        .map(|(weight, s)| (*weight as f64) * ((elapsed as f64 / *s as f64) * (0.9f64).ln()).exp())
        .sum::<f64>()
        .mul_add(PROBABILITY_SCALE, PROBABILITY_EPS);
    let curve_logits_raw = (curve_probs_raw / (1.0 - curve_probs_raw)).ln();

    let right_idx = searchsorted_left(point_space, elapsed);
    let left_idx = right_idx.saturating_sub(1);
    let xl = point_space[left_idx];
    let xr = point_space[right_idx];
    let yl = ahead_logits[left_idx];
    let yr = ahead_logits[right_idx];
    let interpolated = yl + (yr - yl) * (elapsed - xl) / (xr - xl);
    let ahead_logit_residual = PROBABILITY_EPS + PROBABILITY_SCALE * interpolated as f64;

    Ok(sigmoid_scalar(curve_logits_raw + ahead_logit_residual))
}

fn sigmoid_scalar(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn point_space() -> Vec<f32> {
    let scale = (MAX_E - POINT_SPREAD).exp();
    linspace(0.0, POINT_SPREAD, NUM_POINTS)
        .into_iter()
        .map(|x| 0.5 + (x.exp() - 1.0) * scale)
        .collect()
}

fn linspace(start: f32, end: f32, count: usize) -> Vec<f32> {
    if count == 1 {
        return vec![start];
    }
    let step = (end - start) / (count - 1) as f32;
    (0..count)
        .map(|index| start + step * index as f32)
        .collect()
}

fn searchsorted_left(space: &[f32], value: f32) -> usize {
    space.partition_point(|candidate| *candidate < value)
}

fn clamped_elapsed_vec(elapsed_seconds: &Tensor, batch_size: usize) -> Result<Vec<f32>> {
    let elapsed = elapsed_seconds.flatten_all()?.to_vec1::<f32>()?;
    if elapsed.len() != batch_size {
        bail!(
            "expected {batch_size} elapsed_seconds values, got {}",
            elapsed.len()
        );
    }
    Ok(elapsed.into_iter().map(|value| value.max(1.0)).collect())
}

fn elapsed_tensor(elapsed_seconds: Vec<f32>, batch_size: usize) -> Result<Tensor> {
    if elapsed_seconds.len() != batch_size {
        bail!(
            "expected {batch_size} elapsed_seconds values, got {}",
            elapsed_seconds.len()
        );
    }
    Tensor::from_vec(elapsed_seconds, (batch_size, 1), &Device::Cpu)
}

fn py_value_error(err: candle_core::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction(name = "simd_status")]
pub fn simd_status_py(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    let selected_linear_kernel = if native_linear_kernel_enabled() {
        cpu_simd_kernel(!native_linear_simd_disabled())
    } else {
        TimestepOutputKernel::Scalar
    };
    dict.set_item("disabled_by_env", simd_disabled())?;
    dict.set_item("requested_backend", simd_backend_preference().as_str())?;
    dict.set_item("state_simd_enabled_by_env", state_simd_enabled())?;
    dict.set_item("avx2_available", avx2_available())?;
    dict.set_item("fma_available", fma_available())?;
    dict.set_item("custom_avx2_fma_available", custom_avx2_fma_available())?;
    dict.set_item("pulp_available", pulp_simd_available())?;
    dict.set_item("pulp_arch", pulp_arch_name())?;
    dict.set_item("pulp_f32_lanes", pulp_f32_lanes())?;
    dict.set_item(
        "linear_kernel",
        if matches!(selected_linear_kernel, TimestepOutputKernel::Scalar) {
            "candle_fallback"
        } else {
            timestep_kernel_name(selected_linear_kernel)
        },
    )?;
    dict.set_item(
        "predict_output_kernel",
        timestep_kernel_name(timestep_output_kernel()),
    )?;
    dict.set_item(
        "state_output_kernel",
        timestep_kernel_name(timestep_state_output_kernel()),
    )?;
    Ok(dict.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn forgetting_curve_clamps_elapsed_seconds_to_one() {
        let weights = Tensor::from_vec(
            vec![1.0 / NUM_CURVES as f32; NUM_CURVES],
            (1usize, NUM_CURVES),
            &Device::Cpu,
        )
        .unwrap();

        let elapsed_zero = Tensor::from_vec(vec![0.0f32], (1usize, 1usize), &Device::Cpu).unwrap();
        let elapsed_one = Tensor::from_vec(vec![1.0f32], (1usize, 1usize), &Device::Cpu).unwrap();

        let zero = forgetting_curve(&weights, &elapsed_zero).unwrap();
        let one = forgetting_curve(&weights, &elapsed_one).unwrap();

        assert_close(
            &zero.to_vec1::<f32>().unwrap(),
            &one.to_vec1::<f32>().unwrap(),
            0.0,
        );
    }

    #[test]
    fn group_norm_2d_normalizes_each_group() {
        let values = Tensor::new(&[[1.0f32, 3.0, 2.0, 6.0]], &Device::Cpu).unwrap();
        let weight = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &Device::Cpu).unwrap();
        let bias = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &Device::Cpu).unwrap();

        let actual = group_norm_2d(&values, 2, &weight, &bias, 0.0).unwrap();

        assert_close(
            &actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &[-1.0, 1.0, -1.0, 1.0],
            1e-6,
        );
    }

    #[test]
    fn native_centered_layer_norm_matches_candle_slow_rank2() {
        let values = Tensor::new(
            &[[1.0f32, 3.0, 2.0, 6.0], [-2.0, 0.5, 4.0, 8.0]],
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::new(&[1.0f32, 0.5, -1.0, 2.0], &Device::Cpu).unwrap();
        let bias = Tensor::new(&[0.0f32, -0.25, 1.0, 0.5], &Device::Cpu).unwrap();

        let expected = nn_ops::layer_norm_slow(&values, &weight, &bias, 1e-5).unwrap();
        let actual = layer_norm_last_dim_native_centered(&values, &weight, &bias, 1e-5).unwrap();

        assert_close(
            &actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            1e-6,
        );
    }

    #[test]
    fn native_centered_layer_norm_matches_candle_slow_rank3() {
        let values = Tensor::from_vec(
            vec![1.0f32, 3.0, 2.0, 6.0, -2.0, 0.5, 4.0, 8.0],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::new(&[1.0f32, 0.5, -1.0, 2.0], &Device::Cpu).unwrap();
        let bias = Tensor::new(&[0.0f32, -0.25, 1.0, 0.5], &Device::Cpu).unwrap();

        let expected = nn_ops::layer_norm_slow(&values, &weight, &bias, 1e-5).unwrap();
        let actual = layer_norm_last_dim_native_centered(&values, &weight, &bias, 1e-5).unwrap();

        assert_close(
            &actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            1e-6,
        );
    }

    #[test]
    fn l2_normalize_scaled_b1hk_normalizes_and_scales_heads() {
        let values = Tensor::from_vec(
            vec![3.0f32, 4.0, 1.0, 2.0, 2.0, 4.0, 0.0, 5.0],
            (1usize, 1usize, 2usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let scale =
            Tensor::from_vec(vec![2.0f32, 0.5], (1usize, 1usize, 2usize), &Device::Cpu).unwrap();

        let actual = l2_normalize_scaled_b1hk(&values, &scale).unwrap();

        assert_close(
            &actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &[
                1.095445, 1.4605935, 0.36514837, 0.73029673, 0.1490712, 0.2981424, 0.0, 0.372678,
            ],
            1e-6,
        );
    }

    #[test]
    fn time_lerp_parts_b1c_matches_expected_parts() {
        let xs = Tensor::new(&[[[1.0f32, 2.0, 3.0]], [[-1.0, -2.0, -3.0]]], &Device::Cpu).unwrap();
        let x_shift =
            Tensor::new(&[[[2.0f32, 4.0, 6.0]], [[1.0, 2.0, 3.0]]], &Device::Cpu).unwrap();
        let lerp = Tensor::from_vec(
            (0..24).map(|value| value as f32 / 24.0).collect::<Vec<_>>(),
            (8usize, 1usize, 1usize, 3usize),
            &Device::Cpu,
        )
        .unwrap();

        let parts = time_lerp_parts_b1c(&xs, &x_shift, &lerp).unwrap();

        assert_eq!(parts.len(), 8);
        assert_eq!(parts[0].dims(), &[2, 1, 3]);
        assert_close(
            &parts[0].flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &[1.0, 2.0833333, 3.25, -1.0, -1.8333334, -2.5],
            1e-6,
        );
        assert_close(
            &parts[7].flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &[1.875, 3.8333333, 5.875, 0.75, 1.6666667, 2.75],
            1e-6,
        );
    }

    #[test]
    fn time_decay_w_b1c_matches_candle_expression() {
        let values =
            Tensor::new(&[[[-1.0f32, 0.0, 2.0]], [[3.0, -4.0, 0.5]]], &Device::Cpu).unwrap();
        let w = time_decay_w_b1c(&values).unwrap();

        let expected_decay = ((values.neg().unwrap().exp().unwrap() + 1.0)
            .unwrap()
            .log()
            .unwrap()
            + 0.5)
            .unwrap()
            .neg()
            .unwrap();
        let expected_w = expected_decay.exp().unwrap().neg().unwrap().exp().unwrap();

        assert_close(
            &w.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &expected_w.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            1e-6,
        );
    }

    #[test]
    fn fast_time_decay_formula_matches_reference_formula() {
        let values = [
            -50.0f32, -20.0, -10.0, -4.0, -1.0, -0.5, 0.0, 0.5, 1.0, 4.0, 10.0, 20.0, 50.0,
        ];
        let reference = time_decay_w_values_reference(&values);
        let fast = time_decay_w_values_fast(&values);

        assert_close(&fast, &reference, 1e-6);
    }

    #[test]
    fn timestep_output_avx2_matches_scalar_when_available() {
        let len = 37usize;
        let state = (0..len)
            .map(|index| index as f32 * 0.01 - 0.17)
            .collect::<Vec<_>>();
        let r = (0..len)
            .map(|index| 0.2 - index as f32 * 0.003)
            .collect::<Vec<_>>();
        let k = (0..len)
            .map(|index| index as f32 * 0.004 - 0.08)
            .collect::<Vec<_>>();
        let w = (0..len)
            .map(|index| 0.93 + index as f32 * 0.0007)
            .collect::<Vec<_>>();
        let a = (0..len)
            .map(|index| -0.11 + index as f32 * 0.002)
            .collect::<Vec<_>>();
        let k_deformed = (0..len)
            .map(|index| 0.07 - index as f32 * 0.001)
            .collect::<Vec<_>>();

        let scalar = timestep_output_row_scalar(&state, &r, &k, 0.31, &w, &a, &k_deformed);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let simd =
                unsafe { timestep_output_row_avx2_fma(&state, &r, &k, 0.31, &w, &a, &k_deformed) };
            assert!(
                (scalar - simd).abs() <= 1e-6,
                "scalar {scalar} != avx2 {simd}"
            );
        }
    }

    #[test]
    fn algebraic_timestep_output_head_matches_row_recurrence() {
        let head_size = 32usize;
        let state = (0..head_size * head_size)
            .map(|index| (index as f32 * 0.0017).sin() * 0.2)
            .collect::<Vec<_>>();
        let r = (0..head_size)
            .map(|index| 0.2 - index as f32 * 0.003)
            .collect::<Vec<_>>();
        let k_deformed = (0..head_size)
            .map(|index| index as f32 * 0.004 - 0.08)
            .collect::<Vec<_>>();
        let a = (0..head_size)
            .map(|index| 0.4 + index as f32 * 0.002)
            .collect::<Vec<_>>();
        let k = a
            .iter()
            .zip(&k_deformed)
            .map(|(a, k_deformed)| a * k_deformed)
            .collect::<Vec<_>>();
        let v = (0..head_size)
            .map(|index| index as f32 * 0.005 - 0.04)
            .collect::<Vec<_>>();
        let w = (0..head_size)
            .map(|index| 0.93 + index as f32 * 0.0007)
            .collect::<Vec<_>>();

        let expected = (0..head_size)
            .map(|row_index| {
                let row_base = row_index * head_size;
                timestep_output_row_scalar(
                    &state[row_base..row_base + head_size],
                    &r,
                    &k,
                    v[row_index],
                    &w,
                    &a,
                    &k_deformed,
                )
            })
            .collect::<Vec<_>>();
        let mut scalar = vec![0.0f32; head_size];
        timestep_output_head_algebraic_scalar(&state, &mut scalar, &r, &k, &v, &w, &a, &k_deformed);
        assert_close(&scalar, &expected, 2e-6);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut simd = vec![0.0f32; head_size];
            unsafe {
                timestep_output_head_algebraic_avx2_fma(
                    &state,
                    &mut simd,
                    &r,
                    &k,
                    &v,
                    &w,
                    &a,
                    &k_deformed,
                )
            };
            assert_close(&simd, &expected, 2e-6);
        }
    }

    #[test]
    fn timestep_state_output_avx2_matches_scalar_when_available() {
        let len = 37usize;
        let state = (0..len)
            .map(|index| index as f32 * 0.01 - 0.17)
            .collect::<Vec<_>>();
        let r = (0..len)
            .map(|index| 0.2 - index as f32 * 0.003)
            .collect::<Vec<_>>();
        let k = (0..len)
            .map(|index| index as f32 * 0.004 - 0.08)
            .collect::<Vec<_>>();
        let w = (0..len)
            .map(|index| 0.93 + index as f32 * 0.0007)
            .collect::<Vec<_>>();
        let a = (0..len)
            .map(|index| -0.11 + index as f32 * 0.002)
            .collect::<Vec<_>>();
        let k_deformed = (0..len)
            .map(|index| 0.07 - index as f32 * 0.001)
            .collect::<Vec<_>>();

        let mut scalar_state = vec![0.0f32; len];
        let scalar = timestep_state_output_row_scalar(
            &state,
            &mut scalar_state,
            &r,
            &k,
            0.31,
            &w,
            &a,
            &k_deformed,
        );

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut simd_state = vec![0.0f32; len];
            let simd = unsafe {
                timestep_state_output_row_avx2_fma(
                    &state,
                    &mut simd_state,
                    &r,
                    &k,
                    0.31,
                    &w,
                    &a,
                    &k_deformed,
                )
            };
            assert!(
                (scalar - simd).abs() <= 1e-6,
                "scalar {scalar} != avx2 {simd}"
            );
            assert_close(&simd_state, &scalar_state, 1e-6);
        }
    }

    #[test]
    fn single_timestep_preserves_expected_shapes() {
        let input = Tensor::from_vec(
            vec![0.25f32, -0.5, 0.75, 1.0],
            (1usize, 2usize, 2usize),
            &Device::Cpu,
        )
        .unwrap();
        let state = Tensor::zeros(
            (1usize, 2usize, 2usize, 2usize),
            candle_core::DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let (out, next_state) =
            single_timestep(&input, &input, &input, &input, &input, &input, &state).unwrap();

        assert_eq!(out.dims(), &[1, 2, 2]);
        assert_eq!(next_state.dims(), &[1, 2, 2, 2]);
    }

    #[test]
    fn single_timestep_b1c_matches_reshape_sequence() {
        let heads = 2usize;
        let head_size = 2usize;
        let r_b1c = Tensor::from_vec(
            vec![0.1f32, 0.2, -0.3, 0.4, 0.7, -0.6, 0.5, -0.2],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let k_b1c = Tensor::from_vec(
            vec![0.3f32, -0.5, 0.25, 0.75, -0.4, 0.9, 0.6, -0.1],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let v_b1c = Tensor::from_vec(
            vec![1.0f32, -0.25, 0.5, 0.75, -1.0, 0.3, 0.2, -0.6],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let w_b1c = Tensor::from_vec(
            vec![0.8f32, 0.6, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let a_b1c = Tensor::from_vec(
            vec![0.9f32, -0.8, 0.7, 0.6, 0.5, -0.4, 0.3, -0.2],
            (2usize, 1usize, 4usize),
            &Device::Cpu,
        )
        .unwrap();
        let k_scale_b1h =
            Tensor::from_vec(vec![1.2f32, 0.8, 0.7, 1.1], (2, 1, 2), &Device::Cpu).unwrap();
        let v_scale_b1h =
            Tensor::from_vec(vec![0.9f32, 1.3, 1.4, 0.6], (2, 1, 2), &Device::Cpu).unwrap();
        let state_b1hkk = Tensor::from_vec(
            (0..16)
                .map(|index| index as f32 * 0.03 - 0.2)
                .collect::<Vec<_>>(),
            (2usize, 1usize, heads, head_size, head_size),
            &Device::Cpu,
        )
        .unwrap();

        let mut k_b1hk = l2_normalize_scaled_b1hk(
            &k_b1c.reshape((2usize, 1usize, heads, head_size)).unwrap(),
            &k_scale_b1h,
        )
        .unwrap();
        let k_deformed_b1hk = k_b1hk.clone();
        let a_b1hk = a_b1c.reshape((2usize, 1usize, heads, head_size)).unwrap();
        k_b1hk = k_b1hk.broadcast_mul(&a_b1hk).unwrap();
        let v_b1hk = l2_normalize_scaled_b1hk(
            &v_b1c.reshape((2usize, 1usize, heads, head_size)).unwrap(),
            &v_scale_b1h,
        )
        .unwrap();
        let r_bhk = r_b1c
            .reshape((2usize, 1usize, heads, head_size))
            .unwrap()
            .squeeze(1)
            .unwrap();
        let k_bhk = k_b1hk.squeeze(1).unwrap();
        let v_bhk = v_b1hk.squeeze(1).unwrap();
        let w_bhk = w_b1c
            .reshape((2usize, 1usize, heads, head_size))
            .unwrap()
            .squeeze(1)
            .unwrap();
        let a_bhk = a_b1hk.squeeze(1).unwrap();
        let k_deformed_bhk = k_deformed_b1hk.squeeze(1).unwrap();
        let state_bhkk = state_b1hkk.squeeze(1).unwrap();
        let (expected_out, expected_state) = single_timestep_profiled(
            &r_bhk,
            &k_bhk,
            &v_bhk,
            &w_bhk,
            &a_bhk,
            &k_deformed_bhk,
            &state_bhkk,
            None,
        )
        .unwrap();

        let (actual_out, actual_state) = single_timestep_b1c_profiled(
            &r_b1c,
            &k_b1c,
            &v_b1c,
            &w_b1c,
            &a_b1c,
            &k_scale_b1h,
            &v_scale_b1h,
            &state_b1hkk,
            heads,
            head_size,
            None,
        )
        .unwrap();
        let middle_actual = time_mixer_middle_scratch_profiled(
            &r_b1c,
            &k_b1c,
            &v_b1c,
            &w_b1c,
            &a_b1c,
            &k_scale_b1h,
            &v_scale_b1h,
            &state_b1hkk,
            heads,
            head_size,
            true,
            true,
            None,
        )
        .unwrap();
        let middle_output_only = time_mixer_middle_scratch_profiled(
            &r_b1c,
            &k_b1c,
            &v_b1c,
            &w_b1c,
            &a_b1c,
            &k_scale_b1h,
            &v_scale_b1h,
            &state_b1hkk,
            heads,
            head_size,
            false,
            true,
            None,
        )
        .unwrap();
        let w_values = w_b1c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let a_values = a_b1c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let middle_from_values = time_mixer_middle_scratch_values_profiled(
            &r_b1c,
            &k_b1c,
            &v_b1c,
            &w_values,
            &a_values,
            &k_scale_b1h,
            &v_scale_b1h,
            &state_b1hkk,
            heads,
            head_size,
            true,
            true,
            None,
        )
        .unwrap();
        let v_values = v_b1c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let middle_from_wav_values = time_mixer_middle_scratch_wav_values_profiled(
            &r_b1c,
            &k_b1c,
            &v_values,
            &w_values,
            &a_values,
            &k_scale_b1h,
            &v_scale_b1h,
            &state_b1hkk,
            heads,
            head_size,
            true,
            true,
            None,
        )
        .unwrap();
        let r_values = r_b1c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let k_source_values = k_b1c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let k_scale_values = k_scale_b1h.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let v_scale_values = v_scale_b1h.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let middle_from_all_values = time_mixer_middle_scratch_all_values_profiled(
            &r_values,
            &k_source_values,
            &v_values,
            &w_values,
            &a_values,
            &k_scale_values,
            &v_scale_values,
            &state_b1hkk,
            2,
            heads,
            head_size,
            true,
            true,
            None,
        )
        .unwrap();

        assert_close(
            &actual_out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &expected_out
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            1e-6,
        );
        assert_close(
            &middle_actual
                .out_bhk
                .as_ref()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &expected_out
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            1e-6,
        );
        assert_close(
            &middle_output_only
                .out_bhk
                .as_ref()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &expected_out
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            1e-6,
        );
        assert!(middle_output_only.next_state_bhkk.is_none());
        assert_close(
            &actual_state
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &expected_state
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            1e-6,
        );
        assert_close(
            &middle_actual
                .next_state_bhkk
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &expected_state
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            1e-6,
        );
        assert_close(
            &middle_actual.k_values,
            &k_b1hk.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            1e-6,
        );
        assert_close(
            &middle_actual.v_values,
            &v_b1hk.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            1e-6,
        );
        assert_close(
            &middle_from_values
                .out_bhk
                .as_ref()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &expected_out
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            1e-6,
        );
        assert_close(
            &middle_from_wav_values
                .out_bhk
                .as_ref()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &expected_out
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            1e-6,
        );
        assert_close(
            &middle_from_all_values
                .out_bhk
                .as_ref()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &expected_out
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            1e-6,
        );
        assert_close(
            &middle_from_values
                .next_state_bhkk
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            &expected_state
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            1e-6,
        );
    }
}
