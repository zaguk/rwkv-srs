//! Architecture-neutral SIMD selection and Pulp-backed numerical kernels.
//!
//! The existing hand-tuned AVX2/FMA kernels remain the default on supported
//! x86 hosts. Pulp supplies the portable x86-v3 and AArch64/NEON backend used
//! by the flat Fast pipeline on other supported targets and by forced-backend
//! qualification on x86.

use std::array;
use std::sync::OnceLock;

use pulp::{Arch, Simd, WithSimd};

use crate::cpu_config::{simd_backend_preference, SimdBackendPreference};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CpuSimdKernel {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2Fma,
    Pulp,
}

pub(crate) fn cpu_simd_kernel(enabled: bool) -> CpuSimdKernel {
    if !enabled {
        return CpuSimdKernel::Scalar;
    }

    static ENABLED_KERNEL: OnceLock<CpuSimdKernel> = OnceLock::new();
    *ENABLED_KERNEL.get_or_init(select_enabled_cpu_simd_kernel)
}

fn select_enabled_cpu_simd_kernel() -> CpuSimdKernel {
    match simd_backend_preference() {
        SimdBackendPreference::Auto => {
            if custom_avx2_fma_available() {
                return custom_avx2_kernel();
            }
            if pulp_simd_available() {
                CpuSimdKernel::Pulp
            } else {
                CpuSimdKernel::Scalar
            }
        }
        SimdBackendPreference::Avx2 => {
            if custom_avx2_fma_available() {
                custom_avx2_kernel()
            } else {
                CpuSimdKernel::Scalar
            }
        }
        SimdBackendPreference::Pulp => {
            if pulp_simd_available() {
                CpuSimdKernel::Pulp
            } else {
                CpuSimdKernel::Scalar
            }
        }
    }
}

#[inline]
fn custom_avx2_kernel() -> CpuSimdKernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        CpuSimdKernel::Avx2Fma
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        CpuSimdKernel::Scalar
    }
}

pub(crate) fn cpu_simd_kernel_name(kernel: CpuSimdKernel) -> &'static str {
    match kernel {
        CpuSimdKernel::Scalar => "scalar",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        CpuSimdKernel::Avx2Fma => "avx2_fma",
        CpuSimdKernel::Pulp => "pulp",
    }
}

pub(crate) fn custom_avx2_fma_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

fn pulp_arch() -> Arch {
    static ARCH: OnceLock<Arch> = OnceLock::new();
    *ARCH.get_or_init(Arch::new)
}

pub(crate) fn pulp_simd_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        pulp::x86::V3::is_available()
    }
    #[cfg(target_arch = "aarch64")]
    {
        pulp::aarch64::Neon::is_available()
    }
    #[cfg(target_arch = "wasm32")]
    {
        !matches!(pulp_arch(), Arch::Scalar)
    }
    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        false
    }
}

pub(crate) fn pulp_arch_name() -> &'static str {
    if !pulp_simd_available() {
        return "scalar";
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return "x86_v3";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    #[cfg(target_arch = "wasm32")]
    {
        return "simd128";
    }
    #[allow(unreachable_code)]
    "scalar"
}

struct LaneCount;

impl WithSimd for LaneCount {
    type Output = usize;

    #[inline(always)]
    fn with_simd<S: Simd>(self, _simd: S) -> Self::Output {
        S::F32_LANES
    }
}

pub(crate) fn pulp_f32_lanes() -> usize {
    if pulp_simd_available() {
        pulp_arch().dispatch(LaneCount)
    } else {
        1
    }
}

#[inline]
fn dispatch_pulp<Op: WithSimd>(op: Op) -> Op::Output {
    debug_assert!(pulp_simd_available());
    pulp_arch().dispatch(op)
}

struct LinearDot<'a> {
    xs: &'a [f32],
    weights: &'a [f32],
}

impl WithSimd for LinearDot<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        linear_dot_simd(simd, self.xs, self.weights)
    }
}

pub(crate) fn pulp_linear_dot(xs: &[f32], weights: &[f32]) -> f32 {
    debug_assert_eq!(xs.len(), weights.len());
    dispatch_pulp(LinearDot { xs, weights })
}

#[inline(always)]
fn linear_dot_simd<S: Simd>(simd: S, xs: &[f32], weights: &[f32]) -> f32 {
    let (xs_vectors, xs_tail) = S::as_simd_f32s(xs);
    let (weight_vectors, weight_tail) = S::as_simd_f32s(weights);
    let mut acc = simd.splat_f32s(0.0);
    for index in 0..xs_vectors.len() {
        acc = simd.mul_add_f32s(xs_vectors[index], weight_vectors[index], acc);
    }
    let mut output = simd.reduce_sum_f32s(acc);
    for index in 0..xs_tail.len() {
        output = xs_tail[index].mul_add(weight_tail[index], output);
    }
    output
}

struct LinearDotPair<'a> {
    left_xs: &'a [f32],
    left_weights: &'a [f32],
    right_xs: &'a [f32],
    right_weights: &'a [f32],
}

impl WithSimd for LinearDotPair<'_> {
    type Output = (f32, f32);

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (left_xs, left_tail) = S::as_simd_f32s(self.left_xs);
        let (left_weights, left_weight_tail) = S::as_simd_f32s(self.left_weights);
        let (right_xs, right_tail) = S::as_simd_f32s(self.right_xs);
        let (right_weights, right_weight_tail) = S::as_simd_f32s(self.right_weights);
        let mut left_acc = simd.splat_f32s(0.0);
        let mut right_acc = simd.splat_f32s(0.0);
        for index in 0..left_xs.len() {
            left_acc = simd.mul_add_f32s(left_xs[index], left_weights[index], left_acc);
            right_acc = simd.mul_add_f32s(right_xs[index], right_weights[index], right_acc);
        }
        let mut left = simd.reduce_sum_f32s(left_acc);
        let mut right = simd.reduce_sum_f32s(right_acc);
        for index in 0..left_tail.len() {
            left = left_tail[index].mul_add(left_weight_tail[index], left);
            right = right_tail[index].mul_add(right_weight_tail[index], right);
        }
        (left, right)
    }
}

pub(crate) fn pulp_linear_dot_pair(
    left_xs: &[f32],
    left_weights: &[f32],
    right_xs: &[f32],
    right_weights: &[f32],
) -> (f32, f32) {
    debug_assert_eq!(left_xs.len(), left_weights.len());
    debug_assert_eq!(right_xs.len(), right_weights.len());
    debug_assert_eq!(left_xs.len(), right_xs.len());
    dispatch_pulp(LinearDotPair {
        left_xs,
        left_weights,
        right_xs,
        right_weights,
    })
}

struct LinearProject<'a> {
    input: &'a [f32],
    row_count: usize,
    in_dim: usize,
    weights: &'a [f32],
    bias: Option<&'a [f32]>,
    out_dim: usize,
    output: &'a mut [f32],
}

impl WithSimd for LinearProject<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        for row in 0..self.row_count {
            linear_project_row_simd(
                simd,
                &self.input[row * self.in_dim..(row + 1) * self.in_dim],
                self.weights,
                self.bias,
                self.out_dim,
                self.in_dim,
                &mut self.output[row * self.out_dim..(row + 1) * self.out_dim],
            );
        }
    }
}

pub(crate) fn pulp_linear_project_batch(
    input: &[f32],
    row_count: usize,
    in_dim: usize,
    weights: &[f32],
    bias: Option<&[f32]>,
    out_dim: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), row_count * in_dim);
    debug_assert_eq!(weights.len(), out_dim * in_dim);
    debug_assert_eq!(output.len(), row_count * out_dim);
    debug_assert!(bias.is_none_or(|values| values.len() == out_dim));
    dispatch_pulp(LinearProject {
        input,
        row_count,
        in_dim,
        weights,
        bias,
        out_dim,
        output,
    });
}

pub(crate) fn pulp_linear_project_row(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    out_dim: usize,
    in_dim: usize,
    output: &mut [f32],
) {
    pulp_linear_project_batch(input, 1, in_dim, weights, bias, out_dim, output);
}

#[inline(always)]
fn linear_project_row_simd<S: Simd>(
    simd: S,
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    out_dim: usize,
    in_dim: usize,
    output: &mut [f32],
) {
    let mut out_base = 0usize;
    while out_base + 12 <= out_dim {
        linear_project_chunk_simd::<S, 12>(simd, input, weights, bias, out_base, in_dim, output);
        out_base += 12;
    }
    while out_base + 8 <= out_dim {
        linear_project_chunk_simd::<S, 8>(simd, input, weights, bias, out_base, in_dim, output);
        out_base += 8;
    }
    while out_base + 4 <= out_dim {
        linear_project_chunk_simd::<S, 4>(simd, input, weights, bias, out_base, in_dim, output);
        out_base += 4;
    }
    while out_base < out_dim {
        let weight_row = &weights[out_base * in_dim..(out_base + 1) * in_dim];
        let value = linear_dot_simd(simd, input, weight_row);
        output[out_base] = value + bias.map_or(0.0, |values| values[out_base]);
        out_base += 1;
    }
}

#[inline(always)]
fn linear_project_chunk_simd<S: Simd, const OUTPUTS: usize>(
    simd: S,
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    out_base: usize,
    in_dim: usize,
    output: &mut [f32],
) {
    let (input_vectors, input_tail) = S::as_simd_f32s(input);
    let weight_parts: [(&[S::f32s], &[f32]); OUTPUTS] = array::from_fn(|offset| {
        let base = (out_base + offset) * in_dim;
        S::as_simd_f32s(&weights[base..base + in_dim])
    });
    let mut accumulators = [simd.splat_f32s(0.0); OUTPUTS];
    for (input_index, x) in input_vectors.iter().copied().enumerate() {
        for output_offset in 0..OUTPUTS {
            accumulators[output_offset] = simd.mul_add_f32s(
                x,
                weight_parts[output_offset].0[input_index],
                accumulators[output_offset],
            );
        }
    }
    for output_offset in 0..OUTPUTS {
        let mut value = simd.reduce_sum_f32s(accumulators[output_offset]);
        let weight_tail = weight_parts[output_offset].1;
        for input_index in 0..input_tail.len() {
            value = input_tail[input_index].mul_add(weight_tail[input_index], value);
        }
        output[out_base + output_offset] =
            value + bias.map_or(0.0, |values| values[out_base + output_offset]);
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RecurrenceOptions {
    pub(crate) force_skip_deformation: bool,
    pub(crate) deformation_threshold: Option<f32>,
    pub(crate) reuse_deformed_key: bool,
}

struct TimestepOutputRow<'a> {
    state: &'a [f32],
    next_state: Option<&'a mut [f32]>,
    r: &'a [f32],
    k: &'a [f32],
    value: f32,
    w: &'a [f32],
    a: &'a [f32],
    k_deformed: &'a [f32],
    options: RecurrenceOptions,
}

impl WithSimd for TimestepOutputRow<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        timestep_output_row_simd(
            simd,
            self.state,
            self.next_state,
            self.r,
            self.k,
            self.value,
            self.w,
            self.a,
            self.k_deformed,
            self.options,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pulp_timestep_output_row(
    state: &[f32],
    r: &[f32],
    k: &[f32],
    value: f32,
    w: &[f32],
    a: &[f32],
    k_deformed: &[f32],
    options: RecurrenceOptions,
) -> f32 {
    dispatch_pulp(TimestepOutputRow {
        state,
        next_state: None,
        r,
        k,
        value,
        w,
        a,
        k_deformed,
        options,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pulp_timestep_state_output_row(
    state: &[f32],
    next_state: &mut [f32],
    r: &[f32],
    k: &[f32],
    value: f32,
    w: &[f32],
    a: &[f32],
    k_deformed: &[f32],
    options: RecurrenceOptions,
) -> f32 {
    dispatch_pulp(TimestepOutputRow {
        state,
        next_state: Some(next_state),
        r,
        k,
        value,
        w,
        a,
        k_deformed,
        options,
    })
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn timestep_output_row_simd<S: Simd>(
    simd: S,
    state: &[f32],
    mut next_state: Option<&mut [f32]>,
    r: &[f32],
    k: &[f32],
    value: f32,
    w: &[f32],
    a: &[f32],
    k_deformed: &[f32],
    options: RecurrenceOptions,
) -> f32 {
    let (state_vectors, state_tail) = S::as_simd_f32s(state);
    let (r_vectors, r_tail) = S::as_simd_f32s(r);
    let (k_vectors, k_tail) = S::as_simd_f32s(k);
    let (w_vectors, w_tail) = S::as_simd_f32s(w);
    let (a_vectors, a_tail) = S::as_simd_f32s(a);
    let (k_deformed_vectors, k_deformed_tail) = S::as_simd_f32s(k_deformed);

    let mut deformation_left = 0.0f32;
    if !options.force_skip_deformation {
        let mut deformation_acc = simd.splat_f32s(0.0);
        for index in 0..state_vectors.len() {
            deformation_acc = simd.mul_add_f32s(
                state_vectors[index],
                k_deformed_vectors[index],
                deformation_acc,
            );
        }
        deformation_left = simd.reduce_sum_f32s(deformation_acc);
        for index in 0..state_tail.len() {
            deformation_left = state_tail[index].mul_add(k_deformed_tail[index], deformation_left);
        }
    }
    let skip_deformation = options.force_skip_deformation
        || options
            .deformation_threshold
            .is_some_and(|threshold| deformation_left.abs() <= threshold);

    let value_vector = simd.splat_f32s(value);
    let deformation_vector = simd.splat_f32s(deformation_left);
    let mut output_acc = simd.splat_f32s(0.0);
    let mut next_vectors = next_state
        .as_deref_mut()
        .map(|values| S::as_mut_simd_f32s(values).0);
    for index in 0..state_vectors.len() {
        let state_decay = simd.mul_f32s(state_vectors[index], w_vectors[index]);
        let next = if skip_deformation {
            simd.mul_add_f32s(value_vector, k_vectors[index], state_decay)
        } else {
            let deformed_key = if options.reuse_deformed_key {
                k_vectors[index]
            } else {
                simd.mul_f32s(a_vectors[index], k_deformed_vectors[index])
            };
            let base = simd.sub_f32s(state_decay, simd.mul_f32s(deformation_vector, deformed_key));
            simd.mul_add_f32s(value_vector, k_vectors[index], base)
        };
        if let Some(next_vectors) = next_vectors.as_deref_mut() {
            next_vectors[index] = next;
        }
        output_acc = simd.mul_add_f32s(next, r_vectors[index], output_acc);
    }
    let mut output = simd.reduce_sum_f32s(output_acc);
    let next_tail_base = state_vectors.len() * S::F32_LANES;
    for index in 0..state_tail.len() {
        let next = if skip_deformation {
            value.mul_add(k_tail[index], state_tail[index] * w_tail[index])
        } else {
            let deformed_key = if options.reuse_deformed_key {
                k_tail[index]
            } else {
                a_tail[index] * k_deformed_tail[index]
            };
            value.mul_add(
                k_tail[index],
                state_tail[index] * w_tail[index] - deformation_left * deformed_key,
            )
        };
        if let Some(next_state) = next_state.as_deref_mut() {
            next_state[next_tail_base + index] = next;
        }
        output = next.mul_add(r_tail[index], output);
    }
    output
}

struct AlgebraicOutputHead<'a> {
    state: &'a [f32],
    output: &'a mut [f32],
    r: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    w: &'a [f32],
    a: &'a [f32],
    k_deformed: &'a [f32],
    skip_deformation: bool,
    reuse_deformed_key: bool,
}

impl WithSimd for AlgebraicOutputHead<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let head_size = self.output.len();
        let (r_vectors, r_tail) = S::as_simd_f32s(self.r);
        let (k_vectors, k_tail) = S::as_simd_f32s(self.k);
        let (a_vectors, a_tail) = S::as_simd_f32s(self.a);
        let (k_deformed_vectors, k_deformed_tail) = S::as_simd_f32s(self.k_deformed);
        let mut key_acc = simd.splat_f32s(0.0);
        let mut deformation_acc = simd.splat_f32s(0.0);
        for index in 0..r_vectors.len() {
            key_acc = simd.mul_add_f32s(k_vectors[index], r_vectors[index], key_acc);
            if !self.skip_deformation {
                let deformed_key = if self.reuse_deformed_key {
                    k_vectors[index]
                } else {
                    simd.mul_f32s(a_vectors[index], k_deformed_vectors[index])
                };
                deformation_acc =
                    simd.mul_add_f32s(deformed_key, r_vectors[index], deformation_acc);
            }
        }
        let mut key_readout = simd.reduce_sum_f32s(key_acc);
        let mut deformation_readout = simd.reduce_sum_f32s(deformation_acc);
        for index in 0..r_tail.len() {
            key_readout = k_tail[index].mul_add(r_tail[index], key_readout);
            if !self.skip_deformation {
                let deformed_key = if self.reuse_deformed_key {
                    k_tail[index]
                } else {
                    a_tail[index] * k_deformed_tail[index]
                };
                deformation_readout = deformed_key.mul_add(r_tail[index], deformation_readout);
            }
        }

        let mut query = vec![0.0f32; head_size];
        let (query_vectors, query_tail) = S::as_mut_simd_f32s(&mut query);
        let (w_vectors, w_tail) = S::as_simd_f32s(self.w);
        let deformation = simd.splat_f32s(deformation_readout);
        for index in 0..query_vectors.len() {
            let weighted_r = simd.mul_f32s(w_vectors[index], r_vectors[index]);
            query_vectors[index] = simd.sub_f32s(
                weighted_r,
                simd.mul_f32s(deformation, k_deformed_vectors[index]),
            );
        }
        for index in 0..query_tail.len() {
            query_tail[index] =
                w_tail[index] * r_tail[index] - deformation_readout * k_deformed_tail[index];
        }

        for row in 0..head_size {
            let state_row = &self.state[row * head_size..(row + 1) * head_size];
            self.output[row] = linear_dot_simd(simd, state_row, &query) + self.v[row] * key_readout;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pulp_timestep_output_head_algebraic(
    state: &[f32],
    output: &mut [f32],
    r: &[f32],
    k: &[f32],
    v: &[f32],
    w: &[f32],
    a: &[f32],
    k_deformed: &[f32],
    skip_deformation: bool,
    reuse_deformed_key: bool,
) {
    dispatch_pulp(AlgebraicOutputHead {
        state,
        output,
        r,
        k,
        v,
        w,
        a,
        k_deformed,
        skip_deformation,
        reuse_deformed_key,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(len: usize, multiplier: usize, divisor: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((index * multiplier + 7) % 251) as f32 / divisor + offset)
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "value {index}: {actual} != {expected} within {tolerance}"
            );
        }
    }

    #[test]
    fn pulp_linear_batch_matches_independent_scalar_reference() {
        if !pulp_simd_available() {
            return;
        }
        let row_count = 3usize;
        let in_dim = 37usize;
        let out_dim = 29usize;
        let input = fixture(row_count * in_dim, 17, 113.0, -0.9);
        let weights = fixture(out_dim * in_dim, 29, 97.0, -1.1);
        let bias = fixture(out_dim, 11, 211.0, -0.3);
        let mut expected = vec![0.0f32; row_count * out_dim];
        for row in 0..row_count {
            for output in 0..out_dim {
                let mut value = bias[output];
                for input_index in 0..in_dim {
                    value +=
                        input[row * in_dim + input_index] * weights[output * in_dim + input_index];
                }
                expected[row * out_dim + output] = value;
            }
        }

        let mut actual = vec![0.0f32; row_count * out_dim];
        pulp_linear_project_batch(
            &input,
            row_count,
            in_dim,
            &weights,
            Some(&bias),
            out_dim,
            &mut actual,
        );
        assert_close(&actual, &expected, 2e-5);
    }

    #[test]
    fn pulp_predict_and_state_recurrence_match_scalar_reference() {
        if !pulp_simd_available() {
            return;
        }
        let len = 37usize;
        let state = fixture(len, 17, 193.0, -0.5);
        let r = fixture(len, 19, 223.0, -0.4);
        let k = fixture(len, 23, 181.0, -0.6);
        let w = fixture(len, 13, 509.0, 0.45);
        let a = fixture(len, 31, 271.0, -0.3);
        let k_deformed = fixture(len, 37, 331.0, -0.35);
        let value = 0.31f32;

        for options in [
            RecurrenceOptions {
                force_skip_deformation: false,
                deformation_threshold: None,
                reuse_deformed_key: false,
            },
            RecurrenceOptions {
                force_skip_deformation: false,
                deformation_threshold: Some(f32::MAX),
                reuse_deformed_key: true,
            },
            RecurrenceOptions {
                force_skip_deformation: true,
                deformation_threshold: None,
                reuse_deformed_key: false,
            },
        ] {
            let mut deformation_left = 0.0f32;
            if !options.force_skip_deformation {
                for index in 0..len {
                    deformation_left += state[index] * k_deformed[index];
                }
            }
            let skip_deformation = options.force_skip_deformation
                || options
                    .deformation_threshold
                    .is_some_and(|threshold| deformation_left.abs() <= threshold);
            let mut expected_state = vec![0.0f32; len];
            let mut expected_output = 0.0f32;
            for index in 0..len {
                let next = if skip_deformation {
                    state[index] * w[index] + value * k[index]
                } else {
                    let deformed_key = if options.reuse_deformed_key {
                        k[index]
                    } else {
                        a[index] * k_deformed[index]
                    };
                    state[index] * w[index] - deformation_left * deformed_key + value * k[index]
                };
                expected_state[index] = next;
                expected_output += next * r[index];
            }

            let predicted =
                pulp_timestep_output_row(&state, &r, &k, value, &w, &a, &k_deformed, options);
            let mut actual_state = vec![0.0f32; len];
            let processed = pulp_timestep_state_output_row(
                &state,
                &mut actual_state,
                &r,
                &k,
                value,
                &w,
                &a,
                &k_deformed,
                options,
            );
            assert!((predicted - expected_output).abs() <= 2e-5);
            assert!((processed - expected_output).abs() <= 2e-5);
            assert_close(&actual_state, &expected_state, 2e-6);
        }
    }

    #[test]
    fn pulp_algebraic_output_head_matches_independent_row_recurrence() {
        if !pulp_simd_available() {
            return;
        }
        let head_size = 32usize;
        let state = fixture(head_size * head_size, 17, 521.0, -0.25);
        let r = fixture(head_size, 19, 307.0, -0.3);
        let k_deformed = fixture(head_size, 23, 401.0, -0.2);
        let a = fixture(head_size, 29, 617.0, 0.1);
        let k = a
            .iter()
            .zip(&k_deformed)
            .map(|(a, k_deformed)| a * k_deformed)
            .collect::<Vec<_>>();
        let v = fixture(head_size, 31, 431.0, -0.15);
        let w = fixture(head_size, 37, 701.0, 0.65);
        let mut expected = vec![0.0f32; head_size];
        for row in 0..head_size {
            let mut deformation_left = 0.0f32;
            for index in 0..head_size {
                deformation_left += state[row * head_size + index] * k_deformed[index];
            }
            for index in 0..head_size {
                let next = state[row * head_size + index] * w[index] - deformation_left * k[index]
                    + v[row] * k[index];
                expected[row] += next * r[index];
            }
        }

        let mut actual = vec![0.0f32; head_size];
        pulp_timestep_output_head_algebraic(
            &state,
            &mut actual,
            &r,
            &k,
            &v,
            &w,
            &a,
            &k_deformed,
            false,
            true,
        );
        assert_close(&actual, &expected, 3e-5);
    }
}
