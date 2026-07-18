//! Shared layer-normalization, linear projection, and scalar/SIMD kernel code.

use super::*;
use crate::portable_simd::{
    cpu_simd_kernel, pulp_linear_dot, pulp_linear_dot_pair, pulp_linear_project_batch,
    pulp_linear_project_row, CpuSimdKernel,
};

pub(super) fn layer_norm(xs: &Tensor, weights: &LayerNormWeights) -> Result<Tensor> {
    layer_norm_last_dim(xs, &weights.weight, &weights.bias, LAYER_NORM_EPS)
}

pub(super) fn layer_norm_values_native_centered(
    xs_values: &[f32],
    row_count: usize,
    width: usize,
    weights: &LayerNormWeights,
) -> Result<Option<Vec<f32>>> {
    layer_norm_values_native_centered_into(
        xs_values,
        row_count,
        width,
        weights,
        vec![0.0f32; xs_values.len()],
    )
}

pub(super) fn layer_norm_values_native_centered_into(
    xs_values: &[f32],
    row_count: usize,
    width: usize,
    weights: &LayerNormWeights,
    mut out: Vec<f32>,
) -> Result<Option<Vec<f32>>> {
    if width == 0
        || xs_values.len() != row_count * width
        || weights.weight.dims1()? != width
        || weights.bias.dims1()? != width
    {
        return Ok(None);
    }
    let weight_data = f32_tensor_data(&weights.weight)?;
    let bias_data = f32_tensor_data(&weights.bias)?;
    let weight_values = weight_data.as_slice()?;
    let bias_values = bias_data.as_slice()?;
    if weight_values.len() != width || bias_values.len() != width {
        return Ok(None);
    }

    out.resize(xs_values.len(), 0.0);
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
        let inv_std = 1.0f32 / (variance + LAYER_NORM_EPS).sqrt();
        let out_row = &mut out[row_base..row_base + width];
        for channel_index in 0..width {
            out_row[channel_index] =
                (row[channel_index] - mean) * inv_std * weight_values[channel_index]
                    + bias_values[channel_index];
        }
    }

    Ok(Some(out))
}

pub(super) fn layer_norm_values_native_centered_predict_x_buffer(
    xs_values: &[f32],
    row_count: usize,
    width: usize,
    weights: &LayerNormWeights,
) -> Result<Option<Vec<f32>>> {
    let out = take_time_mixer_predict_x_values(xs_values.len());
    layer_norm_values_native_centered_into(xs_values, row_count, width, weights, out)
}

pub(super) fn linear(xs: &Tensor, weights: &LinearWeights) -> Result<Tensor> {
    Linear::new(weights.weight.clone(), weights.bias.clone()).forward(xs)
}

pub(super) fn linear_profiled(
    xs: &Tensor,
    weights: &LinearWeights,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<Tensor> {
    record_linear_layout(profile, xs);
    if native_linear_kernel_enabled() {
        if let Some((output, kernel)) = linear_native_f32(xs, weights)? {
            record_native_linear(profile, xs, &output, kernel);
            return Ok(output);
        }
        record_native_linear_fallback(profile);
    }
    linear(xs, weights)
}

pub(super) fn linear_profiled_rowwise_native_exact(
    xs: &Tensor,
    weights: &LinearWeights,
    profile: &mut Option<&mut ForwardProfile>,
) -> Result<Option<Tensor>> {
    record_linear_layout(profile, xs);
    if !native_linear_kernel_enabled() {
        return Ok(None);
    }
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };

    let xs_dims = xs.dims();
    if xs_dims.len() < 2 {
        return Ok(None);
    }
    let (out_dim, in_dim) = weights.weight.dims2()?;
    if xs_dims[xs_dims.len() - 1] != in_dim {
        return Ok(None);
    }
    if let Some(bias) = &weights.bias {
        if bias.dims() != [out_dim] {
            return Ok(None);
        }
    }

    let row_count = xs_dims[..xs_dims.len() - 1].iter().product::<usize>();
    if row_count == 0 || out_dim == 0 || in_dim == 0 {
        return Ok(None);
    }

    let xs_data = f32_tensor_data(xs)?;
    let weight_data = f32_tensor_data(&weights.weight)?;
    let bias_data = weights.bias.as_ref().map(f32_tensor_data).transpose()?;
    let xs_values = xs_data.as_slice()?;
    let weight_values = weight_data.as_slice()?;
    let bias_values = bias_data.as_ref().map(|data| data.as_slice()).transpose()?;

    let expected_input_values = row_count * in_dim;
    let expected_weight_values = out_dim * in_dim;
    if xs_values.len() != expected_input_values || weight_values.len() != expected_weight_values {
        return Ok(None);
    }

    let mut out = vec![0.0f32; row_count * out_dim];
    for row_index in 0..row_count {
        let input_base = row_index * in_dim;
        let output_base = row_index * out_dim;
        linear_project_row_same_x(
            dot_kernel,
            &xs_values[input_base..input_base + in_dim],
            weight_values,
            out_dim,
            in_dim,
            &mut out[output_base..output_base + out_dim],
        );
    }
    if let Some(bias_values) = bias_values {
        for row_index in 0..row_count {
            let out_row = &mut out[row_index * out_dim..(row_index + 1) * out_dim];
            for out_index in 0..out_dim {
                out_row[out_index] += bias_values[out_index];
            }
        }
    }

    let mut out_dims = xs_dims.to_vec();
    *out_dims
        .last_mut()
        .expect("linear input has at least two dimensions") = out_dim;
    let output = Tensor::from_vec(out, Shape::from_dims(&out_dims), xs.device())?;
    record_native_linear(profile, xs, &output, dot_kernel);
    Ok(Some(output))
}

pub(super) type NativeLinearDotKernel = CpuSimdKernel;

pub(super) fn native_linear_dot_kernel() -> Option<NativeLinearDotKernel> {
    static KERNEL: OnceLock<Option<NativeLinearDotKernel>> = OnceLock::new();
    *KERNEL.get_or_init(|| match cpu_simd_kernel(!native_linear_simd_disabled()) {
        NativeLinearDotKernel::Scalar => None,
        kernel => Some(kernel),
    })
}

pub(super) fn linear_native_f32(
    xs: &Tensor,
    weights: &LinearWeights,
) -> Result<Option<(Tensor, NativeLinearDotKernel)>> {
    let Some(dot_kernel) = native_linear_dot_kernel() else {
        return Ok(None);
    };
    linear_native_f32_with_kernel(xs, weights, dot_kernel)
        .map(|output| output.map(|output| (output, dot_kernel)))
}

pub(super) fn linear_native_f32_with_kernel(
    xs: &Tensor,
    weights: &LinearWeights,
    dot_kernel: NativeLinearDotKernel,
) -> Result<Option<Tensor>> {
    let xs_dims = xs.dims();
    if xs_dims.len() < 2 {
        return Ok(None);
    }
    let (out_dim, in_dim) = weights.weight.dims2()?;
    if xs_dims[xs_dims.len() - 1] != in_dim {
        return Ok(None);
    }
    if let Some(bias) = &weights.bias {
        if bias.dims() != [out_dim] {
            return Ok(None);
        }
    }

    let row_count = xs_dims[..xs_dims.len() - 1].iter().product::<usize>();
    if row_count == 0 || out_dim == 0 || in_dim == 0 {
        return Ok(None);
    }

    let xs_data = f32_tensor_data(xs)?;
    let weight_data = f32_tensor_data(&weights.weight)?;
    let bias_data = weights.bias.as_ref().map(f32_tensor_data).transpose()?;
    let xs_values = xs_data.as_slice()?;
    let weight_values = weight_data.as_slice()?;
    let bias_values = bias_data.as_ref().map(|data| data.as_slice()).transpose()?;

    let expected_input_values = row_count * in_dim;
    let expected_weight_values = out_dim * in_dim;
    if xs_values.len() != expected_input_values || weight_values.len() != expected_weight_values {
        return Ok(None);
    }

    let mut out = vec![0.0f32; row_count * out_dim];
    if let Some(bias_values) = bias_values {
        if native_linear_fused_bias_add_enabled() {
            if !linear_project_batch_same_x_with_bias(
                dot_kernel,
                xs_values,
                row_count,
                in_dim,
                weight_values,
                bias_values,
                out_dim,
                &mut out,
            ) {
                for row_index in 0..row_count {
                    let xs_row = &xs_values[row_index * in_dim..(row_index + 1) * in_dim];
                    let out_row = &mut out[row_index * out_dim..(row_index + 1) * out_dim];
                    linear_project_row_same_x_with_bias(
                        dot_kernel,
                        xs_row,
                        weight_values,
                        bias_values,
                        out_dim,
                        in_dim,
                        out_row,
                    );
                }
            }
        } else {
            if !linear_project_batch_same_x(
                dot_kernel,
                xs_values,
                row_count,
                in_dim,
                weight_values,
                out_dim,
                &mut out,
            ) {
                for row_index in 0..row_count {
                    let xs_row = &xs_values[row_index * in_dim..(row_index + 1) * in_dim];
                    let out_row = &mut out[row_index * out_dim..(row_index + 1) * out_dim];
                    linear_project_row_same_x(
                        dot_kernel,
                        xs_row,
                        weight_values,
                        out_dim,
                        in_dim,
                        out_row,
                    );
                }
            }
            add_linear_bias_to_output_rows(&mut out, row_count, out_dim, bias_values);
        }
    } else if !linear_project_batch_same_x(
        dot_kernel,
        xs_values,
        row_count,
        in_dim,
        weight_values,
        out_dim,
        &mut out,
    ) {
        for row_index in 0..row_count {
            let xs_row = &xs_values[row_index * in_dim..(row_index + 1) * in_dim];
            let out_row = &mut out[row_index * out_dim..(row_index + 1) * out_dim];
            linear_project_row_same_x(dot_kernel, xs_row, weight_values, out_dim, in_dim, out_row);
        }
    }

    let mut out_dims = xs_dims.to_vec();
    *out_dims
        .last_mut()
        .expect("linear input has at least two dimensions") = out_dim;
    Tensor::from_vec(out, Shape::from_dims(&out_dims), xs.device()).map(Some)
}

pub(super) fn linear_f32_weights(weights: &LinearWeights) -> Result<LinearF32Weights> {
    linear_f32_weights_with_blocked(weights, false)
}

pub(super) fn linear_f32_weights_with_blocked(
    weights: &LinearWeights,
    enable_blocked: bool,
) -> Result<LinearF32Weights> {
    let (out_dim, in_dim) = weights.weight.dims2()?;
    let weight_data = f32_tensor_data(&weights.weight)?;
    let weight_values = weight_data.as_slice()?;
    if weight_values.len() != out_dim * in_dim {
        bail!(
            "linear weight expected {} values, got {}",
            out_dim * in_dim,
            weight_values.len()
        );
    }

    let bias = if let Some(bias) = &weights.bias {
        if bias.dims() != [out_dim] {
            bail!(
                "linear bias expected shape [{out_dim}], got {:?}",
                bias.dims()
            );
        }
        let bias_data = f32_tensor_data(bias)?;
        let bias_values = bias_data.as_slice()?;
        if bias_values.len() != out_dim {
            bail!(
                "linear bias expected {out_dim} values, got {}",
                bias_values.len()
            );
        }
        Some(bias_values.to_vec())
    } else {
        None
    };

    let weight = weight_values.to_vec();
    let blocked_input8_dot12 = if cfg!(any(target_arch = "x86", target_arch = "x86_64"))
        && enable_blocked
        && native_linear_blocked_input8_dot12_layout_enabled()
        && matches!(
            (out_dim, in_dim),
            (128, 128) | (192, 128) | (256, 128) | (128, 192) | (128, 256)
        ) {
        linear_pack_input8_dot12_weights(&weight, out_dim, in_dim)
    } else {
        None
    };
    Ok(LinearF32Weights {
        out_dim,
        in_dim,
        weight,
        bias,
        blocked_input8_dot12,
    })
}

pub(super) fn linear_dot(kernel: NativeLinearDotKernel, xs_row: &[f32], weight_row: &[f32]) -> f32 {
    match kernel {
        NativeLinearDotKernel::Scalar => linear_dot_scalar(xs_row, weight_row),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        NativeLinearDotKernel::Avx2Fma => unsafe { linear_dot_avx2_fma(xs_row, weight_row) },
        NativeLinearDotKernel::Pulp => pulp_linear_dot(xs_row, weight_row),
    }
}

pub(super) fn linear_dot_scalar(xs_row: &[f32], weight_row: &[f32]) -> f32 {
    debug_assert_eq!(xs_row.len(), weight_row.len());
    let mut acc = 0.0f32;
    for index in 0..xs_row.len() {
        acc += xs_row[index] * weight_row[index];
    }
    acc
}

pub(super) fn linear_dot_pair(
    kernel: NativeLinearDotKernel,
    left_xs_row: &[f32],
    left_weight_row: &[f32],
    right_xs_row: &[f32],
    right_weight_row: &[f32],
) -> (f32, f32) {
    match kernel {
        NativeLinearDotKernel::Scalar => {
            linear_dot_pair_scalar(left_xs_row, left_weight_row, right_xs_row, right_weight_row)
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        NativeLinearDotKernel::Avx2Fma => unsafe {
            linear_dot_pair_avx2_fma(left_xs_row, left_weight_row, right_xs_row, right_weight_row)
        },
        NativeLinearDotKernel::Pulp => {
            pulp_linear_dot_pair(left_xs_row, left_weight_row, right_xs_row, right_weight_row)
        }
    }
}

pub(super) fn linear_dot_pair_scalar(
    left_xs_row: &[f32],
    left_weight_row: &[f32],
    right_xs_row: &[f32],
    right_weight_row: &[f32],
) -> (f32, f32) {
    debug_assert_eq!(left_xs_row.len(), left_weight_row.len());
    debug_assert_eq!(right_xs_row.len(), right_weight_row.len());
    debug_assert_eq!(left_xs_row.len(), right_xs_row.len());
    let mut left_acc = 0.0f32;
    let mut right_acc = 0.0f32;
    for index in 0..left_xs_row.len() {
        left_acc += left_xs_row[index] * left_weight_row[index];
        right_acc += right_xs_row[index] * right_weight_row[index];
    }
    (left_acc, right_acc)
}

pub(super) fn linear_project_batch_same_x(
    kernel: NativeLinearDotKernel,
    input_values: &[f32],
    row_count: usize,
    in_dim: usize,
    weight_values: &[f32],
    out_dim: usize,
    output_values: &mut [f32],
) -> bool {
    debug_assert_eq!(input_values.len(), row_count * in_dim);
    debug_assert_eq!(weight_values.len(), out_dim * in_dim);
    debug_assert_eq!(output_values.len(), row_count * out_dim);

    if !native_linear_batched_same_x_enabled() || row_count < 2 || out_dim < 4 {
        return false;
    }

    if matches!(kernel, NativeLinearDotKernel::Pulp) {
        pulp_linear_project_batch(
            input_values,
            row_count,
            in_dim,
            weight_values,
            None,
            out_dim,
            output_values,
        );
        return true;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if matches!(kernel, NativeLinearDotKernel::Avx2Fma) {
        let mut row_index = 0usize;
        if native_linear_batched_four_row_same_x_enabled() {
            while row_index + 3 < row_count {
                let input_base0 = row_index * in_dim;
                let input_base1 = (row_index + 1) * in_dim;
                let input_base2 = (row_index + 2) * in_dim;
                let input_base3 = (row_index + 3) * in_dim;
                let output_base0 = row_index * out_dim;
                let output_base1 = (row_index + 1) * out_dim;
                let (before_row1, from_row1) = output_values.split_at_mut(output_base1);
                let output_row0 = &mut before_row1[output_base0..output_base0 + out_dim];
                let (output_row1, from_row2) = from_row1.split_at_mut(out_dim);
                let (output_row2, from_row3) = from_row2.split_at_mut(out_dim);
                let output_row3 = &mut from_row3[..out_dim];
                unsafe {
                    linear_project_four_rows_same_x_avx2_fma(
                        &input_values[input_base0..input_base0 + in_dim],
                        &input_values[input_base1..input_base1 + in_dim],
                        &input_values[input_base2..input_base2 + in_dim],
                        &input_values[input_base3..input_base3 + in_dim],
                        weight_values,
                        out_dim,
                        in_dim,
                        output_row0,
                        output_row1,
                        output_row2,
                        output_row3,
                    );
                }
                row_index += 4;
            }
        }
        while row_index + 1 < row_count {
            let input_base0 = row_index * in_dim;
            let input_base1 = (row_index + 1) * in_dim;
            let output_base0 = row_index * out_dim;
            let output_base1 = (row_index + 1) * out_dim;
            let (before_row1, from_row1) = output_values.split_at_mut(output_base1);
            let output_row0 = &mut before_row1[output_base0..output_base0 + out_dim];
            let output_row1 = &mut from_row1[..out_dim];
            unsafe {
                linear_project_two_rows_same_x_avx2_fma(
                    &input_values[input_base0..input_base0 + in_dim],
                    &input_values[input_base1..input_base1 + in_dim],
                    weight_values,
                    out_dim,
                    in_dim,
                    output_row0,
                    output_row1,
                );
            }
            row_index += 2;
        }
        if row_index < row_count {
            let input_base = row_index * in_dim;
            let output_base = row_index * out_dim;
            linear_project_row_same_x(
                kernel,
                &input_values[input_base..input_base + in_dim],
                weight_values,
                out_dim,
                in_dim,
                &mut output_values[output_base..output_base + out_dim],
            );
        }
        return true;
    }

    false
}

pub(super) fn linear_project_batch_same_x_with_bias(
    kernel: NativeLinearDotKernel,
    input_values: &[f32],
    row_count: usize,
    in_dim: usize,
    weight_values: &[f32],
    bias_values: &[f32],
    out_dim: usize,
    output_values: &mut [f32],
) -> bool {
    debug_assert_eq!(input_values.len(), row_count * in_dim);
    debug_assert_eq!(weight_values.len(), out_dim * in_dim);
    debug_assert_eq!(bias_values.len(), out_dim);
    debug_assert_eq!(output_values.len(), row_count * out_dim);

    if !native_linear_batched_same_x_enabled() || row_count < 2 || out_dim < 4 {
        return false;
    }

    if matches!(kernel, NativeLinearDotKernel::Pulp) {
        pulp_linear_project_batch(
            input_values,
            row_count,
            in_dim,
            weight_values,
            Some(bias_values),
            out_dim,
            output_values,
        );
        return true;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if matches!(kernel, NativeLinearDotKernel::Avx2Fma) {
        let mut row_index = 0usize;
        if native_linear_batched_four_row_same_x_enabled() {
            while row_index + 3 < row_count {
                let input_base0 = row_index * in_dim;
                let input_base1 = (row_index + 1) * in_dim;
                let input_base2 = (row_index + 2) * in_dim;
                let input_base3 = (row_index + 3) * in_dim;
                let output_base0 = row_index * out_dim;
                let output_base1 = (row_index + 1) * out_dim;
                let (before_row1, from_row1) = output_values.split_at_mut(output_base1);
                let output_row0 = &mut before_row1[output_base0..output_base0 + out_dim];
                let (output_row1, from_row2) = from_row1.split_at_mut(out_dim);
                let (output_row2, from_row3) = from_row2.split_at_mut(out_dim);
                let output_row3 = &mut from_row3[..out_dim];
                unsafe {
                    linear_project_four_rows_same_x_avx2_fma(
                        &input_values[input_base0..input_base0 + in_dim],
                        &input_values[input_base1..input_base1 + in_dim],
                        &input_values[input_base2..input_base2 + in_dim],
                        &input_values[input_base3..input_base3 + in_dim],
                        weight_values,
                        out_dim,
                        in_dim,
                        output_row0,
                        output_row1,
                        output_row2,
                        output_row3,
                    );
                }
                add_linear_bias_to_output_row(output_row0, bias_values);
                add_linear_bias_to_output_row(output_row1, bias_values);
                add_linear_bias_to_output_row(output_row2, bias_values);
                add_linear_bias_to_output_row(output_row3, bias_values);
                row_index += 4;
            }
        }
        while row_index + 1 < row_count {
            let input_base0 = row_index * in_dim;
            let input_base1 = (row_index + 1) * in_dim;
            let output_base0 = row_index * out_dim;
            let output_base1 = (row_index + 1) * out_dim;
            let (before_row1, from_row1) = output_values.split_at_mut(output_base1);
            let output_row0 = &mut before_row1[output_base0..output_base0 + out_dim];
            let output_row1 = &mut from_row1[..out_dim];
            unsafe {
                linear_project_two_rows_same_x_avx2_fma(
                    &input_values[input_base0..input_base0 + in_dim],
                    &input_values[input_base1..input_base1 + in_dim],
                    weight_values,
                    out_dim,
                    in_dim,
                    output_row0,
                    output_row1,
                );
            }
            add_linear_bias_to_output_row(output_row0, bias_values);
            add_linear_bias_to_output_row(output_row1, bias_values);
            row_index += 2;
        }
        if row_index < row_count {
            let input_base = row_index * in_dim;
            let output_base = row_index * out_dim;
            linear_project_row_same_x_with_bias(
                kernel,
                &input_values[input_base..input_base + in_dim],
                weight_values,
                bias_values,
                out_dim,
                in_dim,
                &mut output_values[output_base..output_base + out_dim],
            );
        }
        return true;
    }

    false
}

#[inline(always)]
pub(super) fn linear_project_row_same_x_with_bias(
    kernel: NativeLinearDotKernel,
    xs_row: &[f32],
    weight_values: &[f32],
    bias_values: &[f32],
    out_dim: usize,
    in_dim: usize,
    output_row: &mut [f32],
) {
    debug_assert_eq!(bias_values.len(), out_dim);
    linear_project_row_same_x(kernel, xs_row, weight_values, out_dim, in_dim, output_row);
    add_linear_bias_to_output_row(output_row, bias_values);
}

#[inline(always)]
pub(super) fn add_linear_bias_to_output_rows(
    output_values: &mut [f32],
    row_count: usize,
    out_dim: usize,
    bias_values: &[f32],
) {
    debug_assert_eq!(output_values.len(), row_count * out_dim);
    for row_index in 0..row_count {
        let output_base = row_index * out_dim;
        add_linear_bias_to_output_row(
            &mut output_values[output_base..output_base + out_dim],
            bias_values,
        );
    }
}

#[inline(always)]
pub(super) fn add_linear_bias_to_output_row(output_row: &mut [f32], bias_values: &[f32]) {
    debug_assert_eq!(output_row.len(), bias_values.len());
    for out_index in 0..output_row.len() {
        output_row[out_index] += bias_values[out_index];
    }
}

#[inline(always)]
pub(super) fn linear_project_row_same_x(
    kernel: NativeLinearDotKernel,
    xs_row: &[f32],
    weight_values: &[f32],
    out_dim: usize,
    in_dim: usize,
    output_row: &mut [f32],
) {
    debug_assert_eq!(xs_row.len(), in_dim);
    debug_assert_eq!(weight_values.len(), out_dim * in_dim);
    debug_assert_eq!(output_row.len(), out_dim);
    if matches!(kernel, NativeLinearDotKernel::Pulp) {
        pulp_linear_project_row(xs_row, weight_values, None, out_dim, in_dim, output_row);
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if native_linear_fixed_in_dim_same_x_enabled()
        && native_linear_dot12_same_x_enabled()
        && native_linear_dot8_same_x_enabled()
        && native_linear_dot4_same_x_enabled()
        && out_dim >= 32
        && matches!(kernel, NativeLinearDotKernel::Avx2Fma)
    {
        match in_dim {
            16 => unsafe {
                linear_project_row_same_x_avx2_fma_dot12_fixed::<16>(
                    xs_row,
                    weight_values,
                    out_dim,
                    output_row,
                )
            },
            128 => unsafe {
                linear_project_row_same_x_avx2_fma_dot12_fixed::<128>(
                    xs_row,
                    weight_values,
                    out_dim,
                    output_row,
                )
            },
            512 => unsafe {
                linear_project_row_same_x_avx2_fma_dot12_fixed::<512>(
                    xs_row,
                    weight_values,
                    out_dim,
                    output_row,
                )
            },
            _ => {}
        }
        if matches!(in_dim, 16 | 128 | 512) {
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if native_linear_dot12_same_x_enabled()
        && native_linear_dot8_same_x_enabled()
        && native_linear_dot4_same_x_enabled()
        && out_dim >= 32
        && matches!(kernel, NativeLinearDotKernel::Avx2Fma)
    {
        unsafe {
            linear_project_row_same_x_avx2_fma_dot12(
                xs_row,
                weight_values,
                out_dim,
                in_dim,
                output_row,
            )
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if native_linear_fixed_in_dim_same_x_enabled()
        && native_linear_dot8_same_x_enabled()
        && native_linear_dot4_same_x_enabled()
        && out_dim >= 8
        && matches!(kernel, NativeLinearDotKernel::Avx2Fma)
    {
        match in_dim {
            16 => unsafe {
                linear_project_row_same_x_avx2_fma_dot8_fixed::<16>(
                    xs_row,
                    weight_values,
                    out_dim,
                    output_row,
                )
            },
            128 => unsafe {
                linear_project_row_same_x_avx2_fma_dot8_fixed::<128>(
                    xs_row,
                    weight_values,
                    out_dim,
                    output_row,
                )
            },
            512 => unsafe {
                linear_project_row_same_x_avx2_fma_dot8_fixed::<512>(
                    xs_row,
                    weight_values,
                    out_dim,
                    output_row,
                )
            },
            _ => {}
        }
        if matches!(in_dim, 16 | 128 | 512) {
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if native_linear_dot8_same_x_enabled()
        && native_linear_dot4_same_x_enabled()
        && out_dim >= 8
        && matches!(kernel, NativeLinearDotKernel::Avx2Fma)
    {
        unsafe {
            linear_project_row_same_x_avx2_fma_dot8(
                xs_row,
                weight_values,
                out_dim,
                in_dim,
                output_row,
            )
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if native_linear_fixed_in_dim_same_x_enabled()
        && native_linear_dot4_same_x_enabled()
        && out_dim >= 4
        && matches!(kernel, NativeLinearDotKernel::Avx2Fma)
    {
        match in_dim {
            16 => unsafe {
                linear_project_row_same_x_avx2_fma_fixed::<16>(
                    xs_row,
                    weight_values,
                    out_dim,
                    output_row,
                )
            },
            128 => unsafe {
                linear_project_row_same_x_avx2_fma_fixed::<128>(
                    xs_row,
                    weight_values,
                    out_dim,
                    output_row,
                )
            },
            512 => unsafe {
                linear_project_row_same_x_avx2_fma_fixed::<512>(
                    xs_row,
                    weight_values,
                    out_dim,
                    output_row,
                )
            },
            _ => {}
        }
        if matches!(in_dim, 16 | 128 | 512) {
            return;
        }
    }

    if native_linear_dot4_same_x_enabled() && out_dim >= 4 {
        match kernel {
            NativeLinearDotKernel::Scalar => {
                linear_project_row_same_x_scalar(xs_row, weight_values, out_dim, in_dim, output_row)
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            NativeLinearDotKernel::Avx2Fma => unsafe {
                linear_project_row_same_x_avx2_fma(
                    xs_row,
                    weight_values,
                    out_dim,
                    in_dim,
                    output_row,
                )
            },
            NativeLinearDotKernel::Pulp => unreachable!("Pulp returned above"),
        }
        return;
    }

    for out_index in 0..out_dim {
        let weight_row = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        output_row[out_index] = linear_dot(kernel, xs_row, weight_row);
    }
}

pub(super) fn linear_project_row_same_x_scalar(
    xs_row: &[f32],
    weight_values: &[f32],
    out_dim: usize,
    in_dim: usize,
    output_row: &mut [f32],
) {
    let mut out_index = 0usize;
    while out_index + 4 <= out_dim {
        let w0 = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        let w1 = &weight_values[(out_index + 1) * in_dim..(out_index + 2) * in_dim];
        let w2 = &weight_values[(out_index + 2) * in_dim..(out_index + 3) * in_dim];
        let w3 = &weight_values[(out_index + 3) * in_dim..(out_index + 4) * in_dim];
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;
        for index in 0..in_dim {
            let x = xs_row[index];
            acc0 += x * w0[index];
            acc1 += x * w1[index];
            acc2 += x * w2[index];
            acc3 += x * w3[index];
        }
        output_row[out_index] = acc0;
        output_row[out_index + 1] = acc1;
        output_row[out_index + 2] = acc2;
        output_row[out_index + 3] = acc3;
        out_index += 4;
    }

    while out_index < out_dim {
        let weight_row = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        output_row[out_index] = linear_dot_scalar(xs_row, weight_row);
        out_index += 1;
    }
}

macro_rules! horizontal_sum_avx2_linear {
    ($use_direct_sum:expr, $values:expr) => {{
        if $use_direct_sum {
            horizontal_sum_avx2_register($values)
        } else {
            horizontal_sum_avx2($values)
        }
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_project_four_rows_same_x_avx2_fma(
    xs_row0: &[f32],
    xs_row1: &[f32],
    xs_row2: &[f32],
    xs_row3: &[f32],
    weight_values: &[f32],
    out_dim: usize,
    in_dim: usize,
    output_row0: &mut [f32],
    output_row1: &mut [f32],
    output_row2: &mut [f32],
    output_row3: &mut [f32],
) {
    debug_assert_eq!(xs_row0.len(), in_dim);
    debug_assert_eq!(xs_row1.len(), in_dim);
    debug_assert_eq!(xs_row2.len(), in_dim);
    debug_assert_eq!(xs_row3.len(), in_dim);
    debug_assert_eq!(weight_values.len(), out_dim * in_dim);
    debug_assert_eq!(output_row0.len(), out_dim);
    debug_assert_eq!(output_row1.len(), out_dim);
    debug_assert_eq!(output_row2.len(), out_dim);
    debug_assert_eq!(output_row3.len(), out_dim);

    let mut out_index = 0usize;
    let use_pointer_rows = native_linear_batched_pointer_rows_enabled();
    while out_index + 2 <= out_dim {
        let values = if use_pointer_rows {
            linear_dot2_four_rows_same_x_avx2_fma_ptr(
                xs_row0,
                xs_row1,
                xs_row2,
                xs_row3,
                weight_values.as_ptr().add(out_index * in_dim),
                in_dim,
            )
        } else {
            let w0 = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
            let w1 = &weight_values[(out_index + 1) * in_dim..(out_index + 2) * in_dim];
            linear_dot2_four_rows_same_x_avx2_fma(xs_row0, xs_row1, xs_row2, xs_row3, w0, w1)
        };
        output_row0[out_index] = values[0][0];
        output_row0[out_index + 1] = values[0][1];
        output_row1[out_index] = values[1][0];
        output_row1[out_index + 1] = values[1][1];
        output_row2[out_index] = values[2][0];
        output_row2[out_index + 1] = values[2][1];
        output_row3[out_index] = values[3][0];
        output_row3[out_index + 1] = values[3][1];
        out_index += 2;
    }

    while out_index < out_dim {
        let weight_row = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        output_row0[out_index] = linear_dot_avx2_fma(xs_row0, weight_row);
        output_row1[out_index] = linear_dot_avx2_fma(xs_row1, weight_row);
        output_row2[out_index] = linear_dot_avx2_fma(xs_row2, weight_row);
        output_row3[out_index] = linear_dot_avx2_fma(xs_row3, weight_row);
        out_index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_project_two_rows_same_x_avx2_fma(
    xs_row0: &[f32],
    xs_row1: &[f32],
    weight_values: &[f32],
    out_dim: usize,
    in_dim: usize,
    output_row0: &mut [f32],
    output_row1: &mut [f32],
) {
    debug_assert_eq!(xs_row0.len(), in_dim);
    debug_assert_eq!(xs_row1.len(), in_dim);
    debug_assert_eq!(weight_values.len(), out_dim * in_dim);
    debug_assert_eq!(output_row0.len(), out_dim);
    debug_assert_eq!(output_row1.len(), out_dim);

    let mut out_index = 0usize;
    let use_pointer_rows = native_linear_batched_pointer_rows_enabled();
    while out_index + 4 <= out_dim {
        let (values0, values1) = if use_pointer_rows {
            linear_dot4_two_rows_same_x_avx2_fma_ptr(
                xs_row0,
                xs_row1,
                weight_values.as_ptr().add(out_index * in_dim),
                in_dim,
            )
        } else {
            let w0 = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
            let w1 = &weight_values[(out_index + 1) * in_dim..(out_index + 2) * in_dim];
            let w2 = &weight_values[(out_index + 2) * in_dim..(out_index + 3) * in_dim];
            let w3 = &weight_values[(out_index + 3) * in_dim..(out_index + 4) * in_dim];
            linear_dot4_two_rows_same_x_avx2_fma(xs_row0, xs_row1, w0, w1, w2, w3)
        };
        output_row0[out_index] = values0[0];
        output_row0[out_index + 1] = values0[1];
        output_row0[out_index + 2] = values0[2];
        output_row0[out_index + 3] = values0[3];
        output_row1[out_index] = values1[0];
        output_row1[out_index + 1] = values1[1];
        output_row1[out_index + 2] = values1[2];
        output_row1[out_index + 3] = values1[3];
        out_index += 4;
    }

    while out_index < out_dim {
        let weight_row = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        output_row0[out_index] = linear_dot_avx2_fma(xs_row0, weight_row);
        output_row1[out_index] = linear_dot_avx2_fma(xs_row1, weight_row);
        out_index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot2_four_rows_same_x_avx2_fma(
    xs_row0: &[f32],
    xs_row1: &[f32],
    xs_row2: &[f32],
    xs_row3: &[f32],
    weight_row0: &[f32],
    weight_row1: &[f32],
) -> [[f32; 2]; 4] {
    debug_assert_eq!(xs_row0.len(), xs_row1.len());
    debug_assert_eq!(xs_row0.len(), xs_row2.len());
    debug_assert_eq!(xs_row0.len(), xs_row3.len());
    debug_assert_eq!(xs_row0.len(), weight_row0.len());
    debug_assert_eq!(xs_row0.len(), weight_row1.len());

    let mut acc00 = x86_arch::_mm256_setzero_ps();
    let mut acc01 = x86_arch::_mm256_setzero_ps();
    let mut acc10 = x86_arch::_mm256_setzero_ps();
    let mut acc11 = x86_arch::_mm256_setzero_ps();
    let mut acc20 = x86_arch::_mm256_setzero_ps();
    let mut acc21 = x86_arch::_mm256_setzero_ps();
    let mut acc30 = x86_arch::_mm256_setzero_ps();
    let mut acc31 = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= xs_row0.len() {
        let xs0 = x86_arch::_mm256_loadu_ps(xs_row0.as_ptr().add(index));
        let xs1 = x86_arch::_mm256_loadu_ps(xs_row1.as_ptr().add(index));
        let xs2 = x86_arch::_mm256_loadu_ps(xs_row2.as_ptr().add(index));
        let xs3 = x86_arch::_mm256_loadu_ps(xs_row3.as_ptr().add(index));
        let weight0 = x86_arch::_mm256_loadu_ps(weight_row0.as_ptr().add(index));
        let weight1 = x86_arch::_mm256_loadu_ps(weight_row1.as_ptr().add(index));
        acc00 = x86_arch::_mm256_fmadd_ps(xs0, weight0, acc00);
        acc01 = x86_arch::_mm256_fmadd_ps(xs0, weight1, acc01);
        acc10 = x86_arch::_mm256_fmadd_ps(xs1, weight0, acc10);
        acc11 = x86_arch::_mm256_fmadd_ps(xs1, weight1, acc11);
        acc20 = x86_arch::_mm256_fmadd_ps(xs2, weight0, acc20);
        acc21 = x86_arch::_mm256_fmadd_ps(xs2, weight1, acc21);
        acc30 = x86_arch::_mm256_fmadd_ps(xs3, weight0, acc30);
        acc31 = x86_arch::_mm256_fmadd_ps(xs3, weight1, acc31);
        index += 8;
    }

    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut sum00 = horizontal_sum_avx2_linear!(use_direct_sum, acc00);
    let mut sum01 = horizontal_sum_avx2_linear!(use_direct_sum, acc01);
    let mut sum10 = horizontal_sum_avx2_linear!(use_direct_sum, acc10);
    let mut sum11 = horizontal_sum_avx2_linear!(use_direct_sum, acc11);
    let mut sum20 = horizontal_sum_avx2_linear!(use_direct_sum, acc20);
    let mut sum21 = horizontal_sum_avx2_linear!(use_direct_sum, acc21);
    let mut sum30 = horizontal_sum_avx2_linear!(use_direct_sum, acc30);
    let mut sum31 = horizontal_sum_avx2_linear!(use_direct_sum, acc31);
    while index < xs_row0.len() {
        let x0 = xs_row0[index];
        let x1 = xs_row1[index];
        let x2 = xs_row2[index];
        let x3 = xs_row3[index];
        sum00 += x0 * weight_row0[index];
        sum01 += x0 * weight_row1[index];
        sum10 += x1 * weight_row0[index];
        sum11 += x1 * weight_row1[index];
        sum20 += x2 * weight_row0[index];
        sum21 += x2 * weight_row1[index];
        sum30 += x3 * weight_row0[index];
        sum31 += x3 * weight_row1[index];
        index += 1;
    }

    [
        [sum00, sum01],
        [sum10, sum11],
        [sum20, sum21],
        [sum30, sum31],
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot2_four_rows_same_x_avx2_fma_ptr(
    xs_row0: &[f32],
    xs_row1: &[f32],
    xs_row2: &[f32],
    xs_row3: &[f32],
    weight_ptr: *const f32,
    in_dim: usize,
) -> [[f32; 2]; 4] {
    debug_assert_eq!(xs_row0.len(), in_dim);
    debug_assert_eq!(xs_row1.len(), in_dim);
    debug_assert_eq!(xs_row2.len(), in_dim);
    debug_assert_eq!(xs_row3.len(), in_dim);

    let mut acc00 = x86_arch::_mm256_setzero_ps();
    let mut acc01 = x86_arch::_mm256_setzero_ps();
    let mut acc10 = x86_arch::_mm256_setzero_ps();
    let mut acc11 = x86_arch::_mm256_setzero_ps();
    let mut acc20 = x86_arch::_mm256_setzero_ps();
    let mut acc21 = x86_arch::_mm256_setzero_ps();
    let mut acc30 = x86_arch::_mm256_setzero_ps();
    let mut acc31 = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= in_dim {
        let xs0 = x86_arch::_mm256_loadu_ps(xs_row0.as_ptr().add(index));
        let xs1 = x86_arch::_mm256_loadu_ps(xs_row1.as_ptr().add(index));
        let xs2 = x86_arch::_mm256_loadu_ps(xs_row2.as_ptr().add(index));
        let xs3 = x86_arch::_mm256_loadu_ps(xs_row3.as_ptr().add(index));
        let weight0 = x86_arch::_mm256_loadu_ps(weight_ptr.add(index));
        let weight1 = x86_arch::_mm256_loadu_ps(weight_ptr.add(in_dim + index));
        acc00 = x86_arch::_mm256_fmadd_ps(xs0, weight0, acc00);
        acc01 = x86_arch::_mm256_fmadd_ps(xs0, weight1, acc01);
        acc10 = x86_arch::_mm256_fmadd_ps(xs1, weight0, acc10);
        acc11 = x86_arch::_mm256_fmadd_ps(xs1, weight1, acc11);
        acc20 = x86_arch::_mm256_fmadd_ps(xs2, weight0, acc20);
        acc21 = x86_arch::_mm256_fmadd_ps(xs2, weight1, acc21);
        acc30 = x86_arch::_mm256_fmadd_ps(xs3, weight0, acc30);
        acc31 = x86_arch::_mm256_fmadd_ps(xs3, weight1, acc31);
        index += 8;
    }

    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut sum00 = horizontal_sum_avx2_linear!(use_direct_sum, acc00);
    let mut sum01 = horizontal_sum_avx2_linear!(use_direct_sum, acc01);
    let mut sum10 = horizontal_sum_avx2_linear!(use_direct_sum, acc10);
    let mut sum11 = horizontal_sum_avx2_linear!(use_direct_sum, acc11);
    let mut sum20 = horizontal_sum_avx2_linear!(use_direct_sum, acc20);
    let mut sum21 = horizontal_sum_avx2_linear!(use_direct_sum, acc21);
    let mut sum30 = horizontal_sum_avx2_linear!(use_direct_sum, acc30);
    let mut sum31 = horizontal_sum_avx2_linear!(use_direct_sum, acc31);
    while index < in_dim {
        let x0 = xs_row0[index];
        let x1 = xs_row1[index];
        let x2 = xs_row2[index];
        let x3 = xs_row3[index];
        sum00 += x0 * *weight_ptr.add(index);
        sum01 += x0 * *weight_ptr.add(in_dim + index);
        sum10 += x1 * *weight_ptr.add(index);
        sum11 += x1 * *weight_ptr.add(in_dim + index);
        sum20 += x2 * *weight_ptr.add(index);
        sum21 += x2 * *weight_ptr.add(in_dim + index);
        sum30 += x3 * *weight_ptr.add(index);
        sum31 += x3 * *weight_ptr.add(in_dim + index);
        index += 1;
    }

    [
        [sum00, sum01],
        [sum10, sum11],
        [sum20, sum21],
        [sum30, sum31],
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot4_two_rows_same_x_avx2_fma(
    xs_row0: &[f32],
    xs_row1: &[f32],
    weight_row0: &[f32],
    weight_row1: &[f32],
    weight_row2: &[f32],
    weight_row3: &[f32],
) -> ([f32; 4], [f32; 4]) {
    debug_assert_eq!(xs_row0.len(), xs_row1.len());
    debug_assert_eq!(xs_row0.len(), weight_row0.len());
    debug_assert_eq!(xs_row0.len(), weight_row1.len());
    debug_assert_eq!(xs_row0.len(), weight_row2.len());
    debug_assert_eq!(xs_row0.len(), weight_row3.len());

    let mut acc00 = x86_arch::_mm256_setzero_ps();
    let mut acc01 = x86_arch::_mm256_setzero_ps();
    let mut acc02 = x86_arch::_mm256_setzero_ps();
    let mut acc03 = x86_arch::_mm256_setzero_ps();
    let mut acc10 = x86_arch::_mm256_setzero_ps();
    let mut acc11 = x86_arch::_mm256_setzero_ps();
    let mut acc12 = x86_arch::_mm256_setzero_ps();
    let mut acc13 = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= xs_row0.len() {
        let xs0 = x86_arch::_mm256_loadu_ps(xs_row0.as_ptr().add(index));
        let xs1 = x86_arch::_mm256_loadu_ps(xs_row1.as_ptr().add(index));
        let weight0 = x86_arch::_mm256_loadu_ps(weight_row0.as_ptr().add(index));
        let weight1 = x86_arch::_mm256_loadu_ps(weight_row1.as_ptr().add(index));
        let weight2 = x86_arch::_mm256_loadu_ps(weight_row2.as_ptr().add(index));
        let weight3 = x86_arch::_mm256_loadu_ps(weight_row3.as_ptr().add(index));
        acc00 = x86_arch::_mm256_fmadd_ps(xs0, weight0, acc00);
        acc01 = x86_arch::_mm256_fmadd_ps(xs0, weight1, acc01);
        acc02 = x86_arch::_mm256_fmadd_ps(xs0, weight2, acc02);
        acc03 = x86_arch::_mm256_fmadd_ps(xs0, weight3, acc03);
        acc10 = x86_arch::_mm256_fmadd_ps(xs1, weight0, acc10);
        acc11 = x86_arch::_mm256_fmadd_ps(xs1, weight1, acc11);
        acc12 = x86_arch::_mm256_fmadd_ps(xs1, weight2, acc12);
        acc13 = x86_arch::_mm256_fmadd_ps(xs1, weight3, acc13);
        index += 8;
    }

    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut sum00 = horizontal_sum_avx2_linear!(use_direct_sum, acc00);
    let mut sum01 = horizontal_sum_avx2_linear!(use_direct_sum, acc01);
    let mut sum02 = horizontal_sum_avx2_linear!(use_direct_sum, acc02);
    let mut sum03 = horizontal_sum_avx2_linear!(use_direct_sum, acc03);
    let mut sum10 = horizontal_sum_avx2_linear!(use_direct_sum, acc10);
    let mut sum11 = horizontal_sum_avx2_linear!(use_direct_sum, acc11);
    let mut sum12 = horizontal_sum_avx2_linear!(use_direct_sum, acc12);
    let mut sum13 = horizontal_sum_avx2_linear!(use_direct_sum, acc13);
    while index < xs_row0.len() {
        let x0 = xs_row0[index];
        let x1 = xs_row1[index];
        sum00 += x0 * weight_row0[index];
        sum01 += x0 * weight_row1[index];
        sum02 += x0 * weight_row2[index];
        sum03 += x0 * weight_row3[index];
        sum10 += x1 * weight_row0[index];
        sum11 += x1 * weight_row1[index];
        sum12 += x1 * weight_row2[index];
        sum13 += x1 * weight_row3[index];
        index += 1;
    }

    ([sum00, sum01, sum02, sum03], [sum10, sum11, sum12, sum13])
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot4_two_rows_same_x_avx2_fma_ptr(
    xs_row0: &[f32],
    xs_row1: &[f32],
    weight_ptr: *const f32,
    in_dim: usize,
) -> ([f32; 4], [f32; 4]) {
    debug_assert_eq!(xs_row0.len(), in_dim);
    debug_assert_eq!(xs_row1.len(), in_dim);

    let mut acc00 = x86_arch::_mm256_setzero_ps();
    let mut acc01 = x86_arch::_mm256_setzero_ps();
    let mut acc02 = x86_arch::_mm256_setzero_ps();
    let mut acc03 = x86_arch::_mm256_setzero_ps();
    let mut acc10 = x86_arch::_mm256_setzero_ps();
    let mut acc11 = x86_arch::_mm256_setzero_ps();
    let mut acc12 = x86_arch::_mm256_setzero_ps();
    let mut acc13 = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= in_dim {
        let xs0 = x86_arch::_mm256_loadu_ps(xs_row0.as_ptr().add(index));
        let xs1 = x86_arch::_mm256_loadu_ps(xs_row1.as_ptr().add(index));
        let weight0 = x86_arch::_mm256_loadu_ps(weight_ptr.add(index));
        let weight1 = x86_arch::_mm256_loadu_ps(weight_ptr.add(in_dim + index));
        let weight2 = x86_arch::_mm256_loadu_ps(weight_ptr.add(2 * in_dim + index));
        let weight3 = x86_arch::_mm256_loadu_ps(weight_ptr.add(3 * in_dim + index));
        acc00 = x86_arch::_mm256_fmadd_ps(xs0, weight0, acc00);
        acc01 = x86_arch::_mm256_fmadd_ps(xs0, weight1, acc01);
        acc02 = x86_arch::_mm256_fmadd_ps(xs0, weight2, acc02);
        acc03 = x86_arch::_mm256_fmadd_ps(xs0, weight3, acc03);
        acc10 = x86_arch::_mm256_fmadd_ps(xs1, weight0, acc10);
        acc11 = x86_arch::_mm256_fmadd_ps(xs1, weight1, acc11);
        acc12 = x86_arch::_mm256_fmadd_ps(xs1, weight2, acc12);
        acc13 = x86_arch::_mm256_fmadd_ps(xs1, weight3, acc13);
        index += 8;
    }

    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut sum00 = horizontal_sum_avx2_linear!(use_direct_sum, acc00);
    let mut sum01 = horizontal_sum_avx2_linear!(use_direct_sum, acc01);
    let mut sum02 = horizontal_sum_avx2_linear!(use_direct_sum, acc02);
    let mut sum03 = horizontal_sum_avx2_linear!(use_direct_sum, acc03);
    let mut sum10 = horizontal_sum_avx2_linear!(use_direct_sum, acc10);
    let mut sum11 = horizontal_sum_avx2_linear!(use_direct_sum, acc11);
    let mut sum12 = horizontal_sum_avx2_linear!(use_direct_sum, acc12);
    let mut sum13 = horizontal_sum_avx2_linear!(use_direct_sum, acc13);
    while index < in_dim {
        let x0 = xs_row0[index];
        let x1 = xs_row1[index];
        sum00 += x0 * *weight_ptr.add(index);
        sum01 += x0 * *weight_ptr.add(in_dim + index);
        sum02 += x0 * *weight_ptr.add(2 * in_dim + index);
        sum03 += x0 * *weight_ptr.add(3 * in_dim + index);
        sum10 += x1 * *weight_ptr.add(index);
        sum11 += x1 * *weight_ptr.add(in_dim + index);
        sum12 += x1 * *weight_ptr.add(2 * in_dim + index);
        sum13 += x1 * *weight_ptr.add(3 * in_dim + index);
        index += 1;
    }

    ([sum00, sum01, sum02, sum03], [sum10, sum11, sum12, sum13])
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_project_row_same_x_avx2_fma_dot12_fixed<const IN_DIM: usize>(
    xs_row: &[f32],
    weight_values: &[f32],
    out_dim: usize,
    output_row: &mut [f32],
) {
    debug_assert_eq!(IN_DIM % 8, 0);
    debug_assert_eq!(xs_row.len(), IN_DIM);
    debug_assert_eq!(weight_values.len(), out_dim * IN_DIM);
    debug_assert_eq!(output_row.len(), out_dim);

    let mut out_index = 0usize;
    while out_index + 12 <= out_dim {
        let values = if native_linear_dot12_pointer_rows_enabled() {
            linear_dot12_same_x_avx2_fma_fixed_ptr::<IN_DIM>(
                xs_row,
                weight_values.as_ptr().add(out_index * IN_DIM),
            )
        } else {
            let w0 = &weight_values[out_index * IN_DIM..(out_index + 1) * IN_DIM];
            let w1 = &weight_values[(out_index + 1) * IN_DIM..(out_index + 2) * IN_DIM];
            let w2 = &weight_values[(out_index + 2) * IN_DIM..(out_index + 3) * IN_DIM];
            let w3 = &weight_values[(out_index + 3) * IN_DIM..(out_index + 4) * IN_DIM];
            let w4 = &weight_values[(out_index + 4) * IN_DIM..(out_index + 5) * IN_DIM];
            let w5 = &weight_values[(out_index + 5) * IN_DIM..(out_index + 6) * IN_DIM];
            let w6 = &weight_values[(out_index + 6) * IN_DIM..(out_index + 7) * IN_DIM];
            let w7 = &weight_values[(out_index + 7) * IN_DIM..(out_index + 8) * IN_DIM];
            let w8 = &weight_values[(out_index + 8) * IN_DIM..(out_index + 9) * IN_DIM];
            let w9 = &weight_values[(out_index + 9) * IN_DIM..(out_index + 10) * IN_DIM];
            let w10 = &weight_values[(out_index + 10) * IN_DIM..(out_index + 11) * IN_DIM];
            let w11 = &weight_values[(out_index + 11) * IN_DIM..(out_index + 12) * IN_DIM];
            linear_dot12_same_x_avx2_fma_fixed::<IN_DIM>(
                xs_row,
                [w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11],
            )
        };
        output_row[out_index..out_index + 12].copy_from_slice(&values);
        out_index += 12;
    }

    while out_index + 8 <= out_dim {
        let w0 = &weight_values[out_index * IN_DIM..(out_index + 1) * IN_DIM];
        let w1 = &weight_values[(out_index + 1) * IN_DIM..(out_index + 2) * IN_DIM];
        let w2 = &weight_values[(out_index + 2) * IN_DIM..(out_index + 3) * IN_DIM];
        let w3 = &weight_values[(out_index + 3) * IN_DIM..(out_index + 4) * IN_DIM];
        let w4 = &weight_values[(out_index + 4) * IN_DIM..(out_index + 5) * IN_DIM];
        let w5 = &weight_values[(out_index + 5) * IN_DIM..(out_index + 6) * IN_DIM];
        let w6 = &weight_values[(out_index + 6) * IN_DIM..(out_index + 7) * IN_DIM];
        let w7 = &weight_values[(out_index + 7) * IN_DIM..(out_index + 8) * IN_DIM];
        let values =
            linear_dot8_same_x_avx2_fma_fixed::<IN_DIM>(xs_row, [w0, w1, w2, w3, w4, w5, w6, w7]);
        output_row[out_index..out_index + 8].copy_from_slice(&values);
        out_index += 8;
    }

    while out_index + 4 <= out_dim {
        let w0 = &weight_values[out_index * IN_DIM..(out_index + 1) * IN_DIM];
        let w1 = &weight_values[(out_index + 1) * IN_DIM..(out_index + 2) * IN_DIM];
        let w2 = &weight_values[(out_index + 2) * IN_DIM..(out_index + 3) * IN_DIM];
        let w3 = &weight_values[(out_index + 3) * IN_DIM..(out_index + 4) * IN_DIM];
        let values = linear_dot4_same_x_avx2_fma_fixed::<IN_DIM>(xs_row, w0, w1, w2, w3);
        output_row[out_index] = values[0];
        output_row[out_index + 1] = values[1];
        output_row[out_index + 2] = values[2];
        output_row[out_index + 3] = values[3];
        out_index += 4;
    }

    while out_index < out_dim {
        let weight_row = &weight_values[out_index * IN_DIM..(out_index + 1) * IN_DIM];
        output_row[out_index] = linear_dot_avx2_fma_fixed::<IN_DIM>(xs_row, weight_row);
        out_index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_project_row_same_x_avx2_fma_dot12(
    xs_row: &[f32],
    weight_values: &[f32],
    out_dim: usize,
    in_dim: usize,
    output_row: &mut [f32],
) {
    let mut out_index = 0usize;
    while out_index + 12 <= out_dim {
        let values = if native_linear_dot12_pointer_rows_enabled() {
            linear_dot12_same_x_avx2_fma_ptr(
                xs_row,
                weight_values.as_ptr().add(out_index * in_dim),
                in_dim,
            )
        } else {
            let w0 = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
            let w1 = &weight_values[(out_index + 1) * in_dim..(out_index + 2) * in_dim];
            let w2 = &weight_values[(out_index + 2) * in_dim..(out_index + 3) * in_dim];
            let w3 = &weight_values[(out_index + 3) * in_dim..(out_index + 4) * in_dim];
            let w4 = &weight_values[(out_index + 4) * in_dim..(out_index + 5) * in_dim];
            let w5 = &weight_values[(out_index + 5) * in_dim..(out_index + 6) * in_dim];
            let w6 = &weight_values[(out_index + 6) * in_dim..(out_index + 7) * in_dim];
            let w7 = &weight_values[(out_index + 7) * in_dim..(out_index + 8) * in_dim];
            let w8 = &weight_values[(out_index + 8) * in_dim..(out_index + 9) * in_dim];
            let w9 = &weight_values[(out_index + 9) * in_dim..(out_index + 10) * in_dim];
            let w10 = &weight_values[(out_index + 10) * in_dim..(out_index + 11) * in_dim];
            let w11 = &weight_values[(out_index + 11) * in_dim..(out_index + 12) * in_dim];
            linear_dot12_same_x_avx2_fma(xs_row, [w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11])
        };
        output_row[out_index..out_index + 12].copy_from_slice(&values);
        out_index += 12;
    }

    while out_index + 8 <= out_dim {
        let w0 = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        let w1 = &weight_values[(out_index + 1) * in_dim..(out_index + 2) * in_dim];
        let w2 = &weight_values[(out_index + 2) * in_dim..(out_index + 3) * in_dim];
        let w3 = &weight_values[(out_index + 3) * in_dim..(out_index + 4) * in_dim];
        let w4 = &weight_values[(out_index + 4) * in_dim..(out_index + 5) * in_dim];
        let w5 = &weight_values[(out_index + 5) * in_dim..(out_index + 6) * in_dim];
        let w6 = &weight_values[(out_index + 6) * in_dim..(out_index + 7) * in_dim];
        let w7 = &weight_values[(out_index + 7) * in_dim..(out_index + 8) * in_dim];
        let values = linear_dot8_same_x_avx2_fma(xs_row, [w0, w1, w2, w3, w4, w5, w6, w7]);
        output_row[out_index..out_index + 8].copy_from_slice(&values);
        out_index += 8;
    }

    while out_index + 4 <= out_dim {
        let w0 = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        let w1 = &weight_values[(out_index + 1) * in_dim..(out_index + 2) * in_dim];
        let w2 = &weight_values[(out_index + 2) * in_dim..(out_index + 3) * in_dim];
        let w3 = &weight_values[(out_index + 3) * in_dim..(out_index + 4) * in_dim];
        let values = linear_dot4_same_x_avx2_fma(xs_row, w0, w1, w2, w3);
        output_row[out_index] = values[0];
        output_row[out_index + 1] = values[1];
        output_row[out_index + 2] = values[2];
        output_row[out_index + 3] = values[3];
        out_index += 4;
    }

    while out_index < out_dim {
        let weight_row = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        output_row[out_index] = linear_dot_avx2_fma(xs_row, weight_row);
        out_index += 1;
    }
}

pub(super) fn linear_pack_input8_dot12_weights(
    weight_values: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Option<LinearBlockedInput8Dot12Weights> {
    if in_dim == 0 || in_dim % 8 != 0 || weight_values.len() != out_dim * in_dim {
        return None;
    }
    let full_out_dim = out_dim / 12 * 12;
    let mut values = Vec::with_capacity(full_out_dim * in_dim);
    for out_base in (0..full_out_dim).step_by(12) {
        for input_base in (0..in_dim).step_by(8) {
            for output_offset in 0..12 {
                let source_base = (out_base + output_offset) * in_dim + input_base;
                values.extend_from_slice(&weight_values[source_base..source_base + 8]);
            }
        }
    }
    Some(LinearBlockedInput8Dot12Weights {
        values,
        full_out_dim,
        in_dim,
    })
}

#[inline(always)]
pub(super) fn linear_project_f32_row_same_x(
    kernel: NativeLinearDotKernel,
    xs_row: &[f32],
    weights: &LinearF32Weights,
    output_row: &mut [f32],
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let (NativeLinearDotKernel::Avx2Fma, Some(blocked)) =
        (kernel, weights.blocked_input8_dot12.as_ref())
    {
        unsafe {
            linear_project_row_same_x_avx2_fma_blocked_input8_dot12(
                xs_row,
                blocked,
                &weights.weight,
                weights.out_dim,
                output_row,
            );
        }
        return;
    }
    linear_project_row_same_x(
        kernel,
        xs_row,
        &weights.weight,
        weights.out_dim,
        weights.in_dim,
        output_row,
    );
}

pub(super) fn linear_project_f32_batch_same_x(
    kernel: NativeLinearDotKernel,
    input_values: &[f32],
    row_count: usize,
    weights: &LinearF32Weights,
    output_values: &mut [f32],
) {
    if row_count == 1 {
        linear_project_f32_row_same_x(kernel, input_values, weights, output_values);
        return;
    }
    if linear_project_batch_same_x(
        kernel,
        input_values,
        row_count,
        weights.in_dim,
        &weights.weight,
        weights.out_dim,
        output_values,
    ) {
        return;
    }
    for row in 0..row_count {
        let input_base = row * weights.in_dim;
        let output_base = row * weights.out_dim;
        linear_project_row_same_x(
            kernel,
            &input_values[input_base..input_base + weights.in_dim],
            &weights.weight,
            weights.out_dim,
            weights.in_dim,
            &mut output_values[output_base..output_base + weights.out_dim],
        );
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_project_row_same_x_avx2_fma_blocked_input8_dot12(
    xs_row: &[f32],
    blocked: &LinearBlockedInput8Dot12Weights,
    row_major_weight_values: &[f32],
    out_dim: usize,
    output_row: &mut [f32],
) {
    let in_dim = blocked.in_dim;
    debug_assert_eq!(xs_row.len(), in_dim);
    debug_assert_eq!(row_major_weight_values.len(), out_dim * in_dim);
    debug_assert_eq!(output_row.len(), out_dim);
    debug_assert_eq!(blocked.full_out_dim, out_dim / 12 * 12);
    debug_assert_eq!(blocked.values.len(), blocked.full_out_dim * in_dim);

    let use_direct_sum =
        native_linear_dot12_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    for out_base in (0..blocked.full_out_dim).step_by(12) {
        let mut acc0 = x86_arch::_mm256_setzero_ps();
        let mut acc1 = x86_arch::_mm256_setzero_ps();
        let mut acc2 = x86_arch::_mm256_setzero_ps();
        let mut acc3 = x86_arch::_mm256_setzero_ps();
        let mut acc4 = x86_arch::_mm256_setzero_ps();
        let mut acc5 = x86_arch::_mm256_setzero_ps();
        let mut acc6 = x86_arch::_mm256_setzero_ps();
        let mut acc7 = x86_arch::_mm256_setzero_ps();
        let mut acc8 = x86_arch::_mm256_setzero_ps();
        let mut acc9 = x86_arch::_mm256_setzero_ps();
        let mut acc10 = x86_arch::_mm256_setzero_ps();
        let mut acc11 = x86_arch::_mm256_setzero_ps();
        let block_base = out_base * in_dim;
        for input_base in (0..in_dim).step_by(8) {
            let xs = x86_arch::_mm256_loadu_ps(xs_row.as_ptr().add(input_base));
            let chunk_base = block_base + input_base * 12;
            let weight_ptr = blocked.values.as_ptr().add(chunk_base);
            acc0 = x86_arch::_mm256_fmadd_ps(xs, x86_arch::_mm256_loadu_ps(weight_ptr), acc0);
            acc1 =
                x86_arch::_mm256_fmadd_ps(xs, x86_arch::_mm256_loadu_ps(weight_ptr.add(8)), acc1);
            acc2 =
                x86_arch::_mm256_fmadd_ps(xs, x86_arch::_mm256_loadu_ps(weight_ptr.add(16)), acc2);
            acc3 =
                x86_arch::_mm256_fmadd_ps(xs, x86_arch::_mm256_loadu_ps(weight_ptr.add(24)), acc3);
            acc4 =
                x86_arch::_mm256_fmadd_ps(xs, x86_arch::_mm256_loadu_ps(weight_ptr.add(32)), acc4);
            acc5 =
                x86_arch::_mm256_fmadd_ps(xs, x86_arch::_mm256_loadu_ps(weight_ptr.add(40)), acc5);
            acc6 =
                x86_arch::_mm256_fmadd_ps(xs, x86_arch::_mm256_loadu_ps(weight_ptr.add(48)), acc6);
            acc7 =
                x86_arch::_mm256_fmadd_ps(xs, x86_arch::_mm256_loadu_ps(weight_ptr.add(56)), acc7);
            acc8 =
                x86_arch::_mm256_fmadd_ps(xs, x86_arch::_mm256_loadu_ps(weight_ptr.add(64)), acc8);
            acc9 =
                x86_arch::_mm256_fmadd_ps(xs, x86_arch::_mm256_loadu_ps(weight_ptr.add(72)), acc9);
            acc10 =
                x86_arch::_mm256_fmadd_ps(xs, x86_arch::_mm256_loadu_ps(weight_ptr.add(80)), acc10);
            acc11 =
                x86_arch::_mm256_fmadd_ps(xs, x86_arch::_mm256_loadu_ps(weight_ptr.add(88)), acc11);
        }
        let sums = [
            horizontal_sum_avx2_linear!(use_direct_sum, acc0),
            horizontal_sum_avx2_linear!(use_direct_sum, acc1),
            horizontal_sum_avx2_linear!(use_direct_sum, acc2),
            horizontal_sum_avx2_linear!(use_direct_sum, acc3),
            horizontal_sum_avx2_linear!(use_direct_sum, acc4),
            horizontal_sum_avx2_linear!(use_direct_sum, acc5),
            horizontal_sum_avx2_linear!(use_direct_sum, acc6),
            horizontal_sum_avx2_linear!(use_direct_sum, acc7),
            horizontal_sum_avx2_linear!(use_direct_sum, acc8),
            horizontal_sum_avx2_linear!(use_direct_sum, acc9),
            horizontal_sum_avx2_linear!(use_direct_sum, acc10),
            horizontal_sum_avx2_linear!(use_direct_sum, acc11),
        ];
        output_row[out_base..out_base + 12].copy_from_slice(&sums);
    }

    let mut out_index = blocked.full_out_dim;
    while out_index + 8 <= out_dim {
        let weight_base = out_index * in_dim;
        let values = linear_dot8_same_x_avx2_fma(
            xs_row,
            [
                &row_major_weight_values[weight_base..weight_base + in_dim],
                &row_major_weight_values[weight_base + in_dim..weight_base + 2 * in_dim],
                &row_major_weight_values[weight_base + 2 * in_dim..weight_base + 3 * in_dim],
                &row_major_weight_values[weight_base + 3 * in_dim..weight_base + 4 * in_dim],
                &row_major_weight_values[weight_base + 4 * in_dim..weight_base + 5 * in_dim],
                &row_major_weight_values[weight_base + 5 * in_dim..weight_base + 6 * in_dim],
                &row_major_weight_values[weight_base + 6 * in_dim..weight_base + 7 * in_dim],
                &row_major_weight_values[weight_base + 7 * in_dim..weight_base + 8 * in_dim],
            ],
        );
        output_row[out_index..out_index + 8].copy_from_slice(&values);
        out_index += 8;
    }
    while out_index + 4 <= out_dim {
        let weight_base = out_index * in_dim;
        let values = linear_dot4_same_x_avx2_fma(
            xs_row,
            &row_major_weight_values[weight_base..weight_base + in_dim],
            &row_major_weight_values[weight_base + in_dim..weight_base + 2 * in_dim],
            &row_major_weight_values[weight_base + 2 * in_dim..weight_base + 3 * in_dim],
            &row_major_weight_values[weight_base + 3 * in_dim..weight_base + 4 * in_dim],
        );
        output_row[out_index..out_index + 4].copy_from_slice(&values);
        out_index += 4;
    }
    while out_index < out_dim {
        let weight_base = out_index * in_dim;
        output_row[out_index] = linear_dot_avx2_fma(
            xs_row,
            &row_major_weight_values[weight_base..weight_base + in_dim],
        );
        out_index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_project_row_same_x_avx2_fma_dot8_fixed<const IN_DIM: usize>(
    xs_row: &[f32],
    weight_values: &[f32],
    out_dim: usize,
    output_row: &mut [f32],
) {
    debug_assert_eq!(IN_DIM % 8, 0);
    debug_assert_eq!(xs_row.len(), IN_DIM);
    debug_assert_eq!(weight_values.len(), out_dim * IN_DIM);
    debug_assert_eq!(output_row.len(), out_dim);

    let mut out_index = 0usize;
    while out_index + 8 <= out_dim {
        let w0 = &weight_values[out_index * IN_DIM..(out_index + 1) * IN_DIM];
        let w1 = &weight_values[(out_index + 1) * IN_DIM..(out_index + 2) * IN_DIM];
        let w2 = &weight_values[(out_index + 2) * IN_DIM..(out_index + 3) * IN_DIM];
        let w3 = &weight_values[(out_index + 3) * IN_DIM..(out_index + 4) * IN_DIM];
        let w4 = &weight_values[(out_index + 4) * IN_DIM..(out_index + 5) * IN_DIM];
        let w5 = &weight_values[(out_index + 5) * IN_DIM..(out_index + 6) * IN_DIM];
        let w6 = &weight_values[(out_index + 6) * IN_DIM..(out_index + 7) * IN_DIM];
        let w7 = &weight_values[(out_index + 7) * IN_DIM..(out_index + 8) * IN_DIM];
        let values =
            linear_dot8_same_x_avx2_fma_fixed::<IN_DIM>(xs_row, [w0, w1, w2, w3, w4, w5, w6, w7]);
        output_row[out_index..out_index + 8].copy_from_slice(&values);
        out_index += 8;
    }

    while out_index + 4 <= out_dim {
        let w0 = &weight_values[out_index * IN_DIM..(out_index + 1) * IN_DIM];
        let w1 = &weight_values[(out_index + 1) * IN_DIM..(out_index + 2) * IN_DIM];
        let w2 = &weight_values[(out_index + 2) * IN_DIM..(out_index + 3) * IN_DIM];
        let w3 = &weight_values[(out_index + 3) * IN_DIM..(out_index + 4) * IN_DIM];
        let values = linear_dot4_same_x_avx2_fma_fixed::<IN_DIM>(xs_row, w0, w1, w2, w3);
        output_row[out_index] = values[0];
        output_row[out_index + 1] = values[1];
        output_row[out_index + 2] = values[2];
        output_row[out_index + 3] = values[3];
        out_index += 4;
    }

    while out_index < out_dim {
        let weight_row = &weight_values[out_index * IN_DIM..(out_index + 1) * IN_DIM];
        output_row[out_index] = linear_dot_avx2_fma_fixed::<IN_DIM>(xs_row, weight_row);
        out_index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_project_row_same_x_avx2_fma_dot8(
    xs_row: &[f32],
    weight_values: &[f32],
    out_dim: usize,
    in_dim: usize,
    output_row: &mut [f32],
) {
    let mut out_index = 0usize;
    while out_index + 8 <= out_dim {
        let w0 = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        let w1 = &weight_values[(out_index + 1) * in_dim..(out_index + 2) * in_dim];
        let w2 = &weight_values[(out_index + 2) * in_dim..(out_index + 3) * in_dim];
        let w3 = &weight_values[(out_index + 3) * in_dim..(out_index + 4) * in_dim];
        let w4 = &weight_values[(out_index + 4) * in_dim..(out_index + 5) * in_dim];
        let w5 = &weight_values[(out_index + 5) * in_dim..(out_index + 6) * in_dim];
        let w6 = &weight_values[(out_index + 6) * in_dim..(out_index + 7) * in_dim];
        let w7 = &weight_values[(out_index + 7) * in_dim..(out_index + 8) * in_dim];
        let values = linear_dot8_same_x_avx2_fma(xs_row, [w0, w1, w2, w3, w4, w5, w6, w7]);
        output_row[out_index..out_index + 8].copy_from_slice(&values);
        out_index += 8;
    }

    while out_index + 4 <= out_dim {
        let w0 = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        let w1 = &weight_values[(out_index + 1) * in_dim..(out_index + 2) * in_dim];
        let w2 = &weight_values[(out_index + 2) * in_dim..(out_index + 3) * in_dim];
        let w3 = &weight_values[(out_index + 3) * in_dim..(out_index + 4) * in_dim];
        let values = linear_dot4_same_x_avx2_fma(xs_row, w0, w1, w2, w3);
        output_row[out_index] = values[0];
        output_row[out_index + 1] = values[1];
        output_row[out_index + 2] = values[2];
        output_row[out_index + 3] = values[3];
        out_index += 4;
    }

    while out_index < out_dim {
        let weight_row = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        output_row[out_index] = linear_dot_avx2_fma(xs_row, weight_row);
        out_index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot_avx2_fma_fixed<const IN_DIM: usize>(
    xs_row: &[f32],
    weight_row: &[f32],
) -> f32 {
    debug_assert_eq!(IN_DIM % 8, 0);
    debug_assert_eq!(xs_row.len(), IN_DIM);
    debug_assert_eq!(weight_row.len(), IN_DIM);

    let mut acc = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index < IN_DIM {
        let xs = x86_arch::_mm256_loadu_ps(xs_row.as_ptr().add(index));
        let weight = x86_arch::_mm256_loadu_ps(weight_row.as_ptr().add(index));
        acc = x86_arch::_mm256_fmadd_ps(xs, weight, acc);
        index += 8;
    }
    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    horizontal_sum_avx2_linear!(use_direct_sum, acc)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot_avx2_fma(xs_row: &[f32], weight_row: &[f32]) -> f32 {
    debug_assert_eq!(xs_row.len(), weight_row.len());
    let mut acc = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= xs_row.len() {
        let xs = x86_arch::_mm256_loadu_ps(xs_row.as_ptr().add(index));
        let weight = x86_arch::_mm256_loadu_ps(weight_row.as_ptr().add(index));
        acc = x86_arch::_mm256_fmadd_ps(xs, weight, acc);
        index += 8;
    }

    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut sum = horizontal_sum_avx2_linear!(use_direct_sum, acc);
    while index < xs_row.len() {
        sum += xs_row[index] * weight_row[index];
        index += 1;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_project_row_same_x_avx2_fma_fixed<const IN_DIM: usize>(
    xs_row: &[f32],
    weight_values: &[f32],
    out_dim: usize,
    output_row: &mut [f32],
) {
    debug_assert_eq!(IN_DIM % 8, 0);
    debug_assert_eq!(xs_row.len(), IN_DIM);
    debug_assert_eq!(weight_values.len(), out_dim * IN_DIM);
    debug_assert_eq!(output_row.len(), out_dim);

    let mut out_index = 0usize;
    while out_index + 4 <= out_dim {
        let w0 = &weight_values[out_index * IN_DIM..(out_index + 1) * IN_DIM];
        let w1 = &weight_values[(out_index + 1) * IN_DIM..(out_index + 2) * IN_DIM];
        let w2 = &weight_values[(out_index + 2) * IN_DIM..(out_index + 3) * IN_DIM];
        let w3 = &weight_values[(out_index + 3) * IN_DIM..(out_index + 4) * IN_DIM];
        let values = linear_dot4_same_x_avx2_fma_fixed::<IN_DIM>(xs_row, w0, w1, w2, w3);
        output_row[out_index] = values[0];
        output_row[out_index + 1] = values[1];
        output_row[out_index + 2] = values[2];
        output_row[out_index + 3] = values[3];
        out_index += 4;
    }

    while out_index < out_dim {
        let weight_row = &weight_values[out_index * IN_DIM..(out_index + 1) * IN_DIM];
        output_row[out_index] = linear_dot_avx2_fma_fixed::<IN_DIM>(xs_row, weight_row);
        out_index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_project_row_same_x_avx2_fma(
    xs_row: &[f32],
    weight_values: &[f32],
    out_dim: usize,
    in_dim: usize,
    output_row: &mut [f32],
) {
    let mut out_index = 0usize;
    while out_index + 4 <= out_dim {
        let w0 = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        let w1 = &weight_values[(out_index + 1) * in_dim..(out_index + 2) * in_dim];
        let w2 = &weight_values[(out_index + 2) * in_dim..(out_index + 3) * in_dim];
        let w3 = &weight_values[(out_index + 3) * in_dim..(out_index + 4) * in_dim];
        let values = linear_dot4_same_x_avx2_fma(xs_row, w0, w1, w2, w3);
        output_row[out_index] = values[0];
        output_row[out_index + 1] = values[1];
        output_row[out_index + 2] = values[2];
        output_row[out_index + 3] = values[3];
        out_index += 4;
    }

    while out_index < out_dim {
        let weight_row = &weight_values[out_index * in_dim..(out_index + 1) * in_dim];
        output_row[out_index] = linear_dot_avx2_fma(xs_row, weight_row);
        out_index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot4_same_x_avx2_fma_fixed<const IN_DIM: usize>(
    xs_row: &[f32],
    weight_row0: &[f32],
    weight_row1: &[f32],
    weight_row2: &[f32],
    weight_row3: &[f32],
) -> [f32; 4] {
    debug_assert_eq!(IN_DIM % 8, 0);
    debug_assert_eq!(xs_row.len(), IN_DIM);
    debug_assert_eq!(weight_row0.len(), IN_DIM);
    debug_assert_eq!(weight_row1.len(), IN_DIM);
    debug_assert_eq!(weight_row2.len(), IN_DIM);
    debug_assert_eq!(weight_row3.len(), IN_DIM);

    let mut acc0 = x86_arch::_mm256_setzero_ps();
    let mut acc1 = x86_arch::_mm256_setzero_ps();
    let mut acc2 = x86_arch::_mm256_setzero_ps();
    let mut acc3 = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index < IN_DIM {
        let xs = x86_arch::_mm256_loadu_ps(xs_row.as_ptr().add(index));
        let weight0 = x86_arch::_mm256_loadu_ps(weight_row0.as_ptr().add(index));
        let weight1 = x86_arch::_mm256_loadu_ps(weight_row1.as_ptr().add(index));
        let weight2 = x86_arch::_mm256_loadu_ps(weight_row2.as_ptr().add(index));
        let weight3 = x86_arch::_mm256_loadu_ps(weight_row3.as_ptr().add(index));
        acc0 = x86_arch::_mm256_fmadd_ps(xs, weight0, acc0);
        acc1 = x86_arch::_mm256_fmadd_ps(xs, weight1, acc1);
        acc2 = x86_arch::_mm256_fmadd_ps(xs, weight2, acc2);
        acc3 = x86_arch::_mm256_fmadd_ps(xs, weight3, acc3);
        index += 8;
    }

    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    [
        horizontal_sum_avx2_linear!(use_direct_sum, acc0),
        horizontal_sum_avx2_linear!(use_direct_sum, acc1),
        horizontal_sum_avx2_linear!(use_direct_sum, acc2),
        horizontal_sum_avx2_linear!(use_direct_sum, acc3),
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot4_same_x_avx2_fma(
    xs_row: &[f32],
    weight_row0: &[f32],
    weight_row1: &[f32],
    weight_row2: &[f32],
    weight_row3: &[f32],
) -> [f32; 4] {
    debug_assert_eq!(xs_row.len(), weight_row0.len());
    debug_assert_eq!(xs_row.len(), weight_row1.len());
    debug_assert_eq!(xs_row.len(), weight_row2.len());
    debug_assert_eq!(xs_row.len(), weight_row3.len());
    let mut acc0 = x86_arch::_mm256_setzero_ps();
    let mut acc1 = x86_arch::_mm256_setzero_ps();
    let mut acc2 = x86_arch::_mm256_setzero_ps();
    let mut acc3 = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= xs_row.len() {
        let xs = x86_arch::_mm256_loadu_ps(xs_row.as_ptr().add(index));
        let weight0 = x86_arch::_mm256_loadu_ps(weight_row0.as_ptr().add(index));
        let weight1 = x86_arch::_mm256_loadu_ps(weight_row1.as_ptr().add(index));
        let weight2 = x86_arch::_mm256_loadu_ps(weight_row2.as_ptr().add(index));
        let weight3 = x86_arch::_mm256_loadu_ps(weight_row3.as_ptr().add(index));
        acc0 = x86_arch::_mm256_fmadd_ps(xs, weight0, acc0);
        acc1 = x86_arch::_mm256_fmadd_ps(xs, weight1, acc1);
        acc2 = x86_arch::_mm256_fmadd_ps(xs, weight2, acc2);
        acc3 = x86_arch::_mm256_fmadd_ps(xs, weight3, acc3);
        index += 8;
    }

    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut sum0 = horizontal_sum_avx2_linear!(use_direct_sum, acc0);
    let mut sum1 = horizontal_sum_avx2_linear!(use_direct_sum, acc1);
    let mut sum2 = horizontal_sum_avx2_linear!(use_direct_sum, acc2);
    let mut sum3 = horizontal_sum_avx2_linear!(use_direct_sum, acc3);
    while index < xs_row.len() {
        let x = xs_row[index];
        sum0 += x * weight_row0[index];
        sum1 += x * weight_row1[index];
        sum2 += x * weight_row2[index];
        sum3 += x * weight_row3[index];
        index += 1;
    }
    [sum0, sum1, sum2, sum3]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot12_same_x_avx2_fma_fixed<const IN_DIM: usize>(
    xs_row: &[f32],
    weight_rows: [&[f32]; 12],
) -> [f32; 12] {
    debug_assert_eq!(IN_DIM % 8, 0);
    debug_assert_eq!(xs_row.len(), IN_DIM);
    debug_assert!(weight_rows.iter().all(|weight| weight.len() == IN_DIM));

    let mut acc0 = x86_arch::_mm256_setzero_ps();
    let mut acc1 = x86_arch::_mm256_setzero_ps();
    let mut acc2 = x86_arch::_mm256_setzero_ps();
    let mut acc3 = x86_arch::_mm256_setzero_ps();
    let mut acc4 = x86_arch::_mm256_setzero_ps();
    let mut acc5 = x86_arch::_mm256_setzero_ps();
    let mut acc6 = x86_arch::_mm256_setzero_ps();
    let mut acc7 = x86_arch::_mm256_setzero_ps();
    let mut acc8 = x86_arch::_mm256_setzero_ps();
    let mut acc9 = x86_arch::_mm256_setzero_ps();
    let mut acc10 = x86_arch::_mm256_setzero_ps();
    let mut acc11 = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index < IN_DIM {
        let xs = x86_arch::_mm256_loadu_ps(xs_row.as_ptr().add(index));
        let weight0 = x86_arch::_mm256_loadu_ps(weight_rows[0].as_ptr().add(index));
        let weight1 = x86_arch::_mm256_loadu_ps(weight_rows[1].as_ptr().add(index));
        let weight2 = x86_arch::_mm256_loadu_ps(weight_rows[2].as_ptr().add(index));
        let weight3 = x86_arch::_mm256_loadu_ps(weight_rows[3].as_ptr().add(index));
        let weight4 = x86_arch::_mm256_loadu_ps(weight_rows[4].as_ptr().add(index));
        let weight5 = x86_arch::_mm256_loadu_ps(weight_rows[5].as_ptr().add(index));
        let weight6 = x86_arch::_mm256_loadu_ps(weight_rows[6].as_ptr().add(index));
        let weight7 = x86_arch::_mm256_loadu_ps(weight_rows[7].as_ptr().add(index));
        let weight8 = x86_arch::_mm256_loadu_ps(weight_rows[8].as_ptr().add(index));
        let weight9 = x86_arch::_mm256_loadu_ps(weight_rows[9].as_ptr().add(index));
        let weight10 = x86_arch::_mm256_loadu_ps(weight_rows[10].as_ptr().add(index));
        let weight11 = x86_arch::_mm256_loadu_ps(weight_rows[11].as_ptr().add(index));
        acc0 = x86_arch::_mm256_fmadd_ps(xs, weight0, acc0);
        acc1 = x86_arch::_mm256_fmadd_ps(xs, weight1, acc1);
        acc2 = x86_arch::_mm256_fmadd_ps(xs, weight2, acc2);
        acc3 = x86_arch::_mm256_fmadd_ps(xs, weight3, acc3);
        acc4 = x86_arch::_mm256_fmadd_ps(xs, weight4, acc4);
        acc5 = x86_arch::_mm256_fmadd_ps(xs, weight5, acc5);
        acc6 = x86_arch::_mm256_fmadd_ps(xs, weight6, acc6);
        acc7 = x86_arch::_mm256_fmadd_ps(xs, weight7, acc7);
        acc8 = x86_arch::_mm256_fmadd_ps(xs, weight8, acc8);
        acc9 = x86_arch::_mm256_fmadd_ps(xs, weight9, acc9);
        acc10 = x86_arch::_mm256_fmadd_ps(xs, weight10, acc10);
        acc11 = x86_arch::_mm256_fmadd_ps(xs, weight11, acc11);
        index += 8;
    }

    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    [
        horizontal_sum_avx2_linear!(use_direct_sum, acc0),
        horizontal_sum_avx2_linear!(use_direct_sum, acc1),
        horizontal_sum_avx2_linear!(use_direct_sum, acc2),
        horizontal_sum_avx2_linear!(use_direct_sum, acc3),
        horizontal_sum_avx2_linear!(use_direct_sum, acc4),
        horizontal_sum_avx2_linear!(use_direct_sum, acc5),
        horizontal_sum_avx2_linear!(use_direct_sum, acc6),
        horizontal_sum_avx2_linear!(use_direct_sum, acc7),
        horizontal_sum_avx2_linear!(use_direct_sum, acc8),
        horizontal_sum_avx2_linear!(use_direct_sum, acc9),
        horizontal_sum_avx2_linear!(use_direct_sum, acc10),
        horizontal_sum_avx2_linear!(use_direct_sum, acc11),
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot12_same_x_avx2_fma_fixed_ptr<const IN_DIM: usize>(
    xs_row: &[f32],
    weight_ptr: *const f32,
) -> [f32; 12] {
    debug_assert_eq!(IN_DIM % 8, 0);
    debug_assert_eq!(xs_row.len(), IN_DIM);

    let mut acc0 = x86_arch::_mm256_setzero_ps();
    let mut acc1 = x86_arch::_mm256_setzero_ps();
    let mut acc2 = x86_arch::_mm256_setzero_ps();
    let mut acc3 = x86_arch::_mm256_setzero_ps();
    let mut acc4 = x86_arch::_mm256_setzero_ps();
    let mut acc5 = x86_arch::_mm256_setzero_ps();
    let mut acc6 = x86_arch::_mm256_setzero_ps();
    let mut acc7 = x86_arch::_mm256_setzero_ps();
    let mut acc8 = x86_arch::_mm256_setzero_ps();
    let mut acc9 = x86_arch::_mm256_setzero_ps();
    let mut acc10 = x86_arch::_mm256_setzero_ps();
    let mut acc11 = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index < IN_DIM {
        let xs = x86_arch::_mm256_loadu_ps(xs_row.as_ptr().add(index));
        let weight0 = x86_arch::_mm256_loadu_ps(weight_ptr.add(index));
        let weight1 = x86_arch::_mm256_loadu_ps(weight_ptr.add(IN_DIM + index));
        let weight2 = x86_arch::_mm256_loadu_ps(weight_ptr.add(2 * IN_DIM + index));
        let weight3 = x86_arch::_mm256_loadu_ps(weight_ptr.add(3 * IN_DIM + index));
        let weight4 = x86_arch::_mm256_loadu_ps(weight_ptr.add(4 * IN_DIM + index));
        let weight5 = x86_arch::_mm256_loadu_ps(weight_ptr.add(5 * IN_DIM + index));
        let weight6 = x86_arch::_mm256_loadu_ps(weight_ptr.add(6 * IN_DIM + index));
        let weight7 = x86_arch::_mm256_loadu_ps(weight_ptr.add(7 * IN_DIM + index));
        let weight8 = x86_arch::_mm256_loadu_ps(weight_ptr.add(8 * IN_DIM + index));
        let weight9 = x86_arch::_mm256_loadu_ps(weight_ptr.add(9 * IN_DIM + index));
        let weight10 = x86_arch::_mm256_loadu_ps(weight_ptr.add(10 * IN_DIM + index));
        let weight11 = x86_arch::_mm256_loadu_ps(weight_ptr.add(11 * IN_DIM + index));
        acc0 = x86_arch::_mm256_fmadd_ps(xs, weight0, acc0);
        acc1 = x86_arch::_mm256_fmadd_ps(xs, weight1, acc1);
        acc2 = x86_arch::_mm256_fmadd_ps(xs, weight2, acc2);
        acc3 = x86_arch::_mm256_fmadd_ps(xs, weight3, acc3);
        acc4 = x86_arch::_mm256_fmadd_ps(xs, weight4, acc4);
        acc5 = x86_arch::_mm256_fmadd_ps(xs, weight5, acc5);
        acc6 = x86_arch::_mm256_fmadd_ps(xs, weight6, acc6);
        acc7 = x86_arch::_mm256_fmadd_ps(xs, weight7, acc7);
        acc8 = x86_arch::_mm256_fmadd_ps(xs, weight8, acc8);
        acc9 = x86_arch::_mm256_fmadd_ps(xs, weight9, acc9);
        acc10 = x86_arch::_mm256_fmadd_ps(xs, weight10, acc10);
        acc11 = x86_arch::_mm256_fmadd_ps(xs, weight11, acc11);
        index += 8;
    }

    if native_linear_dot12_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled() {
        return [
            horizontal_sum_avx2_register(acc0),
            horizontal_sum_avx2_register(acc1),
            horizontal_sum_avx2_register(acc2),
            horizontal_sum_avx2_register(acc3),
            horizontal_sum_avx2_register(acc4),
            horizontal_sum_avx2_register(acc5),
            horizontal_sum_avx2_register(acc6),
            horizontal_sum_avx2_register(acc7),
            horizontal_sum_avx2_register(acc8),
            horizontal_sum_avx2_register(acc9),
            horizontal_sum_avx2_register(acc10),
            horizontal_sum_avx2_register(acc11),
        ];
    }

    [
        horizontal_sum_avx2(acc0),
        horizontal_sum_avx2(acc1),
        horizontal_sum_avx2(acc2),
        horizontal_sum_avx2(acc3),
        horizontal_sum_avx2(acc4),
        horizontal_sum_avx2(acc5),
        horizontal_sum_avx2(acc6),
        horizontal_sum_avx2(acc7),
        horizontal_sum_avx2(acc8),
        horizontal_sum_avx2(acc9),
        horizontal_sum_avx2(acc10),
        horizontal_sum_avx2(acc11),
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot12_same_x_avx2_fma(
    xs_row: &[f32],
    weight_rows: [&[f32]; 12],
) -> [f32; 12] {
    debug_assert!(weight_rows
        .iter()
        .all(|weight| weight.len() == xs_row.len()));
    let mut acc0 = x86_arch::_mm256_setzero_ps();
    let mut acc1 = x86_arch::_mm256_setzero_ps();
    let mut acc2 = x86_arch::_mm256_setzero_ps();
    let mut acc3 = x86_arch::_mm256_setzero_ps();
    let mut acc4 = x86_arch::_mm256_setzero_ps();
    let mut acc5 = x86_arch::_mm256_setzero_ps();
    let mut acc6 = x86_arch::_mm256_setzero_ps();
    let mut acc7 = x86_arch::_mm256_setzero_ps();
    let mut acc8 = x86_arch::_mm256_setzero_ps();
    let mut acc9 = x86_arch::_mm256_setzero_ps();
    let mut acc10 = x86_arch::_mm256_setzero_ps();
    let mut acc11 = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= xs_row.len() {
        let xs = x86_arch::_mm256_loadu_ps(xs_row.as_ptr().add(index));
        let weight0 = x86_arch::_mm256_loadu_ps(weight_rows[0].as_ptr().add(index));
        let weight1 = x86_arch::_mm256_loadu_ps(weight_rows[1].as_ptr().add(index));
        let weight2 = x86_arch::_mm256_loadu_ps(weight_rows[2].as_ptr().add(index));
        let weight3 = x86_arch::_mm256_loadu_ps(weight_rows[3].as_ptr().add(index));
        let weight4 = x86_arch::_mm256_loadu_ps(weight_rows[4].as_ptr().add(index));
        let weight5 = x86_arch::_mm256_loadu_ps(weight_rows[5].as_ptr().add(index));
        let weight6 = x86_arch::_mm256_loadu_ps(weight_rows[6].as_ptr().add(index));
        let weight7 = x86_arch::_mm256_loadu_ps(weight_rows[7].as_ptr().add(index));
        let weight8 = x86_arch::_mm256_loadu_ps(weight_rows[8].as_ptr().add(index));
        let weight9 = x86_arch::_mm256_loadu_ps(weight_rows[9].as_ptr().add(index));
        let weight10 = x86_arch::_mm256_loadu_ps(weight_rows[10].as_ptr().add(index));
        let weight11 = x86_arch::_mm256_loadu_ps(weight_rows[11].as_ptr().add(index));
        acc0 = x86_arch::_mm256_fmadd_ps(xs, weight0, acc0);
        acc1 = x86_arch::_mm256_fmadd_ps(xs, weight1, acc1);
        acc2 = x86_arch::_mm256_fmadd_ps(xs, weight2, acc2);
        acc3 = x86_arch::_mm256_fmadd_ps(xs, weight3, acc3);
        acc4 = x86_arch::_mm256_fmadd_ps(xs, weight4, acc4);
        acc5 = x86_arch::_mm256_fmadd_ps(xs, weight5, acc5);
        acc6 = x86_arch::_mm256_fmadd_ps(xs, weight6, acc6);
        acc7 = x86_arch::_mm256_fmadd_ps(xs, weight7, acc7);
        acc8 = x86_arch::_mm256_fmadd_ps(xs, weight8, acc8);
        acc9 = x86_arch::_mm256_fmadd_ps(xs, weight9, acc9);
        acc10 = x86_arch::_mm256_fmadd_ps(xs, weight10, acc10);
        acc11 = x86_arch::_mm256_fmadd_ps(xs, weight11, acc11);
        index += 8;
    }

    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut sum0 = horizontal_sum_avx2_linear!(use_direct_sum, acc0);
    let mut sum1 = horizontal_sum_avx2_linear!(use_direct_sum, acc1);
    let mut sum2 = horizontal_sum_avx2_linear!(use_direct_sum, acc2);
    let mut sum3 = horizontal_sum_avx2_linear!(use_direct_sum, acc3);
    let mut sum4 = horizontal_sum_avx2_linear!(use_direct_sum, acc4);
    let mut sum5 = horizontal_sum_avx2_linear!(use_direct_sum, acc5);
    let mut sum6 = horizontal_sum_avx2_linear!(use_direct_sum, acc6);
    let mut sum7 = horizontal_sum_avx2_linear!(use_direct_sum, acc7);
    let mut sum8 = horizontal_sum_avx2_linear!(use_direct_sum, acc8);
    let mut sum9 = horizontal_sum_avx2_linear!(use_direct_sum, acc9);
    let mut sum10 = horizontal_sum_avx2_linear!(use_direct_sum, acc10);
    let mut sum11 = horizontal_sum_avx2_linear!(use_direct_sum, acc11);
    while index < xs_row.len() {
        let x = xs_row[index];
        sum0 += x * weight_rows[0][index];
        sum1 += x * weight_rows[1][index];
        sum2 += x * weight_rows[2][index];
        sum3 += x * weight_rows[3][index];
        sum4 += x * weight_rows[4][index];
        sum5 += x * weight_rows[5][index];
        sum6 += x * weight_rows[6][index];
        sum7 += x * weight_rows[7][index];
        sum8 += x * weight_rows[8][index];
        sum9 += x * weight_rows[9][index];
        sum10 += x * weight_rows[10][index];
        sum11 += x * weight_rows[11][index];
        index += 1;
    }
    [
        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11,
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot12_same_x_avx2_fma_ptr(
    xs_row: &[f32],
    weight_ptr: *const f32,
    in_dim: usize,
) -> [f32; 12] {
    let mut acc0 = x86_arch::_mm256_setzero_ps();
    let mut acc1 = x86_arch::_mm256_setzero_ps();
    let mut acc2 = x86_arch::_mm256_setzero_ps();
    let mut acc3 = x86_arch::_mm256_setzero_ps();
    let mut acc4 = x86_arch::_mm256_setzero_ps();
    let mut acc5 = x86_arch::_mm256_setzero_ps();
    let mut acc6 = x86_arch::_mm256_setzero_ps();
    let mut acc7 = x86_arch::_mm256_setzero_ps();
    let mut acc8 = x86_arch::_mm256_setzero_ps();
    let mut acc9 = x86_arch::_mm256_setzero_ps();
    let mut acc10 = x86_arch::_mm256_setzero_ps();
    let mut acc11 = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= in_dim {
        let xs = x86_arch::_mm256_loadu_ps(xs_row.as_ptr().add(index));
        let weight0 = x86_arch::_mm256_loadu_ps(weight_ptr.add(index));
        let weight1 = x86_arch::_mm256_loadu_ps(weight_ptr.add(in_dim + index));
        let weight2 = x86_arch::_mm256_loadu_ps(weight_ptr.add(2 * in_dim + index));
        let weight3 = x86_arch::_mm256_loadu_ps(weight_ptr.add(3 * in_dim + index));
        let weight4 = x86_arch::_mm256_loadu_ps(weight_ptr.add(4 * in_dim + index));
        let weight5 = x86_arch::_mm256_loadu_ps(weight_ptr.add(5 * in_dim + index));
        let weight6 = x86_arch::_mm256_loadu_ps(weight_ptr.add(6 * in_dim + index));
        let weight7 = x86_arch::_mm256_loadu_ps(weight_ptr.add(7 * in_dim + index));
        let weight8 = x86_arch::_mm256_loadu_ps(weight_ptr.add(8 * in_dim + index));
        let weight9 = x86_arch::_mm256_loadu_ps(weight_ptr.add(9 * in_dim + index));
        let weight10 = x86_arch::_mm256_loadu_ps(weight_ptr.add(10 * in_dim + index));
        let weight11 = x86_arch::_mm256_loadu_ps(weight_ptr.add(11 * in_dim + index));
        acc0 = x86_arch::_mm256_fmadd_ps(xs, weight0, acc0);
        acc1 = x86_arch::_mm256_fmadd_ps(xs, weight1, acc1);
        acc2 = x86_arch::_mm256_fmadd_ps(xs, weight2, acc2);
        acc3 = x86_arch::_mm256_fmadd_ps(xs, weight3, acc3);
        acc4 = x86_arch::_mm256_fmadd_ps(xs, weight4, acc4);
        acc5 = x86_arch::_mm256_fmadd_ps(xs, weight5, acc5);
        acc6 = x86_arch::_mm256_fmadd_ps(xs, weight6, acc6);
        acc7 = x86_arch::_mm256_fmadd_ps(xs, weight7, acc7);
        acc8 = x86_arch::_mm256_fmadd_ps(xs, weight8, acc8);
        acc9 = x86_arch::_mm256_fmadd_ps(xs, weight9, acc9);
        acc10 = x86_arch::_mm256_fmadd_ps(xs, weight10, acc10);
        acc11 = x86_arch::_mm256_fmadd_ps(xs, weight11, acc11);
        index += 8;
    }

    let use_direct_sum =
        native_linear_dot12_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut sum0 = if use_direct_sum {
        horizontal_sum_avx2_register(acc0)
    } else {
        horizontal_sum_avx2(acc0)
    };
    let mut sum1 = if use_direct_sum {
        horizontal_sum_avx2_register(acc1)
    } else {
        horizontal_sum_avx2(acc1)
    };
    let mut sum2 = if use_direct_sum {
        horizontal_sum_avx2_register(acc2)
    } else {
        horizontal_sum_avx2(acc2)
    };
    let mut sum3 = if use_direct_sum {
        horizontal_sum_avx2_register(acc3)
    } else {
        horizontal_sum_avx2(acc3)
    };
    let mut sum4 = if use_direct_sum {
        horizontal_sum_avx2_register(acc4)
    } else {
        horizontal_sum_avx2(acc4)
    };
    let mut sum5 = if use_direct_sum {
        horizontal_sum_avx2_register(acc5)
    } else {
        horizontal_sum_avx2(acc5)
    };
    let mut sum6 = if use_direct_sum {
        horizontal_sum_avx2_register(acc6)
    } else {
        horizontal_sum_avx2(acc6)
    };
    let mut sum7 = if use_direct_sum {
        horizontal_sum_avx2_register(acc7)
    } else {
        horizontal_sum_avx2(acc7)
    };
    let mut sum8 = if use_direct_sum {
        horizontal_sum_avx2_register(acc8)
    } else {
        horizontal_sum_avx2(acc8)
    };
    let mut sum9 = if use_direct_sum {
        horizontal_sum_avx2_register(acc9)
    } else {
        horizontal_sum_avx2(acc9)
    };
    let mut sum10 = if use_direct_sum {
        horizontal_sum_avx2_register(acc10)
    } else {
        horizontal_sum_avx2(acc10)
    };
    let mut sum11 = if use_direct_sum {
        horizontal_sum_avx2_register(acc11)
    } else {
        horizontal_sum_avx2(acc11)
    };
    while index < in_dim {
        let x = xs_row[index];
        sum0 += x * *weight_ptr.add(index);
        sum1 += x * *weight_ptr.add(in_dim + index);
        sum2 += x * *weight_ptr.add(2 * in_dim + index);
        sum3 += x * *weight_ptr.add(3 * in_dim + index);
        sum4 += x * *weight_ptr.add(4 * in_dim + index);
        sum5 += x * *weight_ptr.add(5 * in_dim + index);
        sum6 += x * *weight_ptr.add(6 * in_dim + index);
        sum7 += x * *weight_ptr.add(7 * in_dim + index);
        sum8 += x * *weight_ptr.add(8 * in_dim + index);
        sum9 += x * *weight_ptr.add(9 * in_dim + index);
        sum10 += x * *weight_ptr.add(10 * in_dim + index);
        sum11 += x * *weight_ptr.add(11 * in_dim + index);
        index += 1;
    }
    [
        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11,
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot8_same_x_avx2_fma_fixed<const IN_DIM: usize>(
    xs_row: &[f32],
    weight_rows: [&[f32]; 8],
) -> [f32; 8] {
    debug_assert_eq!(IN_DIM % 8, 0);
    debug_assert_eq!(xs_row.len(), IN_DIM);
    debug_assert!(weight_rows.iter().all(|weight| weight.len() == IN_DIM));

    let mut acc0 = x86_arch::_mm256_setzero_ps();
    let mut acc1 = x86_arch::_mm256_setzero_ps();
    let mut acc2 = x86_arch::_mm256_setzero_ps();
    let mut acc3 = x86_arch::_mm256_setzero_ps();
    let mut acc4 = x86_arch::_mm256_setzero_ps();
    let mut acc5 = x86_arch::_mm256_setzero_ps();
    let mut acc6 = x86_arch::_mm256_setzero_ps();
    let mut acc7 = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index < IN_DIM {
        let xs = x86_arch::_mm256_loadu_ps(xs_row.as_ptr().add(index));
        let weight0 = x86_arch::_mm256_loadu_ps(weight_rows[0].as_ptr().add(index));
        let weight1 = x86_arch::_mm256_loadu_ps(weight_rows[1].as_ptr().add(index));
        let weight2 = x86_arch::_mm256_loadu_ps(weight_rows[2].as_ptr().add(index));
        let weight3 = x86_arch::_mm256_loadu_ps(weight_rows[3].as_ptr().add(index));
        let weight4 = x86_arch::_mm256_loadu_ps(weight_rows[4].as_ptr().add(index));
        let weight5 = x86_arch::_mm256_loadu_ps(weight_rows[5].as_ptr().add(index));
        let weight6 = x86_arch::_mm256_loadu_ps(weight_rows[6].as_ptr().add(index));
        let weight7 = x86_arch::_mm256_loadu_ps(weight_rows[7].as_ptr().add(index));
        acc0 = x86_arch::_mm256_fmadd_ps(xs, weight0, acc0);
        acc1 = x86_arch::_mm256_fmadd_ps(xs, weight1, acc1);
        acc2 = x86_arch::_mm256_fmadd_ps(xs, weight2, acc2);
        acc3 = x86_arch::_mm256_fmadd_ps(xs, weight3, acc3);
        acc4 = x86_arch::_mm256_fmadd_ps(xs, weight4, acc4);
        acc5 = x86_arch::_mm256_fmadd_ps(xs, weight5, acc5);
        acc6 = x86_arch::_mm256_fmadd_ps(xs, weight6, acc6);
        acc7 = x86_arch::_mm256_fmadd_ps(xs, weight7, acc7);
        index += 8;
    }

    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    [
        horizontal_sum_avx2_linear!(use_direct_sum, acc0),
        horizontal_sum_avx2_linear!(use_direct_sum, acc1),
        horizontal_sum_avx2_linear!(use_direct_sum, acc2),
        horizontal_sum_avx2_linear!(use_direct_sum, acc3),
        horizontal_sum_avx2_linear!(use_direct_sum, acc4),
        horizontal_sum_avx2_linear!(use_direct_sum, acc5),
        horizontal_sum_avx2_linear!(use_direct_sum, acc6),
        horizontal_sum_avx2_linear!(use_direct_sum, acc7),
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot8_same_x_avx2_fma(
    xs_row: &[f32],
    weight_rows: [&[f32]; 8],
) -> [f32; 8] {
    debug_assert!(weight_rows
        .iter()
        .all(|weight| weight.len() == xs_row.len()));
    let mut acc0 = x86_arch::_mm256_setzero_ps();
    let mut acc1 = x86_arch::_mm256_setzero_ps();
    let mut acc2 = x86_arch::_mm256_setzero_ps();
    let mut acc3 = x86_arch::_mm256_setzero_ps();
    let mut acc4 = x86_arch::_mm256_setzero_ps();
    let mut acc5 = x86_arch::_mm256_setzero_ps();
    let mut acc6 = x86_arch::_mm256_setzero_ps();
    let mut acc7 = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= xs_row.len() {
        let xs = x86_arch::_mm256_loadu_ps(xs_row.as_ptr().add(index));
        let weight0 = x86_arch::_mm256_loadu_ps(weight_rows[0].as_ptr().add(index));
        let weight1 = x86_arch::_mm256_loadu_ps(weight_rows[1].as_ptr().add(index));
        let weight2 = x86_arch::_mm256_loadu_ps(weight_rows[2].as_ptr().add(index));
        let weight3 = x86_arch::_mm256_loadu_ps(weight_rows[3].as_ptr().add(index));
        let weight4 = x86_arch::_mm256_loadu_ps(weight_rows[4].as_ptr().add(index));
        let weight5 = x86_arch::_mm256_loadu_ps(weight_rows[5].as_ptr().add(index));
        let weight6 = x86_arch::_mm256_loadu_ps(weight_rows[6].as_ptr().add(index));
        let weight7 = x86_arch::_mm256_loadu_ps(weight_rows[7].as_ptr().add(index));
        acc0 = x86_arch::_mm256_fmadd_ps(xs, weight0, acc0);
        acc1 = x86_arch::_mm256_fmadd_ps(xs, weight1, acc1);
        acc2 = x86_arch::_mm256_fmadd_ps(xs, weight2, acc2);
        acc3 = x86_arch::_mm256_fmadd_ps(xs, weight3, acc3);
        acc4 = x86_arch::_mm256_fmadd_ps(xs, weight4, acc4);
        acc5 = x86_arch::_mm256_fmadd_ps(xs, weight5, acc5);
        acc6 = x86_arch::_mm256_fmadd_ps(xs, weight6, acc6);
        acc7 = x86_arch::_mm256_fmadd_ps(xs, weight7, acc7);
        index += 8;
    }

    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut sum0 = horizontal_sum_avx2_linear!(use_direct_sum, acc0);
    let mut sum1 = horizontal_sum_avx2_linear!(use_direct_sum, acc1);
    let mut sum2 = horizontal_sum_avx2_linear!(use_direct_sum, acc2);
    let mut sum3 = horizontal_sum_avx2_linear!(use_direct_sum, acc3);
    let mut sum4 = horizontal_sum_avx2_linear!(use_direct_sum, acc4);
    let mut sum5 = horizontal_sum_avx2_linear!(use_direct_sum, acc5);
    let mut sum6 = horizontal_sum_avx2_linear!(use_direct_sum, acc6);
    let mut sum7 = horizontal_sum_avx2_linear!(use_direct_sum, acc7);
    while index < xs_row.len() {
        let x = xs_row[index];
        sum0 += x * weight_rows[0][index];
        sum1 += x * weight_rows[1][index];
        sum2 += x * weight_rows[2][index];
        sum3 += x * weight_rows[3][index];
        sum4 += x * weight_rows[4][index];
        sum5 += x * weight_rows[5][index];
        sum6 += x * weight_rows[6][index];
        sum7 += x * weight_rows[7][index];
        index += 1;
    }
    [sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn linear_dot_pair_avx2_fma(
    left_xs_row: &[f32],
    left_weight_row: &[f32],
    right_xs_row: &[f32],
    right_weight_row: &[f32],
) -> (f32, f32) {
    debug_assert_eq!(left_xs_row.len(), left_weight_row.len());
    debug_assert_eq!(right_xs_row.len(), right_weight_row.len());
    debug_assert_eq!(left_xs_row.len(), right_xs_row.len());
    let mut left_acc = x86_arch::_mm256_setzero_ps();
    let mut right_acc = x86_arch::_mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= left_xs_row.len() {
        let left_xs = x86_arch::_mm256_loadu_ps(left_xs_row.as_ptr().add(index));
        let left_weight = x86_arch::_mm256_loadu_ps(left_weight_row.as_ptr().add(index));
        left_acc = x86_arch::_mm256_fmadd_ps(left_xs, left_weight, left_acc);

        let right_xs = x86_arch::_mm256_loadu_ps(right_xs_row.as_ptr().add(index));
        let right_weight = x86_arch::_mm256_loadu_ps(right_weight_row.as_ptr().add(index));
        right_acc = x86_arch::_mm256_fmadd_ps(right_xs, right_weight, right_acc);
        index += 8;
    }

    let use_direct_sum =
        native_linear_shared_direct_horizontal_sum_enabled() && fast_avx2_horizontal_sum_enabled();
    let mut left_sum = horizontal_sum_avx2_linear!(use_direct_sum, left_acc);
    let mut right_sum = horizontal_sum_avx2_linear!(use_direct_sum, right_acc);
    while index < left_xs_row.len() {
        left_sum += left_xs_row[index] * left_weight_row[index];
        right_sum += right_xs_row[index] * right_weight_row[index];
        index += 1;
    }
    (left_sum, right_sum)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn horizontal_sum_avx2(values: x86_arch::__m256) -> f32 {
    if fast_avx2_horizontal_sum_enabled() {
        return horizontal_sum_avx2_register(values);
    }
    let mut lanes = [0.0f32; 8];
    x86_arch::_mm256_storeu_ps(lanes.as_mut_ptr(), values);
    lanes.iter().copied().sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn horizontal_sum_avx2_register(values: x86_arch::__m256) -> f32 {
    let low = x86_arch::_mm256_castps256_ps128(values);
    let high = x86_arch::_mm256_extractf128_ps(values, 1);
    let sum128 = x86_arch::_mm_add_ps(low, high);
    let high64 = x86_arch::_mm_movehl_ps(sum128, sum128);
    let sum64 = x86_arch::_mm_add_ps(sum128, high64);
    let high32 = x86_arch::_mm_shuffle_ps(sum64, sum64, 0b01_01_01_01);
    x86_arch::_mm_cvtss_f32(x86_arch::_mm_add_ss(sum64, high32))
}

pub(super) fn record_native_linear(
    profile: &mut Option<&mut ForwardProfile>,
    xs: &Tensor,
    output: &Tensor,
    kernel: NativeLinearDotKernel,
) {
    if let Some(profile) = profile.as_deref_mut() {
        profile.layout.linear_native_calls += 1;
        profile.layout.linear_native_output_values += output.elem_count();
        match kernel {
            NativeLinearDotKernel::Scalar => profile.layout.linear_native_scalar_calls += 1,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            NativeLinearDotKernel::Avx2Fma => {
                profile.layout.linear_native_avx2_fma_calls += 1;
            }
            NativeLinearDotKernel::Pulp => profile.layout.linear_native_pulp_calls += 1,
        }
        match xs.rank() {
            2 => profile.layout.linear_native_rank2_calls += 1,
            3 => profile.layout.linear_native_rank3_calls += 1,
            _ => profile.layout.linear_native_other_rank_calls += 1,
        }
    }
}

pub(super) fn record_native_linear_fallback(profile: &mut Option<&mut ForwardProfile>) {
    if let Some(profile) = profile.as_deref_mut() {
        profile.layout.linear_native_fallback_calls += 1;
    }
}

pub(super) fn record_linear_layout(profile: &mut Option<&mut ForwardProfile>, xs: &Tensor) {
    if let Some(profile) = profile.as_deref_mut() {
        profile.layout.linear_calls += 1;
        if !xs.is_contiguous() {
            profile.layout.linear_non_contiguous_inputs += 1;
        }
        match xs.rank() {
            3 => {
                profile.layout.linear_rank3_calls += 1;
                if !xs.is_contiguous() {
                    profile.layout.linear_rank3_non_contiguous_inputs += 1;
                }
            }
            4 => {
                profile.layout.linear_rank4_calls += 1;
                if !xs.is_contiguous() {
                    profile.layout.linear_rank4_non_contiguous_inputs += 1;
                }
            }
            _ => {}
        }
    }
}

pub(super) fn record_reshape_layout(profile: &mut Option<&mut ForwardProfile>, xs: &Tensor) {
    if let Some(profile) = profile.as_deref_mut() {
        profile.layout.reshape_calls += 1;
        if !xs.is_contiguous() {
            profile.layout.reshape_non_contiguous_inputs += 1;
        }
    }
}
