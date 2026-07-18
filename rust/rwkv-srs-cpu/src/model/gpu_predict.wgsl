enable f16;

const ROWS_PER_GROUP: u32 = /*__ROWS_PER_GROUP__*/u;
const LANES_PER_ROW: u32 = /*__LANES_PER_ROW__*/u;
const FULL_F32_MATH: bool = /*__FULL_F32_MATH__*/;
const REDUCTION_LANES: u32 = 64u;
const SMALL_STRIDE: u32 = 128u;
const LARGE_STRIDE: u32 = 512u;
const EXTRA_STRIDE: u32 = /*__EXTRA_STRIDE__*/u;
const CHANNELS: u32 = 128u;
const HEADS: u32 = 4u;
const HEAD_SIZE: u32 = 32u;
const STATE_LAYER_STRIDE: u32 = 4352u;
const TIME_STATE_OFFSET: u32 = 128u;
const CHANNEL_STATE_OFFSET: u32 = 4224u;
const INVALID_SLOT: u32 = 0xffffffffu;
const BUF_X: u32 = 0u;
const BUF_NORM: u32 = 1u;
const BUF_TMP0: u32 = 2u;
const BUF_TMP1: u32 = 3u;
const BUF_R: u32 = 4u;
const BUF_K: u32 = 5u;
const BUF_V: u32 = 6u;
const BUF_D: u32 = 7u;
const BUF_A: u32 = 8u;
const BUF_G: u32 = 9u;
const BUF_K_DEFORMED: u32 = 10u;
const BUF_OUT: u32 = 11u;
const R_OFFSET: u32 = 128u;
const K_OFFSET: u32 = 256u;
const V_OFFSET: u32 = 384u;
const D_OFFSET: u32 = 192u;
const A_OFFSET: u32 = 320u;
const G_OFFSET: u32 = 0u;
const OUT_OFFSET: u32 = 128u;
const V0_OFFSET: u32 = 256u;
const AUX_OFFSET: u32 = 384u;

struct LayerMeta {
    ln_w: u32,
    ln_b: u32,
    lerp: u32,
    bonus: u32,
    wr: u32,
    wk: u32,
    wv: u32,
    wo: u32,
    ks_w: u32,
    ks_b: u32,
    vs_w: u32,
    vs_b: u32,
    vl_a: u32,
    vl_b: u32,
    vl_bias: u32,
    al_a: u32,
    al_b: u32,
    al_bias: u32,
    dl_a: u32,
    dl_b: u32,
    dl_bias: u32,
    gl_a: u32,
    gl_b: u32,
    gn_w: u32,
    gn_b: u32,
    cln_w: u32,
    cln_b: u32,
    clerp: u32,
    cwk: u32,
    cwv: u32,
    channel_dim: u32,
    module_index: u32,
    entity_stride: u32,
    local_layer: u32,
};

struct ModelMetadata {
    layers: array<LayerMeta, 16>,
    normalization_weights: array<f32>,
};

@group(0) @binding(0)
var<storage, read> model_weights: array</*__MODEL_VALUE_TYPE__*/>;
@group(0) @binding(1)
var<storage, read> feature_rows: array</*__FEATURE_VALUE_TYPE__*/>;
@group(0) @binding(2)
var<storage, read> query_slots: array<u32>;
@group(0) @binding(3)
var<storage, read_write> probabilities: array<f32>;
@group(0) @binding(4)
var<storage, read> card_states: array<f16>;
@group(0) @binding(5)
var<storage, read> deck_states: array<f16>;
@group(0) @binding(6)
var<storage, read> note_states: array<f16>;
@group(0) @binding(7)
var<storage, read> preset_states: array<f16>;
@group(0) @binding(8)
var<storage, read> global_states: array<f16>;
@group(0) @binding(9)
var<storage, read> model_metadata: ModelMetadata;

var<workgroup> x: array<f32, /*__SMALL_VALUES__*/>;
var<workgroup> norm_values: array<f32, /*__SMALL_VALUES__*/>;
var<workgroup> tmp0: array<f32, /*__LARGE_VALUES__*/>;
var<workgroup> tmp1: array<f32, /*__LARGE_VALUES__*/>;
// Scratch lifetime aliases, per row:
//
//   tmp0[0..128]     temporary; [128..256] r; [256..384] k; [384..512] v
//   tmp1[0..192]     temporary; [192..320] d; [320..448] a
//   extra[0..128]    g; [128..256] attention output; [256..384] module v0;
//         [384..448] reduction / head scalars
//
// tmp0/tmp1 use all 512 values only in the feature and probability heads,
// when the recurrent aliases are dead. After all lerps, the norm buffer is
// reused for the deformed key until the channel norm replaces it. This cuts
// per-row workgroup storage from 11,568 to 6,912 bytes without changing
// arithmetic.
var<workgroup> extra: array<f32, /*__EXTRA_VALUES__*/>;
var<private> small_base: u32;
var<private> large_base: u32;
var<private> extra_base: u32;
var<private> row_in_group: u32;

fn normalization_weight(offset: u32) -> f32 {
    return model_metadata.normalization_weights[offset];
}

fn state_value(module_index: u32, offset: u32) -> f16 {
    switch module_index {
        case 0u: { return card_states[offset]; }
        case 1u: { return deck_states[offset]; }
        case 2u: { return note_states[offset]; }
        case 3u: { return preset_states[offset]; }
        default: { return global_states[offset]; }
    }
}

fn scratch_value(buffer: u32, index: u32) -> f32 {
    switch buffer {
        case BUF_X: { return x[small_base + index]; }
        case BUF_NORM: { return norm_values[small_base + index]; }
        case BUF_TMP0: { return tmp0[large_base + index]; }
        case BUF_TMP1: { return tmp1[large_base + index]; }
        case BUF_R: { return tmp0[large_base + R_OFFSET + index]; }
        case BUF_K: { return tmp0[large_base + K_OFFSET + index]; }
        case BUF_V: { return tmp0[large_base + V_OFFSET + index]; }
        case BUF_D: { return tmp1[large_base + D_OFFSET + index]; }
        case BUF_A: { return tmp1[large_base + A_OFFSET + index]; }
        case BUF_G: { return extra[extra_base + G_OFFSET + index]; }
        case BUF_K_DEFORMED: { return norm_values[small_base + index]; }
        default: { return extra[extra_base + OUT_OFFSET + index]; }
    }
}

fn set_scratch_value(buffer: u32, index: u32, value: f32) {
    switch buffer {
        case BUF_X: { x[small_base + index] = value; }
        case BUF_NORM: { norm_values[small_base + index] = value; }
        case BUF_TMP0: { tmp0[large_base + index] = value; }
        case BUF_TMP1: { tmp1[large_base + index] = value; }
        case BUF_R: { tmp0[large_base + R_OFFSET + index] = value; }
        case BUF_K: { tmp0[large_base + K_OFFSET + index] = value; }
        case BUF_V: { tmp0[large_base + V_OFFSET + index] = value; }
        case BUF_D: { tmp1[large_base + D_OFFSET + index] = value; }
        case BUF_A: { tmp1[large_base + A_OFFSET + index] = value; }
        case BUF_G: { extra[extra_base + G_OFFSET + index] = value; }
        case BUF_K_DEFORMED: { norm_values[small_base + index] = value; }
        default: { extra[extra_base + OUT_OFFSET + index] = value; }
    }
}

fn scratch_f32_value(buffer: u32, index: u32) -> f32 {
    return scratch_value(buffer, index);
}

fn matrix_product(input: f32, weight: f32) -> f32 {
    if (FULL_F32_MATH) {
        return input * weight;
    }
    return f32(f16(input)) * weight;
}

fn bounded_product2(left: f32, right: f32) -> f32 {
    if (FULL_F32_MATH) {
        return left * right;
    }
    return f32(f16(left) * f16(right));
}

fn bounded_product3(first: f32, second: f32, third: f32) -> f32 {
    if (FULL_F32_MATH) {
        return first * second * third;
    }
    return f32(f16(first) * f16(second) * f16(third));
}

fn copy_values(
    input: u32,
    output: u32,
    count: u32,
    lane: u32,
) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        set_scratch_value(output, index, scratch_value(input, index));
        index += LANES_PER_ROW;
    }
    workgroupBarrier();
}

fn clear_v0(lane: u32) {
    var index = lane;
    loop {
        if (index >= CHANNELS) { break; }
        extra[extra_base + V0_OFFSET + index] = 0.0;
        index += LANES_PER_ROW;
    }
    workgroupBarrier();
}

/*__LINEAR_FUNCTIONS__*/

fn sigmoid(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

fn silu_in_place(values: u32, count: u32, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        let value = scratch_value(values, index);
        set_scratch_value(values, index, value * sigmoid(value));
        index += LANES_PER_ROW;
    }
    workgroupBarrier();
}

fn sigmoid_in_place(values: u32, count: u32, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        set_scratch_value(values, index, sigmoid(scratch_value(values, index)));
        index += LANES_PER_ROW;
    }
    workgroupBarrier();
}

fn tanh_in_place(values: u32, count: u32, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        set_scratch_value(values, index, tanh(scratch_value(values, index)));
        index += LANES_PER_ROW;
    }
    workgroupBarrier();
}

fn relu_in_place(values: u32, count: u32, square: bool, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        let value = max(scratch_value(values, index), 0.0);
        set_scratch_value(values, index, select(value, value * value, square));
        index += LANES_PER_ROW;
    }
    workgroupBarrier();
}

fn layer_norm(
    input: u32,
    output: u32,
    count: u32,
    weight_offset: u32,
    bias_offset: u32,
    lane: u32,
) {
    var sum = 0.0;
    var index = lane;
    if (lane < REDUCTION_LANES) {
        loop {
            if (index >= count) { break; }
            sum += scratch_f32_value(input, index);
            index += REDUCTION_LANES;
        }
        extra[extra_base + AUX_OFFSET + lane] = sum;
    }
    workgroupBarrier();
    var stride = REDUCTION_LANES / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            extra[extra_base + AUX_OFFSET + lane] += extra[extra_base + AUX_OFFSET + lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let mean = extra[extra_base + AUX_OFFSET] / f32(count);

    var variance_sum = 0.0;
    index = lane;
    if (lane < REDUCTION_LANES) {
        loop {
            if (index >= count) { break; }
            let centered = scratch_f32_value(input, index) - mean;
            variance_sum += centered * centered;
            index += REDUCTION_LANES;
        }
        extra[extra_base + AUX_OFFSET + lane] = variance_sum;
    }
    workgroupBarrier();
    stride = REDUCTION_LANES / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            extra[extra_base + AUX_OFFSET + lane] += extra[extra_base + AUX_OFFSET + lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let inverse_std = inverseSqrt(extra[extra_base + AUX_OFFSET] / f32(count) + 0.00001);
    index = lane;
    loop {
        if (index >= count) { break; }
        set_scratch_value(
            output,
            index,
            (scratch_f32_value(input, index) - mean) * inverse_std
                * normalization_weight(weight_offset + index)
                + f32(model_weights[bias_offset + index]),
        );
        index += LANES_PER_ROW;
    }
    workgroupBarrier();
}

fn lerp_part(
    output: u32,
    part: u32,
    layer_info: LayerMeta,
    module_index: u32,
    slot: u32,
    entity_stride: u32,
    local_layer: u32,
    lane: u32,
) {
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let current = norm_values[small_base + channel];
        var shifted = current;
        if (slot != INVALID_SLOT) {
            shifted = f32(state_value(
                module_index,
                slot * entity_stride + local_layer * STATE_LAYER_STRIDE + channel,
            ));
        }
        set_scratch_value(
            output,
            channel,
            current + (shifted - current)
                * f32(model_weights[layer_info.lerp + part * CHANNELS + channel]),
        );
        channel += LANES_PER_ROW;
    }
    workgroupBarrier();
}

fn normalize_kv(lane: u32) {
    if (lane < HEADS) {
        let base = lane * HEAD_SIZE;
        var k_sum = 0.0;
        var v_sum = 0.0;
        var index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let k = tmp0[large_base + K_OFFSET + base + index];
            let v = tmp0[large_base + V_OFFSET + base + index];
            k_sum += k * k;
            v_sum += v * v;
            index += 1u;
        }
        let k_norm = max(sqrt(k_sum), 0.000000059604645);
        let v_norm = max(sqrt(v_sum), 0.000000059604645);
        let k_scale = extra[extra_base + AUX_OFFSET + lane];
        let v_scale = extra[extra_base + AUX_OFFSET + 4u + lane];
        index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let channel = base + index;
            let k_index = large_base + K_OFFSET + channel;
            let v_index = large_base + V_OFFSET + channel;
            let a_index = large_base + A_OFFSET + channel;
            let deformed = tmp0[k_index] / k_norm * k_scale;
            let a = tmp1[a_index];
            norm_values[small_base + channel] = deformed;
            tmp0[k_index] = deformed * a;
            tmp0[v_index] = tmp0[v_index] / v_norm * v_scale;
            index += 1u;
        }
    }
    workgroupBarrier();
}

fn recurrent_output(
    module_index: u32,
    slot: u32,
    entity_stride: u32,
    local_layer: u32,
    lane: u32,
) {
    if (lane < HEADS) {
        let base = lane * HEAD_SIZE;
        var key_readout = 0.0;
        var index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let channel = base + index;
            key_readout += bounded_product2(
                tmp0[large_base + K_OFFSET + channel],
                tmp0[large_base + R_OFFSET + channel],
            );
            index += 1u;
        }
        extra[extra_base + AUX_OFFSET + lane] = key_readout;
    }
    workgroupBarrier();

    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let head = channel / HEAD_SIZE;
        let row = channel % HEAD_SIZE;
        let vector_base = head * HEAD_SIZE;
        var output = 0.0;
        var column = 0u;
        loop {
            if (column >= HEAD_SIZE) { break; }
            let vector_channel = vector_base + column;
            let query = bounded_product2(
                tmp1[large_base + D_OFFSET + vector_channel],
                tmp0[large_base + R_OFFSET + vector_channel],
            ) - extra[extra_base + AUX_OFFSET + head]
                * norm_values[small_base + vector_channel];
            if (slot != INVALID_SLOT) {
                let state_offset = slot * entity_stride
                    + local_layer * STATE_LAYER_STRIDE
                    + TIME_STATE_OFFSET
                    + head * HEAD_SIZE * HEAD_SIZE
                    + row * HEAD_SIZE
                    + column;
                output += f32(state_value(module_index, state_offset)) * query;
            }
            column += 1u;
        }
        extra[extra_base + OUT_OFFSET + channel] = output
            + tmp0[large_base + V_OFFSET + channel]
                * extra[extra_base + AUX_OFFSET + head];
        channel += LANES_PER_ROW;
    }
    workgroupBarrier();
}

fn group_norm_and_bonus(layer_info: LayerMeta, lane: u32) {
    if (lane < HEADS) {
        let base = lane * HEAD_SIZE;
        var sum = 0.0;
        var index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            sum += extra[extra_base + OUT_OFFSET + base + index];
            index += 1u;
        }
        let mean = sum / 32.0;
        var variance = 0.0;
        index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let centered = extra[extra_base + OUT_OFFSET + base + index] - mean;
            variance += centered * centered;
            index += 1u;
        }
        let inverse_std = inverseSqrt(variance / 32.0 + 0.00064);
        var bonus = 0.0;
        index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let absolute_channel = base + index;
            let channel = lane * HEAD_SIZE + index;
            let out_index = extra_base + OUT_OFFSET + absolute_channel;
            extra[out_index] = (extra[out_index] - mean) * inverse_std
                * normalization_weight(layer_info.gn_w + channel)
                + f32(model_weights[layer_info.gn_b + channel]);
            bonus += bounded_product3(
                tmp0[large_base + R_OFFSET + absolute_channel],
                f32(model_weights[layer_info.bonus + channel]),
                tmp0[large_base + K_OFFSET + absolute_channel],
            );
            index += 1u;
        }
        extra[extra_base + AUX_OFFSET + lane] = bonus;
    }
    workgroupBarrier();

    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let head = channel / HEAD_SIZE;
        let out_index = extra_base + OUT_OFFSET + channel;
        extra[out_index] = extra[extra_base + G_OFFSET + channel]
            * (extra[out_index]
                + extra[extra_base + AUX_OFFSET + head]
                    * tmp0[large_base + V_OFFSET + channel]);
        channel += LANES_PER_ROW;
    }
    workgroupBarrier();
}

fn run_layer(
    layer_info: LayerMeta,
    module_index: u32,
    slot: u32,
    entity_stride: u32,
    local_layer: u32,
    lane: u32,
) {
    layer_norm(BUF_X, BUF_NORM, CHANNELS, layer_info.ln_w, layer_info.ln_b, lane);

    lerp_part(BUF_TMP0, 0u, layer_info, module_index, slot, entity_stride, local_layer, lane);
    linear_tmp0(BUF_R, CHANNELS, CHANNELS, layer_info.wr, INVALID_SLOT, lane);
    lerp_part(BUF_TMP0, 1u, layer_info, module_index, slot, entity_stride, local_layer, lane);
    linear_tmp0(BUF_K, CHANNELS, CHANNELS, layer_info.wk, INVALID_SLOT, lane);
    lerp_part(BUF_V, 2u, layer_info, module_index, slot, entity_stride, local_layer, lane);
    lerp_part(BUF_D, 3u, layer_info, module_index, slot, entity_stride, local_layer, lane);
    lerp_part(BUF_A, 4u, layer_info, module_index, slot, entity_stride, local_layer, lane);
    lerp_part(BUF_G, 5u, layer_info, module_index, slot, entity_stride, local_layer, lane);
    lerp_part(BUF_TMP0, 6u, layer_info, module_index, slot, entity_stride, local_layer, lane);
    linear_tmp0(BUF_TMP1, CHANNELS, HEADS, layer_info.ks_w, layer_info.ks_b, lane);
    sigmoid_in_place(BUF_TMP1, HEADS, lane);
    if (lane < HEADS) {
        extra[extra_base + AUX_OFFSET + lane] = tmp1[large_base + lane];
    }
    workgroupBarrier();
    lerp_part(BUF_TMP0, 7u, layer_info, module_index, slot, entity_stride, local_layer, lane);
    linear_tmp0(BUF_TMP1, CHANNELS, HEADS, layer_info.vs_w, layer_info.vs_b, lane);
    sigmoid_in_place(BUF_TMP1, HEADS, lane);
    if (lane < HEADS) {
        extra[extra_base + AUX_OFFSET + 4u + lane] = tmp1[large_base + lane];
    }
    workgroupBarrier();

    if (local_layer == 0u) {
        linear_v(BUF_TMP0, CHANNELS, CHANNELS, layer_info.wv, INVALID_SLOT, lane);
        copy_values(BUF_TMP0, BUF_V, CHANNELS, lane);
        var channel = lane;
        loop {
            if (channel >= CHANNELS) { break; }
            extra[extra_base + V0_OFFSET + channel] =
                tmp0[large_base + V_OFFSET + channel];
            channel += LANES_PER_ROW;
        }
        workgroupBarrier();
    } else {
        linear_v(BUF_TMP0, CHANNELS, 8u, layer_info.vl_a, INVALID_SLOT, lane);
        linear_tmp0(BUF_TMP1, 8u, CHANNELS, layer_info.vl_b, layer_info.vl_bias, lane);
        sigmoid_in_place(BUF_TMP1, CHANNELS, lane);
        linear_v(BUF_OUT, CHANNELS, CHANNELS, layer_info.wv, INVALID_SLOT, lane);
        var channel = lane;
        loop {
            if (channel >= CHANNELS) { break; }
            let projected = extra[extra_base + OUT_OFFSET + channel];
            tmp0[large_base + V_OFFSET + channel] = projected
                + (extra[extra_base + V0_OFFSET + channel] - projected)
                    * tmp1[large_base + channel];
            channel += LANES_PER_ROW;
        }
        workgroupBarrier();
    }

    linear_a(BUF_TMP0, CHANNELS, 16u, layer_info.al_a, INVALID_SLOT, lane);
    linear_tmp0(BUF_TMP1, 16u, CHANNELS, layer_info.al_b, layer_info.al_bias, lane);
    sigmoid_in_place(BUF_TMP1, CHANNELS, lane);
    copy_values(BUF_TMP1, BUF_A, CHANNELS, lane);

    linear_g(BUF_TMP0, CHANNELS, 16u, layer_info.gl_a, INVALID_SLOT, lane);
    sigmoid_in_place(BUF_TMP0, 16u, lane);
    linear_tmp0(BUF_G, 16u, CHANNELS, layer_info.gl_b, INVALID_SLOT, lane);

    linear_d(BUF_TMP0, CHANNELS, 16u, layer_info.dl_a, INVALID_SLOT, lane);
    tanh_in_place(BUF_TMP0, 16u, lane);
    linear_tmp0(BUF_D, 16u, CHANNELS, layer_info.dl_b, layer_info.dl_bias, lane);
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let absolute_channel = large_base + D_OFFSET + channel;
        let value = -tmp1[absolute_channel];
        let softplus = max(value, 0.0) + log(1.0 + exp(-abs(value)));
        tmp1[absolute_channel] = exp(-exp(-0.5 - softplus));
        channel += LANES_PER_ROW;
    }
    workgroupBarrier();

    normalize_kv(lane);
    recurrent_output(module_index, slot, entity_stride, local_layer, lane);
    group_norm_and_bonus(layer_info, lane);
    linear_out(BUF_TMP0, CHANNELS, CHANNELS, layer_info.wo, INVALID_SLOT, lane);
    channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        x[small_base + channel] += tmp0[large_base + channel];
        channel += LANES_PER_ROW;
    }
    workgroupBarrier();

    layer_norm(BUF_X, BUF_NORM, CHANNELS, layer_info.cln_w, layer_info.cln_b, lane);
    channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let norm_index = small_base + channel;
        let tmp_index = large_base + channel;
        var shifted = norm_values[norm_index];
        if (slot != INVALID_SLOT) {
            shifted = f32(state_value(
                module_index,
                slot * entity_stride + local_layer * STATE_LAYER_STRIDE
                    + CHANNEL_STATE_OFFSET + channel,
            ));
        }
        tmp0[tmp_index] = norm_values[norm_index]
            + (shifted - norm_values[norm_index])
                * f32(model_weights[layer_info.clerp + channel]);
        channel += LANES_PER_ROW;
    }
    workgroupBarrier();
    linear_tmp0(BUF_TMP1, CHANNELS, layer_info.channel_dim, layer_info.cwk, INVALID_SLOT, lane);
    relu_in_place(BUF_TMP1, layer_info.channel_dim, true, lane);
    linear_tmp1(BUF_OUT, layer_info.channel_dim, CHANNELS, layer_info.cwv, INVALID_SLOT, lane);
    channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        x[small_base + channel] += extra[extra_base + OUT_OFFSET + channel];
        channel += LANES_PER_ROW;
    }
    workgroupBarrier();
}

/*__SHARED_ROW_FUNCTIONS__*/

@compute @workgroup_size(/*__WORKGROUP_SIZE__*/)
fn predict(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    // Interleave rows within each subgroup. Adjacent invocations consume the
    // same matrix weight for independent rows, allowing the GPU backend
    // backend to broadcast/cache that FP16 load instead of issuing it from
    // separate waves.
    row_in_group = local.x % ROWS_PER_GROUP;
    small_base = row_in_group * SMALL_STRIDE;
    large_base = row_in_group * LARGE_STRIDE;
    extra_base = row_in_group * EXTRA_STRIDE;
    let row = workgroup.x * ROWS_PER_GROUP + row_in_group;
    let lane = local.x / ROWS_PER_GROUP;
    var index = lane;
    loop {
        if (index >= 92u) { break; }
        x[small_base + index] = f32(feature_rows[row * 92u + index]);
        index += LANES_PER_ROW;
    }
    workgroupBarrier();

    linear_x(BUF_TMP0, 92u, 512u, /*__FEATURES_INPUT_W__*/u, /*__FEATURES_INPUT_B__*/u, lane);
    silu_in_place(BUF_TMP0, 512u, lane);
    layer_norm(BUF_TMP0, BUF_TMP1, 512u, /*__FEATURES_NORM_W__*/u, /*__FEATURES_NORM_B__*/u, lane);
    linear_tmp1(BUF_X, 512u, CHANNELS, /*__FEATURES_OUTPUT_W__*/u, /*__FEATURES_OUTPUT_B__*/u, lane);
    silu_in_place(BUF_X, CHANNELS, lane);

    let slot0 = query_slots[row * 5u];
    let slot1 = query_slots[row * 5u + 1u];
    let slot2 = query_slots[row * 5u + 2u];
    let slot3 = query_slots[row * 5u + 3u];
    let slot4 = query_slots[row * 5u + 4u];

    var layer_index = 0u;
    loop {
        if (layer_index >= 16u) { break; }
        let layer_info = model_metadata.layers[layer_index];
        if (layer_info.local_layer == 0u) {
            clear_v0(lane);
        }
        var state_slot = slot4;
        switch layer_info.module_index {
            case 0u: { state_slot = slot0; }
            case 1u: { state_slot = slot1; }
            case 2u: { state_slot = slot2; }
            case 3u: { state_slot = slot3; }
            default: {}
        }
        run_layer(
            layer_info,
            layer_info.module_index,
            state_slot,
            layer_info.entity_stride,
            layer_info.local_layer,
            lane,
        );
        layer_index += 1u;
    }

    layer_norm(BUF_X, BUF_NORM, CHANNELS, /*__PREHEAD_W__*/u, /*__PREHEAD_B__*/u, lane);
    linear_norm(BUF_TMP0, CHANNELS, 512u, /*__HEAD_P_W__*/u, /*__HEAD_P_B__*/u, lane);
    relu_in_place(BUF_TMP0, 512u, false, lane);
    linear_tmp0(BUF_TMP1, 512u, 4u, /*__P_LINEAR_W__*/u, /*__P_LINEAR_B__*/u, lane);
    if (lane == 0u) {
        var maximum = tmp1[large_base];
        maximum = max(maximum, tmp1[large_base + 1u]);
        maximum = max(maximum, tmp1[large_base + 2u]);
        maximum = max(maximum, tmp1[large_base + 3u]);
        let exp0 = exp(tmp1[large_base] - maximum);
        let exp1 = exp(tmp1[large_base + 1u] - maximum);
        let exp2 = exp(tmp1[large_base + 2u] - maximum);
        let exp3 = exp(tmp1[large_base + 3u] - maximum);
        probabilities[row] = 1.0 - exp0 / (exp0 + exp1 + exp2 + exp3);
    }
}
