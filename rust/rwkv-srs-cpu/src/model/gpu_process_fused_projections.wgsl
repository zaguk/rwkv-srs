const CHANNELS: u32 = 128u;
const HEADS: u32 = 4u;
const HEAD_SIZE: u32 = 32u;
const STATE_LAYER_STRIDE: u32 = 4352u;
const TIME_STATE_OFFSET: u32 = 128u;
const CHANNEL_STATE_OFFSET: u32 = 4224u;
const INVALID_INDEX: u32 = 0xffffffffu;
const ROW_HAS_STATE: u32 = 1u;
const ROW_LAST_PROCESS: u32 = 2u;
const PREPARE_ROWS_PER_WORKGROUP: u32 = 2u;
const CHANNEL_ROWS_PER_WORKGROUP: u32 = 4u;
const REDUCTION_LANES_PER_ROW: u32 = 64u;
const SMALL_STRIDE: u32 = 16u;
const SCALE_STRIDE: u32 = 8u;
const CHANNEL_HIDDEN_STRIDE: u32 = 512u;

const TARGET_R: u32 = 0u;
const TARGET_K: u32 = 1u;
const TARGET_V: u32 = 2u;
const TARGET_A_SCRATCH: u32 = 3u;
const TARGET_G: u32 = 4u;
const TARGET_D: u32 = 5u;
const TARGET_V_GATE: u32 = 6u;

const ACTIVATION_NONE: u32 = 0u;
const ACTIVATION_SIGMOID: u32 = 1u;
const ACTIVATION_TANH: u32 = 2u;

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

struct DispatchParams {
    row_count: u32,
    layer_index: u32,
    local_layer: u32,
    layers: u32,
    scan_offset: u32,
    scan_source_b: u32,
    transform_count: u32,
    final_chunk_count: u32,
};

struct RowMeta {
    previous_process_row: u32,
    entity: u32,
    flags: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> model_weights: array<f32>;
@group(0) @binding(1)
var<storage, read> model_metadata: ModelMetadata;
@group(0) @binding(2)
var<storage, read_write> sequence_x: array<f32>;
@group(0) @binding(3)
var<storage, read_write> time_residual_x: array<f32>;
@group(0) @binding(4)
var<storage, read_write> v0_rows: array<f32>;
@group(0) @binding(5)
var<storage, read_write> r_rows: array<f32>;
@group(0) @binding(6)
var<storage, read_write> k_rows: array<f32>;
@group(0) @binding(7)
var<storage, read_write> v_rows: array<f32>;
@group(0) @binding(8)
var<storage, read_write> w_rows: array<f32>;
@group(0) @binding(9)
var<storage, read_write> k_deformed_rows: array<f32>;
@group(0) @binding(10)
var<storage, read_write> g_rows: array<f32>;
@group(0) @binding(12)
var<storage, read> recurrent_states: array<f32>;
@group(0) @binding(13)
var<storage, read> row_metadata: array<RowMeta>;
@group(0) @binding(18)
var<storage, read_write> next_recurrent_states: array<f32>;

@group(1) @binding(0)
var<uniform> params: DispatchParams;

// TimeMixer preparation shares each projection weight load across two rows;
// ChannelMixer shares it across four. The smaller TimeMixer group avoids the
// register pressure measured with four simultaneous LoRA/projection streams.
// Intermediates that cannot be consumed immediately remain bounded to one
// workgroup; no rows x hidden storage buffer or extra projection dispatch is
// required. Keep this complete footprint synchronized with
// PROCESS_WORKGROUP_STORAGE_FLOATS in src/gpu.rs.
var<workgroup> batch_norm: array<f32, 512>;
var<workgroup> batch_shift_delta: array<f32, 512>;
var<workgroup> batch_projection_input: array<f32, 512>;
var<workgroup> batch_small: array<f32, 64>;
var<workgroup> batch_scales: array<f32, 32>;
var<workgroup> batch_channel_hidden: array<f32, 2048>;
var<workgroup> batch_reduction: array<f32, 256>;

fn sigmoid(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

fn activation(value: f32, kind: u32) -> f32 {
    if (kind == ACTIVATION_SIGMOID) {
        return sigmoid(value);
    }
    if (kind == ACTIVATION_TANH) {
        return tanh(value);
    }
    return value;
}

fn state_layer_base(entity: u32) -> u32 {
    return entity * params.layers * STATE_LAYER_STRIDE
        + params.local_layer * STATE_LAYER_STRIDE;
}

fn source_row_value(row: u32, channel: u32, channel_mixer: bool) -> f32 {
    let index = row * CHANNELS + channel;
    if (channel_mixer) {
        return time_residual_x[index];
    }
    return sequence_x[index];
}

fn reduce_rows(lane: u32) {
    var stride = REDUCTION_LANES_PER_ROW / 2u;
    let reduction_lane = lane % REDUCTION_LANES_PER_ROW;
    loop {
        if (stride == 0u) { break; }
        if (reduction_lane < stride) {
            batch_reduction[lane] += batch_reduction[lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
}

fn prepare_current_norm(
    base_row: u32,
    channel_mixer: bool,
    layer_info: LayerMeta,
    lane: u32,
) {
    let slot = lane / REDUCTION_LANES_PER_ROW;
    let reduction_lane = lane % REDUCTION_LANES_PER_ROW;
    let row = base_row + slot;
    let row_valid = row < params.row_count;
    var sum = 0.0;
    var channel = reduction_lane;
    loop {
        if (channel >= CHANNELS) { break; }
        if (row_valid) {
            sum += source_row_value(row, channel, channel_mixer);
        }
        channel += REDUCTION_LANES_PER_ROW;
    }
    batch_reduction[lane] = sum;
    workgroupBarrier();
    reduce_rows(lane);
    let mean = batch_reduction[slot * REDUCTION_LANES_PER_ROW] / f32(CHANNELS);

    var variance_sum = 0.0;
    channel = reduction_lane;
    loop {
        if (channel >= CHANNELS) { break; }
        if (row_valid) {
            let centered = source_row_value(row, channel, channel_mixer) - mean;
            variance_sum += centered * centered;
        }
        channel += REDUCTION_LANES_PER_ROW;
    }
    batch_reduction[lane] = variance_sum;
    workgroupBarrier();
    reduce_rows(lane);
    let inverse_std = inverseSqrt(
        batch_reduction[slot * REDUCTION_LANES_PER_ROW] / f32(CHANNELS) + 0.00001
    );

    let weight_offset = select(layer_info.ln_w, layer_info.cln_w, channel_mixer);
    let bias_offset = select(layer_info.ln_b, layer_info.cln_b, channel_mixer);
    channel = reduction_lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let batch_index = slot * CHANNELS + channel;
        var normalized = 0.0;
        if (row_valid) {
            normalized = (source_row_value(row, channel, channel_mixer) - mean)
                * inverse_std
                * model_metadata.normalization_weights[weight_offset + channel]
                + model_weights[bias_offset + channel];
            let row_info = row_metadata[row];
            if ((row_info.flags & ROW_LAST_PROCESS) != 0u) {
                let state_offset = select(0u, CHANNEL_STATE_OFFSET, channel_mixer);
                next_recurrent_states[
                    state_layer_base(row_info.entity) + state_offset + channel
                ] = normalized;
            }
        }
        batch_norm[batch_index] = normalized;
        channel += REDUCTION_LANES_PER_ROW;
    }
    storageBarrier();
    workgroupBarrier();
}

fn prepare_shift_delta(
    base_row: u32,
    channel_mixer: bool,
    layer_info: LayerMeta,
    lane: u32,
) {
    let slot = lane / REDUCTION_LANES_PER_ROW;
    let reduction_lane = lane % REDUCTION_LANES_PER_ROW;
    let row = base_row + slot;
    let row_valid = row < params.row_count;
    var has_previous = false;
    var previous = INVALID_INDEX;
    if (row_valid) {
        previous = row_metadata[row].previous_process_row;
        has_previous = previous != INVALID_INDEX;
    }

    var sum = 0.0;
    var channel = reduction_lane;
    loop {
        if (channel >= CHANNELS) { break; }
        if (has_previous) {
            sum += source_row_value(previous, channel, channel_mixer);
        }
        channel += REDUCTION_LANES_PER_ROW;
    }
    batch_reduction[lane] = sum;
    workgroupBarrier();
    reduce_rows(lane);
    let mean = batch_reduction[slot * REDUCTION_LANES_PER_ROW] / f32(CHANNELS);

    var variance_sum = 0.0;
    channel = reduction_lane;
    loop {
        if (channel >= CHANNELS) { break; }
        if (has_previous) {
            let centered = source_row_value(previous, channel, channel_mixer) - mean;
            variance_sum += centered * centered;
        }
        channel += REDUCTION_LANES_PER_ROW;
    }
    batch_reduction[lane] = variance_sum;
    workgroupBarrier();
    reduce_rows(lane);
    let inverse_std = inverseSqrt(
        batch_reduction[slot * REDUCTION_LANES_PER_ROW] / f32(CHANNELS) + 0.00001
    );

    let weight_offset = select(layer_info.ln_w, layer_info.cln_w, channel_mixer);
    let bias_offset = select(layer_info.ln_b, layer_info.cln_b, channel_mixer);
    channel = reduction_lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let batch_index = slot * CHANNELS + channel;
        var shifted = 0.0;
        if (row_valid) {
            let row_info = row_metadata[row];
            if (has_previous) {
                shifted = (source_row_value(previous, channel, channel_mixer) - mean)
                    * inverse_std
                    * model_metadata.normalization_weights[weight_offset + channel]
                    + model_weights[bias_offset + channel];
            } else if ((row_info.flags & ROW_HAS_STATE) != 0u) {
                let state_offset = select(0u, CHANNEL_STATE_OFFSET, channel_mixer);
                shifted = recurrent_states[
                    state_layer_base(row_info.entity) + state_offset + channel
                ];
            } else {
                shifted = batch_norm[batch_index];
            }
        }
        batch_shift_delta[batch_index] = shifted - batch_norm[batch_index];
        channel += REDUCTION_LANES_PER_ROW;
    }
    workgroupBarrier();
}

fn prepare_projection_input(
    layer_info: LayerMeta,
    part: u32,
    channel_mixer: bool,
    row_slots: u32,
    workgroup_lanes: u32,
    lane: u32,
) {
    var index = lane;
    loop {
        if (index >= row_slots * CHANNELS) { break; }
        let channel = index % CHANNELS;
        let lerp_offset = select(
            layer_info.lerp + part * CHANNELS,
            layer_info.clerp,
            channel_mixer,
        );
        batch_projection_input[index] = batch_norm[index]
            + batch_shift_delta[index] * model_weights[lerp_offset + channel];
        index += workgroup_lanes;
    }
    workgroupBarrier();
}

fn set_row_target(target_kind: u32, row: u32, output: u32, value: f32) {
    let index = row * CHANNELS + output;
    switch target_kind {
        case TARGET_R: { r_rows[index] = value; }
        case TARGET_K: { k_rows[index] = value; }
        case TARGET_V: { v_rows[index] = value; }
        case TARGET_A_SCRATCH: { w_rows[index] = value; }
        case TARGET_G: { g_rows[index] = value; }
        default: { w_rows[index] = value; }
    }
}

fn project_lerp_to_rows(
    base_row: u32,
    output_dim: u32,
    weight_offset: u32,
    bias_offset: u32,
    target_kind: u32,
    lane: u32,
) {
        var output = lane;
    loop {
        if (output >= output_dim) { break; }
        var sum0 = 0.0;
        var sum1 = 0.0;
        if (bias_offset != INVALID_INDEX) {
            let bias = model_weights[bias_offset + output];
            sum0 = bias;
            sum1 = bias;
        }
        var input = 0u;
        loop {
            if (input >= CHANNELS) { break; }
            let packed = weight_offset + input * output_dim + output * 4u;
            let weight0 = model_weights[packed];
            let weight1 = model_weights[packed + 1u];
            let weight2 = model_weights[packed + 2u];
            let weight3 = model_weights[packed + 3u];
            sum0 += batch_projection_input[input] * weight0;
            sum0 += batch_projection_input[input + 1u] * weight1;
            sum0 += batch_projection_input[input + 2u] * weight2;
            sum0 += batch_projection_input[input + 3u] * weight3;
            sum1 += batch_projection_input[CHANNELS + input] * weight0;
            sum1 += batch_projection_input[CHANNELS + input + 1u] * weight1;
            sum1 += batch_projection_input[CHANNELS + input + 2u] * weight2;
            sum1 += batch_projection_input[CHANNELS + input + 3u] * weight3;
            input += 4u;
        }
        if (base_row < params.row_count) {
            set_row_target(target_kind, base_row, output, sum0);
        }
        if (base_row + 1u < params.row_count) {
            set_row_target(target_kind, base_row + 1u, output, sum1);
        }
        output += 128u;
    }
    storageBarrier();
    workgroupBarrier();
}

fn project_lerp_to_small(
    output_dim: u32,
    weight_offset: u32,
    bias_offset: u32,
    activation_kind: u32,
    lane: u32,
) {
    let output = lane;
    if (output < output_dim) {
        var sums: array<f32, 4>;
        var slot = 0u;
        loop {
            if (slot >= PREPARE_ROWS_PER_WORKGROUP) { break; }
            sums[slot] = 0.0;
            if (bias_offset != INVALID_INDEX) {
                sums[slot] = model_weights[bias_offset + output];
            }
            slot += 1u;
        }
        var input = 0u;
        loop {
            if (input >= CHANNELS) { break; }
            let packed = weight_offset + input * output_dim + output * 4u;
            let weight0 = model_weights[packed];
            let weight1 = model_weights[packed + 1u];
            let weight2 = model_weights[packed + 2u];
            let weight3 = model_weights[packed + 3u];
            slot = 0u;
            loop {
                if (slot >= PREPARE_ROWS_PER_WORKGROUP) { break; }
                let row_base = slot * CHANNELS + input;
                sums[slot] += batch_projection_input[row_base] * weight0;
                sums[slot] += batch_projection_input[row_base + 1u] * weight1;
                sums[slot] += batch_projection_input[row_base + 2u] * weight2;
                sums[slot] += batch_projection_input[row_base + 3u] * weight3;
                slot += 1u;
            }
            input += 4u;
        }
        slot = 0u;
        loop {
            if (slot >= PREPARE_ROWS_PER_WORKGROUP) { break; }
            batch_small[slot * SMALL_STRIDE + output] = activation(sums[slot], activation_kind);
            slot += 1u;
        }
    }
    workgroupBarrier();
}

fn copy_small_to_scales(scale_offset: u32, count: u32, lane: u32) {
    if (lane < count) {
        var slot = 0u;
        loop {
            if (slot >= PREPARE_ROWS_PER_WORKGROUP) { break; }
            batch_scales[slot * SCALE_STRIDE + scale_offset + lane] =
                batch_small[slot * SMALL_STRIDE + lane];
            slot += 1u;
        }
    }
    workgroupBarrier();
}

fn project_small_to_rows(
    base_row: u32,
    input_dim: u32,
    weight_offset: u32,
    bias_offset: u32,
    target_kind: u32,
    lane: u32,
) {
    let output = lane;
    if (output < CHANNELS) {
        var sums: array<f32, 4>;
        var slot = 0u;
        loop {
            if (slot >= PREPARE_ROWS_PER_WORKGROUP) { break; }
            sums[slot] = 0.0;
            if (bias_offset != INVALID_INDEX) {
                sums[slot] = model_weights[bias_offset + output];
            }
            slot += 1u;
        }
        var input = 0u;
        loop {
            if (input >= input_dim) { break; }
            let packed = weight_offset + input * CHANNELS + output * 4u;
            let weight0 = model_weights[packed];
            let weight1 = model_weights[packed + 1u];
            let weight2 = model_weights[packed + 2u];
            let weight3 = model_weights[packed + 3u];
            slot = 0u;
            loop {
                if (slot >= PREPARE_ROWS_PER_WORKGROUP) { break; }
                let small_base = slot * SMALL_STRIDE + input;
                sums[slot] += batch_small[small_base] * weight0;
                sums[slot] += batch_small[small_base + 1u] * weight1;
                sums[slot] += batch_small[small_base + 2u] * weight2;
                sums[slot] += batch_small[small_base + 3u] * weight3;
                slot += 1u;
            }
            input += 4u;
        }
        slot = 0u;
        loop {
            if (slot >= PREPARE_ROWS_PER_WORKGROUP) { break; }
            let row = base_row + slot;
            if (row < params.row_count) {
                let row_index = row * CHANNELS + output;
                var value = sums[slot];
                if (target_kind == TARGET_V_GATE) {
                    let projected = v_rows[row_index];
                    value = projected
                        + (v0_rows[row_index] - projected) * sigmoid(value);
                    v_rows[row_index] = value;
                } else if (target_kind == TARGET_A_SCRATCH) {
                    w_rows[row_index] = sigmoid(value);
                } else if (target_kind == TARGET_G) {
                    g_rows[row_index] = value;
                } else {
                    let negated = -value;
                    let softplus = max(negated, 0.0)
                        + log(1.0 + exp(-abs(negated)));
                    w_rows[row_index] = exp(-exp(-0.5 - softplus));
                }
            }
            slot += 1u;
        }
    }
    storageBarrier();
    workgroupBarrier();
}

fn copy_v_to_v0(base_row: u32, lane: u32) {
    if (lane < CHANNELS) {
        var slot = 0u;
        loop {
            if (slot >= PREPARE_ROWS_PER_WORKGROUP) { break; }
            let row = base_row + slot;
            if (row < params.row_count) {
                let index = row * CHANNELS + lane;
                v0_rows[index] = v_rows[index];
            }
            slot += 1u;
        }
    }
    storageBarrier();
    workgroupBarrier();
}

fn normalize_kv(base_row: u32, lane: u32) {
    if (lane < PREPARE_ROWS_PER_WORKGROUP * HEADS) {
        let slot = lane / HEADS;
        let head = lane % HEADS;
        let row = base_row + slot;
        if (row < params.row_count) {
            let head_base = head * HEAD_SIZE;
            var k_sum = 0.0;
            var v_sum = 0.0;
            var index = 0u;
            loop {
                if (index >= HEAD_SIZE) { break; }
                let row_index = row * CHANNELS + head_base + index;
                let k = k_rows[row_index];
                let v = v_rows[row_index];
                k_sum += k * k;
                v_sum += v * v;
                index += 1u;
            }
            let k_norm = max(sqrt(k_sum), 0.000000059604645);
            let v_norm = max(sqrt(v_sum), 0.000000059604645);
            let k_scale = batch_scales[slot * SCALE_STRIDE + head];
            let v_scale = batch_scales[slot * SCALE_STRIDE + HEADS + head];
            index = 0u;
            loop {
                if (index >= HEAD_SIZE) { break; }
                let row_index = row * CHANNELS + head_base + index;
                let deformed = k_rows[row_index] / k_norm * k_scale;
                k_deformed_rows[row_index] = deformed;
                k_rows[row_index] = deformed * w_rows[row_index];
                v_rows[row_index] = v_rows[row_index] / v_norm * v_scale;
                index += 1u;
            }
        }
    }
    storageBarrier();
    workgroupBarrier();
}

fn project_channel_hidden(base_row: u32, layer_info: LayerMeta, lane: u32) {
    var output = lane;
    loop {
        if (output >= layer_info.channel_dim) { break; }
        var sum0 = 0.0;
        var sum1 = 0.0;
        var sum2 = 0.0;
        var sum3 = 0.0;
        var input = 0u;
        loop {
            if (input >= CHANNELS) { break; }
            let packed = layer_info.cwk
                + input * layer_info.channel_dim
                + output * 4u;
            let weight0 = model_weights[packed];
            let weight1 = model_weights[packed + 1u];
            let weight2 = model_weights[packed + 2u];
            let weight3 = model_weights[packed + 3u];
            sum0 += batch_projection_input[input] * weight0;
            sum0 += batch_projection_input[input + 1u] * weight1;
            sum0 += batch_projection_input[input + 2u] * weight2;
            sum0 += batch_projection_input[input + 3u] * weight3;
            sum1 += batch_projection_input[CHANNELS + input] * weight0;
            sum1 += batch_projection_input[CHANNELS + input + 1u] * weight1;
            sum1 += batch_projection_input[CHANNELS + input + 2u] * weight2;
            sum1 += batch_projection_input[CHANNELS + input + 3u] * weight3;
            sum2 += batch_projection_input[2u * CHANNELS + input] * weight0;
            sum2 += batch_projection_input[2u * CHANNELS + input + 1u] * weight1;
            sum2 += batch_projection_input[2u * CHANNELS + input + 2u] * weight2;
            sum2 += batch_projection_input[2u * CHANNELS + input + 3u] * weight3;
            sum3 += batch_projection_input[3u * CHANNELS + input] * weight0;
            sum3 += batch_projection_input[3u * CHANNELS + input + 1u] * weight1;
            sum3 += batch_projection_input[3u * CHANNELS + input + 2u] * weight2;
            sum3 += batch_projection_input[3u * CHANNELS + input + 3u] * weight3;
            input += 4u;
        }
        batch_channel_hidden[output] = max(sum0, 0.0) * max(sum0, 0.0);
        batch_channel_hidden[CHANNEL_HIDDEN_STRIDE + output] = max(sum1, 0.0) * max(sum1, 0.0);
        batch_channel_hidden[2u * CHANNEL_HIDDEN_STRIDE + output] = max(sum2, 0.0) * max(sum2, 0.0);
        batch_channel_hidden[3u * CHANNEL_HIDDEN_STRIDE + output] = max(sum3, 0.0) * max(sum3, 0.0);
        output += 256u;
    }
    workgroupBarrier();
}

fn finish_channel_rows(base_row: u32, layer_info: LayerMeta, lane: u32) {
    let output = lane;
    if (output < CHANNELS) {
        var sum0 = 0.0;
        var sum1 = 0.0;
        var sum2 = 0.0;
        var sum3 = 0.0;
        var input = 0u;
        loop {
            if (input >= layer_info.channel_dim) { break; }
            let packed = layer_info.cwv + input * CHANNELS + output * 4u;
            let weight0 = model_weights[packed];
            let weight1 = model_weights[packed + 1u];
            let weight2 = model_weights[packed + 2u];
            let weight3 = model_weights[packed + 3u];
            sum0 += batch_channel_hidden[input] * weight0;
            sum0 += batch_channel_hidden[input + 1u] * weight1;
            sum0 += batch_channel_hidden[input + 2u] * weight2;
            sum0 += batch_channel_hidden[input + 3u] * weight3;
            sum1 += batch_channel_hidden[CHANNEL_HIDDEN_STRIDE + input] * weight0;
            sum1 += batch_channel_hidden[CHANNEL_HIDDEN_STRIDE + input + 1u] * weight1;
            sum1 += batch_channel_hidden[CHANNEL_HIDDEN_STRIDE + input + 2u] * weight2;
            sum1 += batch_channel_hidden[CHANNEL_HIDDEN_STRIDE + input + 3u] * weight3;
            sum2 += batch_channel_hidden[2u * CHANNEL_HIDDEN_STRIDE + input] * weight0;
            sum2 += batch_channel_hidden[2u * CHANNEL_HIDDEN_STRIDE + input + 1u] * weight1;
            sum2 += batch_channel_hidden[2u * CHANNEL_HIDDEN_STRIDE + input + 2u] * weight2;
            sum2 += batch_channel_hidden[2u * CHANNEL_HIDDEN_STRIDE + input + 3u] * weight3;
            sum3 += batch_channel_hidden[3u * CHANNEL_HIDDEN_STRIDE + input] * weight0;
            sum3 += batch_channel_hidden[3u * CHANNEL_HIDDEN_STRIDE + input + 1u] * weight1;
            sum3 += batch_channel_hidden[3u * CHANNEL_HIDDEN_STRIDE + input + 2u] * weight2;
            sum3 += batch_channel_hidden[3u * CHANNEL_HIDDEN_STRIDE + input + 3u] * weight3;
            input += 4u;
        }
        if (base_row < params.row_count) {
            let index = base_row * CHANNELS + output;
            sequence_x[index] = time_residual_x[index] + sum0;
        }
        if (base_row + 1u < params.row_count) {
            let index = (base_row + 1u) * CHANNELS + output;
            sequence_x[index] = time_residual_x[index] + sum1;
        }
        if (base_row + 2u < params.row_count) {
            let index = (base_row + 2u) * CHANNELS + output;
            sequence_x[index] = time_residual_x[index] + sum2;
        }
        if (base_row + 3u < params.row_count) {
            let index = (base_row + 3u) * CHANNELS + output;
            sequence_x[index] = time_residual_x[index] + sum3;
        }
    }
}

@compute @workgroup_size(128)
fn prepare_layer_fused_rows(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let base_row = workgroup.x * PREPARE_ROWS_PER_WORKGROUP;
    let lane = local.x;
    let layer_info = model_metadata.layers[params.layer_index];

    prepare_current_norm(base_row, false, layer_info, lane);
    prepare_shift_delta(base_row, false, layer_info, lane);

    prepare_projection_input(layer_info, 0u, false, PREPARE_ROWS_PER_WORKGROUP, 128u, lane);
    project_lerp_to_rows(base_row, CHANNELS, layer_info.wr, INVALID_INDEX, TARGET_R, lane);
    prepare_projection_input(layer_info, 1u, false, PREPARE_ROWS_PER_WORKGROUP, 128u, lane);
    project_lerp_to_rows(base_row, CHANNELS, layer_info.wk, INVALID_INDEX, TARGET_K, lane);

    prepare_projection_input(layer_info, 6u, false, PREPARE_ROWS_PER_WORKGROUP, 128u, lane);
    project_lerp_to_small(HEADS, layer_info.ks_w, layer_info.ks_b, ACTIVATION_SIGMOID, lane);
    copy_small_to_scales(0u, HEADS, lane);
    prepare_projection_input(layer_info, 7u, false, PREPARE_ROWS_PER_WORKGROUP, 128u, lane);
    project_lerp_to_small(HEADS, layer_info.vs_w, layer_info.vs_b, ACTIVATION_SIGMOID, lane);
    copy_small_to_scales(HEADS, HEADS, lane);

    prepare_projection_input(layer_info, 2u, false, PREPARE_ROWS_PER_WORKGROUP, 128u, lane);
    project_lerp_to_rows(base_row, CHANNELS, layer_info.wv, INVALID_INDEX, TARGET_V, lane);
    if (params.local_layer == 0u) {
        copy_v_to_v0(base_row, lane);
    } else {
        project_lerp_to_small(8u, layer_info.vl_a, INVALID_INDEX, ACTIVATION_NONE, lane);
        project_small_to_rows(base_row, 8u, layer_info.vl_b, layer_info.vl_bias, TARGET_V_GATE, lane);
    }

    prepare_projection_input(layer_info, 4u, false, PREPARE_ROWS_PER_WORKGROUP, 128u, lane);
    project_lerp_to_small(16u, layer_info.al_a, INVALID_INDEX, ACTIVATION_NONE, lane);
    project_small_to_rows(base_row, 16u, layer_info.al_b, layer_info.al_bias, TARGET_A_SCRATCH, lane);
    normalize_kv(base_row, lane);

    prepare_projection_input(layer_info, 5u, false, PREPARE_ROWS_PER_WORKGROUP, 128u, lane);
    project_lerp_to_small(16u, layer_info.gl_a, INVALID_INDEX, ACTIVATION_SIGMOID, lane);
    project_small_to_rows(base_row, 16u, layer_info.gl_b, INVALID_INDEX, TARGET_G, lane);

    prepare_projection_input(layer_info, 3u, false, PREPARE_ROWS_PER_WORKGROUP, 128u, lane);
    project_lerp_to_small(16u, layer_info.dl_a, INVALID_INDEX, ACTIVATION_TANH, lane);
    project_small_to_rows(base_row, 16u, layer_info.dl_b, layer_info.dl_bias, TARGET_D, lane);
}

@compute @workgroup_size(256)
fn finish_channel_mixer_fused_rows(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let base_row = workgroup.x * CHANNEL_ROWS_PER_WORKGROUP;
    let lane = local.x;
    let layer_info = model_metadata.layers[params.layer_index];

    prepare_current_norm(base_row, true, layer_info, lane);
    prepare_shift_delta(base_row, true, layer_info, lane);
    prepare_projection_input(layer_info, 0u, true, CHANNEL_ROWS_PER_WORKGROUP, 256u, lane);
    project_channel_hidden(base_row, layer_info, lane);
    finish_channel_rows(base_row, layer_info, lane);
}
