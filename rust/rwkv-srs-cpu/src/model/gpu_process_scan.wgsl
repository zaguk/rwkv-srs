const CHANNELS: u32 = 128u;
const HEADS: u32 = 4u;
const HEAD_SIZE: u32 = 32u;
const STATE_LAYER_STRIDE: u32 = 4352u;
const TIME_STATE_OFFSET: u32 = 128u;
const CHANNEL_STATE_OFFSET: u32 = 4224u;
const MATRIX_VALUES: u32 = 4096u;
const TRANSFORM_VALUES: u32 = 8192u;
const INVALID_INDEX: u32 = 0xffffffffu;
const ROW_HAS_STATE: u32 = 1u;
const ROW_LAST_PROCESS: u32 = 2u;
const CHUNK_LAST: u32 = 1u;
const REDUCTION_LANES: u32 = 64u;
const TRANSFORM_LANES_PER_ROW: u32 = /*__TRANSFORM_LANES_PER_ROW__*/u;
const TRANSFORM_COLUMNS_PER_LANE: u32 = /*__TRANSFORM_COLUMNS_PER_LANE__*/u;

/*__TRANSFORM_LANE_SUM__*/

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

struct TransformMeta {
    start_row: u32,
    review_count: u32,
    entity: u32,
    local_index: u32,
};

struct FinalChunkMeta {
    start_row: u32,
    review_count: u32,
    entity: u32,
    prefix_transform: u32,
    flags: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
@group(0) @binding(11)
var<storage, read_write> time_output_rows: array<f32>;
@group(0) @binding(12)
var<storage, read> recurrent_states: array<f32>;
@group(0) @binding(13)
var<storage, read> row_metadata: array<RowMeta>;
@group(0) @binding(14)
var<storage, read> transform_metadata: array<TransformMeta>;
@group(0) @binding(15)
var<storage, read> final_chunk_metadata: array<FinalChunkMeta>;
@group(0) @binding(16)
var<storage, read_write> transforms_a: array<f32>;
@group(0) @binding(17)
var<storage, read_write> transforms_b: array<f32>;
@group(0) @binding(18)
var<storage, read_write> next_recurrent_states: array<f32>;

@group(1) @binding(0)
var<uniform> params: DispatchParams;

var<workgroup> x: array<f32, 128>;
var<workgroup> norm_values: array<f32, 128>;
var<workgroup> tmp0: array<f32, 512>;
var<workgroup> tmp1: array<f32, 512>;
var<workgroup> extra: array<f32, 448>;

fn normalization_weight(offset: u32) -> f32 {
    return model_metadata.normalization_weights[offset];
}

fn scratch_value(buffer: u32, index: u32) -> f32 {
    switch buffer {
        case BUF_X: { return x[index]; }
        case BUF_NORM: { return norm_values[index]; }
        case BUF_TMP0: { return tmp0[index]; }
        case BUF_TMP1: { return tmp1[index]; }
        case BUF_R: { return tmp0[R_OFFSET + index]; }
        case BUF_K: { return tmp0[K_OFFSET + index]; }
        case BUF_V: { return tmp0[V_OFFSET + index]; }
        case BUF_D: { return tmp1[D_OFFSET + index]; }
        case BUF_A: { return tmp1[A_OFFSET + index]; }
        case BUF_G: { return extra[G_OFFSET + index]; }
        case BUF_K_DEFORMED: { return norm_values[index]; }
        default: { return extra[OUT_OFFSET + index]; }
    }
}

fn set_scratch_value(buffer: u32, index: u32, value: f32) {
    switch buffer {
        case BUF_X: { x[index] = value; }
        case BUF_NORM: { norm_values[index] = value; }
        case BUF_TMP0: { tmp0[index] = value; }
        case BUF_TMP1: { tmp1[index] = value; }
        case BUF_R: { tmp0[R_OFFSET + index] = value; }
        case BUF_K: { tmp0[K_OFFSET + index] = value; }
        case BUF_V: { tmp0[V_OFFSET + index] = value; }
        case BUF_D: { tmp1[D_OFFSET + index] = value; }
        case BUF_A: { tmp1[A_OFFSET + index] = value; }
        case BUF_G: { extra[G_OFFSET + index] = value; }
        case BUF_K_DEFORMED: { norm_values[index] = value; }
        default: { extra[OUT_OFFSET + index] = value; }
    }
}

/*__LINEAR_FUNCTIONS__*/

fn sigmoid(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

fn copy_values(input: u32, output: u32, count: u32, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        set_scratch_value(output, index, scratch_value(input, index));
        index += 128u;
    }
    workgroupBarrier();
}

fn sigmoid_in_place(values: u32, count: u32, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        set_scratch_value(values, index, sigmoid(scratch_value(values, index)));
        index += 128u;
    }
    workgroupBarrier();
}

fn tanh_in_place(values: u32, count: u32, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        set_scratch_value(values, index, tanh(scratch_value(values, index)));
        index += 128u;
    }
    workgroupBarrier();
}

fn relu_squared_in_place(values: u32, count: u32, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        let value = max(scratch_value(values, index), 0.0);
        set_scratch_value(values, index, value * value);
        index += 128u;
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
            sum += scratch_value(input, index);
            index += REDUCTION_LANES;
        }
        extra[AUX_OFFSET + lane] = sum;
    }
    workgroupBarrier();
    var stride = REDUCTION_LANES / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            extra[AUX_OFFSET + lane] += extra[AUX_OFFSET + lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let mean = extra[AUX_OFFSET] / f32(count);

    var variance_sum = 0.0;
    index = lane;
    if (lane < REDUCTION_LANES) {
        loop {
            if (index >= count) { break; }
            let centered = scratch_value(input, index) - mean;
            variance_sum += centered * centered;
            index += REDUCTION_LANES;
        }
        extra[AUX_OFFSET + lane] = variance_sum;
    }
    workgroupBarrier();
    stride = REDUCTION_LANES / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            extra[AUX_OFFSET + lane] += extra[AUX_OFFSET + lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let inverse_std = inverseSqrt(extra[AUX_OFFSET] / f32(count) + 0.00001);
    index = lane;
    loop {
        if (index >= count) { break; }
        set_scratch_value(
            output,
            index,
            (scratch_value(input, index) - mean) * inverse_std
                * normalization_weight(weight_offset + index)
                + model_weights[bias_offset + index],
        );
        index += 128u;
    }
    workgroupBarrier();
}

fn state_layer_base(entity: u32) -> u32 {
    return entity * params.layers * STATE_LAYER_STRIDE
        + params.local_layer * STATE_LAYER_STRIDE;
}

fn lerp_part(output: u32, part: u32, layer_info: LayerMeta, lane: u32) {
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let current = norm_values[channel];
        let shifted = x[channel];
        set_scratch_value(
            output,
            channel,
            current + (shifted - current)
                * model_weights[layer_info.lerp + part * CHANNELS + channel],
        );
        channel += 128u;
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
            let k = tmp0[K_OFFSET + base + index];
            let v = tmp0[V_OFFSET + base + index];
            k_sum += k * k;
            v_sum += v * v;
            index += 1u;
        }
        let k_norm = max(sqrt(k_sum), 0.000000059604645);
        let v_norm = max(sqrt(v_sum), 0.000000059604645);
        let k_scale = extra[AUX_OFFSET + lane];
        let v_scale = extra[AUX_OFFSET + 4u + lane];
        index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let channel = base + index;
            let deformed = tmp0[K_OFFSET + channel] / k_norm * k_scale;
            let a = tmp1[A_OFFSET + channel];
            norm_values[channel] = deformed;
            tmp0[K_OFFSET + channel] = deformed * a;
            tmp0[V_OFFSET + channel] = tmp0[V_OFFSET + channel] / v_norm * v_scale;
            index += 1u;
        }
    }
    workgroupBarrier();
}

fn load_current_row(row: u32, source_time_residual: bool, lane: u32) {
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let index = row * CHANNELS + channel;
        x[channel] = select(sequence_x[index], time_residual_x[index], source_time_residual);
        channel += 128u;
    }
    workgroupBarrier();
}

fn load_shifted_norm(
    row: u32,
    layer_info: LayerMeta,
    channel_mixer: bool,
    lane: u32,
) {
    let row_info = row_metadata[row];
    let previous = row_info.previous_process_row;
    if (previous != INVALID_INDEX) {
        load_current_row(previous, channel_mixer, lane);
        if (channel_mixer) {
            layer_norm(
                BUF_X,
                BUF_TMP0,
                CHANNELS,
                layer_info.cln_w,
                layer_info.cln_b,
                lane,
            );
        } else {
            layer_norm(
                BUF_X,
                BUF_TMP0,
                CHANNELS,
                layer_info.ln_w,
                layer_info.ln_b,
                lane,
            );
        }
        copy_values(BUF_TMP0, BUF_X, CHANNELS, lane);
    } else if ((row_info.flags & ROW_HAS_STATE) != 0u) {
        var channel = lane;
        let offset = select(0u, CHANNEL_STATE_OFFSET, channel_mixer);
        let base = state_layer_base(row_info.entity) + offset;
        loop {
            if (channel >= CHANNELS) { break; }
            x[channel] = recurrent_states[base + channel];
            channel += 128u;
        }
        workgroupBarrier();
    } else {
        copy_values(BUF_NORM, BUF_X, CHANNELS, lane);
    }
}

fn transform_value(source_b: bool, index: u32) -> f32 {
    return select(transforms_a[index], transforms_b[index], source_b);
}

fn set_transform_value(target_b: bool, index: u32, value: f32) {
    if (target_b) {
        transforms_b[index] = value;
    } else {
        transforms_a[index] = value;
    }
}

@compute @workgroup_size(128)
fn prepare_layer(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let row = workgroup.x;
    let lane = local.x;
    let layer_info = model_metadata.layers[params.layer_index];
    let row_meta = row_metadata[row];

    load_current_row(row, false, lane);
    layer_norm(BUF_X, BUF_NORM, CHANNELS, layer_info.ln_w, layer_info.ln_b, lane);
    if ((row_meta.flags & ROW_LAST_PROCESS) != 0u) {
        var channel = lane;
        let base = state_layer_base(row_meta.entity);
        loop {
            if (channel >= CHANNELS) { break; }
            next_recurrent_states[base + channel] = norm_values[channel];
            channel += 128u;
        }
    }
    workgroupBarrier();
    load_shifted_norm(row, layer_info, false, lane);

    lerp_part(BUF_TMP0, 0u, layer_info, lane);
    linear_tmp0(BUF_R, CHANNELS, CHANNELS, layer_info.wr, INVALID_INDEX, lane);
    lerp_part(BUF_TMP0, 1u, layer_info, lane);
    linear_tmp0(BUF_K, CHANNELS, CHANNELS, layer_info.wk, INVALID_INDEX, lane);
    lerp_part(BUF_V, 2u, layer_info, lane);
    lerp_part(BUF_D, 3u, layer_info, lane);
    lerp_part(BUF_A, 4u, layer_info, lane);
    lerp_part(BUF_G, 5u, layer_info, lane);
    lerp_part(BUF_TMP0, 6u, layer_info, lane);
    linear_tmp0(BUF_TMP1, CHANNELS, HEADS, layer_info.ks_w, layer_info.ks_b, lane);
    sigmoid_in_place(BUF_TMP1, HEADS, lane);
    if (lane < HEADS) {
        extra[AUX_OFFSET + lane] = tmp1[lane];
    }
    workgroupBarrier();
    lerp_part(BUF_TMP0, 7u, layer_info, lane);
    linear_tmp0(BUF_TMP1, CHANNELS, HEADS, layer_info.vs_w, layer_info.vs_b, lane);
    sigmoid_in_place(BUF_TMP1, HEADS, lane);
    if (lane < HEADS) {
        extra[AUX_OFFSET + 4u + lane] = tmp1[lane];
    }
    workgroupBarrier();

    if (params.local_layer == 0u) {
        linear_v(BUF_TMP0, CHANNELS, CHANNELS, layer_info.wv, INVALID_INDEX, lane);
        copy_values(BUF_TMP0, BUF_V, CHANNELS, lane);
        var channel = lane;
        loop {
            if (channel >= CHANNELS) { break; }
            v0_rows[row * CHANNELS + channel] = tmp0[V_OFFSET + channel];
            channel += 128u;
        }
        workgroupBarrier();
    } else {
        linear_v(BUF_TMP0, CHANNELS, 8u, layer_info.vl_a, INVALID_INDEX, lane);
        linear_tmp0(BUF_TMP1, 8u, CHANNELS, layer_info.vl_b, layer_info.vl_bias, lane);
        sigmoid_in_place(BUF_TMP1, CHANNELS, lane);
        linear_v(BUF_OUT, CHANNELS, CHANNELS, layer_info.wv, INVALID_INDEX, lane);
        var channel = lane;
        loop {
            if (channel >= CHANNELS) { break; }
            let projected = extra[OUT_OFFSET + channel];
            tmp0[V_OFFSET + channel] = projected
                + (v0_rows[row * CHANNELS + channel] - projected) * tmp1[channel];
            channel += 128u;
        }
        workgroupBarrier();
    }

    linear_a(BUF_TMP0, CHANNELS, 16u, layer_info.al_a, INVALID_INDEX, lane);
    linear_tmp0(BUF_TMP1, 16u, CHANNELS, layer_info.al_b, layer_info.al_bias, lane);
    sigmoid_in_place(BUF_TMP1, CHANNELS, lane);
    copy_values(BUF_TMP1, BUF_A, CHANNELS, lane);

    linear_g(BUF_TMP0, CHANNELS, 16u, layer_info.gl_a, INVALID_INDEX, lane);
    sigmoid_in_place(BUF_TMP0, 16u, lane);
    linear_tmp0(BUF_G, 16u, CHANNELS, layer_info.gl_b, INVALID_INDEX, lane);

    linear_d(BUF_TMP0, CHANNELS, 16u, layer_info.dl_a, INVALID_INDEX, lane);
    tanh_in_place(BUF_TMP0, 16u, lane);
    linear_tmp0(BUF_D, 16u, CHANNELS, layer_info.dl_b, layer_info.dl_bias, lane);
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let index = D_OFFSET + channel;
        let value = -tmp1[index];
        let softplus = max(value, 0.0) + log(1.0 + exp(-abs(value)));
        tmp1[index] = exp(-exp(-0.5 - softplus));
        channel += 128u;
    }
    workgroupBarrier();

    normalize_kv(lane);
    channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let output_index = row * CHANNELS + channel;
        r_rows[output_index] = tmp0[R_OFFSET + channel];
        k_rows[output_index] = tmp0[K_OFFSET + channel];
        v_rows[output_index] = tmp0[V_OFFSET + channel];
        w_rows[output_index] = tmp1[D_OFFSET + channel];
        k_deformed_rows[output_index] = norm_values[channel];
        g_rows[output_index] = extra[G_OFFSET + channel];
        channel += 128u;
    }
}

@compute @workgroup_size(/*__TRANSFORM_WORKGROUP_SIZE__*/)
fn build_chunk_transforms(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let chunk = workgroup.x;
    let lane = local.x;
    let matrix_row = lane / TRANSFORM_LANES_PER_ROW;
    let column_lane = lane % TRANSFORM_LANES_PER_ROW;
    let head = matrix_row / HEAD_SIZE;
    let row_in_head = matrix_row % HEAD_SIZE;
    let vector_base = head * HEAD_SIZE;
    var mul: array<f32, TRANSFORM_COLUMNS_PER_LANE>;
    var add: array<f32, TRANSFORM_COLUMNS_PER_LANE>;
    var local_column = 0u;
    loop {
        if (local_column >= TRANSFORM_COLUMNS_PER_LANE) { break; }
        let column = column_lane + local_column * TRANSFORM_LANES_PER_ROW;
        mul[local_column] = select(0.0, 1.0, row_in_head == column);
        add[local_column] = 0.0;
        local_column += 1u;
    }

    let chunk_info = transform_metadata[chunk];
    var review = 0u;
    loop {
        if (review >= chunk_info.review_count) { break; }
        let input_row = chunk_info.start_row + review * 2u + 1u;
        let vector_offset = input_row * CHANNELS + vector_base;
        var mul_deformation_part = 0.0;
        var add_deformation_part = 0.0;
        local_column = 0u;
        loop {
            if (local_column >= TRANSFORM_COLUMNS_PER_LANE) { break; }
            let column = column_lane + local_column * TRANSFORM_LANES_PER_ROW;
            let deformed = k_deformed_rows[vector_offset + column];
            mul_deformation_part += mul[local_column] * deformed;
            add_deformation_part += add[local_column] * deformed;
            local_column += 1u;
        }
        let mul_deformation = transform_lane_sum(mul_deformation_part);
        let add_deformation = transform_lane_sum(add_deformation_part);
        let value = v_rows[input_row * CHANNELS + matrix_row];
        local_column = 0u;
        loop {
            if (local_column >= TRANSFORM_COLUMNS_PER_LANE) { break; }
            let column = column_lane + local_column * TRANSFORM_LANES_PER_ROW;
            let decay = w_rows[vector_offset + column];
            let key = k_rows[vector_offset + column];
            mul[local_column] = mul[local_column] * decay - mul_deformation * key;
            add[local_column] = add[local_column] * decay
                - add_deformation * key + value * key;
            local_column += 1u;
        }
        review += 1u;
    }

    let transform_base = chunk * TRANSFORM_VALUES + matrix_row * HEAD_SIZE;
    local_column = 0u;
    loop {
        if (local_column >= TRANSFORM_COLUMNS_PER_LANE) { break; }
        let column = column_lane + local_column * TRANSFORM_LANES_PER_ROW;
        transforms_a[transform_base + column] = mul[local_column];
        transforms_a[transform_base + MATRIX_VALUES + column] = add[local_column];
        local_column += 1u;
    }
}

@compute @workgroup_size(/*__TRANSFORM_WORKGROUP_SIZE__*/)
fn build_chunk_transforms_state_only(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let chunk = workgroup.x;
    let lane = local.x;
    let matrix_row = lane / TRANSFORM_LANES_PER_ROW;
    let column_lane = lane % TRANSFORM_LANES_PER_ROW;
    let head = matrix_row / HEAD_SIZE;
    let row_in_head = matrix_row % HEAD_SIZE;
    let vector_base = head * HEAD_SIZE;
    var mul: array<f32, TRANSFORM_COLUMNS_PER_LANE>;
    var add: array<f32, TRANSFORM_COLUMNS_PER_LANE>;
    var local_column = 0u;
    loop {
        if (local_column >= TRANSFORM_COLUMNS_PER_LANE) { break; }
        let column = column_lane + local_column * TRANSFORM_LANES_PER_ROW;
        mul[local_column] = select(0.0, 1.0, row_in_head == column);
        add[local_column] = 0.0;
        local_column += 1u;
    }

    let chunk_info = transform_metadata[chunk];
    var review = 0u;
    loop {
        if (review >= chunk_info.review_count) { break; }
        let input_row = chunk_info.start_row + review;
        let vector_offset = input_row * CHANNELS + vector_base;
        var mul_deformation_part = 0.0;
        var add_deformation_part = 0.0;
        local_column = 0u;
        loop {
            if (local_column >= TRANSFORM_COLUMNS_PER_LANE) { break; }
            let column = column_lane + local_column * TRANSFORM_LANES_PER_ROW;
            let deformed = k_deformed_rows[vector_offset + column];
            mul_deformation_part += mul[local_column] * deformed;
            add_deformation_part += add[local_column] * deformed;
            local_column += 1u;
        }
        let mul_deformation = transform_lane_sum(mul_deformation_part);
        let add_deformation = transform_lane_sum(add_deformation_part);
        let value = v_rows[input_row * CHANNELS + matrix_row];
        local_column = 0u;
        loop {
            if (local_column >= TRANSFORM_COLUMNS_PER_LANE) { break; }
            let column = column_lane + local_column * TRANSFORM_LANES_PER_ROW;
            let decay = w_rows[vector_offset + column];
            let key = k_rows[vector_offset + column];
            mul[local_column] = mul[local_column] * decay - mul_deformation * key;
            add[local_column] = add[local_column] * decay
                - add_deformation * key + value * key;
            local_column += 1u;
        }
        review += 1u;
    }

    let transform_base = chunk * TRANSFORM_VALUES + matrix_row * HEAD_SIZE;
    local_column = 0u;
    loop {
        if (local_column >= TRANSFORM_COLUMNS_PER_LANE) { break; }
        let column = column_lane + local_column * TRANSFORM_LANES_PER_ROW;
        transforms_a[transform_base + column] = mul[local_column];
        transforms_a[transform_base + MATRIX_VALUES + column] = add[local_column];
        local_column += 1u;
    }
}

@compute @workgroup_size(/*__TRANSFORM_WORKGROUP_SIZE__*/)
fn scan_chunk_transforms(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let chunk = workgroup.x;
    let lane = local.x;
    let matrix_row = lane / TRANSFORM_LANES_PER_ROW;
    let column_lane = lane % TRANSFORM_LANES_PER_ROW;
    let chunk_info = transform_metadata[chunk];
    let source_b = params.scan_source_b != 0u;
    let target_b = !source_b;
    let output_base = chunk * TRANSFORM_VALUES + matrix_row * HEAD_SIZE;
    if (chunk_info.local_index < params.scan_offset) {
        var local_column = 0u;
        loop {
            if (local_column >= TRANSFORM_COLUMNS_PER_LANE) { break; }
            let column = column_lane + local_column * TRANSFORM_LANES_PER_ROW;
            set_transform_value(
                target_b,
                output_base + column,
                transform_value(source_b, output_base + column),
            );
            set_transform_value(
                target_b,
                output_base + MATRIX_VALUES + column,
                transform_value(source_b, output_base + MATRIX_VALUES + column),
            );
            local_column += 1u;
        }
        return;
    }

    let left_chunk = chunk - params.scan_offset;
    let left_base = left_chunk * TRANSFORM_VALUES + matrix_row * HEAD_SIZE;
    let head = matrix_row / HEAD_SIZE;
    let row_in_head = matrix_row % HEAD_SIZE;
    var local_column = 0u;
    loop {
        if (local_column >= TRANSFORM_COLUMNS_PER_LANE) { break; }
        let column = column_lane + local_column * TRANSFORM_LANES_PER_ROW;
        var mul_sum = 0.0;
        var add_sum = 0.0;
        var inner = 0u;
        loop {
            if (inner >= HEAD_SIZE) { break; }
            let left_mul = transform_value(source_b, left_base + inner);
            let left_add = transform_value(source_b, left_base + MATRIX_VALUES + inner);
            let right_matrix_index = chunk * TRANSFORM_VALUES
                + (head * HEAD_SIZE + inner) * HEAD_SIZE + column;
            let right_mul = transform_value(source_b, right_matrix_index);
            mul_sum += left_mul * right_mul;
            add_sum += left_add * right_mul;
            inner += 1u;
        }
        let right_add_index = chunk * TRANSFORM_VALUES + MATRIX_VALUES
            + (head * HEAD_SIZE + row_in_head) * HEAD_SIZE + column;
        add_sum += transform_value(source_b, right_add_index);
        set_transform_value(target_b, output_base + column, mul_sum);
        set_transform_value(target_b, output_base + MATRIX_VALUES + column, add_sum);
        local_column += 1u;
    }
}

fn candidate_output(input_row: u32, lane: u32, state: ptr<function, array<f32, 32>>) -> f32 {
    let head = lane / HEAD_SIZE;
    let vector_base = input_row * CHANNELS + head * HEAD_SIZE;
    var deformation = 0.0;
    var column = 0u;
    loop {
        if (column >= HEAD_SIZE) { break; }
        deformation += (*state)[column] * k_deformed_rows[vector_base + column];
        column += 1u;
    }
    let value = v_rows[input_row * CHANNELS + lane];
    var output = 0.0;
    column = 0u;
    loop {
        if (column >= HEAD_SIZE) { break; }
        let next = (*state)[column] * w_rows[vector_base + column]
            - deformation * k_rows[vector_base + column]
            + value * k_rows[vector_base + column];
        output += next * r_rows[vector_base + column];
        column += 1u;
    }
    return output;
}

fn process_output(input_row: u32, lane: u32, state: ptr<function, array<f32, 32>>) -> f32 {
    let head = lane / HEAD_SIZE;
    let vector_base = input_row * CHANNELS + head * HEAD_SIZE;
    var deformation = 0.0;
    var column = 0u;
    loop {
        if (column >= HEAD_SIZE) { break; }
        deformation += (*state)[column] * k_deformed_rows[vector_base + column];
        column += 1u;
    }
    let value = v_rows[input_row * CHANNELS + lane];
    var output = 0.0;
    column = 0u;
    loop {
        if (column >= HEAD_SIZE) { break; }
        let next = (*state)[column] * w_rows[vector_base + column]
            - deformation * k_rows[vector_base + column]
            + value * k_rows[vector_base + column];
        (*state)[column] = next;
        output += next * r_rows[vector_base + column];
        column += 1u;
    }
    return output;
}

@compute @workgroup_size(128)
fn replay_chunks(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let chunk = workgroup.x;
    let lane = local.x;
    let head = lane / HEAD_SIZE;
    let chunk_info = final_chunk_metadata[chunk];
    let state_base = state_layer_base(chunk_info.entity) + TIME_STATE_OFFSET + lane * HEAD_SIZE;
    var state: array<f32, 32>;
    var column = 0u;
    loop {
        if (column >= HEAD_SIZE) { break; }
        state[column] = recurrent_states[state_base + column];
        column += 1u;
    }
    if (chunk_info.prefix_transform != INVALID_INDEX) {
        let source_b = params.scan_source_b != 0u;
        let prefix_base = chunk_info.prefix_transform * TRANSFORM_VALUES;
        var transformed: array<f32, 32>;
        column = 0u;
        loop {
            if (column >= HEAD_SIZE) { break; }
            var value = transform_value(
                source_b,
                prefix_base + MATRIX_VALUES + lane * HEAD_SIZE + column,
            );
            var inner = 0u;
            loop {
                if (inner >= HEAD_SIZE) { break; }
                let matrix_index = prefix_base
                    + (head * HEAD_SIZE + inner) * HEAD_SIZE + column;
                value += state[inner] * transform_value(source_b, matrix_index);
                inner += 1u;
            }
            transformed[column] = value;
            column += 1u;
        }
        state = transformed;
    }

    var review = 0u;
    loop {
        if (review >= chunk_info.review_count) { break; }
        let query_row = chunk_info.start_row + review * 2u;
        time_output_rows[query_row * CHANNELS + lane] =
            candidate_output(query_row, lane, &state);
        let process_row = query_row + 1u;
        time_output_rows[process_row * CHANNELS + lane] =
            process_output(process_row, lane, &state);
        review += 1u;
    }

    if ((chunk_info.flags & CHUNK_LAST) != 0u) {
        column = 0u;
        loop {
            if (column >= HEAD_SIZE) { break; }
            next_recurrent_states[state_base + column] = state[column];
            column += 1u;
        }
    }
}

@compute @workgroup_size(128)
fn replay_chunks_state_only(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let chunk = workgroup.x;
    let lane = local.x;
    let head = lane / HEAD_SIZE;
    let chunk_info = final_chunk_metadata[chunk];
    let state_base = state_layer_base(chunk_info.entity) + TIME_STATE_OFFSET + lane * HEAD_SIZE;
    var state: array<f32, 32>;
    var column = 0u;
    loop {
        if (column >= HEAD_SIZE) { break; }
        state[column] = recurrent_states[state_base + column];
        column += 1u;
    }
    if (chunk_info.prefix_transform != INVALID_INDEX) {
        let source_b = params.scan_source_b != 0u;
        let prefix_base = chunk_info.prefix_transform * TRANSFORM_VALUES;
        var transformed: array<f32, 32>;
        column = 0u;
        loop {
            if (column >= HEAD_SIZE) { break; }
            var value = transform_value(
                source_b,
                prefix_base + MATRIX_VALUES + lane * HEAD_SIZE + column,
            );
            var inner = 0u;
            loop {
                if (inner >= HEAD_SIZE) { break; }
                let matrix_index = prefix_base
                    + (head * HEAD_SIZE + inner) * HEAD_SIZE + column;
                value += state[inner] * transform_value(source_b, matrix_index);
                inner += 1u;
            }
            transformed[column] = value;
            column += 1u;
        }
        state = transformed;
    }

    var review = 0u;
    loop {
        if (review >= chunk_info.review_count) { break; }
        let process_row = chunk_info.start_row + review;
        time_output_rows[process_row * CHANNELS + lane] =
            process_output(process_row, lane, &state);
        review += 1u;
    }

    if ((chunk_info.flags & CHUNK_LAST) != 0u) {
        column = 0u;
        loop {
            if (column >= HEAD_SIZE) { break; }
            next_recurrent_states[state_base + column] = state[column];
            column += 1u;
        }
    }
}

fn group_norm_and_bonus(layer_info: LayerMeta, lane: u32) {
    if (lane < HEADS) {
        let base = lane * HEAD_SIZE;
        var sum = 0.0;
        var index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            sum += extra[OUT_OFFSET + base + index];
            index += 1u;
        }
        let mean = sum / 32.0;
        var variance = 0.0;
        index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let centered = extra[OUT_OFFSET + base + index] - mean;
            variance += centered * centered;
            index += 1u;
        }
        let inverse_std = inverseSqrt(variance / 32.0 + 0.00064);
        var bonus = 0.0;
        index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let channel = base + index;
            extra[OUT_OFFSET + channel] = (extra[OUT_OFFSET + channel] - mean)
                * inverse_std * normalization_weight(layer_info.gn_w + channel)
                + model_weights[layer_info.gn_b + channel];
            bonus += tmp0[R_OFFSET + channel]
                * model_weights[layer_info.bonus + channel]
                * tmp0[K_OFFSET + channel];
            index += 1u;
        }
        extra[AUX_OFFSET + lane] = bonus;
    }
    workgroupBarrier();

    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let head = channel / HEAD_SIZE;
        extra[OUT_OFFSET + channel] = extra[G_OFFSET + channel]
            * (extra[OUT_OFFSET + channel]
                + extra[AUX_OFFSET + head] * tmp0[V_OFFSET + channel]);
        channel += 128u;
    }
    workgroupBarrier();
}

@compute @workgroup_size(128)
fn finish_time_mixer(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let row = workgroup.x;
    let lane = local.x;
    let layer_info = model_metadata.layers[params.layer_index];
    load_current_row(row, false, lane);
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let index = row * CHANNELS + channel;
        tmp0[R_OFFSET + channel] = r_rows[index];
        tmp0[K_OFFSET + channel] = k_rows[index];
        tmp0[V_OFFSET + channel] = v_rows[index];
        extra[G_OFFSET + channel] = g_rows[index];
        extra[OUT_OFFSET + channel] = time_output_rows[index];
        channel += 128u;
    }
    workgroupBarrier();
    group_norm_and_bonus(layer_info, lane);
    linear_out(BUF_TMP0, CHANNELS, CHANNELS, layer_info.wo, INVALID_INDEX, lane);
    channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        time_residual_x[row * CHANNELS + channel] = x[channel] + tmp0[channel];
        channel += 128u;
    }
}

@compute @workgroup_size(128)
fn finish_channel_mixer(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let row = workgroup.x;
    let lane = local.x;
    let layer_info = model_metadata.layers[params.layer_index];
    let row_meta = row_metadata[row];
    load_current_row(row, true, lane);
    layer_norm(
        BUF_X,
        BUF_NORM,
        CHANNELS,
        layer_info.cln_w,
        layer_info.cln_b,
        lane,
    );
    if ((row_meta.flags & ROW_LAST_PROCESS) != 0u) {
        var channel = lane;
        let base = state_layer_base(row_meta.entity) + CHANNEL_STATE_OFFSET;
        loop {
            if (channel >= CHANNELS) { break; }
            next_recurrent_states[base + channel] = norm_values[channel];
            channel += 128u;
        }
    }
    workgroupBarrier();
    load_shifted_norm(row, layer_info, true, lane);

    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        tmp0[channel] = norm_values[channel] + (x[channel] - norm_values[channel])
            * model_weights[layer_info.clerp + channel];
        channel += 128u;
    }
    workgroupBarrier();
    linear_tmp0(BUF_TMP1, CHANNELS, layer_info.channel_dim, layer_info.cwk, INVALID_INDEX, lane);
    relu_squared_in_place(BUF_TMP1, layer_info.channel_dim, lane);
    linear_tmp1(BUF_OUT, layer_info.channel_dim, CHANNELS, layer_info.cwv, INVALID_INDEX, lane);
    channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let index = row * CHANNELS + channel;
        sequence_x[index] = time_residual_x[index] + extra[OUT_OFFSET + channel];
        channel += 128u;
    }
}
