const CHANNELS: u32 = 128u;
const FEATURE_DIM: u32 = 92u;
const INVALID_INDEX: u32 = 0xffffffffu;
const REDUCTION_LANES: u32 = 64u;
const BUF_X: u32 = 0u;
const BUF_NORM: u32 = 1u;
const BUF_TMP0: u32 = 2u;
const BUF_TMP1: u32 = 3u;

struct LayerMeta {
    values: array<u32, 34>,
};

struct ModelMetadata {
    layers: array<LayerMeta, 16>,
    normalization_weights: array<f32>,
};

struct HeadParams {
    row_count: u32,
    return_curves: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<storage, read> model_weights: array<f32>;
@group(0) @binding(1)
var<storage, read> model_metadata: ModelMetadata;
@group(0) @binding(2)
var<storage, read> feature_rows: array<f32>;
@group(0) @binding(3)
var<storage, read_write> model_rows: array<f32>;
@group(0) @binding(4)
var<storage, read_write> probabilities: array<f32>;
@group(0) @binding(5)
var<storage, read_write> ahead_logits: array<f32>;
@group(0) @binding(6)
var<storage, read_write> curve_weights: array<f32>;
@group(0) @binding(7)
var<uniform> params: HeadParams;

var<workgroup> x: array<f32, 128>;
var<workgroup> norm_values: array<f32, 512>;
var<workgroup> tmp0: array<f32, 512>;
var<workgroup> tmp1: array<f32, 512>;
var<workgroup> reduction: array<f32, 64>;

fn normalization_weight(offset: u32) -> f32 {
    return model_metadata.normalization_weights[offset];
}

fn scratch_value(buffer: u32, index: u32) -> f32 {
    switch buffer {
        case BUF_X: { return x[index]; }
        case BUF_NORM: { return norm_values[index]; }
        case BUF_TMP0: { return tmp0[index]; }
        default: { return tmp1[index]; }
    }
}

fn set_scratch_value(buffer: u32, index: u32, value: f32) {
    switch buffer {
        case BUF_X: { x[index] = value; }
        case BUF_NORM: { norm_values[index] = value; }
        case BUF_TMP0: { tmp0[index] = value; }
        default: { tmp1[index] = value; }
    }
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
        index += 128u;
    }
    workgroupBarrier();
}

fn relu_in_place(values: u32, count: u32, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        set_scratch_value(values, index, max(scratch_value(values, index), 0.0));
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
        reduction[lane] = sum;
    }
    workgroupBarrier();
    var stride = REDUCTION_LANES / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let mean = reduction[0] / f32(count);

    var variance_sum = 0.0;
    index = lane;
    if (lane < REDUCTION_LANES) {
        loop {
            if (index >= count) { break; }
            let centered = scratch_value(input, index) - mean;
            variance_sum += centered * centered;
            index += REDUCTION_LANES;
        }
        reduction[lane] = variance_sum;
    }
    workgroupBarrier();
    stride = REDUCTION_LANES / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let inverse_std = inverseSqrt(reduction[0] / f32(count) + 0.00001);
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

@compute @workgroup_size(128)
fn encode_features(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let row = workgroup.x;
    let lane = local.x;
    var index = lane;
    loop {
        if (index >= FEATURE_DIM) { break; }
        x[index] = feature_rows[row * FEATURE_DIM + index];
        index += 128u;
    }
    workgroupBarrier();
    linear_x(
        BUF_TMP0,
        FEATURE_DIM,
        512u,
        /*__FEATURES_INPUT_W__*/u,
        /*__FEATURES_INPUT_B__*/u,
        lane,
    );
    silu_in_place(BUF_TMP0, 512u, lane);
    layer_norm(
        BUF_TMP0,
        BUF_TMP1,
        512u,
        /*__FEATURES_NORM_W__*/u,
        /*__FEATURES_NORM_B__*/u,
        lane,
    );
    linear_tmp1(
        BUF_X,
        512u,
        CHANNELS,
        /*__FEATURES_OUTPUT_W__*/u,
        /*__FEATURES_OUTPUT_B__*/u,
        lane,
    );
    silu_in_place(BUF_X, CHANNELS, lane);
    index = lane;
    loop {
        if (index >= CHANNELS) { break; }
        model_rows[row * CHANNELS + index] = x[index];
        index += 128u;
    }
}

@compute @workgroup_size(128)
fn output_heads(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let row = select(
        workgroup.x,
        workgroup.x * 2u,
        params.return_curves == 0u,
    );
    let lane = local.x;
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        x[channel] = model_rows[row * CHANNELS + channel];
        channel += 128u;
    }
    workgroupBarrier();
    layer_norm(
        BUF_X,
        BUF_NORM,
        CHANNELS,
        /*__PREHEAD_W__*/u,
        /*__PREHEAD_B__*/u,
        lane,
    );

    let review = row / 2u;
    if ((row & 1u) == 0u) {
        linear_norm(
            BUF_TMP0,
            CHANNELS,
            512u,
            /*__HEAD_P_W__*/u,
            /*__HEAD_P_B__*/u,
            lane,
        );
        relu_in_place(BUF_TMP0, 512u, lane);
        linear_tmp0(
            BUF_TMP1,
            512u,
            4u,
            /*__P_LINEAR_W__*/u,
            /*__P_LINEAR_B__*/u,
            lane,
        );
        if (lane == 0u) {
            var maximum = tmp1[0];
            maximum = max(maximum, tmp1[1]);
            maximum = max(maximum, tmp1[2]);
            maximum = max(maximum, tmp1[3]);
            let exp0 = exp(tmp1[0] - maximum);
            let exp1 = exp(tmp1[1] - maximum);
            let exp2 = exp(tmp1[2] - maximum);
            let exp3 = exp(tmp1[3] - maximum);
            probabilities[review] = 1.0 - exp0 / (exp0 + exp1 + exp2 + exp3);
        }
        return;
    }

    if (params.return_curves == 0u) { return; }
    linear_norm(
        BUF_TMP0,
        CHANNELS,
        512u,
        /*__HEAD_AHEAD_W__*/u,
        /*__HEAD_AHEAD_B__*/u,
        lane,
    );
    relu_in_place(BUF_TMP0, 512u, lane);
    linear_tmp0(
        BUF_TMP1,
        512u,
        CHANNELS,
        /*__AHEAD_LINEAR_W__*/u,
        /*__AHEAD_LINEAR_B__*/u,
        lane,
    );
    channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        ahead_logits[review * CHANNELS + channel] = tmp1[channel];
        channel += 128u;
    }
    workgroupBarrier();

    linear_norm(
        BUF_TMP0,
        CHANNELS,
        CHANNELS,
        /*__HEAD_W_INPUT_W__*/u,
        /*__HEAD_W_INPUT_B__*/u,
        lane,
    );
    relu_in_place(BUF_TMP0, CHANNELS, lane);
    layer_norm(
        BUF_TMP0,
        BUF_TMP1,
        CHANNELS,
        /*__HEAD_W_NORM_W__*/u,
        /*__HEAD_W_NORM_B__*/u,
        lane,
    );
    linear_tmp1(
        BUF_TMP0,
        CHANNELS,
        512u,
        /*__HEAD_W_OUTPUT_W__*/u,
        /*__HEAD_W_OUTPUT_B__*/u,
        lane,
    );
    linear_tmp0(
        BUF_TMP1,
        512u,
        CHANNELS,
        /*__W_LINEAR_W__*/u,
        /*__W_LINEAR_B__*/u,
        lane,
    );
    if (lane == 0u) {
        var maximum = tmp1[0];
        var index = 1u;
        loop {
            if (index >= CHANNELS) { break; }
            maximum = max(maximum, tmp1[index]);
            index += 1u;
        }
        var denominator = 0.0;
        index = 0u;
        loop {
            if (index >= CHANNELS) { break; }
            denominator += exp(tmp1[index] - maximum);
            index += 1u;
        }
        index = 0u;
        loop {
            if (index >= CHANNELS) { break; }
            curve_weights[review * CHANNELS + index] =
                exp(tmp1[index] - maximum) / denominator;
            index += 1u;
        }
    }
}
