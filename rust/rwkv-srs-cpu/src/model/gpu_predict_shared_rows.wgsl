// Safe two-row prediction candidate. One 128-lane workgroup owns both rows.
// Every linear projection keeps two independent accumulators so each model
// weight is loaded once and applied to both row-local inputs. All recurrent,
// normalization, and pointwise work continues to use the mature single-row
// helpers in a uniform row order, preserving their reduction order and scratch
// ownership.

const SHARED_ROWS: u32 = 2u;
const SHARED_LANES: u32 = 128u;

fn shared_scratch_value(buffer: u32, slot: u32, index: u32) -> f32 {
    let row_small_base = slot * SMALL_STRIDE;
    let row_large_base = slot * LARGE_STRIDE;
    let row_extra_base = slot * EXTRA_STRIDE;
    switch buffer {
        case BUF_X: { return x[row_small_base + index]; }
        case BUF_NORM: { return norm_values[row_small_base + index]; }
        case BUF_TMP0: { return tmp0[row_large_base + index]; }
        case BUF_TMP1: { return tmp1[row_large_base + index]; }
        case BUF_R: { return tmp0[row_large_base + R_OFFSET + index]; }
        case BUF_K: { return tmp0[row_large_base + K_OFFSET + index]; }
        case BUF_V: { return tmp0[row_large_base + V_OFFSET + index]; }
        case BUF_D: { return tmp1[row_large_base + D_OFFSET + index]; }
        case BUF_A: { return tmp1[row_large_base + A_OFFSET + index]; }
        case BUF_G: { return extra[row_extra_base + G_OFFSET + index]; }
        case BUF_K_DEFORMED: { return norm_values[row_small_base + index]; }
        default: { return extra[row_extra_base + OUT_OFFSET + index]; }
    }
}

fn shared_set_scratch_value(buffer: u32, slot: u32, index: u32, value: f32) {
    let row_small_base = slot * SMALL_STRIDE;
    let row_large_base = slot * LARGE_STRIDE;
    let row_extra_base = slot * EXTRA_STRIDE;
    switch buffer {
        case BUF_X: { x[row_small_base + index] = value; }
        case BUF_NORM: { norm_values[row_small_base + index] = value; }
        case BUF_TMP0: { tmp0[row_large_base + index] = value; }
        case BUF_TMP1: { tmp1[row_large_base + index] = value; }
        case BUF_R: { tmp0[row_large_base + R_OFFSET + index] = value; }
        case BUF_K: { tmp0[row_large_base + K_OFFSET + index] = value; }
        case BUF_V: { tmp0[row_large_base + V_OFFSET + index] = value; }
        case BUF_D: { tmp1[row_large_base + D_OFFSET + index] = value; }
        case BUF_A: { tmp1[row_large_base + A_OFFSET + index] = value; }
        case BUF_G: { extra[row_extra_base + G_OFFSET + index] = value; }
        case BUF_K_DEFORMED: { norm_values[row_small_base + index] = value; }
        default: { extra[row_extra_base + OUT_OFFSET + index] = value; }
    }
}

/*__SHARED_LINEAR_FUNCTIONS__*/

fn shared_silu_in_place(values: u32, count: u32, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        let value0 = shared_scratch_value(values, 0u, index);
        let value1 = shared_scratch_value(values, 1u, index);
        shared_set_scratch_value(values, 0u, index, value0 * sigmoid(value0));
        shared_set_scratch_value(values, 1u, index, value1 * sigmoid(value1));
        index += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_sigmoid_in_place(values: u32, count: u32, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        shared_set_scratch_value(
            values,
            0u,
            index,
            sigmoid(shared_scratch_value(values, 0u, index)),
        );
        shared_set_scratch_value(
            values,
            1u,
            index,
            sigmoid(shared_scratch_value(values, 1u, index)),
        );
        index += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_tanh_in_place(values: u32, count: u32, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        shared_set_scratch_value(
            values,
            0u,
            index,
            tanh(shared_scratch_value(values, 0u, index)),
        );
        shared_set_scratch_value(
            values,
            1u,
            index,
            tanh(shared_scratch_value(values, 1u, index)),
        );
        index += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_relu_in_place(values: u32, count: u32, square: bool, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        let value0 = max(shared_scratch_value(values, 0u, index), 0.0);
        let value1 = max(shared_scratch_value(values, 1u, index), 0.0);
        shared_set_scratch_value(values, 0u, index, select(value0, value0 * value0, square));
        shared_set_scratch_value(values, 1u, index, select(value1, value1 * value1, square));
        index += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_layer_norm(
    input: u32,
    output: u32,
    count: u32,
    weight_offset: u32,
    bias_offset: u32,
    lane: u32,
) {
    let slot = lane / REDUCTION_LANES;
    let reduction_lane = lane % REDUCTION_LANES;
    let row_extra_base = slot * EXTRA_STRIDE;
    var sum = 0.0;
    var index = reduction_lane;
    loop {
        if (index >= count) { break; }
        sum += shared_scratch_value(input, slot, index);
        index += REDUCTION_LANES;
    }
    extra[row_extra_base + AUX_OFFSET + reduction_lane] = sum;
    workgroupBarrier();
    var stride = REDUCTION_LANES / 2u;
    loop {
        if (stride == 0u) { break; }
        if (reduction_lane < stride) {
            extra[row_extra_base + AUX_OFFSET + reduction_lane] +=
                extra[row_extra_base + AUX_OFFSET + reduction_lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let mean = extra[row_extra_base + AUX_OFFSET] / f32(count);

    var variance_sum = 0.0;
    index = reduction_lane;
    loop {
        if (index >= count) { break; }
        let centered = shared_scratch_value(input, slot, index) - mean;
        variance_sum += centered * centered;
        index += REDUCTION_LANES;
    }
    extra[row_extra_base + AUX_OFFSET + reduction_lane] = variance_sum;
    workgroupBarrier();
    stride = REDUCTION_LANES / 2u;
    loop {
        if (stride == 0u) { break; }
        if (reduction_lane < stride) {
            extra[row_extra_base + AUX_OFFSET + reduction_lane] +=
                extra[row_extra_base + AUX_OFFSET + reduction_lane + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let inverse_std = inverseSqrt(
        extra[row_extra_base + AUX_OFFSET] / f32(count) + 0.00001
    );
    index = reduction_lane;
    loop {
        if (index >= count) { break; }
        shared_set_scratch_value(
            output,
            slot,
            index,
            (shared_scratch_value(input, slot, index) - mean) * inverse_std
                * normalization_weight(weight_offset + index)
                + f32(model_weights[bias_offset + index]),
        );
        index += REDUCTION_LANES;
    }
    workgroupBarrier();
}

fn shared_copy_values(input: u32, output: u32, count: u32, lane: u32) {
    var index = lane;
    loop {
        if (index >= count) { break; }
        shared_set_scratch_value(output, 0u, index, shared_scratch_value(input, 0u, index));
        shared_set_scratch_value(output, 1u, index, shared_scratch_value(input, 1u, index));
        index += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_clear_v0(lane: u32) {
    var index = lane;
    loop {
        if (index >= CHANNELS) { break; }
        extra[V0_OFFSET + index] = 0.0;
        extra[EXTRA_STRIDE + V0_OFFSET + index] = 0.0;
        index += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_lerp_part(
    output: u32,
    part: u32,
    layer_info: LayerMeta,
    module_index: u32,
    slot0: u32,
    slot1: u32,
    entity_stride: u32,
    local_layer: u32,
    lane: u32,
) {
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let current0 = norm_values[channel];
        var shifted0 = current0;
        if (slot0 != INVALID_SLOT) {
            shifted0 = f32(state_value(
                module_index,
                slot0 * entity_stride + local_layer * STATE_LAYER_STRIDE + channel,
            ));
        }
        shared_set_scratch_value(
            output,
            0u,
            channel,
            current0 + (shifted0 - current0)
                * f32(model_weights[layer_info.lerp + part * CHANNELS + channel]),
        );

        let current1 = norm_values[SMALL_STRIDE + channel];
        var shifted1 = current1;
        if (slot1 != INVALID_SLOT) {
            shifted1 = f32(state_value(
                module_index,
                slot1 * entity_stride + local_layer * STATE_LAYER_STRIDE + channel,
            ));
        }
        shared_set_scratch_value(
            output,
            1u,
            channel,
            current1 + (shifted1 - current1)
                * f32(model_weights[layer_info.lerp + part * CHANNELS + channel]),
        );
        channel += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_store_scales(scale_offset: u32, lane: u32) {
    if (lane < HEADS) {
        extra[AUX_OFFSET + scale_offset + lane] = tmp1[lane];
        extra[EXTRA_STRIDE + AUX_OFFSET + scale_offset + lane] =
            tmp1[LARGE_STRIDE + lane];
    }
    workgroupBarrier();
}

fn shared_copy_v_to_v0(lane: u32) {
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        extra[V0_OFFSET + channel] = tmp0[V_OFFSET + channel];
        extra[EXTRA_STRIDE + V0_OFFSET + channel] =
            tmp0[LARGE_STRIDE + V_OFFSET + channel];
        channel += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_combine_v(lane: u32) {
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let projected0 = extra[OUT_OFFSET + channel];
        tmp0[V_OFFSET + channel] = projected0
            + (extra[V0_OFFSET + channel] - projected0) * tmp1[channel];
        let projected1 = extra[EXTRA_STRIDE + OUT_OFFSET + channel];
        tmp0[LARGE_STRIDE + V_OFFSET + channel] = projected1
            + (extra[EXTRA_STRIDE + V0_OFFSET + channel] - projected1)
                * tmp1[LARGE_STRIDE + channel];
        channel += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_finish_decay(lane: u32) {
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let absolute0 = D_OFFSET + channel;
        let value0 = -tmp1[absolute0];
        let softplus0 = max(value0, 0.0) + log(1.0 + exp(-abs(value0)));
        tmp1[absolute0] = exp(-exp(-0.5 - softplus0));

        let absolute1 = LARGE_STRIDE + D_OFFSET + channel;
        let value1 = -tmp1[absolute1];
        let softplus1 = max(value1, 0.0) + log(1.0 + exp(-abs(value1)));
        tmp1[absolute1] = exp(-exp(-0.5 - softplus1));
        channel += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_normalize_kv(lane: u32) {
    if (lane < SHARED_ROWS * HEADS) {
        let slot = lane / HEADS;
        let head = lane % HEADS;
        let row_large_base = slot * LARGE_STRIDE;
        let row_small_base = slot * SMALL_STRIDE;
        let row_extra_base = slot * EXTRA_STRIDE;
        let base = head * HEAD_SIZE;
        var k_sum = 0.0;
        var v_sum = 0.0;
        var index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let k = tmp0[row_large_base + K_OFFSET + base + index];
            let v = tmp0[row_large_base + V_OFFSET + base + index];
            k_sum += k * k;
            v_sum += v * v;
            index += 1u;
        }
        let k_norm = max(sqrt(k_sum), 0.000000059604645);
        let v_norm = max(sqrt(v_sum), 0.000000059604645);
        let k_scale = extra[row_extra_base + AUX_OFFSET + head];
        let v_scale = extra[row_extra_base + AUX_OFFSET + HEADS + head];
        index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let channel = base + index;
            let k_index = row_large_base + K_OFFSET + channel;
            let v_index = row_large_base + V_OFFSET + channel;
            let a_index = row_large_base + A_OFFSET + channel;
            let deformed = tmp0[k_index] / k_norm * k_scale;
            let a = tmp1[a_index];
            norm_values[row_small_base + channel] = deformed;
            tmp0[k_index] = deformed * a;
            tmp0[v_index] = tmp0[v_index] / v_norm * v_scale;
            index += 1u;
        }
    }
    workgroupBarrier();
}

fn shared_recurrent_output(
    module_index: u32,
    slot0: u32,
    slot1: u32,
    entity_stride: u32,
    local_layer: u32,
    lane: u32,
) {
    if (lane < SHARED_ROWS * HEADS) {
        let slot = lane / HEADS;
        let head = lane % HEADS;
        let row_large_base = slot * LARGE_STRIDE;
        let row_extra_base = slot * EXTRA_STRIDE;
        let base = head * HEAD_SIZE;
        var key_readout = 0.0;
        var index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let channel = base + index;
            key_readout += bounded_product2(
                tmp0[row_large_base + K_OFFSET + channel],
                tmp0[row_large_base + R_OFFSET + channel],
            );
            index += 1u;
        }
        extra[row_extra_base + AUX_OFFSET + head] = key_readout;
    }
    workgroupBarrier();

    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        var slot = 0u;
        loop {
            if (slot >= SHARED_ROWS) { break; }
            let state_slot = select(slot0, slot1, slot == 1u);
            let row_small_base = slot * SMALL_STRIDE;
            let row_large_base = slot * LARGE_STRIDE;
            let row_extra_base = slot * EXTRA_STRIDE;
            let head = channel / HEAD_SIZE;
            let row = channel % HEAD_SIZE;
            let vector_base = head * HEAD_SIZE;
            var output = 0.0;
            var column = 0u;
            loop {
                if (column >= HEAD_SIZE) { break; }
                let vector_channel = vector_base + column;
                let query = bounded_product2(
                    tmp1[row_large_base + D_OFFSET + vector_channel],
                    tmp0[row_large_base + R_OFFSET + vector_channel],
                ) - extra[row_extra_base + AUX_OFFSET + head]
                    * norm_values[row_small_base + vector_channel];
                if (state_slot != INVALID_SLOT) {
                    let state_offset = state_slot * entity_stride
                        + local_layer * STATE_LAYER_STRIDE
                        + TIME_STATE_OFFSET
                        + head * HEAD_SIZE * HEAD_SIZE
                        + row * HEAD_SIZE
                        + column;
                    output += f32(state_value(module_index, state_offset)) * query;
                }
                column += 1u;
            }
            extra[row_extra_base + OUT_OFFSET + channel] = output
                + tmp0[row_large_base + V_OFFSET + channel]
                    * extra[row_extra_base + AUX_OFFSET + head];
            slot += 1u;
        }
        channel += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_group_norm_and_bonus(layer_info: LayerMeta, lane: u32) {
    if (lane < SHARED_ROWS * HEADS) {
        let slot = lane / HEADS;
        let head = lane % HEADS;
        let row_large_base = slot * LARGE_STRIDE;
        let row_extra_base = slot * EXTRA_STRIDE;
        let base = head * HEAD_SIZE;
        var sum = 0.0;
        var index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            sum += extra[row_extra_base + OUT_OFFSET + base + index];
            index += 1u;
        }
        let mean = sum / 32.0;
        var variance = 0.0;
        index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let centered = extra[row_extra_base + OUT_OFFSET + base + index] - mean;
            variance += centered * centered;
            index += 1u;
        }
        let inverse_std = inverseSqrt(variance / 32.0 + 0.00064);
        var bonus = 0.0;
        index = 0u;
        loop {
            if (index >= HEAD_SIZE) { break; }
            let absolute_channel = base + index;
            let out_index = row_extra_base + OUT_OFFSET + absolute_channel;
            extra[out_index] = (extra[out_index] - mean) * inverse_std
                * normalization_weight(layer_info.gn_w + absolute_channel)
                + f32(model_weights[layer_info.gn_b + absolute_channel]);
            bonus += bounded_product3(
                tmp0[row_large_base + R_OFFSET + absolute_channel],
                f32(model_weights[layer_info.bonus + absolute_channel]),
                tmp0[row_large_base + K_OFFSET + absolute_channel],
            );
            index += 1u;
        }
        extra[row_extra_base + AUX_OFFSET + head] = bonus;
    }
    workgroupBarrier();

    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        var slot = 0u;
        loop {
            if (slot >= SHARED_ROWS) { break; }
            let row_large_base = slot * LARGE_STRIDE;
            let row_extra_base = slot * EXTRA_STRIDE;
            let head = channel / HEAD_SIZE;
            let out_index = row_extra_base + OUT_OFFSET + channel;
            extra[out_index] = extra[row_extra_base + G_OFFSET + channel]
                * (extra[out_index]
                    + extra[row_extra_base + AUX_OFFSET + head]
                        * tmp0[row_large_base + V_OFFSET + channel]);
            slot += 1u;
        }
        channel += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_add_time_residual(lane: u32) {
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        x[channel] += tmp0[channel];
        x[SMALL_STRIDE + channel] += tmp0[LARGE_STRIDE + channel];
        channel += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_channel_lerp(
    layer_info: LayerMeta,
    module_index: u32,
    slot0: u32,
    slot1: u32,
    entity_stride: u32,
    local_layer: u32,
    lane: u32,
) {
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        let current0 = norm_values[channel];
        var shifted0 = current0;
        if (slot0 != INVALID_SLOT) {
            shifted0 = f32(state_value(
                module_index,
                slot0 * entity_stride + local_layer * STATE_LAYER_STRIDE
                    + CHANNEL_STATE_OFFSET + channel,
            ));
        }
        tmp0[channel] = current0 + (shifted0 - current0)
            * f32(model_weights[layer_info.clerp + channel]);

        let current1 = norm_values[SMALL_STRIDE + channel];
        var shifted1 = current1;
        if (slot1 != INVALID_SLOT) {
            shifted1 = f32(state_value(
                module_index,
                slot1 * entity_stride + local_layer * STATE_LAYER_STRIDE
                    + CHANNEL_STATE_OFFSET + channel,
            ));
        }
        tmp0[LARGE_STRIDE + channel] = current1 + (shifted1 - current1)
            * f32(model_weights[layer_info.clerp + channel]);
        channel += SHARED_LANES;
    }
    workgroupBarrier();
}

fn shared_add_channel_residual(lane: u32) {
    var channel = lane;
    loop {
        if (channel >= CHANNELS) { break; }
        x[channel] += extra[OUT_OFFSET + channel];
        x[SMALL_STRIDE + channel] += extra[EXTRA_STRIDE + OUT_OFFSET + channel];
        channel += SHARED_LANES;
    }
    workgroupBarrier();
}

fn run_shared_layer(
    layer_info: LayerMeta,
    module_index: u32,
    slot0: u32,
    slot1: u32,
    entity_stride: u32,
    local_layer: u32,
    lane: u32,
) {
    shared_layer_norm(
        BUF_X,
        BUF_NORM,
        CHANNELS,
        layer_info.ln_w,
        layer_info.ln_b,
        lane,
    );

    shared_lerp_part(BUF_TMP0, 0u, layer_info, module_index, slot0, slot1, entity_stride, local_layer, lane);
    shared_linear_tmp0(BUF_R, CHANNELS, CHANNELS, layer_info.wr, INVALID_SLOT, lane);
    shared_lerp_part(BUF_TMP0, 1u, layer_info, module_index, slot0, slot1, entity_stride, local_layer, lane);
    shared_linear_tmp0(BUF_K, CHANNELS, CHANNELS, layer_info.wk, INVALID_SLOT, lane);
    shared_lerp_part(BUF_V, 2u, layer_info, module_index, slot0, slot1, entity_stride, local_layer, lane);
    shared_lerp_part(BUF_D, 3u, layer_info, module_index, slot0, slot1, entity_stride, local_layer, lane);
    shared_lerp_part(BUF_A, 4u, layer_info, module_index, slot0, slot1, entity_stride, local_layer, lane);
    shared_lerp_part(BUF_G, 5u, layer_info, module_index, slot0, slot1, entity_stride, local_layer, lane);
    shared_lerp_part(BUF_TMP0, 6u, layer_info, module_index, slot0, slot1, entity_stride, local_layer, lane);
    shared_linear_tmp0(BUF_TMP1, CHANNELS, HEADS, layer_info.ks_w, layer_info.ks_b, lane);
    shared_sigmoid_in_place(BUF_TMP1, HEADS, lane);
    shared_store_scales(0u, lane);
    shared_lerp_part(BUF_TMP0, 7u, layer_info, module_index, slot0, slot1, entity_stride, local_layer, lane);
    shared_linear_tmp0(BUF_TMP1, CHANNELS, HEADS, layer_info.vs_w, layer_info.vs_b, lane);
    shared_sigmoid_in_place(BUF_TMP1, HEADS, lane);
    shared_store_scales(HEADS, lane);

    if (local_layer == 0u) {
        shared_linear_v(BUF_TMP0, CHANNELS, CHANNELS, layer_info.wv, INVALID_SLOT, lane);
        shared_copy_values(BUF_TMP0, BUF_V, CHANNELS, lane);
        shared_copy_v_to_v0(lane);
    } else {
        shared_linear_v(BUF_TMP0, CHANNELS, 8u, layer_info.vl_a, INVALID_SLOT, lane);
        shared_linear_tmp0(BUF_TMP1, 8u, CHANNELS, layer_info.vl_b, layer_info.vl_bias, lane);
        shared_sigmoid_in_place(BUF_TMP1, CHANNELS, lane);
        shared_linear_v(BUF_OUT, CHANNELS, CHANNELS, layer_info.wv, INVALID_SLOT, lane);
        shared_combine_v(lane);
    }

    shared_linear_a(BUF_TMP0, CHANNELS, 16u, layer_info.al_a, INVALID_SLOT, lane);
    shared_linear_tmp0(BUF_TMP1, 16u, CHANNELS, layer_info.al_b, layer_info.al_bias, lane);
    shared_sigmoid_in_place(BUF_TMP1, CHANNELS, lane);
    shared_copy_values(BUF_TMP1, BUF_A, CHANNELS, lane);

    shared_linear_g(BUF_TMP0, CHANNELS, 16u, layer_info.gl_a, INVALID_SLOT, lane);
    shared_sigmoid_in_place(BUF_TMP0, 16u, lane);
    shared_linear_tmp0(BUF_G, 16u, CHANNELS, layer_info.gl_b, INVALID_SLOT, lane);

    shared_linear_d(BUF_TMP0, CHANNELS, 16u, layer_info.dl_a, INVALID_SLOT, lane);
    shared_tanh_in_place(BUF_TMP0, 16u, lane);
    shared_linear_tmp0(BUF_D, 16u, CHANNELS, layer_info.dl_b, layer_info.dl_bias, lane);
    shared_finish_decay(lane);

    shared_normalize_kv(lane);
    shared_recurrent_output(module_index, slot0, slot1, entity_stride, local_layer, lane);
    shared_group_norm_and_bonus(layer_info, lane);
    shared_linear_out(BUF_TMP0, CHANNELS, CHANNELS, layer_info.wo, INVALID_SLOT, lane);
    shared_add_time_residual(lane);

    shared_layer_norm(
        BUF_X,
        BUF_NORM,
        CHANNELS,
        layer_info.cln_w,
        layer_info.cln_b,
        lane,
    );
    shared_channel_lerp(layer_info, module_index, slot0, slot1, entity_stride, local_layer, lane);
    shared_linear_tmp0(BUF_TMP1, CHANNELS, layer_info.channel_dim, layer_info.cwk, INVALID_SLOT, lane);
    shared_relu_in_place(BUF_TMP1, layer_info.channel_dim, true, lane);
    shared_linear_tmp1(BUF_OUT, layer_info.channel_dim, CHANNELS, layer_info.cwv, INVALID_SLOT, lane);
    shared_add_channel_residual(lane);
}

@compute @workgroup_size(128)
fn predict_shared_rows(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let lane = local.x;
    let row0 = workgroup.x * SHARED_ROWS;
    let row1 = row0 + 1u;
    var index = lane;
    loop {
        if (index >= 92u) { break; }
        x[index] = f32(feature_rows[row0 * 92u + index]);
        x[SMALL_STRIDE + index] = f32(feature_rows[row1 * 92u + index]);
        index += SHARED_LANES;
    }
    workgroupBarrier();

    shared_linear_x(BUF_TMP0, 92u, 512u, /*__FEATURES_INPUT_W__*/u, /*__FEATURES_INPUT_B__*/u, lane);
    shared_silu_in_place(BUF_TMP0, 512u, lane);
    shared_layer_norm(BUF_TMP0, BUF_TMP1, 512u, /*__FEATURES_NORM_W__*/u, /*__FEATURES_NORM_B__*/u, lane);
    shared_linear_tmp1(BUF_X, 512u, CHANNELS, /*__FEATURES_OUTPUT_W__*/u, /*__FEATURES_OUTPUT_B__*/u, lane);
    shared_silu_in_place(BUF_X, CHANNELS, lane);

    let row0_slot0 = query_slots[row0 * 5u];
    let row0_slot1 = query_slots[row0 * 5u + 1u];
    let row0_slot2 = query_slots[row0 * 5u + 2u];
    let row0_slot3 = query_slots[row0 * 5u + 3u];
    let row0_slot4 = query_slots[row0 * 5u + 4u];
    let row1_slot0 = query_slots[row1 * 5u];
    let row1_slot1 = query_slots[row1 * 5u + 1u];
    let row1_slot2 = query_slots[row1 * 5u + 2u];
    let row1_slot3 = query_slots[row1 * 5u + 3u];
    let row1_slot4 = query_slots[row1 * 5u + 4u];

    var layer_index = 0u;
    loop {
        if (layer_index >= 16u) { break; }
        let layer_info = model_metadata.layers[layer_index];
        if (layer_info.local_layer == 0u) {
            shared_clear_v0(lane);
        }
        var state_slot0 = row0_slot4;
        var state_slot1 = row1_slot4;
        switch layer_info.module_index {
            case 0u: {
                state_slot0 = row0_slot0;
                state_slot1 = row1_slot0;
            }
            case 1u: {
                state_slot0 = row0_slot1;
                state_slot1 = row1_slot1;
            }
            case 2u: {
                state_slot0 = row0_slot2;
                state_slot1 = row1_slot2;
            }
            case 3u: {
                state_slot0 = row0_slot3;
                state_slot1 = row1_slot3;
            }
            default: {}
        }
        run_shared_layer(
            layer_info,
            layer_info.module_index,
            state_slot0,
            state_slot1,
            layer_info.entity_stride,
            layer_info.local_layer,
            lane,
        );
        layer_index += 1u;
    }

    shared_layer_norm(BUF_X, BUF_NORM, CHANNELS, /*__PREHEAD_W__*/u, /*__PREHEAD_B__*/u, lane);
    shared_linear_norm(BUF_TMP0, CHANNELS, 512u, /*__HEAD_P_W__*/u, /*__HEAD_P_B__*/u, lane);
    shared_relu_in_place(BUF_TMP0, 512u, false, lane);
    shared_linear_tmp0(BUF_TMP1, 512u, 4u, /*__P_LINEAR_W__*/u, /*__P_LINEAR_B__*/u, lane);
    if (lane == 0u) {
        var slot = 0u;
        loop {
            if (slot >= SHARED_ROWS) { break; }
            let row_large_base = slot * LARGE_STRIDE;
            var maximum = tmp1[row_large_base];
            maximum = max(maximum, tmp1[row_large_base + 1u]);
            maximum = max(maximum, tmp1[row_large_base + 2u]);
            maximum = max(maximum, tmp1[row_large_base + 3u]);
            let exp0 = exp(tmp1[row_large_base] - maximum);
            let exp1 = exp(tmp1[row_large_base + 1u] - maximum);
            let exp2 = exp(tmp1[row_large_base + 2u] - maximum);
            let exp3 = exp(tmp1[row_large_base + 3u] - maximum);
            probabilities[row0 + slot] = 1.0 - exp0 / (exp0 + exp1 + exp2 + exp3);
            slot += 1u;
        }
    }
}
