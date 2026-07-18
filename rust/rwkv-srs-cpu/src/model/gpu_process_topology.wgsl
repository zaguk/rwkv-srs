const CHANNELS: u32 = 128u;

struct TopologyParams {
    row_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read_write> original_rows: array<f32>;
@group(0) @binding(1)
var<storage, read_write> sequence_rows: array<f32>;
@group(0) @binding(2)
var<storage, read> sequence_to_original: array<u32>;
@group(0) @binding(3)
var<uniform> params: TopologyParams;

@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    let value_count = params.row_count * CHANNELS;
    if (index >= value_count) { return; }
    let sequence_row = index / CHANNELS;
    let channel = index % CHANNELS;
    let original_row = sequence_to_original[sequence_row];
    sequence_rows[index] = original_rows[original_row * CHANNELS + channel];
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    let value_count = params.row_count * CHANNELS;
    if (index >= value_count) { return; }
    let sequence_row = index / CHANNELS;
    let channel = index % CHANNELS;
    let original_row = sequence_to_original[sequence_row];
    original_rows[original_row * CHANNELS + channel] = sequence_rows[index];
}
