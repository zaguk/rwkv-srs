struct ArenaTransfer {
    arena_entity: u32,
    plan_entity: u32,
    _pad0: u32,
    _pad1: u32,
};

struct ArenaTransferParams {
    state_stride: u32,
    transfer_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<storage, read> source_state: array<f32>;
@group(0) @binding(1)
var<storage, read_write> target_state: array<f32>;
@group(0) @binding(2)
var<storage, read> transfers: array<ArenaTransfer>;
@group(0) @binding(3)
var<uniform> params: ArenaTransferParams;

@compute @workgroup_size(256)
fn gather_arena_state(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let transfer_index = workgroup.y;
    if (transfer_index >= params.transfer_count) { return; }
    let transfer = transfers[transfer_index];
    let index = workgroup.x * 256u + local.x;
    if (index >= params.state_stride) { return; }
    target_state[transfer.plan_entity * params.state_stride + index] =
        source_state[transfer.arena_entity * params.state_stride + index];
}

@compute @workgroup_size(256)
fn scatter_arena_state(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let transfer_index = workgroup.y;
    if (transfer_index >= params.transfer_count) { return; }
    let transfer = transfers[transfer_index];
    let index = workgroup.x * 256u + local.x;
    if (index >= params.state_stride) { return; }
    target_state[transfer.arena_entity * params.state_stride + index] =
        source_state[transfer.plan_entity * params.state_stride + index];
}
