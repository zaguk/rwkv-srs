const STATE_SOURCE_RESIDENT: u32 = 1u;

struct StateSource {
    entity: u32,
    flags: u32,
    _pad0: u32,
    _pad1: u32,
};

struct StateTransferParams {
    state_stride: u32,
    current_entities: u32,
    evicted_entities: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> resident_state: array<f32>;
@group(0) @binding(1)
var<storage, read> host_state: array<f32>;
@group(0) @binding(2)
var<storage, read> state_sources: array<StateSource>;
@group(0) @binding(3)
var<storage, read_write> current_state: array<f32>;
@group(0) @binding(4)
var<storage, read> evicted_sources: array<u32>;
@group(0) @binding(5)
var<storage, read_write> evicted_state: array<f32>;
@group(0) @binding(6)
var<uniform> params: StateTransferParams;

@compute @workgroup_size(256)
fn gather_current_state(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let entity = workgroup.x;
    if (entity >= params.current_entities) { return; }
    let source = state_sources[entity];
    let destination_base = entity * params.state_stride;
    let source_base = source.entity * params.state_stride;
    var index = local.x;
    loop {
        if (index >= params.state_stride) { break; }
        var value: f32;
        if ((source.flags & STATE_SOURCE_RESIDENT) != 0u) {
            value = resident_state[source_base + index];
        } else {
            value = host_state[source_base + index];
        }
        current_state[destination_base + index] = value;
        index += 256u;
    }
}

@compute @workgroup_size(256)
fn gather_evicted_state(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let entity = workgroup.x;
    if (entity >= params.evicted_entities) { return; }
    let source_base = evicted_sources[entity] * params.state_stride;
    let destination_base = entity * params.state_stride;
    var index = local.x;
    loop {
        if (index >= params.state_stride) { break; }
        evicted_state[destination_base + index] = resident_state[source_base + index];
        index += 256u;
    }
}
