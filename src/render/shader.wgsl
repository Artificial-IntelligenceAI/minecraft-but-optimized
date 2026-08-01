struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

// Local-space chunk vertex position (0..=CHUNK_SIZE per axis, 6 bits each)
// plus a face-normal index (3 bits), packed into a single u32. World-space
// position is `instance.origin + local`. See `world::meshing::pack_vertex`.
struct VertexInput {
    @location(0) packed: u32,
    @location(1) color: vec4<f32>,
};

struct InstanceInput {
    @location(2) origin: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

const FACE_NORMALS = array<vec3<f32>, 6>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, -1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, -1.0),
);

@vertex
fn vs_main(in: VertexInput, instance: InstanceInput) -> VertexOutput {
    let local = vec3<f32>(
        f32(in.packed & 0x3Fu),
        f32((in.packed >> 6u) & 0x3Fu),
        f32((in.packed >> 12u) & 0x3Fu),
    );
    let normal_index = (in.packed >> 18u) & 0x7u;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(instance.origin + local, 1.0);
    out.normal = FACE_NORMALS[normal_index];
    out.color = in.color.rgb;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.4, 0.85, 0.3));
    let ambient = 0.35;
    let diffuse = max(dot(normalize(in.normal), light_dir), 0.0);
    let lit = in.color * (ambient + diffuse * (1.0 - ambient));
    return vec4<f32>(lit, 1.0);
}
