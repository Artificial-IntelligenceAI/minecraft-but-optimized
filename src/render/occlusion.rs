use std::sync::mpsc::Receiver;

use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;

use crate::world::ChunkPos;
use crate::world::chunk::CHUNK_SIZE;

/// Upper bound on chunks tested for occlusion in a single frame. Chunks
/// beyond this (only reachable at very large render distances) skip testing
/// and are just drawn unconditionally — correct, just not optimized.
const MAX_OCCLUSION_QUERIES: u32 = 4096;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoxVertex {
    position: [f32; 3],
}

const CUBE_CORNERS: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

/// 12 triangles. Winding is deliberately not relied on for correctness —
/// the occlusion pipeline disables backface culling (see `OcclusionCuller::new`)
/// so a mistake here can't cause a box to wrongly test as invisible.
const CUBE_INDICES: [u16; 36] = [
    0, 1, 2, 2, 3, 0, // -Z
    4, 6, 5, 6, 4, 7, // +Z
    0, 4, 5, 5, 1, 0, // -Y
    3, 2, 6, 6, 7, 3, // +Y
    0, 3, 7, 7, 4, 0, // -X
    1, 5, 6, 6, 2, 1, // +X
];

/// A GPU buffer whose entire contents are replaced every time it's written
/// (unlike `chunk_arena`'s multi-tenant allocator, there's nothing here
/// worth preserving across writes), so growing it never needs to copy old
/// data — it just recreates at the new size and the next `write` fills it.
struct ScratchBuffer {
    buffer: wgpu::Buffer,
    label: &'static str,
    usage: wgpu::BufferUsages,
    capacity: u64,
}

impl ScratchBuffer {
    fn new(device: &wgpu::Device, label: &'static str, usage: wgpu::BufferUsages) -> Self {
        let capacity = 4096;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity,
            usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            label,
            usage,
            capacity,
        }
    }

    /// Ensures the buffer is at least big enough for `data`, then uploads it.
    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) {
        let needed = data.len() as u64;
        if needed > self.capacity {
            self.capacity = needed.next_power_of_two();
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: self.capacity,
                usage: self.usage,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.buffer, 0, data);
    }

    fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// A readback whose resolve/copy has been recorded but not yet submitted —
/// `map_async` can only be called once that submission actually happens
/// (see `begin_readback`), since a buffer can't be mapped for CPU reads
/// while GPU commands writing to it are still unsubmitted.
struct AwaitingSubmit {
    tested: Vec<ChunkPos>,
    buffer_index: usize,
}

/// A readback that's been submitted to the GPU and had `map_async` called,
/// but not yet confirmed mapped.
struct Pending {
    tested: Vec<ChunkPos>,
    buffer_index: usize,
    receiver: Receiver<Result<(), wgpu::BufferAsyncError>>,
}

/// GPU hardware occlusion queries for chunk bounding boxes, with results
/// applied asynchronously: `poll` never blocks the CPU on the GPU, so a
/// chunk's visibility can lag by more than one frame under heavy load
/// instead of stalling every frame waiting for the query readback (the
/// tradeoff a synchronous version would force). Two readback buffers are
/// ping-ponged so a new query pass never has to wait on the previous one's
/// map still being read.
pub struct OcclusionCuller {
    pipeline: wgpu::RenderPipeline,
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffers: [wgpu::Buffer; 2],
    /// Box-proxy geometry rebuilt (but not reallocated) every call to
    /// `record_queries` — was `create_buffer_init` from scratch every frame.
    box_vertex_buffer: ScratchBuffer,
    box_index_buffer: ScratchBuffer,
    /// CPU-side staging for the same box-proxy geometry, cleared and
    /// refilled each call instead of collecting into fresh `Vec`s.
    box_vertices: Vec<BoxVertex>,
    box_indices: Vec<u32>,
    /// Which of `readback_buffers` the *next* `record_queries` call should
    /// target, alternated independently of `pending` (which is always
    /// `None` by the time a new call is allowed — see
    /// `ready_for_new_queries`) so consecutive passes never reuse the same
    /// buffer while its previous map could theoretically still be settling.
    next_buffer: usize,
    awaiting_submit: Option<AwaitingSubmit>,
    pending: Option<Pending>,
    visible: FxHashMap<ChunkPos, bool>,
}

impl OcclusionCuller {
    pub fn new(
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("occlusion shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("occlusion.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("occlusion pipeline layout"),
            bind_group_layouts: &[Some(camera_bind_group_layout)],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BoxVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("occlusion pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(vertex_layout)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Deliberately not culled: we only care whether *any* pixel of
                // the box would pass the depth test, so a winding mistake must
                // never be able to make a truly-visible chunk test as hidden.
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("occlusion query set"),
            ty: wgpu::QueryType::Occlusion,
            count: MAX_OCCLUSION_QUERIES,
        });

        let buffer_size = (MAX_OCCLUSION_QUERIES as u64) * 8;
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion resolve buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let make_readback_buffer = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("occlusion readback buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let box_vertex_buffer = ScratchBuffer::new(
            device,
            "occlusion box vertex buffer",
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );
        let box_index_buffer = ScratchBuffer::new(
            device,
            "occlusion box index buffer",
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        );

        Self {
            pipeline,
            query_set,
            resolve_buffer,
            readback_buffers: [make_readback_buffer(), make_readback_buffer()],
            box_vertex_buffer,
            box_index_buffer,
            box_vertices: Vec::new(),
            box_indices: Vec::new(),
            next_buffer: 0,
            awaiting_submit: None,
            pending: None,
            visible: FxHashMap::default(),
        }
    }

    pub fn capacity(&self) -> usize {
        MAX_OCCLUSION_QUERIES as usize
    }

    /// Whether `pos` is currently believed visible. Untested chunks (just
    /// loaded, or beyond `capacity`) default to visible.
    pub fn is_visible(&self, pos: ChunkPos) -> bool {
        self.visible.get(&pos).copied().unwrap_or(true)
    }

    /// Drops cached visibility for a chunk that's no longer loaded, so the
    /// map doesn't grow forever as chunks stream in and out.
    pub fn forget(&mut self, pos: ChunkPos) {
        self.visible.remove(&pos);
    }

    /// A new query pass can only be recorded once the previous one's
    /// readback buffer is free again (i.e. its results have been consumed
    /// by `poll`, or it was never used). With two ping-ponged buffers this
    /// is only ever false when the GPU has fallen more than a frame behind.
    pub fn ready_for_new_queries(&self) -> bool {
        self.pending.is_none() && self.awaiting_submit.is_none()
    }

    /// Non-blocking: checks whether the in-flight readback (if any) has
    /// finished mapping yet, and if so applies its results to `visible` and
    /// frees it up for reuse. Call once per frame, before deciding whether
    /// to record new queries.
    pub fn poll(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);

        let Some(pending) = &self.pending else {
            return;
        };
        let Ok(result) = pending.receiver.try_recv() else {
            return;
        };
        result.expect("failed to map occlusion readback buffer");

        let size = (pending.tested.len() as u64) * 8;
        let buffer = &self.readback_buffers[pending.buffer_index];
        let slice = buffer.slice(0..size);
        {
            let view = slice.get_mapped_range().expect("occlusion buffer not mapped");
            let results: &[u64] = bytemuck::cast_slice(&view);
            for (&pos, &passed) in pending.tested.iter().zip(results) {
                self.visible.insert(pos, passed != 0);
            }
        }
        buffer.unmap();
        self.pending = None;
    }

    /// Starts the CPU-side map of the readback buffer that `record_queries`
    /// just wrote into. Must be called only after the encoder containing
    /// that write has actually been submitted — mapping a buffer that a
    /// not-yet-submitted command still targets is a validation error.
    pub fn begin_readback(&mut self) {
        let Some(awaiting) = self.awaiting_submit.take() else {
            return;
        };
        let size = (awaiting.tested.len() as u64) * 8;
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback_buffers[awaiting.buffer_index]
            .slice(0..size)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.pending = Some(Pending {
            tested: awaiting.tested,
            buffer_index: awaiting.buffer_index,
            receiver: rx,
        });
    }

    /// Draws a bounding-box proxy per chunk in `tested`, each wrapped in its
    /// own occlusion query, against the depth buffer as it stands right now
    /// — i.e. call this *after* the real geometry pass for the frame, so
    /// it's testing against what's actually already drawn. Records the
    /// resolve into a readback buffer; call `begin_readback` once this
    /// encoder has actually been submitted to start the async map, then
    /// `poll` on later frames to pick up the result.
    ///
    /// Only call this when `ready_for_new_queries()` is true.
    pub fn record_queries(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        camera_bind_group: &wgpu::BindGroup,
        tested: Vec<ChunkPos>,
    ) {
        debug_assert!(self.pending.is_none() && self.awaiting_submit.is_none());
        if tested.is_empty() {
            return;
        }

        self.box_vertices.clear();
        self.box_indices.clear();
        for &pos in &tested {
            let (min, _) = super::chunk_aabb(pos);
            let base = self.box_vertices.len() as u32;
            for corner in CUBE_CORNERS {
                self.box_vertices.push(BoxVertex {
                    position: [
                        min.x + corner[0] * CHUNK_SIZE as f32,
                        min.y + corner[1] * CHUNK_SIZE as f32,
                        min.z + corner[2] * CHUNK_SIZE as f32,
                    ],
                });
            }
            self.box_indices
                .extend(CUBE_INDICES.iter().map(|&i| base + i as u32));
        }

        let vertex_bytes: &[u8] = bytemuck::cast_slice(&self.box_vertices);
        let index_bytes: &[u8] = bytemuck::cast_slice(&self.box_indices);
        self.box_vertex_buffer.write(device, queue, vertex_bytes);
        self.box_index_buffer.write(device, queue, index_bytes);
        let vertex_buffer = self
            .box_vertex_buffer
            .buffer()
            .slice(0..vertex_bytes.len() as u64);
        let index_buffer = self
            .box_index_buffer
            .buffer()
            .slice(0..index_bytes.len() as u64);

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("occlusion query pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: Some(&self.query_set),
                multiview_mask: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, camera_bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer);
            pass.set_index_buffer(index_buffer, wgpu::IndexFormat::Uint32);

            for i in 0..tested.len() as u32 {
                let base = i * CUBE_INDICES.len() as u32;
                pass.begin_occlusion_query(i);
                pass.draw_indexed(base..base + CUBE_INDICES.len() as u32, 0, 0..1);
                pass.end_occlusion_query();
            }
        }

        let size = (tested.len() as u64) * 8;
        encoder.resolve_query_set(&self.query_set, 0..tested.len() as u32, &self.resolve_buffer, 0);

        let buffer_index = self.next_buffer;
        self.next_buffer = 1 - self.next_buffer;

        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffers[buffer_index],
            0,
            size,
        );

        self.awaiting_submit = Some(AwaitingSubmit {
            tested,
            buffer_index,
        });
    }
}
