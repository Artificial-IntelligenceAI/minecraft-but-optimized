use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

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

/// GPU hardware occlusion queries for chunk bounding boxes, with one frame
/// of latency: a chunk's visibility this frame reflects a test run against
/// *last* frame's depth buffer, since testing against "what's already drawn
/// this frame" necessarily has to happen after the real geometry pass.
/// Chunks default to visible until tested, so nothing waits a frame to
/// appear when it first loads.
pub struct OcclusionCuller {
    pipeline: wgpu::RenderPipeline,
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    visible: HashMap<ChunkPos, bool>,
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
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion readback buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            query_set,
            resolve_buffer,
            readback_buffer,
            visible: HashMap::new(),
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

    /// Draws a bounding-box proxy per chunk in `tested`, each wrapped in its
    /// own occlusion query, against the depth buffer as it stands right now
    /// — i.e. call this *after* the real geometry pass for the frame, so
    /// it's testing against what's actually already drawn. Schedules the
    /// results to be resolved into a readable buffer; call `read_results`
    /// with the same slice once the encoder has been submitted.
    pub fn record_queries(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        camera_bind_group: &wgpu::BindGroup,
        tested: &[ChunkPos],
    ) {
        if tested.is_empty() {
            return;
        }

        let mut vertices = Vec::with_capacity(tested.len() * CUBE_CORNERS.len());
        let mut indices = Vec::with_capacity(tested.len() * CUBE_INDICES.len());
        for &pos in tested {
            let (min, _) = super::chunk_aabb(pos);
            let base = vertices.len() as u32;
            for corner in CUBE_CORNERS {
                vertices.push(BoxVertex {
                    position: [
                        min.x + corner[0] * CHUNK_SIZE as f32,
                        min.y + corner[1] * CHUNK_SIZE as f32,
                        min.z + corner[2] * CHUNK_SIZE as f32,
                    ],
                });
            }
            indices.extend(CUBE_INDICES.iter().map(|&i| base + i as u32));
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("occlusion box vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("occlusion box index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

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
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            for i in 0..tested.len() as u32 {
                let base = i * CUBE_INDICES.len() as u32;
                pass.begin_occlusion_query(i);
                pass.draw_indexed(base..base + CUBE_INDICES.len() as u32, 0, 0..1);
                pass.end_occlusion_query();
            }
        }

        encoder.resolve_query_set(
            &self.query_set,
            0..tested.len() as u32,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            (tested.len() as u64) * 8,
        );
    }

    /// Blocks until the queries from the last `record_queries` call (which
    /// must already have been submitted) resolve, then updates visibility
    /// for each chunk in `tested`.
    ///
    /// This stalls the CPU on the GPU once a frame — an accepted tradeoff
    /// for a first version that prioritizes obvious correctness over a
    /// fully async double-buffered readback, which would trade the stall
    /// for an extra frame of latency instead.
    pub fn read_results(&mut self, device: &wgpu::Device, tested: &[ChunkPos]) {
        if tested.is_empty() {
            return;
        }

        let size = (tested.len() as u64) * 8;
        let slice = self.readback_buffer.slice(0..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv()
            .expect("occlusion readback callback never fired")
            .expect("failed to map occlusion readback buffer");

        {
            let view = slice.get_mapped_range().expect("occlusion buffer not mapped");
            let results: &[u64] = bytemuck::cast_slice(&view);
            for (&pos, &passed) in tested.iter().zip(results) {
                self.visible.insert(pos, passed != 0);
            }
        }
        self.readback_buffer.unmap();
    }
}
