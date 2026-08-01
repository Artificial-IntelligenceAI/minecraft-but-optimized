use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2],
    color: [f32; 4],
}

/// A flat-colored rectangle in pixel space, origin top-left (matching window coordinates).
#[derive(Clone, Copy)]
pub struct UiQuad {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub color: [f32; 4],
}

/// Draws flat-colored rectangles in screen space — chat backdrop bars, the
/// text cursor — as a lightweight companion to the glyph-based text
/// renderer, which only draws glyphs.
pub struct QuadRenderer {
    pipeline: wgpu::RenderPipeline,
}

impl QuadRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui quad shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("quad.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui quad pipeline layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui quad pipeline"),
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
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self { pipeline }
    }

    /// Builds a one-frame vertex buffer for `quads` (pixel space, origin
    /// top-left) and draws them into `pass`.
    pub fn draw(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::RenderPass<'_>,
        screen_width: f32,
        screen_height: f32,
        quads: &[UiQuad],
    ) {
        if quads.is_empty() {
            return;
        }

        let mut vertices = Vec::with_capacity(quads.len() * 6);
        for q in quads {
            let (x0, y0) = to_ndc(q.x, q.y, screen_width, screen_height);
            let (x1, y1) = to_ndc(q.x + q.width, q.y + q.height, screen_width, screen_height);
            let top_left = QuadVertex {
                position: [x0, y0],
                color: q.color,
            };
            let top_right = QuadVertex {
                position: [x1, y0],
                color: q.color,
            };
            let bottom_left = QuadVertex {
                position: [x0, y1],
                color: q.color,
            };
            let bottom_right = QuadVertex {
                position: [x1, y1],
                color: q.color,
            };
            vertices.extend_from_slice(&[
                top_left,
                bottom_left,
                top_right,
                top_right,
                bottom_left,
                bottom_right,
            ]);
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ui quad vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }
}

fn to_ndc(x: f32, y: f32, width: f32, height: f32) -> (f32, f32) {
    (x / width * 2.0 - 1.0, 1.0 - y / height * 2.0)
}
