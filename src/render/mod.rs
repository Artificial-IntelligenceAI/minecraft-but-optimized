pub mod camera;
mod chat_view;
mod chunk_arena;
mod fps_view;
mod occlusion;
mod quad;
mod text;

use std::sync::Arc;

use rustc_hash::FxHashMap;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::chat::Chat;
use crate::world::ChunkPos;
use crate::world::meshing::{ChunkMesh, ChunkVertex};
use camera::CameraUniform;
use chunk_arena::GpuArena;
use occlusion::OcclusionCuller;
use quad::QuadRenderer;
use text::{UiText, UiTextLine};

pub enum RenderOutcome {
    Ok,
    Skip,
    Reconfigure,
    Fatal,
}

/// Where one chunk's geometry lives within the shared vertex/index/origin
/// arenas (see `chunk_arena`), in the units each arena's draw parameter
/// needs: byte offsets for `alloc`/`free`, but element offsets for
/// `draw_indexed`'s `base_vertex`/index-range/instance-range.
struct ChunkSlot {
    vertex_offset: u64,
    vertex_count: u32,
    index_offset: u64,
    index_count: u32,
    origin_offset: u64,
}

const CHUNK_VERTEX_SIZE: u64 = std::mem::size_of::<ChunkVertex>() as u64;
const CHUNK_INDEX_SIZE: u64 = std::mem::size_of::<u32>() as u64;
const CHUNK_ORIGIN_SIZE: u64 = std::mem::size_of::<[f32; 3]>() as u64;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    vertex_arena: GpuArena,
    index_arena: GpuArena,
    origin_arena: GpuArena,
    chunk_slots: FxHashMap<ChunkPos, ChunkSlot>,
    occlusion: OcclusionCuller,
    quad_renderer: QuadRenderer,
    ui_text: UiText,
    pub window: Arc<Window>,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("failed to find a suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                ..Default::default()
            })
            .await
            .expect("failed to request device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            // Vsync off / uncapped by default; actual FPS capping (if any) is
            // done in software via frame pacing (see `fps::FpsCounter`), not
            // by switching present modes, since that's the only way to hit
            // arbitrary cap values instead of just "whatever the display's
            // refresh rate is".
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_view = create_depth_view(&device, config.width, config.height);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera uniform"),
            contents: bytemuck::bytes_of(&CameraUniform::from_camera(&camera::Camera::new(
                glam::Vec3::ZERO,
                1.0,
            ))),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chunk shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("chunk pipeline layout"),
            bind_group_layouts: &[Some(&camera_bind_group_layout)],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: CHUNK_VERTEX_SIZE as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Unorm8x4,
                },
            ],
        };

        // Per-chunk world-space origin, stepped once per instance rather
        // than baked into every vertex — see `chunk_arena` and
        // `world::meshing::ChunkVertex`'s doc comment.
        let origin_layout = wgpu::VertexBufferLayout {
            array_stride: CHUNK_ORIGIN_SIZE as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("chunk pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(vertex_layout), Some(origin_layout)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let occlusion = OcclusionCuller::new(&device, &camera_bind_group_layout, DEPTH_FORMAT);

        // 1 MiB starting capacity for vertices/indices (grows on demand, see
        // `GpuArena::grow`); origins are tiny (12 bytes/chunk) so a few
        // thousand chunks' worth costs nothing to start with generously.
        let vertex_arena = GpuArena::new(
            &device,
            "chunk vertex arena",
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            1 << 20,
        );
        let index_arena = GpuArena::new(
            &device,
            "chunk index arena",
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            1 << 20,
        );
        let origin_arena = GpuArena::new(
            &device,
            "chunk origin arena",
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            CHUNK_ORIGIN_SIZE * 4096,
        );

        let quad_renderer = QuadRenderer::new(&device, config.format);
        let ui_text = UiText::new(&device, &queue, config.format);

        Self {
            surface,
            device,
            queue,
            config,
            depth_view,
            pipeline,
            camera_buffer,
            camera_bind_group,
            vertex_arena,
            index_arena,
            origin_arena,
            chunk_slots: FxHashMap::default(),
            occlusion,
            quad_renderer,
            ui_text,
            window,
        }
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.config.width as f32 / self.config.height.max(1) as f32
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth_view(&self.device, width, height);
    }

    /// Toggles vsync at runtime (see `/settings fps sync`) by reconfiguring
    /// the surface's present mode.
    pub fn set_vsync(&mut self, enabled: bool) {
        self.config.present_mode = if enabled {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        self.surface.configure(&self.device, &self.config);
    }

    /// Uploads/replaces the GPU mesh for a chunk, or drops it if the new mesh is empty.
    pub fn upsert_chunk_mesh(&mut self, chunk_pos: ChunkPos, mesh: &ChunkMesh) {
        self.remove_chunk_mesh(chunk_pos);
        if mesh.is_empty() {
            return;
        }

        let vertex_bytes: &[u8] = bytemuck::cast_slice(&mesh.vertices);
        let vertex_offset =
            self.vertex_arena
                .alloc(&self.device, &self.queue, vertex_bytes.len() as u64);
        self.queue
            .write_buffer(self.vertex_arena.buffer(), vertex_offset, vertex_bytes);

        let index_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);
        let index_offset =
            self.index_arena
                .alloc(&self.device, &self.queue, index_bytes.len() as u64);
        self.queue
            .write_buffer(self.index_arena.buffer(), index_offset, index_bytes);

        let origin = crate::world::chunk_origin(chunk_pos).as_vec3().to_array();
        let origin_offset =
            self.origin_arena
                .alloc(&self.device, &self.queue, CHUNK_ORIGIN_SIZE);
        self.queue.write_buffer(
            self.origin_arena.buffer(),
            origin_offset,
            bytemuck::bytes_of(&origin),
        );

        self.chunk_slots.insert(
            chunk_pos,
            ChunkSlot {
                vertex_offset,
                vertex_count: mesh.vertices.len() as u32,
                index_offset,
                index_count: mesh.indices.len() as u32,
                origin_offset,
            },
        );
    }

    pub fn remove_chunk_mesh(&mut self, chunk_pos: ChunkPos) {
        if let Some(slot) = self.chunk_slots.remove(&chunk_pos) {
            self.vertex_arena.free(
                slot.vertex_offset,
                slot.vertex_count as u64 * CHUNK_VERTEX_SIZE,
            );
            self.index_arena.free(
                slot.index_offset,
                slot.index_count as u64 * CHUNK_INDEX_SIZE,
            );
            self.origin_arena
                .free(slot.origin_offset, CHUNK_ORIGIN_SIZE);
        }
        self.occlusion.forget(chunk_pos);
    }

    pub fn render(
        &mut self,
        camera: &camera::Camera,
        chat: &Chat,
        fps: Option<u32>,
    ) -> RenderOutcome {
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform::from_camera(camera)),
        );

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                self.surface.configure(&self.device, &self.config);
                frame
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return RenderOutcome::Skip;
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                return RenderOutcome::Reconfigure;
            }
            wgpu::CurrentSurfaceTexture::Validation => return RenderOutcome::Fatal,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Non-blocking: applies whichever previous occlusion readback (if
        // any) has finished mapping since last frame.
        self.occlusion.poll(&self.device);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        let frustum = camera::Frustum::from_view_proj(camera.view_proj());
        let mut frustum_visible: Vec<ChunkPos> = self
            .chunk_slots
            .keys()
            .copied()
            .filter(|&pos| {
                let (min, max) = chunk_aabb(pos);
                frustum.intersects_aabb(min, max)
            })
            .collect();

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("chunk pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.53,
                            g: 0.72,
                            b: 0.93,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            // Bound once for every chunk this frame — chunks are shared
            // arena allocations now, not individually-owned buffers, so
            // there's nothing to rebind per draw (see `chunk_arena`).
            render_pass.set_vertex_buffer(0, self.vertex_arena.buffer().slice(..));
            render_pass.set_vertex_buffer(1, self.origin_arena.buffer().slice(..));
            render_pass
                .set_index_buffer(self.index_arena.buffer().slice(..), wgpu::IndexFormat::Uint32);

            for &chunk_pos in &frustum_visible {
                if !self.occlusion.is_visible(chunk_pos) {
                    continue;
                }
                let slot = &self.chunk_slots[&chunk_pos];
                let base_vertex = (slot.vertex_offset / CHUNK_VERTEX_SIZE) as i32;
                let index_start = (slot.index_offset / CHUNK_INDEX_SIZE) as u32;
                let instance = (slot.origin_offset / CHUNK_ORIGIN_SIZE) as u32;
                render_pass.draw_indexed(
                    index_start..index_start + slot.index_count,
                    base_vertex,
                    instance..instance + 1,
                );
            }
        }

        // Re-test frustum-visible chunks against the depth buffer we just
        // drew, so `self.occlusion` reflects this frame's geometry for use
        // in a future frame's draw decision (see `OcclusionCuller`). Only
        // when the previous readback has already been applied — otherwise
        // this frame just keeps drawing by last-known visibility.
        if self.occlusion.ready_for_new_queries() {
            frustum_visible.truncate(self.occlusion.capacity());
            self.occlusion.record_queries(
                &self.device,
                &self.queue,
                &mut encoder,
                &self.depth_view,
                &self.camera_bind_group,
                frustum_visible,
            );
        }

        let draw_data = chat_view::build(chat, self.config.width as f32, self.config.height as f32);
        let fps_data = fps.map(fps_view::build);

        let mut quads = draw_data.quads;
        if let Some(fps_data) = &fps_data {
            quads.push(fps_data.quad);
        }

        let mut text_lines = draw_data.text_lines;
        if let Some(ghost) = &draw_data.ghost {
            text_lines.push(UiTextLine {
                text: &ghost.text,
                x: ghost.x,
                y: ghost.y,
                color: [255, 255, 85, 220],
                font_size: ghost.font_size,
                max_width: ghost.max_width,
            });
        }
        if let Some(fps_data) = &fps_data {
            text_lines.push(fps_view::text_line(fps_data));
        }

        self.ui_text.prepare(
            &self.device,
            &self.queue,
            self.config.width,
            self.config.height,
            &text_lines,
        );

        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui overlay pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            self.quad_renderer.draw(
                &self.device,
                &mut ui_pass,
                self.config.width as f32,
                self.config.height as f32,
                &quads,
            );
            self.ui_text.render(&mut ui_pass);
        }
        self.ui_text.trim();

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);

        // Only valid now that the copy recorded in `record_queries` has
        // actually been submitted (see `OcclusionCuller::begin_readback`).
        self.occlusion.begin_readback();

        RenderOutcome::Ok
    }
}

/// World-space (min, max) corners of a chunk's cube, used for culling.
fn chunk_aabb(pos: ChunkPos) -> (glam::Vec3, glam::Vec3) {
    let origin = crate::world::chunk_origin(pos).as_vec3();
    (
        origin,
        origin + glam::Vec3::splat(crate::world::chunk::CHUNK_SIZE as f32),
    )
}

fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
