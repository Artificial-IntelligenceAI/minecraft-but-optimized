use bytemuck::{Pod, Zeroable};
use glam::IVec3;

use super::block::{self, BlockId};
use super::chunk::CHUNK_SIZE_I32;
use super::{ChunkPos, World, chunk_origin};

/// A packed 8-byte chunk vertex (down from 36 bytes as separate f32 fields).
/// `packed` holds local-space position (0..=CHUNK_SIZE, 6 bits per axis) plus
/// a face-normal index (3 bits, see `shader.wgsl`'s `FACE_NORMALS`), since
/// greedy-meshed chunk faces are always axis-aligned. World-space position is
/// reconstructed in the vertex shader from this plus a per-chunk origin,
/// supplied separately as an instanced vertex attribute (see
/// `render::chunk_arena`) rather than baked into every vertex.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ChunkVertex {
    pub packed: u32,
    pub color: [u8; 4],
}

fn pack_vertex(local: [i32; 3], normal_index: u32) -> u32 {
    debug_assert!(local.iter().all(|&c| (0..=CHUNK_SIZE_I32).contains(&c)));
    (local[0] as u32) | ((local[1] as u32) << 6) | ((local[2] as u32) << 12) | (normal_index << 18)
}

pub struct ChunkMesh {
    pub vertices: Vec<ChunkVertex>,
    pub indices: Vec<u32>,
}

impl ChunkMesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct MaskEntry {
    block: BlockId,
    /// True when the face normal points in the negative direction of the sweep axis.
    backface: bool,
}

/// Greedy-meshes a single chunk against the live world (so faces against
/// already-loaded neighbor chunks are culled correctly, not just faces
/// against air within the chunk itself).
pub fn mesh_chunk(world: &World, chunk_pos: ChunkPos) -> ChunkMesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let origin = chunk_origin(chunk_pos);
    let dims = [CHUNK_SIZE_I32; 3];

    // Of the three axes checked per voxel face, only the swept axis `d` ever
    // steps outside this chunk (by exactly one voxel, to see the neighbor
    // across the boundary) — `u`/`v` always stay in `0..CHUNK_SIZE`. So the
    // overwhelming majority of lookups are against this same chunk; fetch it
    // once and index straight into it instead of paying `World::get_block`'s
    // div_euclid/rem_euclid-and-hashmap-lookup cost for every voxel.
    let this_chunk = world.chunk(chunk_pos);
    let in_bounds = |c: i32| (0..CHUNK_SIZE_I32).contains(&c);

    let solid_block_at = |local: [i32; 3]| -> Option<BlockId> {
        let id = if local.iter().copied().all(in_bounds) {
            this_chunk.map_or(block::AIR, |chunk| {
                chunk.get(local[0] as usize, local[1] as usize, local[2] as usize)
            })
        } else {
            let world_pos = origin + IVec3::new(local[0], local[1], local[2]);
            world.get_block(world_pos)
        };
        if block::is_solid(id) { Some(id) } else { None }
    };

    for d in 0..3usize {
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;

        let mut x = [0i32; 3];

        let mask_w = dims[u] as usize;
        let mask_h = dims[v] as usize;
        let mut mask: Vec<Option<MaskEntry>> = vec![None; mask_w * mask_h];

        x[d] = -1;
        while x[d] < dims[d] {
            let mut n = 0;
            x[v] = 0;
            while x[v] < dims[v] {
                x[u] = 0;
                while x[u] < dims[u] {
                    // `solid_block_at` resolves arbitrary world coordinates via the world's
                    // chunk lookup, so no special-casing is needed at chunk boundaries: `a`/`b`
                    // transparently query the neighbor chunk when `x`/`x+q` fall outside this one.
                    let a = solid_block_at(x);
                    let mut xq = x;
                    xq[d] += 1;
                    let b = solid_block_at(xq);

                    mask[n] = match (a, b) {
                        (Some(_), Some(_)) => None,
                        (None, None) => None,
                        (Some(block), None) => Some(MaskEntry {
                            block,
                            backface: false,
                        }),
                        (None, Some(block)) => Some(MaskEntry {
                            block,
                            backface: true,
                        }),
                    };

                    n += 1;
                    x[u] += 1;
                }
                x[v] += 1;
            }

            x[d] += 1;

            n = 0;
            for j in 0..mask_h {
                let mut i = 0;
                while i < mask_w {
                    let Some(entry) = mask[n] else {
                        i += 1;
                        n += 1;
                        continue;
                    };

                    let mut w = 1;
                    while i + w < mask_w && mask[n + w] == Some(entry) {
                        w += 1;
                    }

                    let mut h = 1;
                    'grow_h: while j + h < mask_h {
                        for k in 0..w {
                            if mask[n + k + h * mask_w] != Some(entry) {
                                break 'grow_h;
                            }
                        }
                        h += 1;
                    }

                    emit_quad(
                        &mut vertices,
                        &mut indices,
                        d,
                        u,
                        v,
                        x[d],
                        i as i32,
                        j as i32,
                        w as i32,
                        h as i32,
                        entry,
                    );

                    for l in 0..h {
                        for k in 0..w {
                            mask[n + k + l * mask_w] = None;
                        }
                    }

                    i += w;
                    n += w;
                }
            }
        }
    }

    ChunkMesh { vertices, indices }
}

#[allow(clippy::too_many_arguments)]
fn emit_quad(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    d: usize,
    u: usize,
    v: usize,
    plane: i32,
    i: i32,
    j: i32,
    w: i32,
    h: i32,
    entry: MaskEntry,
) {
    let corner = |along_u: i32, along_v: i32| -> [i32; 3] {
        let mut p = [0i32; 3];
        p[d] = plane;
        p[u] = i + along_u;
        p[v] = j + along_v;
        p
    };

    let c00 = corner(0, 0);
    let c10 = corner(w, 0);
    let c11 = corner(w, h);
    let c01 = corner(0, h);

    // Matches `FACE_NORMALS` in shader.wgsl: axis `d` contributes index
    // `d * 2`, with `+ 1` for the negative-facing (backface) direction.
    let normal_index = (d as u32) * 2 + entry.backface as u32;

    let [r, g, b] = block::color(entry.block);
    let color = [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
        255,
    ];
    let base = vertices.len() as u32;

    let quad = if entry.backface {
        [c00, c01, c11, c10]
    } else {
        [c00, c10, c11, c01]
    };

    for local in quad {
        vertices.push(ChunkVertex {
            packed: pack_vertex(local, normal_index),
            color,
        });
    }

    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::STONE;
    use crate::world::chunk::Chunk;

    fn world_with_dense(setup: impl FnOnce(&mut Chunk)) -> World {
        let mut chunk = Chunk::empty();
        setup(&mut chunk);
        let mut world = World::new();
        world.insert_chunk(ChunkPos::new(0, 0, 0), chunk);
        world
    }

    #[test]
    fn single_voxel_produces_six_unmerged_faces() {
        let world = world_with_dense(|c| c.set(0, 0, 0, STONE));
        let mesh = mesh_chunk(&world, ChunkPos::new(0, 0, 0));
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn two_adjacent_voxels_merge_side_faces() {
        let world = world_with_dense(|c| {
            c.set(0, 0, 0, STONE);
            c.set(1, 0, 0, STONE);
        });
        let mesh = mesh_chunk(&world, ChunkPos::new(0, 0, 0));
        // top, bottom, +z, -z faces merge into 2x1 quads; the two x-end faces stay 1x1.
        assert_eq!(mesh.indices.len(), 6 * 6);
        assert_eq!(mesh.vertices.len(), 6 * 4);
    }

    #[test]
    fn empty_chunk_has_no_faces() {
        let world = world_with_dense(|_| {});
        let mesh = mesh_chunk(&world, ChunkPos::new(0, 0, 0));
        assert!(mesh.is_empty());
    }
}
