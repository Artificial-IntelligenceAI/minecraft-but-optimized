use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use rayon::prelude::*;

use super::block::{self, BlockId};
use super::chunk::CHUNK_SIZE_I32;
use super::{ChunkPos, World, chunk_origin};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ChunkVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
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

    let solid_block_at = |local: [i32; 3]| -> Option<BlockId> {
        let world_pos = origin + IVec3::new(local[0], local[1], local[2]);
        let id = world.get_block(world_pos);
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
                        origin,
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

pub fn mesh_world(world: &World) -> Vec<(ChunkPos, ChunkMesh)> {
    let positions: Vec<ChunkPos> = world.loaded_chunk_positions().collect();
    positions
        .par_iter()
        .map(|&pos| (pos, mesh_chunk(world, pos)))
        .filter(|(_, mesh)| !mesh.is_empty())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn emit_quad(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    origin: IVec3,
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
    let origin = [origin.x, origin.y, origin.z];
    let corner = |along_u: i32, along_v: i32| -> [f32; 3] {
        let mut p = [0i32; 3];
        p[d] = plane;
        p[u] = i + along_u;
        p[v] = j + along_v;
        [
            (p[0] + origin[0]) as f32,
            (p[1] + origin[1]) as f32,
            (p[2] + origin[2]) as f32,
        ]
    };

    let c00 = corner(0, 0);
    let c10 = corner(w, 0);
    let c11 = corner(w, h);
    let c01 = corner(0, h);

    let mut normal = [0.0f32; 3];
    normal[d] = if entry.backface { -1.0 } else { 1.0 };

    let color = block::color(entry.block);
    let base = vertices.len() as u32;

    let quad = if entry.backface {
        [c00, c01, c11, c10]
    } else {
        [c00, c10, c11, c01]
    };

    for pos in quad {
        vertices.push(ChunkVertex {
            position: pos,
            normal,
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
