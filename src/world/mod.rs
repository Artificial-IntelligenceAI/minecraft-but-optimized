pub mod block;
pub mod chunk;
pub mod generation;
pub mod meshing;

use std::collections::HashMap;

use glam::IVec3;

use chunk::{CHUNK_SIZE_I32, Chunk};

pub type ChunkPos = IVec3;

/// Converts a world-space block coordinate into (chunk position, local voxel coordinate).
#[inline]
pub fn world_to_chunk(world_pos: IVec3) -> (ChunkPos, [usize; 3]) {
    let chunk_pos = world_pos.div_euclid(IVec3::splat(CHUNK_SIZE_I32));
    let local = world_pos.rem_euclid(IVec3::splat(CHUNK_SIZE_I32));
    (
        chunk_pos,
        [local.x as usize, local.y as usize, local.z as usize],
    )
}

#[inline]
pub fn chunk_origin(chunk_pos: ChunkPos) -> IVec3 {
    chunk_pos * CHUNK_SIZE_I32
}

pub struct World {
    chunks: HashMap<ChunkPos, Chunk>,
}

impl World {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    pub fn insert_chunk(&mut self, pos: ChunkPos, chunk: Chunk) {
        self.chunks.insert(pos, chunk);
    }

    pub fn chunk(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&pos)
    }

    pub fn loaded_chunk_positions(&self) -> impl Iterator<Item = ChunkPos> + '_ {
        self.chunks.keys().copied()
    }

    pub fn get_block(&self, world_pos: IVec3) -> block::BlockId {
        let (chunk_pos, [x, y, z]) = world_to_chunk(world_pos);
        match self.chunks.get(&chunk_pos) {
            Some(chunk) => chunk.get(x, y, z),
            None => block::AIR,
        }
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}
