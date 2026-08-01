use noise::{Fbm, NoiseFn, Perlin};

use super::block::{AIR, DIRT, GRASS, SAND, STONE};
use super::chunk::{CHUNK_SIZE, CHUNK_SIZE_I32, Chunk, VOLUME, voxel_index};
use super::{ChunkPos, chunk_origin};

const SEA_LEVEL: i32 = 62;
const BASE_HEIGHT: f64 = 68.0;
const AMPLITUDE: f64 = 28.0;
const NOISE_FREQUENCY: f64 = 0.008;

/// Chunks span world y in `[0, VERTICAL_CHUNKS * CHUNK_SIZE)`, which comfortably
/// covers the height range produced by `BASE_HEIGHT`/`AMPLITUDE` above. Loaded
/// and unloaded per-column (all `VERTICAL_CHUNKS` chunks at once) by chunk streaming.
pub const VERTICAL_CHUNKS: i32 = 4;

pub struct TerrainGenerator {
    noise: Fbm<Perlin>,
}

impl TerrainGenerator {
    pub fn new(seed: u32) -> Self {
        Self {
            noise: Fbm::new(seed),
        }
    }

    fn height_at(&self, world_x: i32, world_z: i32) -> i32 {
        let n = self.noise.get([
            world_x as f64 * NOISE_FREQUENCY,
            world_z as f64 * NOISE_FREQUENCY,
        ]);
        (BASE_HEIGHT + n * AMPLITUDE).round() as i32
    }

    /// Generates every vertically-stacked chunk in an (x, z) column at once.
    /// `height_at` doesn't depend on the vertical chunk index, but chunk
    /// streaming loads/unloads a whole column's `VERTICAL_CHUNKS` chunks
    /// together (see `streaming::ChunkStreamer`) — generating them one
    /// `ChunkPos` at a time would resample the same column's heightmap
    /// noise up to `VERTICAL_CHUNKS` times over. Sampling it once here and
    /// reusing it across all of them cuts that noise evaluation by 4x.
    pub fn generate_column(&self, x: i32, z: i32) -> Vec<(ChunkPos, Chunk)> {
        let world_origin_x = x * CHUNK_SIZE_I32;
        let world_origin_z = z * CHUNK_SIZE_I32;
        let mut heights = vec![0i32; CHUNK_SIZE * CHUNK_SIZE];
        for cz in 0..CHUNK_SIZE {
            for cx in 0..CHUNK_SIZE {
                heights[cz * CHUNK_SIZE + cx] =
                    self.height_at(world_origin_x + cx as i32, world_origin_z + cz as i32);
            }
        }

        (0..VERTICAL_CHUNKS)
            .map(|y| {
                let pos = ChunkPos::new(x, y, z);
                (pos, Self::fill_chunk(chunk_origin(pos), &heights))
            })
            .collect()
    }

    fn fill_chunk(origin: ChunkPos, heights: &[i32]) -> Chunk {
        let mut dense = vec![AIR; VOLUME];

        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let height = heights[z * CHUNK_SIZE + x];

                for y in 0..CHUNK_SIZE {
                    let world_y = origin.y + y as i32;
                    if world_y > height {
                        continue;
                    }
                    let block = if world_y == height {
                        if height <= SEA_LEVEL + 1 { SAND } else { GRASS }
                    } else if world_y > height - 4 {
                        DIRT
                    } else {
                        STONE
                    };
                    dense[voxel_index(x, y, z)] = block;
                }
            }
        }

        Chunk::from_dense(&dense)
    }
}
