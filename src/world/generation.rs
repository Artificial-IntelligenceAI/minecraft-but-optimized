use noise::{Fbm, NoiseFn, Perlin};
use rayon::prelude::*;

use super::block::{AIR, DIRT, GRASS, SAND, STONE};
use super::chunk::{CHUNK_SIZE, Chunk, VOLUME, voxel_index};
use super::{ChunkPos, World, chunk_origin};

const SEA_LEVEL: i32 = 62;
const BASE_HEIGHT: f64 = 68.0;
const AMPLITUDE: f64 = 28.0;
const NOISE_FREQUENCY: f64 = 0.008;

/// Chunks span world y in `[0, VERTICAL_CHUNKS * CHUNK_SIZE)`, which comfortably
/// covers the height range produced by `BASE_HEIGHT`/`AMPLITUDE` above.
const VERTICAL_CHUNKS: i32 = 4;

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

    pub fn generate_chunk(&self, chunk_pos: ChunkPos) -> Chunk {
        let origin = chunk_origin(chunk_pos);
        let mut dense = vec![AIR; VOLUME];

        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let world_x = origin.x + x as i32;
                let world_z = origin.z + z as i32;
                let height = self.height_at(world_x, world_z);

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

/// Generates a square grid of chunks (`2 * radius_chunks + 1` on a side, in x/z)
/// covering the full vertical range, in parallel.
pub fn generate_world(radius_chunks: i32, seed: u32) -> World {
    let generator = TerrainGenerator::new(seed);

    let mut positions = Vec::new();
    for cz in -radius_chunks..=radius_chunks {
        for cx in -radius_chunks..=radius_chunks {
            for cy in 0..VERTICAL_CHUNKS {
                positions.push(ChunkPos::new(cx, cy, cz));
            }
        }
    }

    let chunks: Vec<(ChunkPos, Chunk)> = positions
        .par_iter()
        .map(|&pos| (pos, generator.generate_chunk(pos)))
        .collect();

    let mut world = World::new();
    for (pos, chunk) in chunks {
        world.insert_chunk(pos, chunk);
    }
    world
}
