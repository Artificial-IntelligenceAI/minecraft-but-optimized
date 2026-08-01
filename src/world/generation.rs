use noise::{Fbm, NoiseFn, Perlin};

use super::block::{AIR, DIRT, GRASS, SAND, STONE};
use super::chunk::{CHUNK_SIZE, CHUNK_SIZE_I32, Chunk, VOLUME, voxel_index};
use super::{ChunkPos, chunk_origin};

const SEA_LEVEL: i32 = 62;
const BASE_HEIGHT: f64 = 68.0;
const AMPLITUDE: f64 = 28.0;
const NOISE_FREQUENCY: f64 = 0.008;

/// How many blocks of dirt/grass/sand sit under the surface block before
/// stone starts.
const SURFACE_LAYER_DEPTH: i32 = 4;

/// Caves carve stone away wherever this 3D noise field's value is close to
/// zero — thresholding a single continuous field's zero-level-set naturally
/// produces winding, tunnel-like voids rather than isolated round blobs.
const CAVE_FREQUENCY: f64 = 0.008;
const CAVE_THRESHOLD: f64 = 0.02;
/// How far below the surface caves are allowed to start, kept comfortably
/// past `SURFACE_LAYER_DEPTH` so a cave can never hole through the visible
/// ground layer.
const CAVE_SURFACE_MARGIN: i32 = SURFACE_LAYER_DEPTH + 2;

/// Chunks span world y in `[0, VERTICAL_CHUNKS * CHUNK_SIZE)`, which comfortably
/// covers the height range produced by `BASE_HEIGHT`/`AMPLITUDE` above. Loaded
/// and unloaded per-column (all `VERTICAL_CHUNKS` chunks at once) by chunk streaming.
pub const VERTICAL_CHUNKS: i32 = 4;

pub struct TerrainGenerator {
    noise: Fbm<Perlin>,
    /// Separately seeded so cave shapes don't correlate with the heightmap
    /// (e.g. tunnels always tracking ridgelines).
    cave_noise: Fbm<Perlin>,
}

impl TerrainGenerator {
    pub fn new(seed: u32) -> Self {
        Self {
            noise: Fbm::new(seed),
            cave_noise: Fbm::new(seed.wrapping_add(1)),
        }
    }

    fn height_at(&self, world_x: i32, world_z: i32) -> i32 {
        let n = self.noise.get([
            world_x as f64 * NOISE_FREQUENCY,
            world_z as f64 * NOISE_FREQUENCY,
        ]);
        (BASE_HEIGHT + n * AMPLITUDE).round() as i32
    }

    fn is_cave(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        let n = self.cave_noise.get([
            world_x as f64 * CAVE_FREQUENCY,
            world_y as f64 * CAVE_FREQUENCY,
            world_z as f64 * CAVE_FREQUENCY,
        ]);
        n.abs() < CAVE_THRESHOLD
    }

    /// Whether a voxel that would otherwise be solid should be carved into
    /// a cave: within the cave noise's threshold band *and* deep enough
    /// underground not to breach the surface layer.
    fn should_carve_cave(&self, world_x: i32, world_y: i32, world_z: i32, height: i32) -> bool {
        world_y <= height - CAVE_SURFACE_MARGIN && self.is_cave(world_x, world_y, world_z)
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
                (pos, self.fill_chunk(chunk_origin(pos), &heights))
            })
            .collect()
    }

    fn fill_chunk(&self, origin: ChunkPos, heights: &[i32]) -> Chunk {
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
                    } else if world_y > height - SURFACE_LAYER_DEPTH {
                        DIRT
                    } else {
                        STONE
                    };

                    let world_x = origin.x + x as i32;
                    let world_z = origin.z + z as i32;
                    let block = if self.should_carve_cave(world_x, world_y, world_z, height) {
                        AIR
                    } else {
                        block
                    };

                    dense[voxel_index(x, y, z)] = block;
                }
            }
        }

        Chunk::from_dense(&dense)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression coverage for the cave threshold: catches it silently
    /// becoming so narrow caves never spawn, or so wide they swallow
    /// most of the underground into open voids.
    #[test]
    fn caves_carve_a_reasonable_fraction_of_deep_stone() {
        let generator = TerrainGenerator::new(0);
        let mut carved = 0u32;
        let mut total = 0u32;
        for x in 0..200 {
            for z in 0..200 {
                for y in 0..40 {
                    total += 1;
                    if generator.is_cave(x, y, z) {
                        carved += 1;
                    }
                }
            }
        }
        let fraction = f64::from(carved) / f64::from(total);
        assert!(fraction > 0.005, "caves are too sparse: {fraction}");
        assert!(fraction < 0.2, "caves are too dense: {fraction}");
    }

    /// A density check alone doesn't catch caves that are technically
    /// carved but only ever one voxel wide (uncrossable cracks, not
    /// tunnels) — measure actual passage width via the run of carved
    /// voxels through each cave point along the narrowest of the three
    /// axes, which is what an earlier, higher-frequency/tighter-threshold
    /// version of these constants got wrong.
    #[test]
    fn caves_are_wide_enough_to_walk_through() {
        let generator = TerrainGenerator::new(0);
        let run_length = |x: i32, y: i32, z: i32, axis: usize| -> i32 {
            let step = |d: i32| -> [i32; 3] {
                let mut p = [x, y, z];
                p[axis] += d;
                p
            };
            let mut len = 1;
            let mut d = 1;
            while {
                let p = step(d);
                generator.is_cave(p[0], p[1], p[2])
            } {
                len += 1;
                d += 1;
            }
            let mut d = -1;
            while {
                let p = step(d);
                generator.is_cave(p[0], p[1], p[2])
            } {
                len += 1;
                d -= 1;
            }
            len
        };

        let mut widths = Vec::new();
        for x in (0..300).step_by(3) {
            for z in (0..300).step_by(3) {
                for y in (0..40).step_by(3) {
                    if generator.is_cave(x, y, z) {
                        let w = run_length(x, y, z, 0)
                            .min(run_length(x, y, z, 1))
                            .min(run_length(x, y, z, 2));
                        widths.push(w);
                    }
                }
            }
        }
        widths.sort_unstable();
        let median = widths[widths.len() / 2];
        assert!(
            median >= 3,
            "caves are too narrow to walk through: median width {median}"
        );
    }

    /// Caves must never reach into the dirt/grass/sand layer, so the
    /// visible ground surface never grows holes.
    #[test]
    fn caves_never_reach_the_surface_layer() {
        let generator = TerrainGenerator::new(0);
        for x in 0..64 {
            for z in 0..64 {
                let height = generator.height_at(x, z);
                for depth in 0..CAVE_SURFACE_MARGIN {
                    let world_y = height - depth;
                    assert!(
                        !generator.should_carve_cave(x, world_y, z, height),
                        "cave reached within {depth} blocks of the surface at ({x}, {z})"
                    );
                }
            }
        }
    }
}
