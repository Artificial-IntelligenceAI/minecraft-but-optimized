use std::collections::HashSet;

use glam::Vec3;
use rayon::prelude::*;

use super::chunk::{CHUNK_SIZE_I32, Chunk};
use super::generation::{TerrainGenerator, VERTICAL_CHUNKS};
use super::meshing::{self, ChunkMesh};
use super::{ChunkPos, World};

/// (x, z) chunk-column coordinate: one column covers every chunk from
/// y = 0..VERTICAL_CHUNKS at that x/z, and is the unit of loading/unloading.
type ColumnPos = (i32, i32);

const NEIGHBOR_OFFSETS: [ColumnPos; 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// Chunk meshes to upload/replace, and chunks that no longer exist and
/// should have their GPU meshes dropped.
pub struct StreamingUpdate {
    pub meshes: Vec<(ChunkPos, ChunkMesh)>,
    pub removed: Vec<ChunkPos>,
}

/// Loads/unloads chunk columns around a moving point (the camera), keeping
/// the world within `load_radius` chunks populated. Columns are only
/// unloaded once they exceed `unload_radius` (a wider radius than
/// `load_radius`) so hovering near a boundary doesn't thrash load/unload
/// every frame.
pub struct ChunkStreamer {
    generator: TerrainGenerator,
    loaded_columns: HashSet<ColumnPos>,
    load_radius: i32,
    unload_radius: i32,
}

impl ChunkStreamer {
    pub fn new(seed: u32, load_radius: i32, unload_radius: i32) -> Self {
        assert!(unload_radius >= load_radius);
        Self {
            generator: TerrainGenerator::new(seed),
            loaded_columns: HashSet::new(),
            load_radius,
            unload_radius,
        }
    }

    pub fn load_radius(&self) -> i32 {
        self.load_radius
    }

    /// Changes the load radius at runtime (e.g. from a `/settings rd` chat
    /// command), keeping the existing gap between load and unload radii so
    /// hysteresis still holds. Columns aren't force-reloaded immediately —
    /// the next `update()` calls converge on the new radius at the normal
    /// per-frame budget, so a smaller radius unloads gradually and a larger
    /// one streams in gradually rather than both stalling a frame.
    pub fn set_load_radius(&mut self, load_radius: i32) {
        let gap = (self.unload_radius - self.load_radius).max(1);
        self.load_radius = load_radius;
        self.unload_radius = load_radius + gap;
    }

    /// Loads/unloads columns around `focus` (typically the camera position),
    /// generating and remeshing at most `load_budget` new columns this call
    /// (pass `usize::MAX` to load everything needed in one go, e.g. at startup).
    pub fn update(
        &mut self,
        world: &mut World,
        focus: Vec3,
        load_budget: usize,
    ) -> StreamingUpdate {
        let center = world_pos_to_column(focus);
        let load_radius_sq = self.load_radius * self.load_radius;

        // Both this and `to_unload` below use circular (Euclidean) distance from `center`.
        // Mixing a square load region with a circular unload region would mean the load
        // region's corners sit outside the unload circle whenever unload_radius <
        // load_radius * sqrt(2), which is true for any unload_radius the assertion in
        // `new` allows equal to load_radius — those corners would load, then immediately
        // exceed the unload check, then reload next frame: permanent churn even standing still.
        let mut to_load: Vec<ColumnPos> = (-self.load_radius..=self.load_radius)
            .flat_map(|dz| (-self.load_radius..=self.load_radius).map(move |dx| (dx, dz)))
            .map(|(dx, dz)| (center.0 + dx, center.1 + dz))
            .filter(|&col| column_dist_sq(center, col) <= load_radius_sq)
            .filter(|col| !self.loaded_columns.contains(col))
            .collect();
        to_load.sort_by_key(|&(x, z)| column_dist_sq(center, (x, z)));
        to_load.truncate(load_budget);

        let unload_radius_sq = self.unload_radius * self.unload_radius;
        let to_unload: Vec<ColumnPos> = self
            .loaded_columns
            .iter()
            .copied()
            .filter(|&col| column_dist_sq(center, col) > unload_radius_sq)
            .collect();

        let mut affected: HashSet<ColumnPos> = HashSet::new();
        for &col in to_load.iter().chain(&to_unload) {
            affected.insert(col);
            for (dx, dz) in NEIGHBOR_OFFSETS {
                affected.insert((col.0 + dx, col.1 + dz));
            }
        }

        let new_chunk_positions: Vec<ChunkPos> = to_load
            .iter()
            .flat_map(|&(x, z)| (0..VERTICAL_CHUNKS).map(move |y| ChunkPos::new(x, y, z)))
            .collect();
        let generated: Vec<(ChunkPos, Chunk)> = new_chunk_positions
            .par_iter()
            .map(|&pos| (pos, self.generator.generate_chunk(pos)))
            .collect();
        for (pos, chunk) in generated {
            world.insert_chunk(pos, chunk);
        }
        for &col in &to_load {
            self.loaded_columns.insert(col);
        }

        let mut removed = Vec::with_capacity(to_unload.len() * VERTICAL_CHUNKS as usize);
        for &(x, z) in &to_unload {
            self.loaded_columns.remove(&(x, z));
            for y in 0..VERTICAL_CHUNKS {
                let pos = ChunkPos::new(x, y, z);
                world.remove_chunk(pos);
                removed.push(pos);
            }
        }

        let remesh_positions: Vec<ChunkPos> = affected
            .into_iter()
            .filter(|col| self.loaded_columns.contains(col))
            .flat_map(|(x, z)| (0..VERTICAL_CHUNKS).map(move |y| ChunkPos::new(x, y, z)))
            .collect();
        let meshes: Vec<(ChunkPos, ChunkMesh)> = remesh_positions
            .par_iter()
            .map(|&pos| (pos, meshing::mesh_chunk(world, pos)))
            .collect();

        StreamingUpdate { meshes, removed }
    }
}

#[inline]
fn world_pos_to_column(pos: Vec3) -> ColumnPos {
    (
        (pos.x / CHUNK_SIZE_I32 as f32).floor() as i32,
        (pos.z / CHUNK_SIZE_I32 as f32).floor() as i32,
    )
}

#[inline]
fn column_dist_sq(a: ColumnPos, b: ColumnPos) -> i32 {
    let dx = a.0 - b.0;
    let dz = a.1 - b.1;
    dx * dx + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_update_loads_a_circular_area_not_a_square() {
        let mut world = World::new();
        let mut streamer = ChunkStreamer::new(0, 2, 3);

        streamer.update(&mut world, Vec3::ZERO, usize::MAX);

        assert!(world.chunk(ChunkPos::new(0, 0, 0)).is_some());
        assert!(world.chunk(ChunkPos::new(2, 0, 0)).is_some());
        assert!(world.chunk(ChunkPos::new(1, 0, 1)).is_some());
        // (2, 2) has squared distance 8 from the center, outside load_radius=2 (squared 4):
        // a square load region would include it, a circular one must not.
        assert!(world.chunk(ChunkPos::new(2, 0, 2)).is_none());
    }

    #[test]
    fn standing_still_produces_no_further_load_or_unload_churn() {
        let mut world = World::new();
        let mut streamer = ChunkStreamer::new(0, 6, 8);

        streamer.update(&mut world, Vec3::ZERO, usize::MAX);
        // Regression test: an earlier version mixed a square load region with a
        // circular unload region, so corner columns loaded and then immediately
        // exceeded the unload radius, reloading every subsequent call forever
        // even with a stationary focus point.
        for _ in 0..5 {
            let update = streamer.update(&mut world, Vec3::ZERO, usize::MAX);
            assert!(update.meshes.is_empty());
            assert!(update.removed.is_empty());
        }
    }

    #[test]
    fn load_budget_caps_columns_loaded_per_call() {
        let mut world = World::new();
        let mut streamer = ChunkStreamer::new(0, 5, 6);

        streamer.update(&mut world, Vec3::ZERO, 3);

        assert_eq!(streamer.loaded_columns.len(), 3);
    }

    #[test]
    fn moving_far_away_unloads_old_columns_and_loads_new_ones() {
        let mut world = World::new();
        let mut streamer = ChunkStreamer::new(0, 2, 3);

        streamer.update(&mut world, Vec3::ZERO, usize::MAX);
        assert!(world.chunk(ChunkPos::new(0, 0, 0)).is_some());

        let far = Vec3::new(10_000.0, 0.0, 0.0);
        let update = streamer.update(&mut world, far, usize::MAX);

        assert!(world.chunk(ChunkPos::new(0, 0, 0)).is_none());
        assert!(!update.removed.is_empty());

        let far_column = world_pos_to_column(far);
        assert!(
            world
                .chunk(ChunkPos::new(far_column.0, 0, far_column.1))
                .is_some()
        );
    }
}
