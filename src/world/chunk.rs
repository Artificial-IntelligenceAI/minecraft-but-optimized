use super::block::{AIR, BlockId};

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

#[inline]
pub fn voxel_index(x: usize, y: usize, z: usize) -> usize {
    x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE
}

/// A packed array of fixed-width unsigned integers, backed by `u32` words.
/// Entries may straddle a word boundary.
struct BitBuffer {
    len: usize,
    bits_per_entry: u8,
    words: Vec<u32>,
}

impl BitBuffer {
    fn new(len: usize, bits_per_entry: u8) -> Self {
        let total_bits = len * bits_per_entry as usize;
        Self {
            len,
            bits_per_entry,
            words: vec![0u32; total_bits.div_ceil(32)],
        }
    }

    #[inline]
    fn get(&self, index: usize) -> u32 {
        let bit_start = index * self.bits_per_entry as usize;
        let word_idx = bit_start / 32;
        let bit_off = bit_start % 32;
        let mask = (1u64 << self.bits_per_entry) - 1;

        let lo = self.words[word_idx] as u64;
        let value = if bit_off + self.bits_per_entry as usize <= 32 {
            lo >> bit_off
        } else {
            let hi = self.words[word_idx + 1] as u64;
            (lo >> bit_off) | (hi << (32 - bit_off))
        };
        (value & mask) as u32
    }

    #[inline]
    fn set(&mut self, index: usize, value: u32) {
        let bit_start = index * self.bits_per_entry as usize;
        let word_idx = bit_start / 32;
        let bit_off = bit_start % 32;
        let mask = (1u64 << self.bits_per_entry) - 1;
        let value = value as u64 & mask;

        if bit_off + self.bits_per_entry as usize <= 32 {
            let w = &mut self.words[word_idx];
            *w = ((*w as u64 & !(mask << bit_off)) | (value << bit_off)) as u32;
        } else {
            let split = 32 - bit_off;
            let lo_mask = mask & ((1u64 << split) - 1);
            let w0 = &mut self.words[word_idx];
            *w0 = ((*w0 as u64 & !(lo_mask << bit_off)) | ((value & lo_mask) << bit_off)) as u32;
            let hi_mask = mask >> split;
            let w1 = &mut self.words[word_idx + 1];
            *w1 = ((*w1 as u64 & !hi_mask) | (value >> split)) as u32;
        }
    }

    fn grow(&mut self, new_bits_per_entry: u8) {
        let mut grown = BitBuffer::new(self.len, new_bits_per_entry);
        for i in 0..self.len {
            grown.set(i, self.get(i));
        }
        *self = grown;
    }
}

fn bits_needed(palette_len: usize) -> u8 {
    if palette_len <= 1 {
        return 1;
    }
    (usize::BITS - (palette_len - 1).leading_zeros()).max(1) as u8
}

enum Storage {
    /// Every voxel in the chunk is this block. The common case for air/stone-only chunks.
    Uniform(BlockId),
    Palette {
        palette: Vec<BlockId>,
        data: BitBuffer,
    },
}

/// A `CHUNK_SIZE`^3 volume of blocks, stored with palette compression: chunks
/// typically contain only a handful of distinct block types, so voxels are
/// stored as small bit-packed indices into a per-chunk palette rather than
/// as full `BlockId`s.
pub struct Chunk {
    storage: Storage,
}

impl Chunk {
    pub fn uniform(block: BlockId) -> Self {
        Self {
            storage: Storage::Uniform(block),
        }
    }

    pub fn empty() -> Self {
        Self::uniform(AIR)
    }

    /// Builds a chunk from a dense, row-major (`voxel_index` order) array of
    /// block ids, deduplicating into a palette in one pass. Prefer this over
    /// repeated `set` calls when generating a chunk from scratch.
    pub fn from_dense(dense: &[BlockId]) -> Self {
        debug_assert_eq!(dense.len(), VOLUME);

        let mut palette: Vec<BlockId> = Vec::new();
        for &b in dense {
            if !palette.contains(&b) {
                palette.push(b);
            }
        }

        if palette.len() <= 1 {
            return Self::uniform(palette.first().copied().unwrap_or(AIR));
        }

        let bits = bits_needed(palette.len());
        let mut data = BitBuffer::new(VOLUME, bits);
        for (i, &b) in dense.iter().enumerate() {
            let palette_idx = palette.iter().position(|&p| p == b).unwrap() as u32;
            data.set(i, palette_idx);
        }

        Self {
            storage: Storage::Palette { palette, data },
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockId {
        match &self.storage {
            Storage::Uniform(b) => *b,
            Storage::Palette { palette, data } => palette[data.get(voxel_index(x, y, z)) as usize],
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, block: BlockId) {
        match &mut self.storage {
            Storage::Uniform(current) if *current == block => {}
            Storage::Uniform(current) => {
                let mut data = BitBuffer::new(VOLUME, 1);
                data.set(voxel_index(x, y, z), 1);
                self.storage = Storage::Palette {
                    palette: vec![*current, block],
                    data,
                };
            }
            Storage::Palette { palette, data } => {
                let palette_idx = match palette.iter().position(|&p| p == block) {
                    Some(i) => i,
                    None => {
                        palette.push(block);
                        let needed_bits = bits_needed(palette.len());
                        if needed_bits > data.bits_per_entry {
                            data.grow(needed_bits);
                        }
                        palette.len() - 1
                    }
                };
                data.set(voxel_index(x, y, z), palette_idx as u32);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self.storage, Storage::Uniform(AIR))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::{DIRT, GRASS, STONE};

    #[test]
    fn uniform_chunk_reads_back_everywhere() {
        let chunk = Chunk::uniform(STONE);
        assert_eq!(chunk.get(0, 0, 0), STONE);
        assert_eq!(chunk.get(31, 31, 31), STONE);
        assert!(!chunk.is_empty());
        assert!(Chunk::empty().is_empty());
    }

    #[test]
    fn set_promotes_uniform_to_palette_and_back_reads_correctly() {
        let mut chunk = Chunk::uniform(AIR);
        chunk.set(5, 6, 7, STONE);
        assert_eq!(chunk.get(5, 6, 7), STONE);
        assert_eq!(chunk.get(0, 0, 0), AIR);
    }

    #[test]
    fn palette_grows_as_more_distinct_blocks_are_set() {
        let mut chunk = Chunk::uniform(AIR);
        let blocks = [STONE, DIRT, GRASS, 10, 11, 12, 13, 14, 15, 16, 17];
        for (i, &b) in blocks.iter().enumerate() {
            chunk.set(i, 0, 0, b);
        }
        for (i, &b) in blocks.iter().enumerate() {
            assert_eq!(chunk.get(i, 0, 0), b);
        }
        assert_eq!(chunk.get(20, 0, 0), AIR);
    }

    #[test]
    fn from_dense_matches_manual_sets() {
        let mut dense = vec![AIR; VOLUME];
        dense[voxel_index(1, 2, 3)] = STONE;
        dense[voxel_index(4, 5, 6)] = DIRT;

        let chunk = Chunk::from_dense(&dense);
        assert_eq!(chunk.get(1, 2, 3), STONE);
        assert_eq!(chunk.get(4, 5, 6), DIRT);
        assert_eq!(chunk.get(0, 0, 0), AIR);
    }

    #[test]
    fn bit_buffer_roundtrips_values_at_every_width() {
        for bits in 1u8..=16 {
            let mut buf = BitBuffer::new(100, bits);
            let max = (1u32 << bits) - 1;
            for i in 0..100 {
                buf.set(i, (i as u32 * 37) % (max + 1));
            }
            for i in 0..100 {
                assert_eq!(buf.get(i), (i as u32 * 37) % (max + 1));
            }
        }
    }
}
