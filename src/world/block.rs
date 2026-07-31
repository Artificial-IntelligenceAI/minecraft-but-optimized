pub type BlockId = u16;

pub const AIR: BlockId = 0;
pub const STONE: BlockId = 1;
pub const DIRT: BlockId = 2;
pub const GRASS: BlockId = 3;
pub const SAND: BlockId = 4;

pub struct BlockInfo {
    pub name: &'static str,
    pub solid: bool,
    /// Flat RGB used until a texture atlas exists.
    pub color: [f32; 3],
}

pub const BLOCKS: &[BlockInfo] = &[
    BlockInfo {
        name: "air",
        solid: false,
        color: [0.0, 0.0, 0.0],
    },
    BlockInfo {
        name: "stone",
        solid: true,
        color: [0.50, 0.50, 0.52],
    },
    BlockInfo {
        name: "dirt",
        solid: true,
        color: [0.40, 0.28, 0.16],
    },
    BlockInfo {
        name: "grass",
        solid: true,
        color: [0.30, 0.58, 0.22],
    },
    BlockInfo {
        name: "sand",
        solid: true,
        color: [0.82, 0.76, 0.52],
    },
];

#[inline]
pub fn is_solid(id: BlockId) -> bool {
    BLOCKS[id as usize].solid
}

#[inline]
pub fn color(id: BlockId) -> [f32; 3] {
    BLOCKS[id as usize].color
}
