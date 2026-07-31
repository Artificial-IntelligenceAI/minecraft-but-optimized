# minecraft-but-optimized

A Minecraft-like voxel game/engine built from scratch in Rust, with performance as the core design goal rather than an afterthought.

## Why

Vanilla Minecraft (Java Edition) is famously CPU-bound and single-threaded in a lot of hot paths (chunk meshing, lighting, entity ticking). This project explores how far a modern, data-oriented, multithreaded architecture can push voxel world simulation and rendering — greedy meshing, parallel chunk generation, cache-friendly data layouts, and a GPU-driven renderer.

This is not aiming for feature parity with Minecraft; it's a playground for building the same *kind* of game with a much more optimized foundation.

## Status

A flyable, streamed voxel world with a chat console. No survival gameplay yet (no collision, inventory, or block breaking/placing).

- Fly around with **WASD** + mouse look (click the window to grab the cursor, **Escape** to release)
- Terrain streams in/out around you as you move, instead of loading a fixed area once
- Press **T** or **/** to open chat, like Minecraft — **Enter** to send, **Escape** to cancel, **↑/↓** to recall previous messages, **Page Up/Down** to scroll history
- Chat commands: `/settings rd <chunks>` changes render distance live, `/help` lists commands

## Tech stack

- **Language:** Rust
- **Rendering:** [`wgpu`](https://wgpu.rs/) (Vulkan/Metal/DX12/GL backend)
- **Windowing/input:** [`winit`](https://github.com/rust-windowing/winit)
- **Text rendering:** [`glyphon`](https://github.com/grovesNL/glyphon) (cosmic-text + wgpu)
- **World generation:** [`noise`](https://github.com/Razaekel/noise-rs) (Perlin/Fbm heightmaps), parallelized with [`rayon`](https://github.com/rayon-rs/rayon)

## Building

```bash
cargo run --release
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
