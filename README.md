# minecraft-but-optimized

A Minecraft-like voxel game/engine built from scratch in Rust, with performance as the core design goal rather than an afterthought.

## Why

Vanilla Minecraft (Java Edition) is famously CPU-bound and single-threaded in a lot of hot paths (chunk meshing, lighting, entity ticking). This project explores how far a modern, data-oriented, multithreaded architecture can push voxel world simulation and rendering — greedy meshing, parallel chunk generation, cache-friendly data layouts, and a GPU-driven renderer.

This is not aiming for feature parity with Minecraft; it's a playground for building the same *kind* of game with a much more optimized foundation.

## Status

Early scaffolding. Nothing playable yet.

## Tech stack

- **Language:** Rust
- **Rendering:** [`wgpu`](https://wgpu.rs/) (Vulkan/Metal/DX12/GL backend)
- **Windowing/input:** [`winit`](https://github.com/rust-windowing/winit)

(Stack will grow as the project does — chunk data structures, meshing, world generation, etc. are still being designed.)

## Building

```bash
cargo run
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
