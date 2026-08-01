/// A growable GPU buffer with a free-list byte allocator, so many chunks can
/// share one vertex/index/instance buffer instead of each owning its own —
/// letting the whole chunk pass bind buffers once per frame instead of once
/// per chunk (see `Renderer::render`'s chunk draw loop).
pub struct GpuArena {
    buffer: wgpu::Buffer,
    label: &'static str,
    usage: wgpu::BufferUsages,
    capacity: u64,
    /// Free byte ranges as `(offset, size)`, sorted by offset and merged
    /// wherever adjacent, so fragmentation never grows unbounded.
    free: Vec<(u64, u64)>,
}

impl GpuArena {
    pub fn new(
        device: &wgpu::Device,
        label: &'static str,
        usage: wgpu::BufferUsages,
        initial_capacity: u64,
    ) -> Self {
        let capacity = initial_capacity.max(4);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity,
            usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            label,
            usage,
            capacity,
            free: vec![(0, capacity)],
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Reserves `size` bytes and returns their offset, growing the backing
    /// buffer (via a GPU-side copy of the old contents, queued so it's
    /// ordered before any writes into the new space — no CPU stall) if
    /// nothing free is big enough.
    pub fn alloc(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, size: u64) -> u64 {
        if size == 0 {
            return 0;
        }
        if let Some(offset) = self.try_alloc(size) {
            return offset;
        }
        self.grow(device, queue, size);
        self.try_alloc(size)
            .expect("arena grow did not create enough space")
    }

    pub fn free(&mut self, offset: u64, size: u64) {
        if size == 0 {
            return;
        }
        let idx = self.free.partition_point(|&(o, _)| o < offset);
        self.free.insert(idx, (offset, size));
        if idx + 1 < self.free.len() {
            let (o, s) = self.free[idx];
            let (next_o, next_s) = self.free[idx + 1];
            if o + s == next_o {
                self.free[idx] = (o, s + next_s);
                self.free.remove(idx + 1);
            }
        }
        if idx > 0 {
            let (prev_o, prev_s) = self.free[idx - 1];
            let (o, s) = self.free[idx];
            if prev_o + prev_s == o {
                self.free[idx - 1] = (prev_o, prev_s + s);
                self.free.remove(idx);
            }
        }
    }

    fn try_alloc(&mut self, size: u64) -> Option<u64> {
        let idx = self.free.iter().position(|&(_, s)| s >= size)?;
        let (offset, free_size) = self.free[idx];
        if free_size == size {
            self.free.remove(idx);
        } else {
            self.free[idx] = (offset + size, free_size - size);
        }
        Some(offset)
    }

    fn grow(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, min_extra: u64) {
        let new_capacity = (self.capacity * 2).max(self.capacity + min_extra);
        let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: new_capacity,
            usage: self.usage,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("arena grow copy"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, self.capacity);
        queue.submit(std::iter::once(encoder.finish()));

        self.free.push((self.capacity, new_capacity - self.capacity));
        self.free.sort_by_key(|&(o, _)| o);
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(self.free.len());
        for &(o, s) in &self.free {
            if let Some(last) = merged.last_mut() {
                let (last_o, last_s): &mut (u64, u64) = last;
                if *last_o + *last_s == o {
                    *last_s += s;
                    continue;
                }
            }
            merged.push((o, s));
        }
        self.free = merged;

        self.buffer = new_buffer;
        self.capacity = new_capacity;
    }
}
