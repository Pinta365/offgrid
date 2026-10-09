use offgrid_sys as sys;

/// Owned `llama_batch` with a fixed token capacity.
pub(crate) struct Batch {
    raw: sys::llama_batch,
    capacity: usize,
    n_seq_max: usize,
}

impl Batch {
    pub fn new(capacity: usize, n_seq_max: usize) -> Self {
        let raw = unsafe { sys::llama_batch_init(capacity as i32, 0, n_seq_max as i32) };
        Self { raw, capacity, n_seq_max }
    }

    pub fn clear(&mut self) {
        self.raw.n_tokens = 0;
    }

    pub fn len(&self) -> usize {
        self.raw.n_tokens as usize
    }

    pub fn add(&mut self, token: sys::llama_token, pos: i32, seq_ids: &[i32], output: bool) {
        let i = self.len();
        assert!(i < self.capacity, "batch overflow");
        assert!(seq_ids.len() <= self.n_seq_max, "too many seq ids");
        unsafe {
            *self.raw.token.add(i) = token;
            *self.raw.pos.add(i) = pos;
            *self.raw.n_seq_id.add(i) = seq_ids.len() as i32;
            let dst = *self.raw.seq_id.add(i);
            for (j, s) in seq_ids.iter().enumerate() {
                *dst.add(j) = *s;
            }
            *self.raw.logits.add(i) = output as i8;
        }
        self.raw.n_tokens += 1;
    }

    pub fn raw(&self) -> sys::llama_batch {
        self.raw
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { sys::llama_batch_free(self.raw) };
    }
}
