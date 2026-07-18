use std::collections::BTreeMap;

pub const ID_SPLIT: u32 = 4;
pub const ID_CENTER: f32 = ((ID_SPLIT - 1) as f32) / 2.0;
pub const ID_SUBMODULES: [(&str, usize); 4] = [
    ("card_id", 12),
    ("note_id", 12),
    ("deck_id", 8),
    ("preset_id", 8),
];
pub const ID_ENCODING_LEN: usize = 12 + 12 + 8 + 8;

const MERSENNE_STATE_N: usize = 624;
const MERSENNE_STATE_M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UMASK: u32 = 0x8000_0000;
const LMASK: u32 = 0x7fff_ffff;
const TORCH_RNG_STATE_U64_LEN: usize = 632;
pub(crate) const TORCH_RNG_STATE_BYTE_LEN: usize = TORCH_RNG_STATE_U64_LEN * 8;
const TORCH_RNG_STATE_HEADER_LEN: usize = 3;
const TORCH_RNG_STATE_TAIL_LEN: usize = 5;

pub type IdEncodings = BTreeMap<&'static str, BTreeMap<i64, Vec<f32>>>;

pub fn empty_id_encodings() -> IdEncodings {
    ID_SUBMODULES
        .into_iter()
        .map(|(name, _)| (name, BTreeMap::new()))
        .collect()
}

#[derive(Debug, Clone)]
pub struct TorchMt19937 {
    seed: u64,
    left: usize,
    next: usize,
    state: [u32; MERSENNE_STATE_N],
}

impl Default for TorchMt19937 {
    fn default() -> Self {
        Self::seed_from_u64(5489)
    }
}

impl TorchMt19937 {
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut state = [0; MERSENNE_STATE_N];
        state[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..MERSENNE_STATE_N {
            state[j] = 1_812_433_253u32
                .wrapping_mul(state[j - 1] ^ (state[j - 1] >> 30))
                .wrapping_add(j as u32);
        }
        Self {
            seed,
            left: 1,
            next: 0,
            state,
        }
    }

    pub fn from_torch_rng_state(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != TORCH_RNG_STATE_BYTE_LEN {
            return Err(format!(
                "torch RNG state must contain {TORCH_RNG_STATE_BYTE_LEN} bytes, got {}",
                bytes.len()
            ));
        }

        let mut values = [0u64; TORCH_RNG_STATE_U64_LEN];
        for (index, chunk) in bytes.chunks_exact(8).enumerate() {
            values[index] = u64::from_le_bytes(
                chunk
                    .try_into()
                    .expect("chunks_exact(8) always yields 8-byte chunks"),
            );
        }

        let packed_left = values[1];
        if packed_left >> 32 != 1 {
            return Err("unsupported torch RNG state header".to_string());
        }
        let left = (packed_left & 0xffff_ffff) as usize;
        let next = usize::try_from(values[2])
            .map_err(|_| "torch RNG state next index does not fit usize".to_string())?;
        if left > MERSENNE_STATE_N {
            return Err(format!(
                "torch RNG state left count must be <= {MERSENNE_STATE_N}, got {left}"
            ));
        }
        if next > MERSENNE_STATE_N {
            return Err(format!(
                "torch RNG state next index must be <= {MERSENNE_STATE_N}, got {next}"
            ));
        }

        let mut state = [0u32; MERSENNE_STATE_N];
        for (index, value) in values
            [TORCH_RNG_STATE_HEADER_LEN..TORCH_RNG_STATE_HEADER_LEN + MERSENNE_STATE_N]
            .iter()
            .enumerate()
        {
            state[index] = u32::try_from(*value)
                .map_err(|_| format!("torch RNG state word {index} exceeds u32 range"))?;
        }

        if values[TORCH_RNG_STATE_HEADER_LEN + MERSENNE_STATE_N..]
            .iter()
            .any(|value| *value != 0)
        {
            return Err("unsupported nonzero torch RNG state tail".to_string());
        }

        Ok(Self {
            seed: values[0],
            left,
            next,
            state,
        })
    }

    pub fn to_torch_rng_state_bytes(&self) -> Vec<u8> {
        let mut values = [0u64; TORCH_RNG_STATE_U64_LEN];
        values[0] = self.seed;
        values[1] = (1u64 << 32) | self.left as u64;
        values[2] = self.next as u64;
        for (index, word) in self.state.iter().enumerate() {
            values[TORCH_RNG_STATE_HEADER_LEN + index] = u64::from(*word);
        }
        debug_assert_eq!(
            values[TORCH_RNG_STATE_HEADER_LEN + MERSENNE_STATE_N..].len(),
            TORCH_RNG_STATE_TAIL_LEN
        );

        let mut bytes = Vec::with_capacity(TORCH_RNG_STATE_BYTE_LEN);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    pub fn random_u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    pub fn id_encoding(&mut self, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|_| (self.random_u32() % ID_SPLIT) as f32 - ID_CENTER)
            .collect()
    }

    fn next_state(&mut self) {
        self.left = MERSENNE_STATE_N;
        self.next = 0;

        for i in 0..(MERSENNE_STATE_N - MERSENNE_STATE_M) {
            self.state[i] =
                self.state[i + MERSENNE_STATE_M] ^ twist(self.state[i], self.state[i + 1]);
        }

        for i in (MERSENNE_STATE_N - MERSENNE_STATE_M)..(MERSENNE_STATE_N - 1) {
            self.state[i] = self.state[i + MERSENNE_STATE_M - MERSENNE_STATE_N]
                ^ twist(self.state[i], self.state[i + 1]);
        }

        self.state[MERSENNE_STATE_N - 1] = self.state[MERSENNE_STATE_M - 1]
            ^ twist(self.state[MERSENNE_STATE_N - 1], self.state[0]);
    }
}

fn mix_bits(u: u32, v: u32) -> u32 {
    (u & UMASK) | (v & LMASK)
}

fn twist(u: u32, v: u32) -> u32 {
    (mix_bits(u, v) >> 1) ^ if v & 1 != 0 { MATRIX_A } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_torch_randint_for_seeded_id_values() {
        let mut rng = TorchMt19937::seed_from_u64(12_345);
        let values = rng.id_encoding(12);
        assert_eq!(
            values,
            vec![0.5, -0.5, -0.5, -0.5, -1.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5]
        );

        let note_values = rng.id_encoding(12);
        assert_eq!(
            note_values,
            vec![0.5, -0.5, 1.5, 0.5, -0.5, 1.5, 1.5, 0.5, -1.5, 0.5, -0.5, 1.5]
        );
    }

    #[test]
    fn torch_rng_state_round_trips_native_state() {
        let mut rng = TorchMt19937::seed_from_u64(12_345);
        let expected = rng.id_encoding(40);
        let restored = TorchMt19937::from_torch_rng_state(&rng.to_torch_rng_state_bytes()).unwrap();

        assert_eq!(restored.seed, 12_345);
        assert_eq!(restored.left, rng.left);
        assert_eq!(restored.next, rng.next);
        assert_eq!(restored.state, rng.state);

        let mut seeded = TorchMt19937::seed_from_u64(12_345);
        assert_eq!(seeded.id_encoding(40), expected);
    }
}
