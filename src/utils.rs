use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};

/// Hash a byte slice using xxHash3 with a given seed
pub fn hash_with_seed(data: &[u8], seed: u64) -> u64 {
    xxh3_64_with_seed(data, seed)
}

/// Hash a string using xxHash3
pub fn hash_str(s: &str, seed: u64) -> u64 {
    xxh3_64_with_seed(s.as_bytes(), seed)
}

/// Generate two independent hashes for double hashing
pub fn hash_pair(data: &[u8], seed: u64) -> (u64, u64) {
    let h1 = xxh3_64_with_seed(data, seed);
    let h2 = xxh3_64_with_seed(data, seed.wrapping_add(0x9E3779B97F4A7C15));
    (h1, h2)
}

/// Generate multiple hash values using double hashing technique
/// This uses the formula: h_i(x) = h1(x) + i * h2(x)
pub fn double_hash(h1: u64, h2: u64, i: u32, size: usize) -> usize {
    let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
    (h % size as u64) as usize
}

/// Calculate optimal number of bits for a Bloom filter
pub fn optimal_bits(expected_items: usize, false_positive_rate: f64) -> usize {
    let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
    let n = expected_items as f64;
    let p = false_positive_rate;
    let m = -(n * p.ln()) / ln2_squared;
    m.ceil() as usize
}

/// Calculate optimal number of hash functions for a Bloom filter
pub fn optimal_hashes(num_bits: usize, expected_items: usize) -> u32 {
    let m = num_bits as f64;
    let n = expected_items as f64;
    let k = (m / n) * std::f64::consts::LN_2;
    std::cmp::max(1, k.round() as u32)
}

/// Calculate the expected false positive rate for current parameters
pub fn expected_fpr(num_bits: usize, num_hashes: u32, num_items: usize) -> f64 {
    let m = num_bits as f64;
    let k = num_hashes as f64;
    let n = num_items as f64;
    (1.0 - (-k * n / m).exp()).powf(k)
}

/// MurmurHash3 finalizer for better bit mixing
pub fn murmur_finalize(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

/// Count leading zeros (for HyperLogLog)
pub fn leading_zeros(hash: u64) -> u8 {
    hash.leading_zeros() as u8
}

/// Get bucket index for HyperLogLog
pub fn hll_bucket(hash: u64, precision: u8) -> usize {
    (hash >> (64 - precision)) as usize
}

/// Get remaining bits for HyperLogLog rank calculation
pub fn hll_rank(hash: u64, precision: u8) -> u8 {
    // Get the remaining bits after removing precision bits
    let remaining = hash << precision;
    // Count leading zeros + 1 (minimum rank is 1)
    remaining.leading_zeros() as u8 + 1
}

/// Serialize to compact binary format
pub fn to_bincode<T: serde::Serialize>(data: &T) -> Result<Vec<u8>, String> {
    bincode::serialize(data).map_err(|e| e.to_string())
}

/// Deserialize from compact binary format
pub fn from_bincode<'a, T: serde::Deserialize<'a>>(data: &'a [u8]) -> Result<T, String> {
    bincode::deserialize(data).map_err(|e| e.to_string())
}
