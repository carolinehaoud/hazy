use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const MAX_KICKS: usize = 500;
const BUCKET_SIZE: usize = 4;

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
struct Bucket {
    fingerprints: [u16; BUCKET_SIZE],
}

impl Bucket {
    fn insert(&mut self, fingerprint: u16) -> bool {
        for slot in &mut self.fingerprints {
            if *slot == 0 {
                *slot = fingerprint;
                return true;
            }
        }
        false
    }

    fn contains(&self, fingerprint: u16) -> bool {
        self.fingerprints.iter().any(|&fp| fp == fingerprint)
    }

    fn remove(&mut self, fingerprint: u16) -> bool {
        for slot in &mut self.fingerprints {
            if *slot == fingerprint {
                *slot = 0;
                return true;
            }
        }
        false
    }

    fn swap(&mut self, index: usize, fingerprint: u16) -> u16 {
        let old = self.fingerprints[index];
        self.fingerprints[index] = fingerprint;
        old
    }
}

#[derive(Serialize, Deserialize)]
struct CuckooFilterData {
    buckets: Vec<Bucket>,
    num_buckets: usize,
    count: usize,
    seed: u64,
}

/// A space-efficient probabilistic data structure for set membership testing with deletion support.
///
/// Cuckoo filters offer better space efficiency than Bloom filters for low false positive rates
/// and support deletion without risk of false negatives (unlike Counting Bloom filters).
#[pyclass]
pub struct CuckooFilter {
    buckets: Vec<Bucket>,
    num_buckets: usize,
    count: usize,
    seed: u64,
}

impl CuckooFilter {
    fn hash(&self, item: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        hasher.finish() ^ self.seed
    }

    fn fingerprint(&self, item: &str) -> u16 {
        let mut hasher = DefaultHasher::new();
        (item, "fingerprint", self.seed).hash(&mut hasher);
        // Ensure fingerprint is non-zero (0 means empty slot)
        let fp = (hasher.finish() % 65535) as u16;
        if fp == 0 { 1 } else { fp }
    }

    fn index1(&self, item: &str) -> usize {
        (self.hash(item) as usize) % self.num_buckets
    }

    fn index2(&self, i1: usize, fingerprint: u16) -> usize {
        // Use partial-key cuckoo hashing: i2 = i1 XOR hash(fingerprint)
        let mut hasher = DefaultHasher::new();
        fingerprint.hash(&mut hasher);
        let fp_hash = hasher.finish() as usize;
        (i1 ^ fp_hash) % self.num_buckets
    }
}

#[pymethods]
impl CuckooFilter {
    /// Create a new Cuckoo filter.
    ///
    /// Args:
    ///     capacity: Maximum number of items the filter can hold
    ///     seed: Random seed for hash functions (default: 0)
    #[new]
    #[pyo3(signature = (capacity, seed=0))]
    fn new(capacity: usize, seed: u64) -> PyResult<Self> {
        if capacity == 0 {
            return Err(PyValueError::new_err("capacity must be positive"));
        }

        // Calculate number of buckets (round up, with some extra space)
        let num_buckets = ((capacity as f64 / BUCKET_SIZE as f64) * 1.1).ceil() as usize;
        let num_buckets = std::cmp::max(1, num_buckets);

        Ok(Self {
            buckets: vec![Bucket::default(); num_buckets],
            num_buckets,
            count: 0,
            seed,
        })
    }

    /// Add an item to the filter.
    ///
    /// Returns True if the item was added successfully, False if the filter is full.
    fn add(&mut self, item: &str) -> bool {
        let fingerprint = self.fingerprint(item);
        let i1 = self.index1(item);
        let i2 = self.index2(i1, fingerprint);

        // Try to insert in either bucket
        if self.buckets[i1].insert(fingerprint) {
            self.count += 1;
            return true;
        }
        if self.buckets[i2].insert(fingerprint) {
            self.count += 1;
            return true;
        }

        // Need to relocate existing items
        let mut current_index = if rand::random() { i1 } else { i2 };
        let mut current_fp = fingerprint;

        for _ in 0..MAX_KICKS {
            // Pick a random slot to evict
            let slot = rand::random::<usize>() % BUCKET_SIZE;
            current_fp = self.buckets[current_index].swap(slot, current_fp);

            // Find alternate bucket for evicted fingerprint
            current_index = self.index2(current_index, current_fp);

            if self.buckets[current_index].insert(current_fp) {
                self.count += 1;
                return true;
            }
        }

        // Filter is too full
        false
    }

    /// Add multiple items to the filter.
    ///
    /// Returns the number of items successfully added.
    fn update(&mut self, items: Vec<String>) -> usize {
        let mut added = 0;
        for item in items {
            if self.add(&item) {
                added += 1;
            }
        }
        added
    }

    /// Check if an item might be in the filter.
    fn contains(&self, item: &str) -> bool {
        let fingerprint = self.fingerprint(item);
        let i1 = self.index1(item);
        let i2 = self.index2(i1, fingerprint);

        self.buckets[i1].contains(fingerprint) || self.buckets[i2].contains(fingerprint)
    }

    /// Check if an item might be in the filter (alias for contains).
    fn query(&self, item: &str) -> bool {
        self.contains(item)
    }

    /// Remove an item from the filter.
    ///
    /// Returns True if the item was found and removed, False otherwise.
    fn remove(&mut self, item: &str) -> bool {
        let fingerprint = self.fingerprint(item);
        let i1 = self.index1(item);
        let i2 = self.index2(i1, fingerprint);

        if self.buckets[i1].remove(fingerprint) {
            self.count -= 1;
            return true;
        }
        if self.buckets[i2].remove(fingerprint) {
            self.count -= 1;
            return true;
        }

        false
    }

    /// Create a deep copy of this filter.
    fn copy(&self) -> Self {
        Self {
            buckets: self.buckets.clone(),
            num_buckets: self.num_buckets,
            count: self.count,
            seed: self.seed,
        }
    }

    /// Clear all items from the filter.
    fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.fingerprints = [0; BUCKET_SIZE];
        }
        self.count = 0;
    }

    /// Serialize the filter to bytes.
    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        let data = CuckooFilterData {
            buckets: self.buckets.clone(),
            num_buckets: self.num_buckets,
            count: self.count,
            seed: self.seed,
        };

        serde_json::to_vec(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }

    /// Deserialize a filter from bytes.
    #[staticmethod]
    fn from_bytes(data: Vec<u8>) -> PyResult<Self> {
        let cf_data: CuckooFilterData = serde_json::from_slice(&data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            buckets: cf_data.buckets,
            num_buckets: cf_data.num_buckets,
            count: cf_data.count,
            seed: cf_data.seed,
        })
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        let data = CuckooFilterData {
            buckets: self.buckets.clone(),
            num_buckets: self.num_buckets,
            count: self.count,
            seed: self.seed,
        };

        serde_json::to_string_pretty(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }

    /// Deserialize from JSON string.
    #[staticmethod]
    fn from_json(data: &str) -> PyResult<Self> {
        let cf_data: CuckooFilterData = serde_json::from_str(data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            buckets: cf_data.buckets,
            num_buckets: cf_data.num_buckets,
            count: cf_data.count,
            seed: cf_data.seed,
        })
    }

    fn __len__(&self) -> usize {
        self.count
    }

    fn __contains__(&self, item: &str) -> bool {
        self.contains(item)
    }

    fn __repr__(&self) -> String {
        format!(
            "CuckooFilter(capacity={}, count={}, load_factor={:.2}%)",
            self.capacity(),
            self.count,
            self.load_factor() * 100.0
        )
    }

    /// Get the maximum capacity of the filter.
    fn capacity(&self) -> usize {
        self.num_buckets * BUCKET_SIZE
    }

    /// Get the current load factor.
    fn load_factor(&self) -> f64 {
        self.count as f64 / self.capacity() as f64
    }

    #[getter]
    fn size_in_bytes(&self) -> usize {
        self.num_buckets * BUCKET_SIZE * 2  // 2 bytes per fingerprint
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }
}
