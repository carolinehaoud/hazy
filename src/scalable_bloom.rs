use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyString;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use crate::utils::{double_hash, hash_pair, optimal_bits, optimal_hashes};

const MAGIC: &[u8; 4] = b"HZSB"; // Hazy Scalable Bloom
const VERSION: u8 = 1;
const DEFAULT_GROWTH_RATIO: f64 = 2.0;
const DEFAULT_FPR_RATIO: f64 = 0.5; // Each filter has tighter FPR

#[derive(Clone, Serialize, Deserialize)]
struct BloomFilterSlice {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: u32,
    count: usize,
    capacity: usize,
    fpr: f64,
}

impl BloomFilterSlice {
    fn new(capacity: usize, fpr: f64, seed: u64) -> Self {
        let num_bits = optimal_bits(capacity, fpr);
        let num_hashes = optimal_hashes(num_bits, capacity);
        let num_words = (num_bits + 63) / 64;

        Self {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes,
            count: 0,
            capacity,
            fpr,
        }
    }

    fn add(&mut self, h1: u64, h2: u64) {
        for i in 0..self.num_hashes {
            let index = double_hash(h1, h2, i, self.num_bits);
            let word = index / 64;
            let bit = index % 64;
            self.bits[word] |= 1u64 << bit;
        }
        self.count += 1;
    }

    fn contains(&self, h1: u64, h2: u64) -> bool {
        for i in 0..self.num_hashes {
            let index = double_hash(h1, h2, i, self.num_bits);
            let word = index / 64;
            let bit = index % 64;
            if (self.bits[word] >> bit) & 1 == 0 {
                return false;
            }
        }
        true
    }

    fn is_full(&self) -> bool {
        self.count >= self.capacity
    }

    fn size_in_bytes(&self) -> usize {
        self.bits.len() * 8
    }
}

#[derive(Serialize, Deserialize)]
struct ScalableBloomFilterData {
    slices: Vec<BloomFilterSlice>,
    initial_capacity: usize,
    initial_fpr: f64,
    growth_ratio: f64,
    fpr_ratio: f64,
    seed: u64,
}

/// A Bloom filter that automatically scales to accommodate more items.
///
/// Unlike a standard Bloom filter, a Scalable Bloom Filter grows dynamically
/// by adding new filter slices when the current one fills up. This allows
/// it to handle an unknown number of items while maintaining the target
/// false positive rate.
///
/// Example:
///     >>> sbf = ScalableBloomFilter(initial_capacity=1000)
///     >>> for i in range(100000):  # Much more than initial capacity
///     ...     sbf.add(f"item_{i}")
///     >>> "item_500" in sbf
///     True
#[pyclass]
pub struct ScalableBloomFilter {
    slices: Vec<BloomFilterSlice>,
    initial_capacity: usize,
    initial_fpr: f64,
    growth_ratio: f64,
    fpr_ratio: f64,
    seed: u64,
}

impl ScalableBloomFilter {
    fn hash_item(&self, item: &[u8]) -> (u64, u64) {
        hash_pair(item, self.seed)
    }

    fn next_slice_params(&self) -> (usize, f64) {
        if self.slices.is_empty() {
            (self.initial_capacity, self.initial_fpr)
        } else {
            let last = self.slices.last().unwrap();
            let new_capacity = (last.capacity as f64 * self.growth_ratio) as usize;
            let new_fpr = last.fpr * self.fpr_ratio;
            (new_capacity, new_fpr)
        }
    }

    fn add_slice(&mut self) {
        let (capacity, fpr) = self.next_slice_params();
        self.slices.push(BloomFilterSlice::new(capacity, fpr, self.seed));
    }

    fn add_bytes(&mut self, item: &[u8]) {
        // Ensure we have at least one slice
        if self.slices.is_empty() {
            self.add_slice();
        }

        let (h1, h2) = self.hash_item(item);

        // Check if already present (to avoid duplicates inflating count)
        if self.contains_hashes(h1, h2) {
            return;
        }

        // Add to the last slice, creating new one if needed
        if self.slices.last().unwrap().is_full() {
            self.add_slice();
        }
        self.slices.last_mut().unwrap().add(h1, h2);
    }

    fn contains_hashes(&self, h1: u64, h2: u64) -> bool {
        self.slices.iter().any(|slice| slice.contains(h1, h2))
    }

    fn contains_bytes(&self, item: &[u8]) -> bool {
        let (h1, h2) = self.hash_item(item);
        self.contains_hashes(h1, h2)
    }

    fn item_to_bytes(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        if let Ok(s) = item.downcast::<PyString>() {
            return Ok(s.to_string().into_bytes());
        }
        if let Ok(bytes) = item.extract::<Vec<u8>>() {
            return Ok(bytes);
        }
        let s = item.str()?;
        Ok(s.to_string().into_bytes())
    }
}

#[pymethods]
impl ScalableBloomFilter {
    /// Create a new Scalable Bloom Filter.
    ///
    /// Args:
    ///     initial_capacity: Initial number of items before first resize (default: 1000)
    ///     false_positive_rate: Target false positive rate (default: 0.01)
    ///     growth_ratio: Factor by which capacity grows for each new slice (default: 2.0)
    ///     fpr_ratio: Factor by which FPR decreases for each new slice (default: 0.5)
    ///     seed: Random seed for hash functions (default: 0)
    #[new]
    #[pyo3(signature = (initial_capacity=1000, false_positive_rate=0.01, growth_ratio=2.0, fpr_ratio=0.5, seed=0))]
    fn new(
        initial_capacity: usize,
        false_positive_rate: f64,
        growth_ratio: f64,
        fpr_ratio: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if initial_capacity == 0 {
            return Err(PyValueError::new_err("initial_capacity must be positive"));
        }
        if false_positive_rate <= 0.0 || false_positive_rate >= 1.0 {
            return Err(PyValueError::new_err(
                "false_positive_rate must be between 0 and 1 (exclusive)"
            ));
        }
        if growth_ratio <= 1.0 {
            return Err(PyValueError::new_err("growth_ratio must be greater than 1"));
        }
        if fpr_ratio <= 0.0 || fpr_ratio >= 1.0 {
            return Err(PyValueError::new_err(
                "fpr_ratio must be between 0 and 1 (exclusive)"
            ));
        }

        Ok(Self {
            slices: Vec::new(),
            initial_capacity,
            initial_fpr: false_positive_rate,
            growth_ratio,
            fpr_ratio,
            seed,
        })
    }

    /// Add an item to the filter.
    fn add(&mut self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = self.item_to_bytes(py, item)?;
        self.add_bytes(&bytes);
        Ok(())
    }

    /// Add multiple items to the filter.
    fn update(&mut self, py: Python<'_>, items: &Bound<'_, PyAny>) -> PyResult<()> {
        for item in items.iter()? {
            let item = item?;
            let bytes = self.item_to_bytes(py, &item)?;
            self.add_bytes(&bytes);
        }
        Ok(())
    }

    /// Check if an item might be in the filter.
    fn contains(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        let bytes = self.item_to_bytes(py, item)?;
        Ok(self.contains_bytes(&bytes))
    }

    /// Check if an item might be in the filter (alias for contains).
    fn query(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.contains(py, item)
    }

    /// Create a deep copy.
    fn copy(&self) -> Self {
        Self {
            slices: self.slices.clone(),
            initial_capacity: self.initial_capacity,
            initial_fpr: self.initial_fpr,
            growth_ratio: self.growth_ratio,
            fpr_ratio: self.fpr_ratio,
            seed: self.seed,
        }
    }

    /// Clear all items.
    fn clear(&mut self) {
        self.slices.clear();
    }

    /// Serialize to bytes.
    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        let data = ScalableBloomFilterData {
            slices: self.slices.clone(),
            initial_capacity: self.initial_capacity,
            initial_fpr: self.initial_fpr,
            growth_ratio: self.growth_ratio,
            fpr_ratio: self.fpr_ratio,
            seed: self.seed,
        };

        let mut result = Vec::new();
        result.extend_from_slice(MAGIC);
        result.push(VERSION);

        let encoded = bincode::serialize(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))?;
        result.extend(encoded);

        Ok(result)
    }

    /// Deserialize from bytes.
    #[staticmethod]
    fn from_bytes(data: Vec<u8>) -> PyResult<Self> {
        if data.len() < 5 || &data[0..4] != MAGIC {
            return Err(PyValueError::new_err("Invalid data format"));
        }
        if data[4] != VERSION {
            return Err(PyValueError::new_err(format!(
                "Unsupported version: {} (expected {})", data[4], VERSION
            )));
        }

        let sbf_data: ScalableBloomFilterData = bincode::deserialize(&data[5..])
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            slices: sbf_data.slices,
            initial_capacity: sbf_data.initial_capacity,
            initial_fpr: sbf_data.initial_fpr,
            growth_ratio: sbf_data.growth_ratio,
            fpr_ratio: sbf_data.fpr_ratio,
            seed: sbf_data.seed,
        })
    }

    /// Save to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        let file = File::create(path)
            .map_err(|e| PyValueError::new_err(format!("Cannot create file: {}", e)))?;
        let mut writer = BufWriter::new(file);
        let data = self.to_bytes()?;
        writer.write_all(&data)
            .map_err(|e| PyValueError::new_err(format!("Write error: {}", e)))?;
        Ok(())
    }

    /// Load from a file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|e| PyValueError::new_err(format!("Cannot open file: {}", e)))?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)
            .map_err(|e| PyValueError::new_err(format!("Read error: {}", e)))?;
        Self::from_bytes(data)
    }

    /// Total number of items added.
    fn __len__(&self) -> usize {
        self.slices.iter().map(|s| s.count).sum()
    }

    /// Check membership using 'in' operator.
    fn __contains__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.contains(py, item)
    }

    fn __repr__(&self) -> String {
        format!(
            "ScalableBloomFilter(slices={}, count={}, size={}B)",
            self.slices.len(),
            self.__len__(),
            self.size_in_bytes()
        )
    }

    /// Number of filter slices.
    #[getter]
    fn num_slices(&self) -> usize {
        self.slices.len()
    }

    /// Total memory footprint in bytes.
    #[getter]
    fn size_in_bytes(&self) -> usize {
        self.slices.iter().map(|s| s.size_in_bytes()).sum()
    }

    /// Get the seed.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Current total capacity across all slices.
    fn capacity(&self) -> usize {
        self.slices.iter().map(|s| s.capacity).sum()
    }

    /// Expected false positive rate.
    ///
    /// For a scalable Bloom filter, the total FPR is approximately
    /// the sum of FPRs of all slices.
    fn false_positive_rate(&self) -> f64 {
        // Total FPR ≈ sum of individual FPRs
        self.slices.iter().map(|s| s.fpr).sum()
    }
}
