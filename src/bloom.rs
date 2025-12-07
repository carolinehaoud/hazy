use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyString;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::utils::{double_hash, expected_fpr, hash_pair, optimal_bits, optimal_hashes};

const MAGIC: &[u8; 4] = b"HZBF"; // Hazy Bloom Filter
const VERSION: u8 = 1;

#[derive(Serialize, Deserialize)]
struct BloomFilterData {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: u32,
    count: usize,
    seed: u64,
}

/// A space-efficient probabilistic data structure for set membership testing.
///
/// Bloom filters can have false positives but never false negatives.
/// This means if `contains` returns False, the item is definitely not in the set.
/// If it returns True, the item is probably in the set.
///
/// Example:
///     >>> bf = BloomFilter(expected_items=10000)
///     >>> bf.add("hello")
///     >>> "hello" in bf
///     True
///     >>> "world" in bf
///     False
#[pyclass]
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: u32,
    count: usize,
    seed: u64,
}

impl BloomFilter {
    fn hash_item(&self, item: &[u8]) -> (u64, u64) {
        hash_pair(item, self.seed)
    }

    fn set_bit(&mut self, index: usize) {
        let word = index / 64;
        let bit = index % 64;
        self.bits[word] |= 1u64 << bit;
    }

    fn get_bit(&self, index: usize) -> bool {
        let word = index / 64;
        let bit = index % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    fn add_bytes(&mut self, item: &[u8]) {
        let (h1, h2) = self.hash_item(item);

        for i in 0..self.num_hashes {
            let index = double_hash(h1, h2, i, self.num_bits);
            self.set_bit(index);
        }

        self.count += 1;
    }

    fn contains_bytes(&self, item: &[u8]) -> bool {
        let (h1, h2) = self.hash_item(item);

        for i in 0..self.num_hashes {
            let index = double_hash(h1, h2, i, self.num_bits);
            if !self.get_bit(index) {
                return false;
            }
        }

        true
    }
}

#[pymethods]
impl BloomFilter {
    /// Create a new Bloom filter.
    ///
    /// Args:
    ///     expected_items: Expected number of items to be added (used if num_bits not provided)
    ///     false_positive_rate: Target false positive rate (default: 0.01)
    ///     num_bits: Explicit number of bits (overrides expected_items calculation)
    ///     num_hashes: Number of hash functions (calculated automatically if not provided)
    ///     seed: Random seed for hash functions (default: 0)
    #[new]
    #[pyo3(signature = (expected_items=None, false_positive_rate=0.01, num_bits=None, num_hashes=None, seed=0))]
    fn new(
        expected_items: Option<usize>,
        false_positive_rate: f64,
        num_bits: Option<usize>,
        num_hashes: Option<u32>,
        seed: u64,
    ) -> PyResult<Self> {
        // Validate false positive rate
        if false_positive_rate <= 0.0 || false_positive_rate >= 1.0 {
            return Err(PyValueError::new_err(
                "false_positive_rate must be between 0 and 1 (exclusive)"
            ));
        }

        // Calculate num_bits if not provided
        let num_bits = match (num_bits, expected_items) {
            (Some(nb), _) => {
                if nb == 0 {
                    return Err(PyValueError::new_err("num_bits must be positive"));
                }
                nb
            }
            (None, Some(ei)) => {
                if ei == 0 {
                    return Err(PyValueError::new_err("expected_items must be positive"));
                }
                optimal_bits(ei, false_positive_rate)
            }
            (None, None) => {
                return Err(PyValueError::new_err(
                    "Either expected_items or num_bits must be provided"
                ));
            }
        };

        // Calculate num_hashes if not provided
        let expected = expected_items.unwrap_or(num_bits / 10);
        let num_hashes = num_hashes.unwrap_or_else(|| optimal_hashes(num_bits, expected));

        if num_hashes == 0 {
            return Err(PyValueError::new_err("num_hashes must be positive"));
        }

        // Allocate bit array (rounded up to 64-bit words)
        let num_words = (num_bits + 63) / 64;

        Ok(Self {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes,
            count: 0,
            seed,
        })
    }

    /// Add an item to the Bloom filter.
    ///
    /// The item can be a string, bytes, or any object with __hash__.
    fn add(&mut self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = self.item_to_bytes(py, item)?;
        self.add_bytes(&bytes);
        Ok(())
    }

    /// Add multiple items to the Bloom filter.
    fn update(&mut self, py: Python<'_>, items: &Bound<'_, PyAny>) -> PyResult<()> {
        for item in items.iter()? {
            let item = item?;
            let bytes = self.item_to_bytes(py, &item)?;
            self.add_bytes(&bytes);
        }
        Ok(())
    }

    /// Check if an item might be in the Bloom filter.
    ///
    /// Returns True if the item might be in the set (possible false positive).
    /// Returns False if the item is definitely not in the set.
    fn contains(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        let bytes = self.item_to_bytes(py, item)?;
        Ok(self.contains_bytes(&bytes))
    }

    /// Check if an item might be in the Bloom filter (alias for contains).
    fn query(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.contains(py, item)
    }

    /// Merge another Bloom filter into this one (union operation).
    ///
    /// Both filters must have the same configuration (num_bits, num_hashes, seed).
    fn merge(&mut self, other: &BloomFilter) -> PyResult<()> {
        if self.num_bits != other.num_bits {
            return Err(PyValueError::new_err(
                format!("Cannot merge: num_bits mismatch ({} vs {})",
                    self.num_bits, other.num_bits)
            ));
        }
        if self.num_hashes != other.num_hashes {
            return Err(PyValueError::new_err(
                format!("Cannot merge: num_hashes mismatch ({} vs {})",
                    self.num_hashes, other.num_hashes)
            ));
        }
        if self.seed != other.seed {
            return Err(PyValueError::new_err(
                format!("Cannot merge: seed mismatch ({} vs {})",
                    self.seed, other.seed)
            ));
        }

        for i in 0..self.bits.len() {
            self.bits[i] |= other.bits[i];
        }

        // Count is approximate after merge
        self.count += other.count;

        Ok(())
    }

    /// Compute intersection with another Bloom filter.
    ///
    /// Returns a new filter containing items that might be in both filters.
    fn intersection(&self, other: &BloomFilter) -> PyResult<Self> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes || self.seed != other.seed {
            return Err(PyValueError::new_err(
                "Cannot intersect: filters must have same configuration"
            ));
        }

        let bits: Vec<u64> = self.bits.iter()
            .zip(other.bits.iter())
            .map(|(a, b)| a & b)
            .collect();

        Ok(Self {
            bits,
            num_bits: self.num_bits,
            num_hashes: self.num_hashes,
            count: 0,  // Unknown after intersection
            seed: self.seed,
        })
    }

    /// Create a deep copy of this Bloom filter.
    fn copy(&self) -> Self {
        Self {
            bits: self.bits.clone(),
            num_bits: self.num_bits,
            num_hashes: self.num_hashes,
            count: self.count,
            seed: self.seed,
        }
    }

    /// Clear all items from the Bloom filter.
    fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
        self.count = 0;
    }

    /// Serialize the Bloom filter to bytes (compact binary format).
    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        let data = BloomFilterData {
            bits: self.bits.clone(),
            num_bits: self.num_bits,
            num_hashes: self.num_hashes,
            count: self.count,
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

    /// Deserialize a Bloom filter from bytes.
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

        let bf_data: BloomFilterData = bincode::deserialize(&data[5..])
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            bits: bf_data.bits,
            num_bits: bf_data.num_bits,
            num_hashes: bf_data.num_hashes,
            count: bf_data.count,
            seed: bf_data.seed,
        })
    }

    /// Serialize the Bloom filter to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        let data = BloomFilterData {
            bits: self.bits.clone(),
            num_bits: self.num_bits,
            num_hashes: self.num_hashes,
            count: self.count,
            seed: self.seed,
        };

        serde_json::to_string_pretty(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }

    /// Deserialize a Bloom filter from a JSON string.
    #[staticmethod]
    fn from_json(data: &str) -> PyResult<Self> {
        let bf_data: BloomFilterData = serde_json::from_str(data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            bits: bf_data.bits,
            num_bits: bf_data.num_bits,
            num_hashes: bf_data.num_hashes,
            count: bf_data.count,
            seed: bf_data.seed,
        })
    }

    /// Save the Bloom filter to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        let file = File::create(path)
            .map_err(|e| PyValueError::new_err(format!("Cannot create file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        let data = self.to_bytes()?;
        writer.write_all(&data)
            .map_err(|e| PyValueError::new_err(format!("Write error: {}", e)))?;

        Ok(())
    }

    /// Load a Bloom filter from a file.
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

    /// Number of items added to the filter.
    fn __len__(&self) -> usize {
        self.count
    }

    /// Check membership using 'in' operator.
    fn __contains__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.contains(py, item)
    }

    /// Union of two Bloom filters using '|' operator.
    fn __or__(&self, other: &BloomFilter) -> PyResult<Self> {
        let mut result = self.copy();
        result.merge(other)?;
        Ok(result)
    }

    /// In-place union using '|=' operator.
    fn __ior__(&mut self, other: &BloomFilter) -> PyResult<()> {
        self.merge(other)
    }

    /// Intersection using '&' operator.
    fn __and__(&self, other: &BloomFilter) -> PyResult<Self> {
        self.intersection(other)
    }

    fn __repr__(&self) -> String {
        format!(
            "BloomFilter(num_bits={}, num_hashes={}, count={}, fpr={:.4})",
            self.num_bits, self.num_hashes, self.count, self.false_positive_rate()
        )
    }

    /// Get the number of bits in the filter.
    #[getter]
    fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Get the number of hash functions used.
    #[getter]
    fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Get the memory footprint in bytes.
    #[getter]
    fn size_in_bytes(&self) -> usize {
        self.bits.len() * 8
    }

    /// Get the expected false positive rate based on current fill.
    #[getter]
    fn false_positive_rate(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        expected_fpr(self.num_bits, self.num_hashes, self.count)
    }

    /// Get the seed used for hashing.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Get the number of bits set to 1.
    fn bit_count(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Get the fill ratio (fraction of bits set).
    fn fill_ratio(&self) -> f64 {
        self.bit_count() as f64 / self.num_bits as f64
    }

    /// Estimate the number of items in the filter based on fill ratio.
    ///
    /// This can be more accurate than `len()` after merging filters.
    fn estimate_count(&self) -> f64 {
        let x = self.bit_count() as f64;
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;

        // n ≈ -m/k * ln(1 - x/m)
        let ratio = x / m;
        if ratio >= 1.0 {
            return f64::INFINITY;
        }
        -m / k * (1.0 - ratio).ln()
    }
}

impl BloomFilter {
    /// Convert a Python object to bytes for hashing.
    fn item_to_bytes(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        // Fast path for strings
        if let Ok(s) = item.downcast::<PyString>() {
            return Ok(s.to_string().into_bytes());
        }

        // Fast path for bytes
        if let Ok(bytes) = item.extract::<Vec<u8>>() {
            return Ok(bytes);
        }

        // Fall back to using str() representation
        let s = item.str()?;
        Ok(s.to_string().into_bytes())
    }
}
