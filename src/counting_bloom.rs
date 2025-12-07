use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyString;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use crate::utils::{double_hash, hash_pair, optimal_bits, optimal_hashes};

const MAGIC: &[u8; 4] = b"HZCB"; // Hazy Counting Bloom
const VERSION: u8 = 1;

#[derive(Serialize, Deserialize)]
struct CountingBloomFilterData {
    counters: Vec<u8>,
    num_counters: usize,
    num_hashes: u32,
    count: usize,
    seed: u64,
}

/// A Bloom filter variant that supports deletion by using counters instead of bits.
///
/// Each position uses a counter (8 bits). This allows items to be removed from
/// the filter, but uses more memory than a standard Bloom filter.
#[pyclass]
pub struct CountingBloomFilter {
    counters: Vec<u8>,
    num_counters: usize,
    num_hashes: u32,
    count: usize,
    seed: u64,
}

impl CountingBloomFilter {
    fn hash_item(&self, item: &[u8]) -> (u64, u64) {
        hash_pair(item, self.seed)
    }

    fn add_bytes(&mut self, item: &[u8]) {
        let (h1, h2) = self.hash_item(item);
        for i in 0..self.num_hashes {
            let index = double_hash(h1, h2, i, self.num_counters);
            self.counters[index] = self.counters[index].saturating_add(1);
        }
        self.count += 1;
    }

    fn contains_bytes(&self, item: &[u8]) -> bool {
        let (h1, h2) = self.hash_item(item);
        for i in 0..self.num_hashes {
            let index = double_hash(h1, h2, i, self.num_counters);
            if self.counters[index] == 0 {
                return false;
            }
        }
        true
    }

    fn remove_bytes(&mut self, item: &[u8]) -> bool {
        if !self.contains_bytes(item) {
            return false;
        }

        let (h1, h2) = self.hash_item(item);
        for i in 0..self.num_hashes {
            let index = double_hash(h1, h2, i, self.num_counters);
            self.counters[index] = self.counters[index].saturating_sub(1);
        }

        if self.count > 0 {
            self.count -= 1;
        }
        true
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
impl CountingBloomFilter {
    /// Create a new Counting Bloom filter.
    #[new]
    #[pyo3(signature = (expected_items=None, false_positive_rate=0.01, num_counters=None, num_hashes=None, seed=0))]
    fn new(
        expected_items: Option<usize>,
        false_positive_rate: f64,
        num_counters: Option<usize>,
        num_hashes: Option<u32>,
        seed: u64,
    ) -> PyResult<Self> {
        if false_positive_rate <= 0.0 || false_positive_rate >= 1.0 {
            return Err(PyValueError::new_err(
                "false_positive_rate must be between 0 and 1 (exclusive)"
            ));
        }

        let num_counters = match (num_counters, expected_items) {
            (Some(nc), _) => {
                if nc == 0 {
                    return Err(PyValueError::new_err("num_counters must be positive"));
                }
                nc
            }
            (None, Some(ei)) => {
                if ei == 0 {
                    return Err(PyValueError::new_err("expected_items must be positive"));
                }
                optimal_bits(ei, false_positive_rate)
            }
            (None, None) => {
                return Err(PyValueError::new_err(
                    "Either expected_items or num_counters must be provided"
                ));
            }
        };

        let expected = expected_items.unwrap_or(num_counters / 10);
        let num_hashes = num_hashes.unwrap_or_else(|| optimal_hashes(num_counters, expected));

        if num_hashes == 0 {
            return Err(PyValueError::new_err("num_hashes must be positive"));
        }

        Ok(Self {
            counters: vec![0u8; num_counters],
            num_counters,
            num_hashes,
            count: 0,
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

    /// Remove an item from the filter.
    fn remove(&mut self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        let bytes = self.item_to_bytes(py, item)?;
        Ok(self.remove_bytes(&bytes))
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

    /// Merge another Counting Bloom filter into this one.
    fn merge(&mut self, other: &CountingBloomFilter) -> PyResult<()> {
        if self.num_counters != other.num_counters {
            return Err(PyValueError::new_err(
                format!("Cannot merge: num_counters mismatch ({} vs {})",
                    self.num_counters, other.num_counters)
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

        for i in 0..self.counters.len() {
            self.counters[i] = self.counters[i].saturating_add(other.counters[i]);
        }

        self.count += other.count;

        Ok(())
    }

    /// Create a deep copy of this filter.
    fn copy(&self) -> Self {
        Self {
            counters: self.counters.clone(),
            num_counters: self.num_counters,
            num_hashes: self.num_hashes,
            count: self.count,
            seed: self.seed,
        }
    }

    /// Clear all items from the filter.
    fn clear(&mut self) {
        for counter in &mut self.counters {
            *counter = 0;
        }
        self.count = 0;
    }

    /// Serialize the filter to bytes.
    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        let data = CountingBloomFilterData {
            counters: self.counters.clone(),
            num_counters: self.num_counters,
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

    /// Deserialize a filter from bytes.
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

        let bf_data: CountingBloomFilterData = bincode::deserialize(&data[5..])
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            counters: bf_data.counters,
            num_counters: bf_data.num_counters,
            num_hashes: bf_data.num_hashes,
            count: bf_data.count,
            seed: bf_data.seed,
        })
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        let data = CountingBloomFilterData {
            counters: self.counters.clone(),
            num_counters: self.num_counters,
            num_hashes: self.num_hashes,
            count: self.count,
            seed: self.seed,
        };

        serde_json::to_string_pretty(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }

    /// Deserialize from JSON string.
    #[staticmethod]
    fn from_json(data: &str) -> PyResult<Self> {
        let bf_data: CountingBloomFilterData = serde_json::from_str(data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            counters: bf_data.counters,
            num_counters: bf_data.num_counters,
            num_hashes: bf_data.num_hashes,
            count: bf_data.count,
            seed: bf_data.seed,
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

    fn __len__(&self) -> usize {
        self.count
    }

    fn __contains__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.contains(py, item)
    }

    fn __repr__(&self) -> String {
        format!(
            "CountingBloomFilter(num_counters={}, num_hashes={}, count={})",
            self.num_counters, self.num_hashes, self.count
        )
    }

    #[getter]
    fn num_counters(&self) -> usize {
        self.num_counters
    }

    #[getter]
    fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    #[getter]
    fn size_in_bytes(&self) -> usize {
        self.counters.len()
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }
}
