use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Serialize, Deserialize)]
struct MinHashData {
    signature: Vec<u64>,
    num_hashes: usize,
    seed: u64,
}

/// A probabilistic data structure for estimating set similarity using Jaccard index.
///
/// MinHash creates a compact signature of a set that can be used to estimate
/// the Jaccard similarity between sets without storing the actual elements.
#[pyclass]
pub struct MinHash {
    signature: Vec<u64>,
    num_hashes: usize,
    seed: u64,
}

impl MinHash {
    fn hash(&self, item: &str, hash_index: usize) -> u64 {
        let mut hasher = DefaultHasher::new();
        (item, hash_index, self.seed).hash(&mut hasher);
        hasher.finish()
    }
}

#[pymethods]
impl MinHash {
    /// Create a new MinHash signature.
    ///
    /// Args:
    ///     num_hashes: Number of hash functions (higher = more accurate, default: 128)
    ///     seed: Random seed for hash functions (default: 0)
    #[new]
    #[pyo3(signature = (num_hashes=128, seed=0))]
    fn new(num_hashes: usize, seed: u64) -> PyResult<Self> {
        if num_hashes == 0 {
            return Err(PyValueError::new_err("num_hashes must be positive"));
        }

        Ok(Self {
            signature: vec![u64::MAX; num_hashes],
            num_hashes,
            seed,
        })
    }

    /// Add an item to the MinHash signature.
    fn add(&mut self, item: &str) {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            if hash < self.signature[i] {
                self.signature[i] = hash;
            }
        }
    }

    /// Add multiple items to the signature.
    fn update(&mut self, items: Vec<String>) {
        for item in items {
            self.add(&item);
        }
    }

    /// Estimate the Jaccard similarity with another MinHash.
    ///
    /// Returns a value between 0.0 (completely different) and 1.0 (identical).
    fn jaccard(&self, other: &MinHash) -> PyResult<f64> {
        if self.num_hashes != other.num_hashes {
            return Err(PyValueError::new_err(
                format!("Cannot compare: num_hashes mismatch ({} vs {})",
                    self.num_hashes, other.num_hashes)
            ));
        }
        if self.seed != other.seed {
            return Err(PyValueError::new_err(
                format!("Cannot compare: seed mismatch ({} vs {})",
                    self.seed, other.seed)
            ));
        }

        let matches: usize = self.signature
            .iter()
            .zip(other.signature.iter())
            .filter(|(a, b)| a == b)
            .count();

        Ok(matches as f64 / self.num_hashes as f64)
    }

    /// Query similarity with another MinHash (alias for jaccard).
    fn query(&self, other: &MinHash) -> PyResult<f64> {
        self.jaccard(other)
    }

    /// Merge another MinHash into this one (union of the underlying sets).
    fn merge(&mut self, other: &MinHash) -> PyResult<()> {
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

        for i in 0..self.num_hashes {
            self.signature[i] = std::cmp::min(self.signature[i], other.signature[i]);
        }

        Ok(())
    }

    /// Create a deep copy of this MinHash.
    fn copy(&self) -> Self {
        Self {
            signature: self.signature.clone(),
            num_hashes: self.num_hashes,
            seed: self.seed,
        }
    }

    /// Clear the signature (reset to initial state).
    fn clear(&mut self) {
        for h in &mut self.signature {
            *h = u64::MAX;
        }
    }

    /// Check if the signature is empty (no items added).
    fn is_empty(&self) -> bool {
        self.signature.iter().all(|&h| h == u64::MAX)
    }

    /// Serialize to bytes.
    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        let data = MinHashData {
            signature: self.signature.clone(),
            num_hashes: self.num_hashes,
            seed: self.seed,
        };

        serde_json::to_vec(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }

    /// Deserialize from bytes.
    #[staticmethod]
    fn from_bytes(data: Vec<u8>) -> PyResult<Self> {
        let mh_data: MinHashData = serde_json::from_slice(&data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            signature: mh_data.signature,
            num_hashes: mh_data.num_hashes,
            seed: mh_data.seed,
        })
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        let data = MinHashData {
            signature: self.signature.clone(),
            num_hashes: self.num_hashes,
            seed: self.seed,
        };

        serde_json::to_string_pretty(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }

    /// Deserialize from JSON string.
    #[staticmethod]
    fn from_json(data: &str) -> PyResult<Self> {
        let mh_data: MinHashData = serde_json::from_str(data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            signature: mh_data.signature,
            num_hashes: mh_data.num_hashes,
            seed: mh_data.seed,
        })
    }

    /// Union of two MinHash signatures using '|' operator.
    fn __or__(&self, other: &MinHash) -> PyResult<Self> {
        let mut result = self.copy();
        result.merge(other)?;
        Ok(result)
    }

    /// In-place union using '|=' operator.
    fn __ior__(&mut self, other: &MinHash) -> PyResult<()> {
        self.merge(other)
    }

    fn __repr__(&self) -> String {
        format!(
            "MinHash(num_hashes={}, is_empty={})",
            self.num_hashes,
            self.is_empty()
        )
    }

    /// Get the number of hash functions.
    #[getter]
    fn num_hashes(&self) -> usize {
        self.num_hashes
    }

    /// Get the memory footprint in bytes.
    #[getter]
    fn size_in_bytes(&self) -> usize {
        self.signature.len() * 8  // 8 bytes per u64
    }

    /// Get the seed.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Get the signature as a list of integers.
    fn get_signature(&self) -> Vec<u64> {
        self.signature.clone()
    }

    /// Get the standard error of the Jaccard estimate.
    fn standard_error(&self) -> f64 {
        1.0 / (self.num_hashes as f64).sqrt()
    }
}
