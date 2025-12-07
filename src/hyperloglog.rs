use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::utils::hll_rank;

const ALPHA_16: f64 = 0.673;
const ALPHA_32: f64 = 0.697;
const ALPHA_64: f64 = 0.709;

fn alpha(m: usize) -> f64 {
    match m {
        16 => ALPHA_16,
        32 => ALPHA_32,
        64 => ALPHA_64,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

#[derive(Serialize, Deserialize)]
struct HyperLogLogData {
    registers: Vec<u8>,
    precision: u8,
    count: usize,
    seed: u64,
}

/// A probabilistic data structure for cardinality estimation.
///
/// HyperLogLog can estimate the number of distinct elements in a multiset
/// using a small, fixed amount of memory. It provides ~2% typical error
/// with just a few kilobytes of memory.
#[pyclass]
pub struct HyperLogLog {
    registers: Vec<u8>,
    precision: u8,
    count: usize,  // Number of add operations (not cardinality)
    seed: u64,
}

impl HyperLogLog {
    fn hash(&self, item: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        (item, self.seed).hash(&mut hasher);
        hasher.finish()
    }

    fn num_registers(&self) -> usize {
        1 << self.precision
    }
}

#[pymethods]
impl HyperLogLog {
    /// Create a new HyperLogLog counter.
    ///
    /// Args:
    ///     precision: Number of bits for register indexing (4-18, default: 14)
    ///                Higher precision = more accuracy but more memory
    ///                Memory usage = 2^precision bytes
    ///     seed: Random seed for hash function (default: 0)
    #[new]
    #[pyo3(signature = (precision=14, seed=0))]
    fn new(precision: u8, seed: u64) -> PyResult<Self> {
        if precision < 4 || precision > 18 {
            return Err(PyValueError::new_err(
                "precision must be between 4 and 18"
            ));
        }

        let num_registers = 1 << precision;

        Ok(Self {
            registers: vec![0u8; num_registers],
            precision,
            count: 0,
            seed,
        })
    }

    /// Add an item to the HyperLogLog counter.
    fn add(&mut self, item: &str) {
        let hash = self.hash(item);

        // Get register index from first precision bits
        let index = (hash >> (64 - self.precision)) as usize;

        // Get rank (position of first 1 bit) from remaining bits
        let rank = hll_rank(hash, self.precision);

        // Update register if we found a higher rank
        if rank > self.registers[index] {
            self.registers[index] = rank;
        }

        self.count += 1;
    }

    /// Add multiple items to the counter.
    fn update(&mut self, items: Vec<String>) {
        for item in items {
            self.add(&item);
        }
    }

    /// Estimate the cardinality (number of distinct elements).
    fn cardinality(&self) -> f64 {
        let m = self.num_registers() as f64;
        let alpha_m = alpha(self.num_registers());

        // Calculate harmonic mean of 2^(-register)
        let sum: f64 = self.registers
            .iter()
            .map(|&r| 2f64.powi(-(r as i32)))
            .sum();

        let raw_estimate = alpha_m * m * m / sum;

        // Apply bias corrections
        if raw_estimate <= 2.5 * m {
            // Small range correction using linear counting
            let zeros = self.registers.iter().filter(|&&r| r == 0).count() as f64;
            if zeros > 0.0 {
                m * (m / zeros).ln()
            } else {
                raw_estimate
            }
        } else if raw_estimate > (1.0 / 30.0) * 2f64.powi(32) {
            // Large range correction
            -2f64.powi(32) * (1.0 - raw_estimate / 2f64.powi(32)).ln()
        } else {
            raw_estimate
        }
    }

    /// Query the estimated cardinality (alias for cardinality).
    fn query(&self) -> f64 {
        self.cardinality()
    }

    /// Merge another HyperLogLog into this one.
    ///
    /// After merging, this counter will estimate the cardinality of the union.
    fn merge(&mut self, other: &HyperLogLog) -> PyResult<()> {
        if self.precision != other.precision {
            return Err(PyValueError::new_err(
                format!("Cannot merge: precision mismatch ({} vs {})",
                    self.precision, other.precision)
            ));
        }

        for i in 0..self.registers.len() {
            self.registers[i] = std::cmp::max(self.registers[i], other.registers[i]);
        }

        self.count += other.count;

        Ok(())
    }

    /// Create a deep copy of this counter.
    fn copy(&self) -> Self {
        Self {
            registers: self.registers.clone(),
            precision: self.precision,
            count: self.count,
            seed: self.seed,
        }
    }

    /// Clear the counter.
    fn clear(&mut self) {
        for reg in &mut self.registers {
            *reg = 0;
        }
        self.count = 0;
    }

    /// Serialize to bytes.
    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        let data = HyperLogLogData {
            registers: self.registers.clone(),
            precision: self.precision,
            count: self.count,
            seed: self.seed,
        };

        serde_json::to_vec(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }

    /// Deserialize from bytes.
    #[staticmethod]
    fn from_bytes(data: Vec<u8>) -> PyResult<Self> {
        let hll_data: HyperLogLogData = serde_json::from_slice(&data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            registers: hll_data.registers,
            precision: hll_data.precision,
            count: hll_data.count,
            seed: hll_data.seed,
        })
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        let data = HyperLogLogData {
            registers: self.registers.clone(),
            precision: self.precision,
            count: self.count,
            seed: self.seed,
        };

        serde_json::to_string_pretty(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }

    /// Deserialize from JSON string.
    #[staticmethod]
    fn from_json(data: &str) -> PyResult<Self> {
        let hll_data: HyperLogLogData = serde_json::from_str(data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            registers: hll_data.registers,
            precision: hll_data.precision,
            count: hll_data.count,
            seed: hll_data.seed,
        })
    }

    /// Return the estimated cardinality as an integer.
    fn __len__(&self) -> usize {
        self.cardinality().round() as usize
    }

    /// Union of two HyperLogLog counters using '|' operator.
    fn __or__(&self, other: &HyperLogLog) -> PyResult<Self> {
        let mut result = self.copy();
        result.merge(other)?;
        Ok(result)
    }

    /// In-place union using '|=' operator.
    fn __ior__(&mut self, other: &HyperLogLog) -> PyResult<()> {
        self.merge(other)
    }

    fn __repr__(&self) -> String {
        format!(
            "HyperLogLog(precision={}, cardinality={:.0}, memory={}B)",
            self.precision,
            self.cardinality(),
            self.size_in_bytes()
        )
    }

    /// Get the precision (number of bits for indexing).
    #[getter]
    fn precision(&self) -> u8 {
        self.precision
    }

    /// Get the memory footprint in bytes.
    #[getter]
    fn size_in_bytes(&self) -> usize {
        self.registers.len()
    }

    /// Get the seed.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Get the standard error of the estimate.
    #[getter]
    fn standard_error(&self) -> f64 {
        1.04 / (self.num_registers() as f64).sqrt()
    }

    /// Get relative error (as a percentage).
    fn relative_error(&self) -> f64 {
        self.standard_error() * 100.0
    }
}
