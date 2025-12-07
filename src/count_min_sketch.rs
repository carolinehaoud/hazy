use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Serialize, Deserialize)]
struct CountMinSketchData {
    table: Vec<Vec<u64>>,
    width: usize,
    depth: usize,
    count: usize,
    seed: u64,
}

/// A probabilistic data structure for frequency estimation in data streams.
///
/// Count-Min Sketch provides approximate frequency counts for items with
/// guaranteed error bounds. It may overestimate counts but never underestimates.
#[pyclass]
pub struct CountMinSketch {
    table: Vec<Vec<u64>>,
    width: usize,
    depth: usize,
    count: usize,  // Total number of items added
    seed: u64,
}

impl CountMinSketch {
    fn hash(&self, item: &str, row: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        (item, row, self.seed).hash(&mut hasher);
        (hasher.finish() as usize) % self.width
    }
}

#[pymethods]
impl CountMinSketch {
    /// Create a new Count-Min Sketch.
    ///
    /// Args:
    ///     width: Number of counters per row (higher = more accuracy)
    ///     depth: Number of hash functions/rows (higher = lower error probability)
    ///     error_rate: Target error rate (alternative to width)
    ///     confidence: Confidence level (alternative to depth)
    ///     seed: Random seed for hash functions (default: 0)
    ///
    /// Either provide (width, depth) or (error_rate, confidence).
    #[new]
    #[pyo3(signature = (width=None, depth=None, error_rate=None, confidence=None, seed=0))]
    fn new(
        width: Option<usize>,
        depth: Option<usize>,
        error_rate: Option<f64>,
        confidence: Option<f64>,
        seed: u64,
    ) -> PyResult<Self> {
        let (w, d) = match (width, depth, error_rate, confidence) {
            (Some(w), Some(d), _, _) => (w, d),
            (None, None, Some(e), Some(c)) => {
                if e <= 0.0 || e >= 1.0 {
                    return Err(PyValueError::new_err(
                        "error_rate must be between 0 and 1 (exclusive)"
                    ));
                }
                if c <= 0.0 || c >= 1.0 {
                    return Err(PyValueError::new_err(
                        "confidence must be between 0 and 1 (exclusive)"
                    ));
                }
                // width = ceil(e / epsilon)
                // depth = ceil(ln(1 / delta))
                let w = (std::f64::consts::E / e).ceil() as usize;
                let d = (1.0 / (1.0 - c)).ln().ceil() as usize;
                (w, d)
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Either provide (width, depth) or (error_rate, confidence)"
                ));
            }
        };

        if w == 0 {
            return Err(PyValueError::new_err("width must be positive"));
        }
        if d == 0 {
            return Err(PyValueError::new_err("depth must be positive"));
        }

        Ok(Self {
            table: vec![vec![0u64; w]; d],
            width: w,
            depth: d,
            count: 0,
            seed,
        })
    }

    /// Add an item to the sketch (increment its count by 1).
    fn add(&mut self, item: &str) {
        self.add_count(item, 1);
    }

    /// Add an item with a specific count.
    #[pyo3(signature = (item, count=1))]
    fn add_count(&mut self, item: &str, count: u64) {
        for row in 0..self.depth {
            let col = self.hash(item, row);
            self.table[row][col] = self.table[row][col].saturating_add(count);
        }
        self.count += count as usize;
    }

    /// Add multiple items to the sketch.
    fn update(&mut self, items: Vec<String>) {
        for item in items {
            self.add(&item);
        }
    }

    /// Query the estimated frequency of an item.
    fn query(&self, item: &str) -> u64 {
        let mut min_count = u64::MAX;

        for row in 0..self.depth {
            let col = self.hash(item, row);
            min_count = std::cmp::min(min_count, self.table[row][col]);
        }

        min_count
    }

    /// Get estimated frequency (alias for query).
    fn count(&self, item: &str) -> u64 {
        self.query(item)
    }

    /// Merge another Count-Min Sketch into this one.
    fn merge(&mut self, other: &CountMinSketch) -> PyResult<()> {
        if self.width != other.width {
            return Err(PyValueError::new_err(
                format!("Cannot merge: width mismatch ({} vs {})",
                    self.width, other.width)
            ));
        }
        if self.depth != other.depth {
            return Err(PyValueError::new_err(
                format!("Cannot merge: depth mismatch ({} vs {})",
                    self.depth, other.depth)
            ));
        }
        if self.seed != other.seed {
            return Err(PyValueError::new_err(
                format!("Cannot merge: seed mismatch ({} vs {})",
                    self.seed, other.seed)
            ));
        }

        for row in 0..self.depth {
            for col in 0..self.width {
                self.table[row][col] = self.table[row][col].saturating_add(other.table[row][col]);
            }
        }

        self.count += other.count;

        Ok(())
    }

    /// Create a deep copy of this sketch.
    fn copy(&self) -> Self {
        Self {
            table: self.table.clone(),
            width: self.width,
            depth: self.depth,
            count: self.count,
            seed: self.seed,
        }
    }

    /// Clear the sketch.
    fn clear(&mut self) {
        for row in &mut self.table {
            for counter in row {
                *counter = 0;
            }
        }
        self.count = 0;
    }

    /// Serialize to bytes.
    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        let data = CountMinSketchData {
            table: self.table.clone(),
            width: self.width,
            depth: self.depth,
            count: self.count,
            seed: self.seed,
        };

        serde_json::to_vec(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }

    /// Deserialize from bytes.
    #[staticmethod]
    fn from_bytes(data: Vec<u8>) -> PyResult<Self> {
        let cms_data: CountMinSketchData = serde_json::from_slice(&data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            table: cms_data.table,
            width: cms_data.width,
            depth: cms_data.depth,
            count: cms_data.count,
            seed: cms_data.seed,
        })
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        let data = CountMinSketchData {
            table: self.table.clone(),
            width: self.width,
            depth: self.depth,
            count: self.count,
            seed: self.seed,
        };

        serde_json::to_string_pretty(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }

    /// Deserialize from JSON string.
    #[staticmethod]
    fn from_json(data: &str) -> PyResult<Self> {
        let cms_data: CountMinSketchData = serde_json::from_str(data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        Ok(Self {
            table: cms_data.table,
            width: cms_data.width,
            depth: cms_data.depth,
            count: cms_data.count,
            seed: cms_data.seed,
        })
    }

    /// Total number of items added.
    fn __len__(&self) -> usize {
        self.count
    }

    /// Get item frequency using indexing.
    fn __getitem__(&self, item: &str) -> u64 {
        self.query(item)
    }

    fn __repr__(&self) -> String {
        format!(
            "CountMinSketch(width={}, depth={}, total_count={})",
            self.width, self.depth, self.count
        )
    }

    /// Get the width (number of counters per row).
    #[getter]
    fn width(&self) -> usize {
        self.width
    }

    /// Get the depth (number of rows/hash functions).
    #[getter]
    fn depth(&self) -> usize {
        self.depth
    }

    /// Get the memory footprint in bytes.
    #[getter]
    fn size_in_bytes(&self) -> usize {
        self.width * self.depth * 8  // 8 bytes per u64 counter
    }

    /// Get the seed.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    /// Get the total count of all items.
    #[getter]
    fn total_count(&self) -> usize {
        self.count
    }

    /// Get the error rate (epsilon).
    fn error_rate(&self) -> f64 {
        std::f64::consts::E / self.width as f64
    }

    /// Get the confidence level (1 - delta).
    fn confidence(&self) -> f64 {
        1.0 - std::f64::consts::E.powf(-(self.depth as f64))
    }

    /// Estimate the inner product with another sketch.
    fn inner_product(&self, other: &CountMinSketch) -> PyResult<u64> {
        if self.width != other.width || self.depth != other.depth {
            return Err(PyValueError::new_err(
                "Cannot compute inner product: dimension mismatch"
            ));
        }

        let mut min_product = u64::MAX;

        for row in 0..self.depth {
            let product: u64 = (0..self.width)
                .map(|col| self.table[row][col].saturating_mul(other.table[row][col]))
                .sum();
            min_product = std::cmp::min(min_product, product);
        }

        Ok(min_product)
    }
}
