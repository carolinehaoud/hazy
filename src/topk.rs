use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize)]
struct Counter {
    item: String,
    count: u64,
    error: u64,  // Maximum possible error in count
}

#[derive(Serialize, Deserialize)]
struct TopKData {
    counters: Vec<Counter>,
    k: usize,
}

/// A probabilistic data structure for finding the most frequent items in a stream.
///
/// This implements the Space-Saving algorithm, which maintains a fixed-size
/// list of items and their approximate frequencies. It guarantees that all
/// items with frequency >= n/k are tracked (where n is total count).
#[pyclass]
pub struct TopK {
    counters: Vec<Counter>,
    k: usize,
    item_index: HashMap<String, usize>,  // Maps item to its position in counters
}

impl TopK {
    fn rebuild_index(&mut self) {
        self.item_index.clear();
        for (i, counter) in self.counters.iter().enumerate() {
            self.item_index.insert(counter.item.clone(), i);
        }
    }
}

#[pymethods]
impl TopK {
    /// Create a new Top-K tracker.
    ///
    /// Args:
    ///     k: Maximum number of items to track
    #[new]
    fn new(k: usize) -> PyResult<Self> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be positive"));
        }

        Ok(Self {
            counters: Vec::with_capacity(k),
            k,
            item_index: HashMap::with_capacity(k),
        })
    }

    /// Add an item to the tracker (increment its count by 1).
    fn add(&mut self, item: &str) {
        self.add_count(item, 1);
    }

    /// Add an item with a specific count.
    #[pyo3(signature = (item, count=1))]
    fn add_count(&mut self, item: &str, count: u64) {
        // Check if item is already tracked
        if let Some(&index) = self.item_index.get(item) {
            self.counters[index].count += count;

            // Maintain sorted order by moving item up if needed
            let mut i = index;
            while i > 0 && self.counters[i].count > self.counters[i - 1].count {
                self.counters.swap(i, i - 1);
                // Update index for both swapped items
                self.item_index.insert(self.counters[i].item.clone(), i);
                self.item_index.insert(self.counters[i - 1].item.clone(), i - 1);
                i -= 1;
            }
            return;
        }

        // Item not tracked - need to add or replace
        if self.counters.len() < self.k {
            // Still have space, just add
            let new_counter = Counter {
                item: item.to_string(),
                count,
                error: 0,
            };
            self.counters.push(new_counter);
            self.item_index.insert(item.to_string(), self.counters.len() - 1);

            // Sort to maintain order
            let mut i = self.counters.len() - 1;
            while i > 0 && self.counters[i].count > self.counters[i - 1].count {
                self.counters.swap(i, i - 1);
                self.item_index.insert(self.counters[i].item.clone(), i);
                self.item_index.insert(self.counters[i - 1].item.clone(), i - 1);
                i -= 1;
            }
        } else {
            // Replace the item with minimum count
            let min_idx = self.counters.len() - 1;
            let min_count = self.counters[min_idx].count;

            // Remove old item from index
            self.item_index.remove(&self.counters[min_idx].item);

            // Replace with new item
            self.counters[min_idx] = Counter {
                item: item.to_string(),
                count: min_count + count,  // Start from min_count + new count
                error: min_count,  // Error is the evicted count
            };
            self.item_index.insert(item.to_string(), min_idx);

            // Sort to maintain order
            let mut i = min_idx;
            while i > 0 && self.counters[i].count > self.counters[i - 1].count {
                self.counters.swap(i, i - 1);
                self.item_index.insert(self.counters[i].item.clone(), i);
                self.item_index.insert(self.counters[i - 1].item.clone(), i - 1);
                i -= 1;
            }
        }
    }

    /// Add multiple items.
    fn update(&mut self, items: Vec<String>) {
        for item in items {
            self.add(&item);
        }
    }

    /// Query the estimated count of an item.
    ///
    /// Returns 0 if the item is not being tracked.
    fn query(&self, item: &str) -> u64 {
        if let Some(&index) = self.item_index.get(item) {
            self.counters[index].count
        } else {
            0
        }
    }

    /// Get the count of an item (alias for query).
    fn count(&self, item: &str) -> u64 {
        self.query(item)
    }

    /// Check if an item is being tracked.
    fn contains(&self, item: &str) -> bool {
        self.item_index.contains_key(item)
    }

    /// Get the top-k items with their counts.
    ///
    /// Returns a list of (item, count) tuples sorted by count (descending).
    #[pyo3(signature = (n=None))]
    fn top(&self, n: Option<usize>) -> Vec<(String, u64)> {
        let limit = n.unwrap_or(self.k).min(self.counters.len());
        self.counters[..limit]
            .iter()
            .map(|c| (c.item.clone(), c.count))
            .collect()
    }

    /// Get the top-k items with their counts and error bounds.
    ///
    /// Returns a list of (item, count, error) tuples.
    /// True count is in range [count - error, count].
    #[pyo3(signature = (n=None))]
    fn top_with_error(&self, n: Option<usize>) -> Vec<(String, u64, u64)> {
        let limit = n.unwrap_or(self.k).min(self.counters.len());
        self.counters[..limit]
            .iter()
            .map(|c| (c.item.clone(), c.count, c.error))
            .collect()
    }

    /// Get the minimum count among tracked items.
    fn min_count(&self) -> u64 {
        self.counters.last().map(|c| c.count).unwrap_or(0)
    }

    /// Merge another TopK into this one.
    fn merge(&mut self, other: &TopK) -> PyResult<()> {
        // Add all items from other
        for counter in &other.counters {
            self.add_count(&counter.item, counter.count);
        }
        Ok(())
    }

    /// Create a deep copy.
    fn copy(&self) -> Self {
        let mut result = Self {
            counters: self.counters.clone(),
            k: self.k,
            item_index: HashMap::new(),
        };
        result.rebuild_index();
        result
    }

    /// Clear the tracker.
    fn clear(&mut self) {
        self.counters.clear();
        self.item_index.clear();
    }

    /// Serialize to bytes.
    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        let data = TopKData {
            counters: self.counters.clone(),
            k: self.k,
        };

        serde_json::to_vec(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }

    /// Deserialize from bytes.
    #[staticmethod]
    fn from_bytes(data: Vec<u8>) -> PyResult<Self> {
        let tk_data: TopKData = serde_json::from_slice(&data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        let mut result = Self {
            counters: tk_data.counters,
            k: tk_data.k,
            item_index: HashMap::new(),
        };
        result.rebuild_index();
        Ok(result)
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        let data = TopKData {
            counters: self.counters.clone(),
            k: self.k,
        };

        serde_json::to_string_pretty(&data)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
    }

    /// Deserialize from JSON string.
    #[staticmethod]
    fn from_json(data: &str) -> PyResult<Self> {
        let tk_data: TopKData = serde_json::from_str(data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization error: {}", e)))?;

        let mut result = Self {
            counters: tk_data.counters,
            k: tk_data.k,
            item_index: HashMap::new(),
        };
        result.rebuild_index();
        Ok(result)
    }

    /// Number of items currently tracked.
    fn __len__(&self) -> usize {
        self.counters.len()
    }

    /// Check if item is tracked using 'in' operator.
    fn __contains__(&self, item: &str) -> bool {
        self.contains(item)
    }

    /// Get item count using indexing.
    fn __getitem__(&self, item: &str) -> u64 {
        self.query(item)
    }

    fn __repr__(&self) -> String {
        format!(
            "TopK(k={}, tracked={}, min_count={})",
            self.k,
            self.counters.len(),
            self.min_count()
        )
    }

    /// Maximum number of items to track.
    #[getter]
    fn k(&self) -> usize {
        self.k
    }

    /// Memory footprint in bytes (approximate).
    #[getter]
    fn size_in_bytes(&self) -> usize {
        // Rough estimate: item string + 2 * u64 per counter + hashmap overhead
        self.counters.iter()
            .map(|c| c.item.len() + 16)
            .sum::<usize>()
            + self.k * 32  // HashMap overhead estimate
    }
}
