use pyo3::prelude::*;

mod bloom;
mod counting_bloom;
mod scalable_bloom;
mod cuckoo;
mod hyperloglog;
mod count_min_sketch;
mod minhash;
mod topk;
mod utils;

use bloom::BloomFilter;
use counting_bloom::CountingBloomFilter;
use scalable_bloom::ScalableBloomFilter;
use cuckoo::CuckooFilter;
use hyperloglog::HyperLogLog;
use count_min_sketch::CountMinSketch;
use minhash::MinHash;
use topk::TopK;

/// A Python module for probabilistic data structures implemented in Rust.
#[pymodule]
fn _hazy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BloomFilter>()?;
    m.add_class::<CountingBloomFilter>()?;
    m.add_class::<ScalableBloomFilter>()?;
    m.add_class::<CuckooFilter>()?;
    m.add_class::<HyperLogLog>()?;
    m.add_class::<CountMinSketch>()?;
    m.add_class::<MinHash>()?;
    m.add_class::<TopK>()?;
    Ok(())
}
