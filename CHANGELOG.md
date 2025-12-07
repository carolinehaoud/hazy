# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-06

### Added

#### Data Structures
- **BloomFilter**: Space-efficient probabilistic set membership testing
  - Configurable expected items, false positive rate, or explicit bit/hash counts
  - Union (`|`) and intersection (`&`) operators
  - `estimate_count()` for estimating items from fill ratio
- **CountingBloomFilter**: Bloom filter with deletion support using 8-bit counters
- **ScalableBloomFilter**: Auto-scaling Bloom filter for unknown cardinality
  - Automatically adds filter slices as capacity is reached
  - Configurable growth ratio and FPR tightening
- **CuckooFilter**: Space-efficient filter with deletion support
  - Better space efficiency than Bloom filters for low FPR
  - Uses partial-key cuckoo hashing
- **HyperLogLog**: Cardinality estimation with ~1-2% error
  - Configurable precision (4-18 bits)
  - Union operation via `merge()` or `|` operator
- **CountMinSketch**: Frequency estimation for streaming data
  - Configurable via width/depth or error_rate/confidence
  - `inner_product()` for similarity estimation
- **MinHash**: Set similarity estimation using Jaccard index
  - Configurable number of hash functions
  - `jaccard()` method for similarity queries
- **TopK**: Find most frequent items using Space-Saving algorithm
  - Error bounds tracking for count estimates
  - `top_with_error()` for counts with error bounds

#### Core Features
- **Rust backend** with PyO3 for near-native performance
- **xxHash3** for fast, high-quality hashing
- **Binary serialization** with bincode for compact `to_bytes()`/`from_bytes()`
- **JSON serialization** via `to_json()`/`from_json()` for debugging
- **File I/O** with `save()`/`load()` methods
- **Generic type support**: `add()` accepts strings, bytes, or any Python object
- **Pythonic API**: `in` operator, `len()`, `|` for union, `&` for intersection

#### Parameter Helpers
- `estimate_bloom_params()`: Calculate optimal Bloom filter parameters
- `estimate_counting_bloom_params()`: Calculate Counting Bloom parameters
- `estimate_cuckoo_params()`: Calculate Cuckoo filter parameters
- `estimate_hll_params()`: Calculate HyperLogLog parameters
- `estimate_cms_params()`: Calculate Count-Min Sketch parameters
- `estimate_minhash_params()`: Calculate MinHash parameters

#### Visualization (`pip install hazy[viz]`)
- `plot_bloom()`: Bit array heatmap with statistics
- `plot_bloom_fill_curve()`: Theoretical fill ratio and FPR curves
- `plot_hll()`: Register value histogram
- `plot_cms()`: 2D counter heatmap (log scale optional)
- `plot_topk()`: Horizontal bar chart with error bars
- `plot_minhash_comparison()`: Side-by-side signature comparison

#### Jupyter Integration
- `hazy.enable_notebook_display()`: Rich HTML display for all types
- Progress bars, statistics tables, and mini visualizations

### Technical Details
- Python 3.9+ support
- Type hints with `.pyi` stub files
- PEP 561 compatible (`py.typed` marker)
- Optional dependencies: `numpy`, `matplotlib`

[0.1.0]: https://github.com/carolinehaoud/hazy/releases/tag/v0.1.0
