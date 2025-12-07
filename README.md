# Hazy

A modern Python library for probabilistic data structures, implemented in Rust for maximum performance.

## Why Hazy?

When working with massive datasets, exact answers become expensive. Counting unique visitors across billions of events? A precise solution needs gigabytes of memory. Checking if a URL exists in a blocklist of millions? Exact lookups are slow.

Probabilistic data structures solve this by trading perfect accuracy for dramatic improvements in speed and memory. A HyperLogLog can count 1 billion unique items using just 16KB of memory (with ~1% error). A Bloom filter can check membership in a set of 10 million items using 12MB instead of hundreds of megabytes.

Hazy provides battle-tested implementations of these structures with a clean Python API and Rust performance. Use it when you need to:

- **Count unique items** in streams too large to fit in memory
- **Check set membership** without storing every element
- **Estimate frequencies** of items in high-throughput streams
- **Find similar documents** without comparing every pair
- **Track top-K items** in real-time leaderboards

## Features

- **Bloom Filter**: Space-efficient set membership testing
- **Counting Bloom Filter**: Bloom filter with deletion support
- **Scalable Bloom Filter**: Auto-scaling Bloom filter for unknown cardinality
- **Cuckoo Filter**: Better space efficiency than Bloom filters for low FPR, with deletion
- **HyperLogLog**: Cardinality estimation with ~2% error using minimal memory
- **Count-Min Sketch**: Frequency estimation for streaming data
- **MinHash**: Set similarity estimation using Jaccard index
- **Top-K (Space-Saving)**: Find the most frequent items in a stream

### Performance

- **Rust backend** with PyO3 for near-native performance
- **xxHash3** for fast, high-quality hashing
- **Binary serialization** with bincode for compact storage
- **File I/O** with save()/load() for persistence

## Installation

```bash
pip install hazy
```

### Building from source

Requires Rust 1.70+ and Python 3.9+:

```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build and install
pip install maturin
maturin develop --release
```

## Quick Start

```python
from hazy import BloomFilter, HyperLogLog, CountMinSketch

# Bloom Filter - set membership
bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)
bf.add("hello")
print("hello" in bf)  # True
print("world" in bf)  # False (probably)

# HyperLogLog - count unique items
hll = HyperLogLog(precision=14)
for i in range(1000000):
    hll.add(f"user_{i}")
print(f"Unique users: {hll.cardinality():.0f}")  # ~1,000,000

# Count-Min Sketch - frequency estimation
cms = CountMinSketch(width=10000, depth=5)
cms.add("apple")
cms.add("apple")
cms.add("banana")
print(f"Apple count: {cms['apple']}")  # >= 2
```

## Parameter Selection

Use the estimation helpers to choose optimal parameters:

```python
from hazy import estimate_bloom_params, estimate_hll_params

# Bloom filter for 1M items at 1% FPR
params = estimate_bloom_params(expected_items=1_000_000, false_positive_rate=0.01)
print(f"Memory needed: {params.memory_mb:.1f} MB")
print(f"Hash functions: {params.num_hashes}")

# HyperLogLog for 1% error
params = estimate_hll_params(expected_cardinality=1_000_000, error_rate=0.01)
print(f"Precision: {params.precision}")
print(f"Memory: {params.memory_bytes} bytes")
```

## API Overview

All structures share a consistent API:

```python
# Add items
structure.add(item)           # Add single item
structure.update(items)       # Add multiple items

# Query
structure.query(item)         # Query (meaning varies by structure)
item in structure             # Membership test (where applicable)

# Combine
structure.merge(other)        # Combine two structures
result = structure | other    # Union operator (where applicable)

# Serialize
data = structure.to_bytes()   # Binary serialization
structure = Type.from_bytes(data)

json_str = structure.to_json()
structure = Type.from_json(json_str)

# Introspection
len(structure)                # Approximate count
structure.size_in_bytes       # Memory footprint

# File I/O
structure.save("filter.hazy")
structure = Type.load("filter.hazy")
```

## Data Structures

### BloomFilter

```python
from hazy import BloomFilter

bf = BloomFilter(expected_items=10000)
bf.add("hello")
assert "hello" in bf
assert bf.false_positive_rate < 0.02
```

### CountingBloomFilter

```python
from hazy import CountingBloomFilter

cbf = CountingBloomFilter(expected_items=10000)
cbf.add("hello")
cbf.add("hello")
cbf.remove("hello")  # Still contains "hello"
cbf.remove("hello")  # Now removed
```

### ScalableBloomFilter

```python
from hazy import ScalableBloomFilter

# Automatically grows as you add items
sbf = ScalableBloomFilter(initial_capacity=1000)
for i in range(1_000_000):  # Way more than initial capacity
    sbf.add(f"item_{i}")

print(f"Slices: {sbf.num_slices}")  # Multiple slices created
print("item_500" in sbf)  # True
```

### CuckooFilter

```python
from hazy import CuckooFilter

cf = CuckooFilter(capacity=10000)
cf.add("hello")
cf.remove("hello")
assert "hello" not in cf
```

### HyperLogLog

```python
from hazy import HyperLogLog

hll = HyperLogLog(precision=14)  # 16KB memory, ~0.8% error
hll.update([f"item_{i}" for i in range(1000000)])
print(f"Cardinality: {hll.cardinality():.0f}")
```

### CountMinSketch

```python
from hazy import CountMinSketch

cms = CountMinSketch(width=10000, depth=5)
# Or: cms = CountMinSketch(error_rate=0.001, confidence=0.99)
cms.add_count("apple", 10)
print(f"Apple frequency: {cms['apple']}")
```

### MinHash

```python
from hazy import MinHash

mh1 = MinHash(num_hashes=128)
mh1.update(["a", "b", "c", "d"])

mh2 = MinHash(num_hashes=128)
mh2.update(["c", "d", "e", "f"])

print(f"Jaccard similarity: {mh1.jaccard(mh2):.2f}")  # ~0.33
```

### TopK

```python
from hazy import TopK

tk = TopK(k=10)
for word in ["apple"] * 100 + ["banana"] * 50 + ["cherry"] * 25:
    tk.add(word)

for item, count in tk.top(3):
    print(f"{item}: {count}")
```

## Visualization

Install with visualization support:

```bash
pip install hazy[viz]
```

### Plotting

```python
from hazy import BloomFilter, HyperLogLog, CountMinSketch, TopK
from hazy.viz import plot_bloom, plot_hll, plot_cms, plot_topk, show

# Bloom filter bit array heatmap
bf = BloomFilter(expected_items=10000)
bf.update([f"item_{i}" for i in range(5000)])
plot_bloom(bf)

# HyperLogLog register histogram
hll = HyperLogLog(precision=12)
hll.update([f"user_{i}" for i in range(100000)])
plot_hll(hll)

# Count-Min Sketch heatmap
cms = CountMinSketch(width=100, depth=5)
for word in ["apple"] * 50 + ["banana"] * 30 + ["cherry"] * 10:
    cms.add(word)
plot_cms(cms)

# Top-K bar chart
tk = TopK(k=10)
tk.update(["apple"] * 100 + ["banana"] * 50 + ["cherry"] * 25)
plot_topk(tk)

show()  # Display all figures
```

### Jupyter Notebooks

Enable rich HTML display in Jupyter:

```python
import hazy
hazy.enable_notebook_display()

bf = hazy.BloomFilter(expected_items=1000)
bf.add("hello")
bf  # Displays rich HTML with stats and progress bar
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Build release wheel
maturin build --release
```

## License

MIT
