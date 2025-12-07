---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Hazy

**Fast probabilistic data structures for Python, powered by Rust**

<div class="badges">
  <a href="https://pypi.org/project/hazy/"><img src="https://img.shields.io/pypi/v/hazy?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://github.com/caroline/hazy"><img src="https://img.shields.io/github/stars/caroline/hazy?style=social" alt="GitHub Stars"></a>
  <img src="https://img.shields.io/pypi/pyversions/hazy" alt="Python Versions">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</div>

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/caroline/hazy){ .md-button }

</div>

---

<div class="performance-box" markdown>

### ⚡ Built for Performance

Hazy is implemented in **Rust** with Python bindings, delivering exceptional speed for memory-efficient probabilistic computations.

<div class="stats">
  <div class="stat">
    <div class="stat-value">5-20×</div>
    <div class="stat-label">faster than pure Python</div>
  </div>
  <div class="stat">
    <div class="stat-value">~1KB</div>
    <div class="stat-label">to count 1M uniques</div>
  </div>
  <div class="stat">
    <div class="stat-value">8</div>
    <div class="stat-label">data structures</div>
  </div>
</div>

</div>

## What is Hazy?

Hazy provides **probabilistic data structures** that trade perfect accuracy for massive space and time savings. These structures are essential for big data applications where exact answers are impractical.

```python
from hazy import BloomFilter, HyperLogLog, CountMinSketch

# Check if items exist (with 1% false positive rate)
users = BloomFilter(expected_items=1_000_000, false_positive_rate=0.01)
users.add("alice@example.com")
print("alice@example.com" in users)  # True

# Count unique items using only 16KB of memory
counter = HyperLogLog(precision=14)
for user_id in million_user_ids:
    counter.add(user_id)
print(f"Unique users: {counter.cardinality():,.0f}")  # ~1,000,000

# Track frequencies in streaming data
clicks = CountMinSketch(width=10000, depth=5)
clicks.add("/home")
print(f"Home page clicks: {clicks['/home']}")
```

## Data Structures

<div class="grid cards" markdown>

-   :material-filter-outline:{ .lg .middle } **Bloom Filter**

    ---

    Space-efficient set membership testing. Know if an item is *definitely not* in a set, or *probably* in it.

    [:octicons-arrow-right-24: Learn more](structures/bloom-filter.md)

-   :material-counter:{ .lg .middle } **HyperLogLog**

    ---

    Count unique items with ~1% error using just 16KB. Perfect for cardinality estimation at scale.

    [:octicons-arrow-right-24: Learn more](structures/hyperloglog.md)

-   :material-chart-bar:{ .lg .middle } **Count-Min Sketch**

    ---

    Estimate frequencies in streaming data. Track how often items appear without storing them all.

    [:octicons-arrow-right-24: Learn more](structures/count-min-sketch.md)

-   :material-set-center:{ .lg .middle } **MinHash**

    ---

    Estimate set similarity using Jaccard index. Find similar documents, users, or items efficiently.

    [:octicons-arrow-right-24: Learn more](structures/minhash.md)

-   :material-filter-plus-outline:{ .lg .middle } **Cuckoo Filter**

    ---

    Like Bloom filters but with deletion support. Better space efficiency for low false positive rates.

    [:octicons-arrow-right-24: Learn more](structures/cuckoo-filter.md)

-   :material-trophy-outline:{ .lg .middle } **Top-K**

    ---

    Find the most frequent items in a stream using the Space-Saving algorithm with bounded memory.

    [:octicons-arrow-right-24: Learn more](structures/topk.md)

</div>

## Why Hazy?

### :zap: Rust Performance

Implemented in Rust with PyO3 bindings for near-native speed. Uses xxHash3 for fast, high-quality hashing.

### :package: Simple API

Pythonic interface with `in` operator, `len()`, serialization, and file I/O. Feels natural and intuitive.

### :floppy_disk: Compact Serialization

Binary serialization with bincode for minimal storage. Save filters to disk and load them instantly.

### :bar_chart: Visualization

Built-in plotting with matplotlib for debugging and understanding your data structures.

## Quick Comparison

| Structure | Use Case | Memory | Error |
|-----------|----------|--------|-------|
| **BloomFilter** | Set membership | ~1.2 bytes/item | Configurable FPR |
| **HyperLogLog** | Cardinality | 2^p bytes | ~1.04/√(2^p) |
| **CountMinSketch** | Frequencies | w × d × 8 bytes | ε·N overestimate |
| **MinHash** | Similarity | 8 × k bytes | 1/√k |
| **CuckooFilter** | Membership + delete | ~1 byte/item | ~3% FPR |
| **TopK** | Heavy hitters | O(k) | Bounded |

## Installation

=== "pip"

    ```bash
    pip install hazy
    ```

=== "pip (with viz)"

    ```bash
    pip install hazy[viz]
    ```

=== "From source"

    ```bash
    git clone https://github.com/caroline/hazy
    cd hazy
    pip install maturin
    maturin develop --release
    ```

## Featured Examples

<div class="grid cards" markdown>

-   :material-google-analytics:{ .lg .middle } **Web Analytics**

    ---

    Track unique visitors, page views, and trending content with minimal memory.

    [:octicons-arrow-right-24: See tutorial](tutorial/web-analytics.md)

-   :material-content-duplicate:{ .lg .middle } **Deduplication**

    ---

    Detect duplicate events, URLs, or records in streaming data pipelines.

    [:octicons-arrow-right-24: See tutorial](tutorial/deduplication.md)

-   :material-file-search:{ .lg .middle } **Similarity Search**

    ---

    Find similar documents or detect near-duplicates using MinHash signatures.

    [:octicons-arrow-right-24: See tutorial](tutorial/similarity-search.md)

</div>

---

<div style="text-align: center; margin-top: 3rem;">

**Ready to get started?**

[Installation Guide](getting-started/installation.md){ .md-button .md-button--primary }
[Quick Start Tutorial](getting-started/quickstart.md){ .md-button }

</div>
