# API Reference

Complete API reference for all hazy data structures.

## BloomFilter

Space-efficient probabilistic set membership testing.

```python
class BloomFilter:
    def __init__(
        self,
        expected_items: int = None,
        false_positive_rate: float = 0.01,
        num_bits: int = None,
        num_hashes: int = None
    ): ...
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expected_items` | int | None | Expected number of items to add |
| `false_positive_rate` | float | 0.01 | Target false positive rate |
| `num_bits` | int | None | Explicit bit array size |
| `num_hashes` | int | None | Explicit hash function count |

Either `expected_items` or both `num_bits` and `num_hashes` must be provided.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add(item)` | None | Add an item to the filter |
| `update(items)` | None | Add multiple items |
| `query(item)` | bool | Check if item might be present |
| `__contains__(item)` | bool | Same as `query()` |
| `merge(other)` | None | Merge another filter (in-place union) |
| `intersection(other)` | BloomFilter | Return intersection of two filters |
| `__or__(other)` | BloomFilter | Union operator |
| `__and__(other)` | BloomFilter | Intersection operator |
| `copy()` | BloomFilter | Create a copy |
| `clear()` | None | Reset the filter |
| `estimate_count()` | float | Estimate items added |
| `to_bytes()` | bytes | Binary serialization |
| `from_bytes(data)` | BloomFilter | Deserialize from bytes (classmethod) |
| `to_json()` | str | JSON serialization |
| `from_json(json_str)` | BloomFilter | Deserialize from JSON (classmethod) |
| `save(path)` | None | Save to file |
| `load(path)` | BloomFilter | Load from file (classmethod) |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_bits` | int | Size of bit array |
| `num_hashes` | int | Number of hash functions |
| `fill_ratio` | float | Fraction of bits set |
| `false_positive_rate` | float | Current estimated FPR |
| `size_in_bytes` | int | Memory usage |
| `__len__` | int | Estimated item count |

---

## CountingBloomFilter

Bloom filter with deletion support using counters.

```python
class CountingBloomFilter:
    def __init__(
        self,
        expected_items: int = None,
        false_positive_rate: float = 0.01,
        num_counters: int = None,
        num_hashes: int = None
    ): ...
```

### Additional Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `remove(item)` | None | Remove an item |
| `count(item)` | int | Get estimated count for item |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_counters` | int | Number of counters |

---

## ScalableBloomFilter

Auto-scaling Bloom filter for unknown cardinality.

```python
class ScalableBloomFilter:
    def __init__(
        self,
        initial_capacity: int,
        false_positive_rate: float = 0.01,
        growth_ratio: float = 2.0,
        fpr_ratio: float = 0.9
    ): ...
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capacity` | int | Required | Capacity of first slice |
| `false_positive_rate` | float | 0.01 | Target FPR |
| `growth_ratio` | float | 2.0 | Size multiplier for new slices |
| `fpr_ratio` | float | 0.9 | FPR tightening ratio |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_slices` | int | Number of filter slices |

---

## CuckooFilter

Space-efficient filter with deletion support.

```python
class CuckooFilter:
    def __init__(
        self,
        capacity: int,
        fingerprint_bits: int = 8,
        bucket_size: int = 4
    ): ...
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add(item)` | bool | Add item, returns False if full |
| `remove(item)` | bool | Remove item |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `capacity` | int | Maximum capacity |
| `load_factor` | float | Current load (0.0 to 1.0) |

---

## HyperLogLog

Cardinality estimation with minimal memory.

```python
class HyperLogLog:
    def __init__(self, precision: int = 14): ...
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `precision` | int | 14 | Precision (4-18), determines accuracy |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add(item)` | None | Add an item |
| `update(items)` | None | Add multiple items |
| `cardinality()` | float | Get estimated unique count |
| `merge(other)` | None | Merge another HLL (in-place) |
| `__or__(other)` | HyperLogLog | Union operator |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `precision` | int | Precision parameter |
| `__len__` | int | Same as `cardinality()` |

---

## CountMinSketch

Frequency estimation for streaming data.

```python
class CountMinSketch:
    def __init__(
        self,
        width: int = None,
        depth: int = None,
        error_rate: float = None,
        confidence: float = None
    ): ...
```

### Constructor Parameters

Either `width` and `depth`, or `error_rate` and `confidence` must be provided.

| Parameter | Type | Description |
|-----------|------|-------------|
| `width` | int | Number of counters per row |
| `depth` | int | Number of rows (hash functions) |
| `error_rate` | float | Maximum error as fraction of total |
| `confidence` | float | Probability of being within error bound |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add(item)` | None | Increment item's count by 1 |
| `add_count(item, count)` | None | Increment by specific count |
| `update(items)` | None | Add multiple items |
| `query(item)` | int | Get estimated count |
| `__getitem__(item)` | int | Same as `query()` |
| `merge(other)` | None | Merge another sketch |
| `inner_product(other)` | int | Estimate dot product |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `width` | int | Width of sketch |
| `depth` | int | Depth of sketch |
| `total_count` | int | Sum of all counts |

---

## MinHash

Set similarity estimation using Jaccard index.

```python
class MinHash:
    def __init__(self, num_hashes: int = 128): ...
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add(item)` | None | Add an item to the set |
| `update(items)` | None | Add multiple items |
| `jaccard(other)` | float | Estimate Jaccard similarity |
| `merge(other)` | None | Merge (union) with another MinHash |
| `__or__(other)` | MinHash | Union operator |
| `signature()` | list[int] | Get the signature vector |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_hashes` | int | Number of hash functions |

---

## TopK

Find most frequent items using Space-Saving algorithm.

```python
class TopK:
    def __init__(self, k: int): ...
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add(item)` | None | Add an occurrence |
| `add_count(item, count)` | None | Add multiple occurrences |
| `update(items)` | None | Add multiple items |
| `query(item)` | int | Get count (0 if not tracked) |
| `top(n)` | list[tuple[str, int]] | Get top n items with counts |
| `top_with_error(n)` | list[tuple[str, int, int]] | Get top n with error bounds |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `k` | int | Maximum items tracked |
| `__len__` | int | Current items tracked |

---

## Parameter Estimation Functions

### estimate_bloom_params

```python
def estimate_bloom_params(
    expected_items: int,
    false_positive_rate: float = 0.01
) -> BloomParams: ...
```

Returns `BloomParams` with: `num_bits`, `num_hashes`, `memory_mb`, `bits_per_item`

### estimate_counting_bloom_params

```python
def estimate_counting_bloom_params(
    expected_items: int,
    false_positive_rate: float = 0.01
) -> CountingBloomParams: ...
```

### estimate_cuckoo_params

```python
def estimate_cuckoo_params(
    expected_items: int,
    false_positive_rate: float = 0.01
) -> CuckooParams: ...
```

### estimate_hll_params

```python
def estimate_hll_params(
    expected_cardinality: int = None,
    error_rate: float = 0.01
) -> HLLParams: ...
```

Returns `HLLParams` with: `precision`, `memory_bytes`, `expected_error`

### estimate_cms_params

```python
def estimate_cms_params(
    error_rate: float = 0.01,
    confidence: float = 0.99
) -> CMSParams: ...
```

Returns `CMSParams` with: `width`, `depth`, `memory_mb`

### estimate_minhash_params

```python
def estimate_minhash_params(
    error_rate: float = 0.05
) -> MinHashParams: ...
```

Returns `MinHashParams` with: `num_hashes`, `memory_bytes`, `expected_error`

---

## Visualization Functions

Available with `pip install hazy[viz]`:

```python
from hazy.viz import (
    plot_bloom,
    plot_bloom_fill_curve,
    plot_hll,
    plot_cms,
    plot_topk,
    plot_minhash_comparison,
    show
)
```

### plot_bloom

```python
def plot_bloom(
    bf: BloomFilter,
    title: str = "Bloom Filter",
    cmap: str = "Blues"
) -> None: ...
```

### plot_bloom_fill_curve

```python
def plot_bloom_fill_curve(
    bf: BloomFilter,
    max_items: int = None
) -> None: ...
```

### plot_hll

```python
def plot_hll(
    hll: HyperLogLog,
    title: str = "HyperLogLog Registers"
) -> None: ...
```

### plot_cms

```python
def plot_cms(
    cms: CountMinSketch,
    title: str = "Count-Min Sketch",
    log_scale: bool = True
) -> None: ...
```

### plot_topk

```python
def plot_topk(
    tk: TopK,
    n: int = None,
    title: str = "Top-K Items"
) -> None: ...
```

### plot_minhash_comparison

```python
def plot_minhash_comparison(
    mh1: MinHash,
    mh2: MinHash,
    title: str = "MinHash Comparison"
) -> None: ...
```

---

## Jupyter Integration

```python
import hazy

# Enable rich HTML display
hazy.enable_notebook_display()

# Now all types display nicely
bf = hazy.BloomFilter(expected_items=1000)
bf  # Shows HTML with stats and progress bar
```
