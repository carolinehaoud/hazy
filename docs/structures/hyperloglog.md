# HyperLogLog

HyperLogLog (HLL) is a probabilistic algorithm for **cardinality estimation** — counting the number of unique items in a dataset using minimal memory.

## When to Use

- Counting unique visitors, users, or events
- Estimating distinct values in database columns
- Tracking unique items across distributed systems
- Any scenario where exact counting is impractical

## Basic Usage

```python
from hazy import HyperLogLog

# Create with precision 14 (~16KB memory, ~0.8% error)
hll = HyperLogLog(precision=14)

# Add items
for user_id in stream_of_user_ids():
    hll.add(user_id)

# Get cardinality estimate
unique_users = hll.cardinality()
print(f"Unique users: {unique_users:,.0f}")
```

## Construction Options

```python
# Precision determines accuracy vs memory tradeoff
hll = HyperLogLog(precision=14)  # Default, good balance

# Lower precision = less memory, more error
hll = HyperLogLog(precision=10)  # ~1KB, ~3% error

# Higher precision = more memory, less error
hll = HyperLogLog(precision=16)  # ~64KB, ~0.4% error
```

### Precision vs Accuracy

| Precision | Registers | Memory | Standard Error |
|-----------|-----------|--------|----------------|
| 4 | 16 | 16 B | 26% |
| 8 | 256 | 256 B | 6.5% |
| 10 | 1,024 | 1 KB | 3.25% |
| 12 | 4,096 | 4 KB | 1.625% |
| 14 | 16,384 | 16 KB | 0.81% |
| 16 | 65,536 | 64 KB | 0.41% |
| 18 | 262,144 | 256 KB | 0.20% |

### Parameter Estimation

```python
from hazy import estimate_hll_params

params = estimate_hll_params(
    expected_cardinality=1_000_000,
    error_rate=0.01
)
print(f"Precision: {params.precision}")
print(f"Memory: {params.memory_bytes:,} bytes")
```

## Key Operations

### Adding Items

```python
hll = HyperLogLog(precision=14)

# Single item
hll.add("user_123")

# Multiple items
hll.update(["user_1", "user_2", "user_3"])

# Different types
hll.add(12345)
hll.add(b"binary data")
```

### Getting Cardinality

```python
# Get estimated count
count = hll.cardinality()
print(f"Unique items: {count:,.0f}")

# len() also works
print(f"Unique items: {len(hll):,}")
```

### Merging HLLs

```python
# Create separate HLLs (e.g., from different servers)
hll1 = HyperLogLog(precision=14)
hll2 = HyperLogLog(precision=14)

hll1.update([f"user_{i}" for i in range(1000)])
hll2.update([f"user_{i}" for i in range(500, 1500)])

# Merge to get union cardinality
merged = hll1 | hll2
print(f"Total unique: {merged.cardinality():,.0f}")  # ~1500

# Or merge in-place
hll1.merge(hll2)
print(f"Total unique: {hll1.cardinality():,.0f}")  # ~1500
```

## Statistics

```python
hll = HyperLogLog(precision=14)
hll.update([f"item_{i}" for i in range(100000)])

print(f"Precision: {hll.precision}")
print(f"Cardinality: {hll.cardinality():,.0f}")
print(f"Memory: {hll.size_in_bytes:,} bytes")
```

## Serialization

```python
# Binary (compact)
data = hll.to_bytes()
hll2 = HyperLogLog.from_bytes(data)

# JSON
json_str = hll.to_json()
hll2 = HyperLogLog.from_json(json_str)

# File I/O
hll.save("counter.hazy")
hll2 = HyperLogLog.load("counter.hazy")
```

## How It Works

### The Algorithm

1. **Hash each item** to get a uniformly distributed value
2. **Use first p bits** to select one of 2^p registers
3. **Count leading zeros** in remaining bits
4. **Store maximum** leading zeros seen for each register
5. **Estimate cardinality** using harmonic mean of registers

### Intuition

The key insight: if you flip a coin until you get heads, on average you need 2 flips. If you do this many times and track the **maximum** number of tails seen, that maximum grows logarithmically with the number of experiments.

By tracking maximums across many registers and combining them, HLL achieves accurate estimates with tiny memory.

### Mathematical Basis

The estimate is:

\[
E = \alpha_m \cdot m^2 \cdot \left( \sum_{j=1}^{m} 2^{-M_j} \right)^{-1}
\]

Where:

- \(m = 2^p\) is the number of registers
- \(M_j\) is the value in register \(j\)
- \(\alpha_m\) is a correction constant

## Use Cases

### 1. Unique Visitor Counting

```python
from hazy import HyperLogLog

daily_visitors = HyperLogLog(precision=14)

def log_visit(user_id, page):
    daily_visitors.add(user_id)

def get_daily_unique_visitors():
    return daily_visitors.cardinality()

def reset_daily():
    daily_visitors.clear()
```

### 2. Distributed Counting

```python
from hazy import HyperLogLog

# Each server tracks its own HLL
server_hlls = {}

def add_on_server(server_id, user_id):
    if server_id not in server_hlls:
        server_hlls[server_id] = HyperLogLog(precision=14)
    server_hlls[server_id].add(user_id)

def get_total_unique():
    # Merge all server HLLs
    merged = HyperLogLog(precision=14)
    for hll in server_hlls.values():
        merged.merge(hll)
    return merged.cardinality()
```

### 3. Database Cardinality Estimation

```python
from hazy import HyperLogLog

def estimate_distinct_values(table, column):
    hll = HyperLogLog(precision=14)

    for row in scan_table(table):
        hll.add(row[column])

    return hll.cardinality()

# Much faster than SELECT COUNT(DISTINCT column)
distinct_users = estimate_distinct_values("events", "user_id")
```

### 4. Set Cardinality Operations

```python
from hazy import HyperLogLog

def count_unique_in_union(sets):
    """Count unique items across all sets."""
    merged = HyperLogLog(precision=14)
    for s in sets:
        for item in s:
            merged.add(item)
    return merged.cardinality()

def estimate_intersection_size(hll_a, hll_b, hll_union):
    """Estimate |A ∩ B| using inclusion-exclusion."""
    # |A ∩ B| = |A| + |B| - |A ∪ B|
    return (hll_a.cardinality() + hll_b.cardinality() -
            hll_union.cardinality())
```

## Accuracy Characteristics

### Error Distribution

HLL error follows a normal distribution with standard deviation:

\[
\sigma = \frac{1.04}{\sqrt{m}}
\]

This means ~68% of estimates are within ±σ of the true value.

### Small Range Correction

For small cardinalities, HLL applies **Linear Counting** correction to improve accuracy.

### Large Range Correction

For very large cardinalities approaching 2^32, bias correction is applied.

## Comparison with Exact Counting

| Cardinality | Exact (HashSet) | HyperLogLog (p=14) |
|-------------|-----------------|-------------------|
| 1,000 | ~48 KB | 16 KB |
| 100,000 | ~4.8 MB | 16 KB |
| 10,000,000 | ~480 MB | 16 KB |
| 1,000,000,000 | ~48 GB | 16 KB |

## Best Practices

1. **Choose precision based on accuracy needs** - p=14 is good for most cases

2. **Use same precision for merging** - HLLs must have matching precision to merge

3. **Merge instead of querying multiple times** - Merging is more accurate than adding estimates

4. **Consider memory vs accuracy tradeoff**:
```python
# Memory constrained
hll = HyperLogLog(precision=10)  # 1KB, ~3% error

# Accuracy critical
hll = HyperLogLog(precision=16)  # 64KB, ~0.4% error
```

5. **HLL is additive, not subtractive** - You can't remove items or compute set difference
