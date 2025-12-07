# Parameter Selection

Choosing the right parameters is crucial for getting the best trade-off between accuracy, memory usage, and performance. Hazy provides helper functions to make this easier.

## Bloom Filter Parameters

### Using the Helper

```python
from hazy import estimate_bloom_params

# Estimate parameters for 1 million items at 1% FPR
params = estimate_bloom_params(
    expected_items=1_000_000,
    false_positive_rate=0.01
)

print(f"Bits needed: {params.num_bits:,}")
print(f"Hash functions: {params.num_hashes}")
print(f"Memory: {params.memory_mb:.2f} MB")
print(f"Bits per item: {params.bits_per_item:.2f}")
```

### Parameter Relationships

The false positive rate is determined by:

\[
FPR \approx \left(1 - e^{-kn/m}\right)^k
\]

Where:

- \(m\) = number of bits
- \(n\) = number of items
- \(k\) = number of hash functions

**Rules of thumb:**

| FPR | Bits per item | Hash functions |
|-----|--------------|----------------|
| 10% | 4.8 | 3 |
| 1% | 9.6 | 7 |
| 0.1% | 14.4 | 10 |
| 0.01% | 19.2 | 13 |

### Direct Construction

You can also construct with explicit parameters:

```python
from hazy import BloomFilter

# Using expected items and FPR (recommended)
bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)

# Using explicit bit count and hash count
bf = BloomFilter(num_bits=100000, num_hashes=7)
```

## HyperLogLog Parameters

### Using the Helper

```python
from hazy import estimate_hll_params

# Estimate parameters for 1% error rate
params = estimate_hll_params(
    expected_cardinality=1_000_000,
    error_rate=0.01
)

print(f"Precision: {params.precision}")
print(f"Memory: {params.memory_bytes:,} bytes")
print(f"Expected error: {params.expected_error:.2%}")
```

### Precision vs Error

The standard error of HyperLogLog is approximately:

\[
\sigma \approx \frac{1.04}{\sqrt{2^p}}
\]

Where \(p\) is the precision (4-18).

| Precision | Registers | Memory | Error |
|-----------|-----------|--------|-------|
| 8 | 256 | 256 B | 6.5% |
| 10 | 1,024 | 1 KB | 3.25% |
| 12 | 4,096 | 4 KB | 1.625% |
| 14 | 16,384 | 16 KB | 0.81% |
| 16 | 65,536 | 64 KB | 0.41% |
| 18 | 262,144 | 256 KB | 0.20% |

### Recommendations

- **Precision 12-14** is good for most use cases
- **Precision 14** (~16KB) is the sweet spot for accuracy vs memory
- Use **higher precision** only if you need very accurate counts

```python
from hazy import HyperLogLog

# Good default for most cases
hll = HyperLogLog(precision=14)

# Memory-constrained environments
hll = HyperLogLog(precision=10)

# High accuracy requirements
hll = HyperLogLog(precision=16)
```

## Count-Min Sketch Parameters

### Using the Helper

```python
from hazy import estimate_cms_params

# Estimate parameters for given error bounds
params = estimate_cms_params(
    error_rate=0.001,      # ε: overestimate within ε·N
    confidence=0.99        # 1-δ: probability of being within bounds
)

print(f"Width: {params.width:,}")
print(f"Depth: {params.depth}")
print(f"Memory: {params.memory_mb:.2f} MB")
```

### Error Bounds

Count-Min Sketch guarantees that for any item:

\[
\hat{f}(x) \leq f(x) + \varepsilon \cdot N
\]

with probability at least \(1 - \delta\), where:

- \(\hat{f}(x)\) = estimated count
- \(f(x)\) = true count
- \(\varepsilon\) = error rate = \(e / \text{width}\)
- \(\delta\) = failure probability = \(e^{-\text{depth}}\)
- \(N\) = total count of all items

### Recommendations

| Use Case | Width | Depth | Memory |
|----------|-------|-------|--------|
| Quick estimates | 1,000 | 4 | 32 KB |
| General purpose | 10,000 | 5 | 400 KB |
| High accuracy | 100,000 | 7 | 5.6 MB |

```python
from hazy import CountMinSketch

# Using width and depth directly
cms = CountMinSketch(width=10000, depth=5)

# Using error bounds
cms = CountMinSketch(error_rate=0.001, confidence=0.99)
```

## MinHash Parameters

### Using the Helper

```python
from hazy import estimate_minhash_params

# Estimate parameters for desired accuracy
params = estimate_minhash_params(
    error_rate=0.05  # 5% error in Jaccard estimate
)

print(f"Number of hashes: {params.num_hashes}")
print(f"Memory: {params.memory_bytes} bytes")
print(f"Expected error: {params.expected_error:.2%}")
```

### Error vs Hash Count

The standard error of Jaccard estimation is:

\[
\sigma = \sqrt{\frac{J(1-J)}{k}}
\]

Where \(k\) is the number of hash functions and \(J\) is the true Jaccard.

| Hashes | Memory | Error (J=0.5) |
|--------|--------|---------------|
| 64 | 512 B | 6.25% |
| 128 | 1 KB | 4.4% |
| 256 | 2 KB | 3.1% |
| 512 | 4 KB | 2.2% |

### Recommendations

- **128 hashes** is good for most applications
- Use **256+** for high-precision similarity search
- Use **64** for memory-constrained LSH applications

## Cuckoo Filter Parameters

### Using the Helper

```python
from hazy import estimate_cuckoo_params

params = estimate_cuckoo_params(
    expected_items=100000,
    false_positive_rate=0.01
)

print(f"Capacity: {params.capacity:,}")
print(f"Fingerprint bits: {params.fingerprint_bits}")
print(f"Memory: {params.memory_mb:.2f} MB")
```

### Cuckoo vs Bloom

| Feature | Cuckoo | Bloom |
|---------|--------|-------|
| Deletion | Yes | No |
| Space (low FPR) | Better | Worse |
| Space (high FPR) | Worse | Better |
| Lookup speed | Faster | Similar |
| Insertion speed | Variable | Constant |

**Use Cuckoo when:**

- You need deletion support
- Target FPR is < 3%
- Lookup performance is critical

**Use Bloom when:**

- No deletion needed
- Target FPR is > 3%
- Consistent insertion time matters

## General Guidelines

1. **Start with the helpers** - They encode best practices
2. **Benchmark with real data** - Actual performance may vary
3. **Monitor in production** - Track actual FPR/error rates
4. **Leave headroom** - Plan for 20-50% more items than expected
