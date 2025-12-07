# Data Structures Overview

Hazy provides eight probabilistic data structures, each optimized for specific use cases.

## Quick Reference

| Structure | Primary Use | Query Type | Deletions |
|-----------|-------------|------------|-----------|
| [BloomFilter](bloom-filter.md) | Set membership | "Is X in the set?" | No |
| [CountingBloomFilter](counting-bloom.md) | Set membership | "Is X in the set?" | Yes |
| [ScalableBloomFilter](scalable-bloom.md) | Set membership | "Is X in the set?" | No |
| [CuckooFilter](cuckoo-filter.md) | Set membership | "Is X in the set?" | Yes |
| [HyperLogLog](hyperloglog.md) | Cardinality | "How many unique items?" | N/A |
| [CountMinSketch](count-min-sketch.md) | Frequency | "How often does X appear?" | N/A |
| [MinHash](minhash.md) | Similarity | "How similar are A and B?" | N/A |
| [TopK](topk.md) | Heavy hitters | "What are the most frequent items?" | N/A |

## Choosing the Right Structure

### Set Membership Testing

**"Have I seen this item before?"**

```
Need deletions?
├── No → BloomFilter (most space-efficient)
├── Yes, occasionally → CountingBloomFilter
└── Yes, frequently → CuckooFilter

Unknown total items?
└── Yes → ScalableBloomFilter
```

### Cardinality Estimation

**"How many unique items are there?"**

→ Use **HyperLogLog**

- Constant memory regardless of cardinality
- ~1% error with 16KB memory
- Supports union of multiple counters

### Frequency Estimation

**"How many times has this item appeared?"**

→ Use **CountMinSketch**

- Tracks frequencies for all items
- Never underestimates (may overestimate)
- Good for identifying heavy hitters

### Finding Top Items

**"What are the K most frequent items?"**

→ Use **TopK**

- Space-Saving algorithm
- Exact for truly heavy hitters
- Provides error bounds

### Set Similarity

**"How similar are these two sets?"**

→ Use **MinHash**

- Estimates Jaccard similarity
- Can be used with LSH for nearest neighbor search
- Compact signatures for large sets

## Common API

All structures share a consistent API:

```python
# Construction
structure = Type(params...)

# Adding items
structure.add(item)           # Single item
structure.update(items)       # Multiple items

# Querying
result = structure.query(item)  # Varies by structure
item in structure               # Membership test

# Combining
merged = structure1 | structure2  # Union (where applicable)
structure1.merge(structure2)      # In-place merge

# Serialization
data = structure.to_bytes()
structure = Type.from_bytes(data)

json_str = structure.to_json()
structure = Type.from_json(json_str)

# File I/O
structure.save("path.hazy")
structure = Type.load("path.hazy")

# Introspection
len(structure)              # Item count (estimated)
structure.size_in_bytes     # Memory footprint
```

## Memory Comparison

For tracking 1 million items:

| Structure | Memory | Notes |
|-----------|--------|-------|
| Python set | ~48 MB | Exact |
| BloomFilter (1% FPR) | ~1.2 MB | ~120x smaller |
| CuckooFilter | ~1 MB | With deletion support |
| HyperLogLog | 16 KB | Cardinality only |
| CountMinSketch | 400 KB | Frequency tracking |

## Error Characteristics

| Structure | Error Type | Direction |
|-----------|------------|-----------|
| BloomFilter | False positives | One-way (no false negatives) |
| CuckooFilter | False positives | One-way (no false negatives) |
| HyperLogLog | Estimation error | Both directions |
| CountMinSketch | Count error | One-way (overestimates) |
| MinHash | Similarity error | Both directions |
| TopK | Count error | One-way (overestimates) |
