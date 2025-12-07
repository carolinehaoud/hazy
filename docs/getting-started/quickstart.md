# Quick Start

This guide will get you up and running with hazy in just a few minutes.

## Basic Usage

### Bloom Filter - Set Membership

Use a Bloom filter when you need to test if items are in a set, and can tolerate occasional false positives.

```python
from hazy import BloomFilter

# Create a filter expecting 10,000 items with 1% false positive rate
bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)

# Add items
bf.add("alice@example.com")
bf.add("bob@example.com")

# Or add many at once
bf.update(["charlie@example.com", "diana@example.com"])

# Check membership
print("alice@example.com" in bf)  # True
print("unknown@example.com" in bf)  # False (probably)
```

### HyperLogLog - Counting Unique Items

Use HyperLogLog when you need to count unique items in a large dataset without storing them all.

```python
from hazy import HyperLogLog

# Create with precision 14 (~16KB memory, ~0.8% error)
hll = HyperLogLog(precision=14)

# Add items (duplicates are handled automatically)
for i in range(1_000_000):
    hll.add(f"user_{i}")

# Get cardinality estimate
print(f"Unique users: {hll.cardinality():,.0f}")  # ~1,000,000
```

### Count-Min Sketch - Frequency Estimation

Use Count-Min Sketch to track frequencies of items in a stream.

```python
from hazy import CountMinSketch

# Create with width=10000, depth=5
cms = CountMinSketch(width=10000, depth=5)

# Count occurrences
for word in ["apple", "apple", "banana", "apple", "cherry"]:
    cms.add(word)

# Query frequencies
print(f"apple: {cms['apple']}")    # >= 3
print(f"banana: {cms['banana']}")  # >= 1
print(f"cherry: {cms['cherry']}")  # >= 1
```

### MinHash - Set Similarity

Use MinHash to estimate how similar two sets are (Jaccard similarity).

```python
from hazy import MinHash

# Create MinHash signatures for two documents
doc1 = MinHash(num_hashes=128)
doc2 = MinHash(num_hashes=128)

# Add words from each document
doc1.update(["the", "quick", "brown", "fox"])
doc2.update(["the", "lazy", "brown", "dog"])

# Estimate Jaccard similarity
similarity = doc1.jaccard(doc2)
print(f"Similarity: {similarity:.2%}")  # ~33%
```

## Common Patterns

### Saving and Loading

All structures support persistence:

```python
from hazy import BloomFilter

bf = BloomFilter(expected_items=10000)
bf.update(["item1", "item2", "item3"])

# Save to file
bf.save("my_filter.hazy")

# Load from file
bf2 = BloomFilter.load("my_filter.hazy")
print("item1" in bf2)  # True
```

### Combining Structures

Many structures support union operations:

```python
from hazy import BloomFilter, HyperLogLog

# Combine Bloom filters
bf1 = BloomFilter(expected_items=1000)
bf2 = BloomFilter(expected_items=1000)
bf1.add("a")
bf2.add("b")

combined = bf1 | bf2  # Union
print("a" in combined)  # True
print("b" in combined)  # True

# Combine HyperLogLogs
hll1 = HyperLogLog(precision=14)
hll2 = HyperLogLog(precision=14)
hll1.update(range(1000))
hll2.update(range(500, 1500))

merged = hll1 | hll2
print(f"Combined cardinality: {merged.cardinality():.0f}")  # ~1500
```

### Working with Different Types

Hazy accepts strings, bytes, and any Python object:

```python
from hazy import BloomFilter

bf = BloomFilter(expected_items=1000)

# Strings
bf.add("hello")

# Bytes
bf.add(b"binary data")

# Integers (converted via repr)
bf.add(42)

# Custom objects (converted via repr)
bf.add({"key": "value"})
```

## What's Next?

- [Parameter Selection](parameters.md) - Learn how to choose optimal parameters
- [Data Structures](../structures/index.md) - Detailed guides for each structure
- [Serialization](../guides/serialization.md) - Binary and JSON serialization
- [Visualization](../guides/visualization.md) - Plotting and debugging tools
