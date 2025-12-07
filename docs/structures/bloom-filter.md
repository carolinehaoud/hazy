# Bloom Filter

A Bloom filter is a space-efficient probabilistic data structure for set membership testing. It can tell you if an item is **definitely not** in the set, or **probably** in the set.

## When to Use

- Checking if a username is taken before querying a database
- Filtering URLs that have already been crawled
- Detecting duplicate events in a stream
- Cache lookups (avoid expensive operations for items not in cache)

## Basic Usage

```python
from hazy import BloomFilter

# Create a filter for 10,000 items with 1% false positive rate
bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)

# Add items
bf.add("alice")
bf.add("bob")

# Check membership
print("alice" in bf)    # True (definitely added)
print("charlie" in bf)  # False (probably not added)
print("unknown" in bf)  # False (probably not added)
```

## Construction Options

### Automatic Sizing (Recommended)

```python
# Specify expected items and desired false positive rate
bf = BloomFilter(expected_items=100000, false_positive_rate=0.01)
```

### Manual Sizing

```python
# Specify exact bit count and hash count
bf = BloomFilter(num_bits=1000000, num_hashes=7)
```

### Parameter Estimation

```python
from hazy import estimate_bloom_params

params = estimate_bloom_params(expected_items=100000, false_positive_rate=0.01)
print(f"Bits: {params.num_bits:,}")
print(f"Hashes: {params.num_hashes}")
print(f"Memory: {params.memory_mb:.2f} MB")
```

## Key Operations

### Adding Items

```python
# Single item
bf.add("item")

# Multiple items
bf.update(["item1", "item2", "item3"])

# Different types
bf.add(b"binary data")
bf.add(12345)
bf.add({"key": "value"})
```

### Membership Testing

```python
# Using 'in' operator
if "alice" in bf:
    print("Alice might be in the set")

# Using query method
result = bf.query("alice")  # Returns True/False
```

### Combining Filters

```python
bf1 = BloomFilter(expected_items=1000)
bf2 = BloomFilter(expected_items=1000)

bf1.update(["a", "b", "c"])
bf2.update(["c", "d", "e"])

# Union - items from both filters
union = bf1 | bf2
print("a" in union)  # True
print("d" in union)  # True

# Intersection - items in both filters (approximate)
intersection = bf1 & bf2
print("c" in intersection)  # True
```

!!! warning "Filter Compatibility"
    Union and intersection only work between filters with the same `num_bits` and `num_hashes`.

## Statistics

```python
bf = BloomFilter(expected_items=10000)
bf.update([f"item_{i}" for i in range(5000)])

print(f"Bits: {bf.num_bits:,}")
print(f"Hash functions: {bf.num_hashes}")
print(f"Fill ratio: {bf.fill_ratio:.2%}")
print(f"Estimated count: {len(bf):,}")
print(f"False positive rate: {bf.false_positive_rate:.4f}")
print(f"Memory: {bf.size_in_bytes:,} bytes")
```

## Serialization

### Binary (Compact)

```python
# Serialize
data = bf.to_bytes()
print(f"Serialized size: {len(data):,} bytes")

# Deserialize
bf2 = BloomFilter.from_bytes(data)
```

### JSON (Human-readable)

```python
# Serialize
json_str = bf.to_json()

# Deserialize
bf2 = BloomFilter.from_json(json_str)
```

### File I/O

```python
# Save
bf.save("filter.hazy")

# Load
bf2 = BloomFilter.load("filter.hazy")
```

## Advanced Usage

### Estimating Item Count

```python
bf = BloomFilter(expected_items=10000)
bf.update([f"item_{i}" for i in range(5000)])

# Estimate how many items have been added
estimated = bf.estimate_count()
print(f"Estimated items: {estimated:,.0f}")  # ~5000
```

### Copying

```python
bf1 = BloomFilter(expected_items=1000)
bf1.add("test")

bf2 = bf1.copy()
bf2.add("other")

print("test" in bf1)   # True
print("other" in bf1)  # False (copy is independent)
```

### Clearing

```python
bf.clear()
print(len(bf))  # 0
```

## How It Works

A Bloom filter uses:

1. **A bit array** of size `m`
2. **Multiple hash functions** (`k` functions)

**Adding an item:**
```
For each hash function h_i:
    index = h_i(item) mod m
    bit_array[index] = 1
```

**Checking membership:**
```
For each hash function h_i:
    index = h_i(item) mod m
    if bit_array[index] == 0:
        return "Definitely not in set"
return "Probably in set"
```

### False Positives

False positives occur when all bits for an item are set by other items. The probability is:

\[
FPR \approx \left(1 - e^{-kn/m}\right)^k
\]

where `n` is the number of items added.

### No False Negatives

A Bloom filter **never** has false negatives. If an item was added, it will always be found.

## Use Cases

### 1. Database Query Optimization

```python
# Avoid unnecessary database queries
bf = BloomFilter(expected_items=1_000_000, false_positive_rate=0.01)

# Populate with known usernames
for username in get_all_usernames():
    bf.add(username)

# Check before querying database
def is_username_available(username):
    if username in bf:
        # Might exist - check database
        return not database.username_exists(username)
    else:
        # Definitely doesn't exist
        return True
```

### 2. Web Crawler Deduplication

```python
visited = BloomFilter(expected_items=10_000_000, false_positive_rate=0.001)

def crawl(url):
    if url in visited:
        return  # Already crawled (or false positive)

    visited.add(url)
    page = fetch(url)
    for link in extract_links(page):
        crawl(link)
```

### 3. Spell Checker

```python
dictionary = BloomFilter(expected_items=500_000, false_positive_rate=0.0001)

# Load dictionary
with open("words.txt") as f:
    dictionary.update(word.strip() for word in f)

def is_word_valid(word):
    return word.lower() in dictionary
```

## Performance

| Operation | Time Complexity |
|-----------|----------------|
| Add | O(k) |
| Query | O(k) |
| Union | O(m) |
| Intersection | O(m) |

Where `k` is the number of hash functions and `m` is the bit array size.

## Comparison with Other Structures

| Feature | BloomFilter | CountingBloomFilter | CuckooFilter |
|---------|-------------|--------------------| --------------|
| Memory per item | ~1.2 bytes | ~4 bytes | ~1 byte |
| Deletion | No | Yes | Yes |
| False positives | Yes | Yes | Yes |
| False negatives | No | Possible* | No |

*CountingBloomFilter can have false negatives if counters overflow.
