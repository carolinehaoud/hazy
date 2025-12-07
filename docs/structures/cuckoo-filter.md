# Cuckoo Filter

A Cuckoo filter is an alternative to Bloom filters that supports **deletion** while often using less space, especially for low false positive rates.

## When to Use

- Set membership with deletion support needed
- Low false positive rates (< 3%) are required
- Lookup performance is critical
- Memory efficiency matters more than insertion predictability

## Basic Usage

```python
from hazy import CuckooFilter

# Create a filter with capacity for 10,000 items
cf = CuckooFilter(capacity=10000)

# Add items
cf.add("alice")
cf.add("bob")

# Check membership
print("alice" in cf)  # True

# Delete items
cf.remove("alice")
print("alice" in cf)  # False
```

## Construction Options

```python
# Basic construction
cf = CuckooFilter(capacity=10000)

# With custom fingerprint size
cf = CuckooFilter(capacity=10000, fingerprint_bits=16)

# With bucket size
cf = CuckooFilter(capacity=10000, bucket_size=4)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capacity` | Required | Maximum number of items |
| `fingerprint_bits` | 8 | Bits per fingerprint (affects FPR) |
| `bucket_size` | 4 | Entries per bucket |

### Parameter Estimation

```python
from hazy import estimate_cuckoo_params

params = estimate_cuckoo_params(
    expected_items=100000,
    false_positive_rate=0.01
)
print(f"Capacity: {params.capacity:,}")
print(f"Memory: {params.memory_mb:.2f} MB")
```

## Key Operations

### Adding Items

```python
cf = CuckooFilter(capacity=10000)

# Returns True if successful, False if filter is full
success = cf.add("item")

if not success:
    print("Filter is full!")

# Bulk add
cf.update(["a", "b", "c"])
```

### Membership Testing

```python
if "item" in cf:
    print("Item might be in the set")

# Using query method
result = cf.query("item")  # True/False
```

### Deletion

```python
cf = CuckooFilter(capacity=10000)

cf.add("alice")
cf.add("bob")

# Remove returns True if item was (probably) present
removed = cf.remove("alice")
print(f"Removed: {removed}")  # True

print("alice" in cf)  # False
print("bob" in cf)    # True
```

!!! warning "Deleting Non-existent Items"
    Unlike Counting Bloom filters, deleting a non-existent item from a Cuckoo filter doesn't cause false negatives for other items. However, you might accidentally delete a different item with the same fingerprint.

## Statistics

```python
cf = CuckooFilter(capacity=10000)
cf.update([f"item_{i}" for i in range(5000)])

print(f"Items: {len(cf):,}")
print(f"Capacity: {cf.capacity:,}")
print(f"Load factor: {cf.load_factor:.2%}")
print(f"Memory: {cf.size_in_bytes:,} bytes")
```

## Serialization

```python
# Binary
data = cf.to_bytes()
cf2 = CuckooFilter.from_bytes(data)

# JSON
json_str = cf.to_json()
cf2 = CuckooFilter.from_json(json_str)

# File I/O
cf.save("filter.hazy")
cf2 = CuckooFilter.load("filter.hazy")
```

## How It Works

A Cuckoo filter uses **cuckoo hashing** with fingerprints:

### Structure

- An array of **buckets**, each holding multiple **fingerprints**
- Each item maps to **two possible bucket locations**
- Only a small **fingerprint** is stored, not the full item

### Adding an Item

```
fingerprint = hash(item) & fingerprint_mask
bucket1 = hash(item) mod num_buckets
bucket2 = bucket1 XOR hash(fingerprint)

if bucket1 has space:
    store fingerprint in bucket1
elif bucket2 has space:
    store fingerprint in bucket2
else:
    // Cuckoo displacement
    kick existing fingerprint to its alternate location
    repeat until stable or max_kicks reached
```

### Partial-Key Cuckoo Hashing

The XOR-based alternate location allows finding an item's other bucket using only the fingerprint:

```
alternate_bucket = current_bucket XOR hash(fingerprint)
```

This is what enables deletion without storing the original item.

### False Positives

FPR depends on fingerprint size:

\[
FPR \approx \frac{2b}{2^f}
\]

where `b` is bucket size and `f` is fingerprint bits.

| Fingerprint Bits | Approximate FPR |
|------------------|-----------------|
| 8 | ~3% |
| 12 | ~0.2% |
| 16 | ~0.01% |

## Capacity and Load Factor

Cuckoo filters have a maximum **load factor** (~95%) before insertions fail:

```python
cf = CuckooFilter(capacity=1000)

# Add items until full
count = 0
for i in range(2000):
    if cf.add(f"item_{i}"):
        count += 1
    else:
        break

print(f"Added {count} items before full")
print(f"Load factor: {cf.load_factor:.2%}")
```

!!! tip "Size for Expected Items"
    Create the filter with capacity ~1.2x your expected items to ensure successful insertions.

## Use Cases

### 1. Cache with Eviction

```python
from hazy import CuckooFilter

class FilteredCache:
    def __init__(self, capacity):
        self.filter = CuckooFilter(capacity=capacity)
        self.cache = {}

    def add(self, key, value):
        self.cache[key] = value
        self.filter.add(key)

    def remove(self, key):
        if key in self.cache:
            del self.cache[key]
            self.filter.remove(key)

    def might_contain(self, key):
        return key in self.filter
```

### 2. Distributed Deduplication

```python
from hazy import CuckooFilter

seen = CuckooFilter(capacity=1_000_000)

def process_with_ttl(item_id, ttl_seconds):
    if item_id in seen:
        return  # Duplicate

    seen.add(item_id)
    process(item_id)

    # Schedule removal after TTL
    schedule(ttl_seconds, lambda: seen.remove(item_id))
```

### 3. Negative Cache

```python
not_found = CuckooFilter(capacity=100000)

def lookup(key):
    if key in not_found:
        return None  # Known to not exist

    result = expensive_lookup(key)
    if result is None:
        not_found.add(key)
    return result

def invalidate(key):
    not_found.remove(key)
```

## Comparison with Bloom Filters

| Feature | CuckooFilter | BloomFilter | CountingBloom |
|---------|--------------|-------------|---------------|
| Deletion | Yes | No | Yes |
| Space (1% FPR) | ~12 bits/item | ~10 bits/item | ~40 bits/item |
| Space (0.1% FPR) | ~16 bits/item | ~15 bits/item | ~60 bits/item |
| Space (0.01% FPR) | ~20 bits/item | ~20 bits/item | ~80 bits/item |
| Lookup time | O(1) | O(k) | O(k) |
| Insert time | O(1) avg* | O(k) | O(k) |
| False negatives | No | No | Possible |

*Worst case O(n) during cuckoo displacement.

## Best Practices

1. **Size appropriately** - Create with ~1.2x expected capacity

2. **Monitor load factor** - Insertions fail near 95% load

3. **Handle insertion failures** - Check return value of `add()`

4. **Be cautious with deletions** - Deleting items with same fingerprint can cause issues

```python
cf = CuckooFilter(capacity=10000)

# Always check insertion success
if not cf.add("item"):
    # Handle full filter
    print("Warning: Filter is full")

# Monitor load
if cf.load_factor > 0.9:
    print("Warning: Filter nearly full")
```
