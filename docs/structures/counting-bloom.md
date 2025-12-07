# Counting Bloom Filter

A Counting Bloom filter extends the standard Bloom filter by using counters instead of single bits. This enables **deletion** of items while maintaining the no-false-negatives guarantee (with some caveats).

## When to Use

- When you need Bloom filter functionality **with deletion support**
- Tracking items that may be added and removed over time
- Session management or temporary memberships
- When CuckooFilter's variable insertion time is problematic

## Basic Usage

```python
from hazy import CountingBloomFilter

# Create a filter for 10,000 items with 1% false positive rate
cbf = CountingBloomFilter(expected_items=10000, false_positive_rate=0.01)

# Add items
cbf.add("alice")
cbf.add("alice")  # Increments counter

# Remove items
cbf.remove("alice")  # Decrements counter
cbf.remove("alice")  # Removes completely

# Check membership
print("alice" in cbf)  # False (removed)
```

## Construction Options

```python
# Using expected items and FPR
cbf = CountingBloomFilter(expected_items=10000, false_positive_rate=0.01)

# Using explicit parameters
cbf = CountingBloomFilter(num_counters=100000, num_hashes=7)
```

### Parameter Estimation

```python
from hazy import estimate_counting_bloom_params

params = estimate_counting_bloom_params(
    expected_items=100000,
    false_positive_rate=0.01
)
print(f"Counters: {params.num_counters:,}")
print(f"Memory: {params.memory_mb:.2f} MB")
```

## Key Operations

### Adding and Removing

```python
cbf = CountingBloomFilter(expected_items=1000)

# Add multiple times
cbf.add("item")
cbf.add("item")
cbf.add("item")
print(cbf.count("item"))  # >= 3

# Remove
cbf.remove("item")
print(cbf.count("item"))  # >= 2

# Remove all occurrences
cbf.remove("item")
cbf.remove("item")
print("item" in cbf)  # False
```

### Bulk Operations

```python
cbf.update(["a", "b", "c", "a", "a"])  # a added 3 times
print(cbf.count("a"))  # >= 3
```

### Counting

```python
cbf = CountingBloomFilter(expected_items=1000)
cbf.add("apple")
cbf.add("apple")
cbf.add("banana")

# Get estimated count
print(cbf.count("apple"))   # >= 2
print(cbf.count("banana"))  # >= 1
print(cbf.count("cherry"))  # 0
```

## Statistics

```python
cbf = CountingBloomFilter(expected_items=10000)
cbf.update([f"item_{i}" for i in range(5000)])

print(f"Counters: {cbf.num_counters:,}")
print(f"Hash functions: {cbf.num_hashes}")
print(f"Estimated items: {len(cbf):,}")
print(f"Memory: {cbf.size_in_bytes:,} bytes")
```

## Serialization

```python
# Binary
data = cbf.to_bytes()
cbf2 = CountingBloomFilter.from_bytes(data)

# JSON
json_str = cbf.to_json()
cbf2 = CountingBloomFilter.from_json(json_str)

# File I/O
cbf.save("filter.hazy")
cbf2 = CountingBloomFilter.load("filter.hazy")
```

## How It Works

Instead of single bits, a Counting Bloom filter uses **8-bit counters** (values 0-255):

**Adding an item:**
```
For each hash function h_i:
    index = h_i(item) mod num_counters
    counters[index] += 1  (saturates at 255)
```

**Removing an item:**
```
For each hash function h_i:
    index = h_i(item) mod num_counters
    if counters[index] > 0:
        counters[index] -= 1
```

**Checking membership:**
```
For each hash function h_i:
    index = h_i(item) mod num_counters
    if counters[index] == 0:
        return "Definitely not in set"
return "Probably in set"
```

### Counter Overflow

!!! warning "Counter Saturation"
    If a counter reaches 255, it **saturates** and won't increment further. This prevents wrap-around bugs but means very frequent items may have undercounted values.

```python
cbf = CountingBloomFilter(expected_items=100)

# Add same item many times
for _ in range(300):
    cbf.add("frequent")

# Counter saturates at 255
print(cbf.count("frequent"))  # 255 (not 300)
```

### False Negatives After Removal

!!! danger "Removing Non-existent Items"
    Removing an item that was never added can cause **false negatives** for other items. Only remove items you're certain were added.

```python
cbf = CountingBloomFilter(expected_items=100)
cbf.add("real")

# DON'T DO THIS - removing item that wasn't added
cbf.remove("fake")  # Decrements counters that "real" might use

# May cause false negative!
print("real" in cbf)  # Might be False!
```

## Memory Usage

Counting Bloom filters use **4x more memory** than standard Bloom filters (8-bit counters vs 1-bit):

| Items | Standard Bloom | Counting Bloom |
|-------|---------------|----------------|
| 10,000 | ~12 KB | ~48 KB |
| 100,000 | ~120 KB | ~480 KB |
| 1,000,000 | ~1.2 MB | ~4.8 MB |

## Use Cases

### 1. Session Management

```python
active_sessions = CountingBloomFilter(expected_items=100000)

def login(user_id):
    active_sessions.add(user_id)

def logout(user_id):
    active_sessions.remove(user_id)

def is_logged_in(user_id):
    return user_id in active_sessions
```

### 2. Rate Limiting with Decay

```python
from hazy import CountingBloomFilter

class RateLimiter:
    def __init__(self, max_requests=100):
        self.filter = CountingBloomFilter(expected_items=10000)
        self.max_requests = max_requests

    def allow_request(self, client_id):
        count = self.filter.count(client_id)
        if count >= self.max_requests:
            return False
        self.filter.add(client_id)
        return True

    def decay(self, client_id):
        """Call periodically to allow more requests"""
        if client_id in self.filter:
            self.filter.remove(client_id)
```

### 3. Temporary Blacklist

```python
blacklist = CountingBloomFilter(expected_items=10000)

def ban_ip(ip, duration_minutes=30):
    blacklist.add(ip)
    # Schedule removal
    schedule_task(duration_minutes * 60, lambda: blacklist.remove(ip))

def is_banned(ip):
    return ip in blacklist
```

## Comparison with Alternatives

| Feature | CountingBloom | CuckooFilter | Standard Bloom |
|---------|---------------|--------------|----------------|
| Deletion | Yes | Yes | No |
| Memory | 4x Bloom | ~1x Bloom | Baseline |
| Insertion time | O(k) constant | O(1) avg, O(n) worst | O(k) constant |
| Count queries | Yes | No | No |
| False negatives | Possible* | No | No |

*If non-existent items are removed or counters overflow.

## Best Practices

1. **Only remove items you added** - Removing non-existent items corrupts the filter
2. **Monitor counter saturation** - Very frequent items may saturate counters
3. **Size appropriately** - Account for the 4x memory overhead vs standard Bloom
4. **Consider CuckooFilter** - If you don't need count queries, it's more memory-efficient
