# Scalable Bloom Filter

A Scalable Bloom filter automatically grows as you add items, making it ideal when you don't know the final dataset size in advance.

## When to Use

- Dataset size is unknown or unbounded
- Streaming data where items arrive over time
- When you can't afford to rebuild filters as data grows
- Log/event deduplication

## Basic Usage

```python
from hazy import ScalableBloomFilter

# Create with initial capacity and target FPR
sbf = ScalableBloomFilter(
    initial_capacity=1000,
    false_positive_rate=0.01
)

# Add items - filter grows automatically
for i in range(100000):
    sbf.add(f"item_{i}")

# Check how many slices were created
print(f"Slices: {sbf.num_slices}")  # Multiple slices

# Query membership
print("item_50000" in sbf)  # True
```

## Construction Options

```python
# Basic construction
sbf = ScalableBloomFilter(initial_capacity=1000)

# With custom FPR
sbf = ScalableBloomFilter(
    initial_capacity=1000,
    false_positive_rate=0.01
)

# With growth control
sbf = ScalableBloomFilter(
    initial_capacity=1000,
    false_positive_rate=0.01,
    growth_ratio=2,      # Each slice is 2x larger
    fpr_ratio=0.9        # Each slice has 90% of previous FPR
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capacity` | Required | Expected items in first slice |
| `false_positive_rate` | 0.01 | Target FPR for the entire filter |
| `growth_ratio` | 2 | How much larger each new slice is |
| `fpr_ratio` | 0.9 | FPR tightening ratio for new slices |

## Key Operations

### Adding Items

```python
sbf = ScalableBloomFilter(initial_capacity=100)

# Single item
sbf.add("item")

# Multiple items
sbf.update(["a", "b", "c"])

# The filter grows automatically
for i in range(10000):
    sbf.add(f"item_{i}")

print(f"Total capacity across {sbf.num_slices} slices")
```

### Membership Testing

```python
if "item" in sbf:
    print("Item might be in the set")

# Or using query method
result = sbf.query("item")
```

## Statistics

```python
sbf = ScalableBloomFilter(initial_capacity=1000)
sbf.update([f"item_{i}" for i in range(50000)])

print(f"Number of slices: {sbf.num_slices}")
print(f"Estimated items: {len(sbf):,}")
print(f"Total memory: {sbf.size_in_bytes:,} bytes")
```

## Serialization

```python
# Binary
data = sbf.to_bytes()
sbf2 = ScalableBloomFilter.from_bytes(data)

# JSON
json_str = sbf.to_json()
sbf2 = ScalableBloomFilter.from_json(json_str)

# File I/O
sbf.save("filter.hazy")
sbf2 = ScalableBloomFilter.load("filter.hazy")
```

## How It Works

A Scalable Bloom filter maintains a **series of standard Bloom filters** (called "slices"):

1. Start with one slice sized for `initial_capacity` items
2. When a slice reaches capacity, create a new larger slice
3. New items go into the newest slice
4. Queries check all slices

### Adding an Item

```
if newest_slice.fill_ratio < threshold:
    newest_slice.add(item)
else:
    create_new_slice()
    new_slice.add(item)
```

### Querying

```
for slice in all_slices:
    if item in slice:
        return True
return False
```

### FPR Management

To maintain the overall FPR as slices are added, each new slice gets a tighter FPR:

\[
FPR_i = FPR_0 \times r^i
\]

where `r` is the `fpr_ratio` (default 0.9).

The total FPR is bounded by:

\[
FPR_{total} \leq FPR_0 \times \frac{1}{1-r}
\]

## Growth Behavior

```python
sbf = ScalableBloomFilter(initial_capacity=1000, growth_ratio=2)

# Slice 1: capacity 1000
# Slice 2: capacity 2000
# Slice 3: capacity 4000
# Slice 4: capacity 8000
# ...

for i in range(50000):
    sbf.add(f"item_{i}")

print(f"Slices created: {sbf.num_slices}")
```

## Memory Considerations

Scalable Bloom filters use more memory than a single pre-sized Bloom filter:

| Scenario | Single Bloom | Scalable Bloom |
|----------|--------------|----------------|
| Known size | Optimal | ~1.2-1.5x more |
| Unknown size | May need rebuild | Automatic growth |
| Overestimated size | Wasted space | Efficient |

!!! tip "When Size is Known"
    If you know the final dataset size, a single `BloomFilter` is more memory-efficient.

## Use Cases

### 1. Stream Deduplication

```python
from hazy import ScalableBloomFilter

seen = ScalableBloomFilter(initial_capacity=10000)

def process_event(event_id, data):
    if event_id in seen:
        return  # Duplicate, skip

    seen.add(event_id)
    handle_event(data)
```

### 2. Unbounded URL Crawler

```python
visited = ScalableBloomFilter(
    initial_capacity=100000,
    false_positive_rate=0.001
)

def crawl(url):
    if url in visited:
        return

    visited.add(url)
    page = fetch(url)

    for link in extract_links(page):
        crawl(link)
```

### 3. Long-Running Log Processor

```python
processed_ids = ScalableBloomFilter(initial_capacity=50000)

def process_log_batch(logs):
    for log in logs:
        if log.id in processed_ids:
            continue

        processed_ids.add(log.id)
        process_log(log)
```

## Comparison with Standard Bloom Filter

| Feature | BloomFilter | ScalableBloomFilter |
|---------|-------------|---------------------|
| Fixed size | Yes | No (grows) |
| Memory efficiency | Optimal* | Slightly higher |
| Unknown cardinality | Must guess or rebuild | Handles automatically |
| Query time | O(k) | O(k × slices) |
| Deletion | No | No |

*When size is correctly estimated.

## Best Practices

1. **Set initial_capacity wisely** - Start with a reasonable estimate to minimize slices

2. **Monitor slice count** - Many slices slow down queries

3. **Choose growth_ratio based on growth pattern**:
   - Rapid growth → higher ratio (2-4)
   - Gradual growth → lower ratio (1.5-2)

4. **Adjust fpr_ratio if FPR is critical**:
   - Lower ratio (0.8) → stricter FPR control
   - Higher ratio (0.95) → more memory efficient

```python
# For rapidly growing data
sbf = ScalableBloomFilter(
    initial_capacity=10000,
    growth_ratio=4,
    fpr_ratio=0.85
)

# For gradually growing data
sbf = ScalableBloomFilter(
    initial_capacity=10000,
    growth_ratio=1.5,
    fpr_ratio=0.95
)
```
