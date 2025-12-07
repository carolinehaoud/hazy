# Count-Min Sketch

Count-Min Sketch (CMS) is a probabilistic data structure for **frequency estimation** in streaming data. It answers "approximately how many times has this item appeared?"

## When to Use

- Tracking click/view counts for millions of URLs
- Finding heavy hitters (most frequent items)
- Frequency estimation in network traffic analysis
- Any streaming scenario where exact counts are impractical

## Basic Usage

```python
from hazy import CountMinSketch

# Create with width=10000, depth=5
cms = CountMinSketch(width=10000, depth=5)

# Count items
for url in stream_of_clicks():
    cms.add(url)

# Query frequency
clicks = cms["/home"]
print(f"Clicks on /home: {clicks}")
```

## Construction Options

### Using Width and Depth

```python
# width × depth counters
cms = CountMinSketch(width=10000, depth=5)
```

### Using Error Bounds

```python
# Specify desired accuracy
cms = CountMinSketch(error_rate=0.001, confidence=0.99)
```

This creates a CMS where:

- Overestimate is at most ε × N with probability 1-δ
- ε = error_rate, δ = 1 - confidence, N = total count

### Parameters

| Parameter | Formula | Description |
|-----------|---------|-------------|
| `width` | ⌈e/ε⌉ | Number of counters per row |
| `depth` | ⌈ln(1/δ)⌉ | Number of hash functions/rows |

### Parameter Estimation

```python
from hazy import estimate_cms_params

params = estimate_cms_params(error_rate=0.001, confidence=0.99)
print(f"Width: {params.width:,}")
print(f"Depth: {params.depth}")
print(f"Memory: {params.memory_mb:.2f} MB")
```

## Key Operations

### Adding Items

```python
cms = CountMinSketch(width=10000, depth=5)

# Add single occurrence
cms.add("apple")

# Add multiple occurrences
cms.add_count("apple", 10)

# Bulk add
cms.update(["apple", "banana", "apple", "cherry"])
```

### Querying Frequencies

```python
# Using indexing
count = cms["apple"]

# Using query method
count = cms.query("apple")

# Total count of all items
total = cms.total_count
```

### Merging Sketches

```python
cms1 = CountMinSketch(width=10000, depth=5)
cms2 = CountMinSketch(width=10000, depth=5)

cms1.add_count("apple", 50)
cms2.add_count("apple", 30)

# Merge in-place
cms1.merge(cms2)
print(cms1["apple"])  # >= 80
```

## Statistics

```python
cms = CountMinSketch(width=10000, depth=5)
cms.update([f"item_{i % 100}" for i in range(50000)])

print(f"Width: {cms.width:,}")
print(f"Depth: {cms.depth}")
print(f"Total count: {cms.total_count:,}")
print(f"Memory: {cms.size_in_bytes:,} bytes")
```

## Serialization

```python
# Binary
data = cms.to_bytes()
cms2 = CountMinSketch.from_bytes(data)

# JSON
json_str = cms.to_json()
cms2 = CountMinSketch.from_json(json_str)

# File I/O
cms.save("sketch.hazy")
cms2 = CountMinSketch.load("sketch.hazy")
```

## How It Works

### Structure

CMS maintains a 2D array of counters: `depth` rows × `width` columns.

### Adding an Item

```
for each row i (0 to depth-1):
    col = hash_i(item) mod width
    counters[i][col] += count
```

### Querying

```
estimates = []
for each row i:
    col = hash_i(item) mod width
    estimates.append(counters[i][col])
return min(estimates)
```

The minimum across rows reduces overestimation from hash collisions.

### Error Bounds

For any item with true count `f`:

\[
f \leq \hat{f} \leq f + \varepsilon \cdot N
\]

with probability at least 1 - δ, where:

- \(\hat{f}\) is the estimate
- \(\varepsilon = e/\text{width}\)
- \(\delta = e^{-\text{depth}}\)
- N is total count

## Use Cases

### 1. Click Tracking

```python
from hazy import CountMinSketch

clicks = CountMinSketch(error_rate=0.001, confidence=0.99)

def record_click(url):
    clicks.add(url)

def get_click_count(url):
    return clicks[url]

def get_top_pages():
    # Would need to track candidates separately
    pass
```

### 2. Heavy Hitters Detection

```python
from hazy import CountMinSketch

class HeavyHitterTracker:
    def __init__(self, threshold_ratio=0.01):
        self.cms = CountMinSketch(width=10000, depth=5)
        self.threshold_ratio = threshold_ratio
        self.candidates = set()

    def add(self, item):
        self.cms.add(item)

        # Track items that might be heavy
        if self.cms[item] > self.cms.total_count * self.threshold_ratio:
            self.candidates.add(item)

    def get_heavy_hitters(self):
        threshold = self.cms.total_count * self.threshold_ratio
        return {item for item in self.candidates
                if self.cms[item] >= threshold}
```

### 3. Network Traffic Analysis

```python
from hazy import CountMinSketch

ip_counts = CountMinSketch(width=100000, depth=7)

def process_packet(packet):
    ip_counts.add(packet.source_ip)

def is_potential_dos(ip, threshold=10000):
    return ip_counts[ip] > threshold
```

### 4. Inner Product / Similarity

```python
from hazy import CountMinSketch

def frequency_similarity(cms1, cms2):
    """Estimate dot product of frequency vectors."""
    return cms1.inner_product(cms2)

# Build frequency profiles
profile1 = CountMinSketch(width=10000, depth=5)
profile2 = CountMinSketch(width=10000, depth=5)

for word in document1:
    profile1.add(word)
for word in document2:
    profile2.add(word)

similarity = frequency_similarity(profile1, profile2)
```

## Accuracy Characteristics

### Never Underestimates

CMS **never** reports a count lower than the true count. It may overestimate due to hash collisions.

### Error Increases with Load

As more items are added, collision probability increases:

```python
cms = CountMinSketch(width=1000, depth=5)

# Track error as we add items
for n in [100, 1000, 10000, 100000]:
    cms.clear()
    for i in range(n):
        cms.add(f"item_{i % 100}")

    # Check overestimation
    true_count = n // 100
    estimated = cms["item_0"]
    error = estimated - true_count
    print(f"N={n:,}: true={true_count}, est={estimated}, error={error}")
```

## Memory vs Accuracy Tradeoff

| Width | Depth | Memory | ε (error rate) | δ (failure prob) |
|-------|-------|--------|----------------|------------------|
| 1,000 | 4 | 32 KB | 0.27% | 1.8% |
| 10,000 | 5 | 400 KB | 0.027% | 0.67% |
| 100,000 | 7 | 5.6 MB | 0.0027% | 0.09% |

## Comparison with Other Structures

| Feature | Count-Min Sketch | HyperLogLog | TopK |
|---------|-----------------|-------------|------|
| Query type | Frequency | Cardinality | Top items |
| Error direction | Overestimates | Both | Overestimates |
| Merge | Yes | Yes | No |
| Memory | O(w × d) | O(2^p) | O(k) |

## Best Practices

1. **Size based on error tolerance** - Use `error_rate` and `confidence` parameters

2. **Width matters more than depth** - Wider is better for accuracy; depth 5-7 is usually enough

3. **Track candidates for top-k** - CMS alone can't enumerate all items

4. **Merge for distributed counting** - Same width/depth required

```python
# Good default for general use
cms = CountMinSketch(error_rate=0.001, confidence=0.99)

# Memory constrained
cms = CountMinSketch(width=1000, depth=4)

# High accuracy
cms = CountMinSketch(width=100000, depth=7)
```
