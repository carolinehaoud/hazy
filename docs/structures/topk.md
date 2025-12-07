# Top-K (Space-Saving)

Top-K uses the Space-Saving algorithm to find the **most frequent items** in a data stream using limited memory.

## When to Use

- Finding trending topics or hashtags
- Identifying heavy hitters in network traffic
- Tracking most viewed pages or products
- Any scenario where you need the "top N" frequent items

## Basic Usage

```python
from hazy import TopK

# Track top 10 items
tk = TopK(k=10)

# Add items from a stream
for word in text.split():
    tk.add(word)

# Get most frequent items
for item, count in tk.top(5):
    print(f"{item}: {count}")
```

## Construction

```python
# Track top k items
tk = TopK(k=10)    # Top 10
tk = TopK(k=100)   # Top 100
tk = TopK(k=1000)  # Top 1000
```

## Key Operations

### Adding Items

```python
tk = TopK(k=10)

# Single item
tk.add("apple")

# With explicit count
tk.add_count("apple", 10)

# Bulk add
tk.update(["apple", "banana", "apple", "cherry"])
```

### Getting Top Items

```python
tk = TopK(k=10)
tk.update(["a"] * 100 + ["b"] * 50 + ["c"] * 25)

# Get top 3
for item, count in tk.top(3):
    print(f"{item}: {count}")

# Output:
# a: 100
# b: 50
# c: 25
```

### Getting Top Items with Error Bounds

```python
# Get counts with guaranteed error bounds
for item, count, error in tk.top_with_error(3):
    print(f"{item}: {count} (±{error})")
    print(f"  True count is between {count - error} and {count}")
```

The error bound represents the maximum possible overestimation.

### Querying Specific Items

```python
# Check if item is in top-k
count = tk.query("apple")  # Returns count or 0
```

## Statistics

```python
tk = TopK(k=10)
tk.update([f"item_{i % 100}" for i in range(10000)])

print(f"K: {tk.k}")
print(f"Items tracked: {len(tk)}")
print(f"Memory: {tk.size_in_bytes} bytes")
```

## Serialization

```python
# Binary
data = tk.to_bytes()
tk2 = TopK.from_bytes(data)

# JSON
json_str = tk.to_json()
tk2 = TopK.from_json(json_str)

# File I/O
tk.save("topk.hazy")
tk2 = TopK.load("topk.hazy")
```

## How It Works

### Space-Saving Algorithm

The algorithm maintains a fixed-size dictionary of k counters:

**Adding an item:**
```
if item is already tracked:
    increment its counter
elif fewer than k items tracked:
    add item with count 1
else:
    find item with minimum count
    replace it with new item
    set new item's count = old minimum + 1
    record error = old minimum
```

### Guaranteed Accuracy

For any item with true count f:

- If f > N/k (where N is total count), the item is **guaranteed** to be tracked
- The error bound for any tracked item is at most N/k

### Why It Works

Items that are truly frequent get incrementally larger counts and are rarely evicted. Items that are evicted must have counts below the minimum, so the error is bounded.

## Use Cases

### 1. Trending Topics

```python
from hazy import TopK

trending = TopK(k=100)

def record_hashtag(hashtag):
    trending.add(hashtag.lower())

def get_trending(n=10):
    return trending.top(n)

# Usage
for tweet in tweet_stream():
    for tag in extract_hashtags(tweet):
        record_hashtag(tag)

print("Trending now:")
for tag, count in get_trending(10):
    print(f"  #{tag}: {count}")
```

### 2. Heavy Hitters Detection

```python
from hazy import TopK

ip_tracker = TopK(k=1000)

def process_request(request):
    ip_tracker.add(request.client_ip)

def get_suspicious_ips(threshold=1000):
    """Get IPs with high request counts."""
    return [(ip, count) for ip, count in ip_tracker.top(100)
            if count > threshold]
```

### 3. Product Analytics

```python
from hazy import TopK

class ProductTracker:
    def __init__(self):
        self.views = TopK(k=100)
        self.purchases = TopK(k=100)
        self.searches = TopK(k=100)

    def record_view(self, product_id):
        self.views.add(product_id)

    def record_purchase(self, product_id):
        self.purchases.add(product_id)

    def record_search(self, query):
        self.searches.add(query.lower())

    def report(self):
        print("Most Viewed:")
        for pid, count in self.views.top(10):
            print(f"  {pid}: {count}")

        print("\nBest Sellers:")
        for pid, count in self.purchases.top(10):
            print(f"  {pid}: {count}")

        print("\nTop Searches:")
        for query, count in self.searches.top(10):
            print(f"  {query}: {count}")
```

### 4. Log Analysis

```python
from hazy import TopK

error_tracker = TopK(k=50)

def process_log(log_line):
    if "ERROR" in log_line:
        error_type = extract_error_type(log_line)
        error_tracker.add(error_type)

def get_error_summary():
    print("Most Common Errors:")
    for error, count, bound in error_tracker.top_with_error(10):
        print(f"  {error}: ~{count} (±{bound})")
```

## Accuracy Characteristics

### Guaranteed Heavy Hitters

If an item appears more than N/k times, it is **guaranteed** to be in the top-k list.

### Error Bounds

```python
tk = TopK(k=10)

# Add items
for i in range(10000):
    tk.add(f"item_{i % 100}")

# Check error bounds
for item, count, error in tk.top_with_error(5):
    # True count is in [count - error, count]
    lower_bound = count - error
    upper_bound = count
    print(f"{item}: {count} (true count in [{lower_bound}, {upper_bound}])")
```

### When Counts Are Exact

For the most frequent items (those never evicted), counts are exact:

```python
tk = TopK(k=10)

# "winner" is added way more than others
tk.update(["winner"] * 10000)
tk.update([f"other_{i}" for i in range(1000)])

# Top item has exact count (error = 0)
item, count, error = tk.top_with_error(1)[0]
print(f"{item}: {count} (error: {error})")  # winner: 10000 (error: 0)
```

## Memory Usage

TopK uses O(k) memory:

| k | Approximate Memory |
|---|--------------------|
| 10 | ~1 KB |
| 100 | ~10 KB |
| 1,000 | ~100 KB |
| 10,000 | ~1 MB |

## Comparison with Alternatives

| Method | Space | Error | Can List Items |
|--------|-------|-------|----------------|
| TopK (Space-Saving) | O(k) | ≤ N/k | Yes |
| Count-Min Sketch | O(w×d) | ε×N | No (need candidates) |
| Exact counting | O(n) | 0 | Yes |

## Best Practices

1. **Size k based on what you need**:
   - Top 10 trending → k=100 (some buffer)
   - Heavy hitter detection → k=1000+

2. **Use error bounds for important decisions**:
   ```python
   for item, count, error in tk.top_with_error(10):
       if count - error > threshold:
           # Confident this is truly heavy
           take_action(item)
   ```

3. **Consider the N/k threshold**:
   - Items with count > N/k are guaranteed tracked
   - For 1M items with k=1000, need 1000+ occurrences

4. **Combine with Count-Min Sketch for verification**:
   ```python
   from hazy import TopK, CountMinSketch

   tk = TopK(k=100)
   cms = CountMinSketch(width=10000, depth=5)

   for item in stream:
       tk.add(item)
       cms.add(item)

   # Verify top-k counts with CMS
   for item, count in tk.top(10):
       cms_count = cms[item]
       print(f"{item}: TopK={count}, CMS={cms_count}")
   ```
