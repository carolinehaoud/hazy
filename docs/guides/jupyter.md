# Jupyter Notebooks

Hazy provides rich HTML display for Jupyter notebooks, making it easy to inspect and debug data structures interactively.

## Enabling Rich Display

Call `enable_notebook_display()` once at the start of your notebook:

```python
import hazy
hazy.enable_notebook_display()
```

Now all hazy types will display with rich HTML formatting.

## BloomFilter Display

```python
from hazy import BloomFilter

bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)
bf.update([f"item_{i}" for i in range(5000)])
bf  # Rich display
```

Shows:

- **Type badge**: "BloomFilter"
- **Progress bar**: Visual fill ratio
- **Statistics table**: Items, bits, hashes, fill ratio, FPR, memory

## HyperLogLog Display

```python
from hazy import HyperLogLog

hll = HyperLogLog(precision=14)
hll.update([f"user_{i}" for i in range(100000)])
hll
```

Shows:

- **Type badge**: "HyperLogLog"
- **Cardinality**: Estimated unique count
- **Statistics**: Precision, registers, memory

## CountMinSketch Display

```python
from hazy import CountMinSketch

cms = CountMinSketch(width=10000, depth=5)
for word in ["apple"] * 50 + ["banana"] * 30:
    cms.add(word)
cms
```

Shows:

- **Type badge**: "CountMinSketch"
- **Dimensions**: Width × Depth
- **Statistics**: Total count, memory

## Other Types

All types have rich display:

```python
from hazy import (
    CountingBloomFilter,
    ScalableBloomFilter,
    CuckooFilter,
    MinHash,
    TopK
)

# Each displays relevant statistics
cbf = CountingBloomFilter(expected_items=1000)
sbf = ScalableBloomFilter(initial_capacity=100)
cf = CuckooFilter(capacity=1000)
mh = MinHash(num_hashes=128)
tk = TopK(k=10)
```

## Combining with Plots

Use both rich display and visualizations:

```python
import hazy
from hazy import BloomFilter
from hazy.viz import plot_bloom

hazy.enable_notebook_display()

bf = BloomFilter(expected_items=10000)
bf.update([f"item_{i}" for i in range(5000)])

# Rich text display
display(bf)

# Visual plot
plot_bloom(bf)
```

## Display Customization

The HTML display is styled but can be customized with CSS:

```python
from IPython.display import HTML

# Add custom styles
HTML("""
<style>
.hazy-container {
    font-family: 'Fira Code', monospace;
    border-radius: 8px;
}
.hazy-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
</style>
""")
```

## Interactive Exploration

### Comparing Structures

```python
import hazy
hazy.enable_notebook_display()

from hazy import BloomFilter

# Compare different configurations
configs = [
    {"expected_items": 10000, "false_positive_rate": 0.1},
    {"expected_items": 10000, "false_positive_rate": 0.01},
    {"expected_items": 10000, "false_positive_rate": 0.001},
]

for config in configs:
    bf = BloomFilter(**config)
    bf.update([f"item_{i}" for i in range(5000)])
    print(f"FPR target: {config['false_positive_rate']}")
    display(bf)
    print()
```

### Tracking Progress

```python
import hazy
from hazy import HyperLogLog
from IPython.display import clear_output
import time

hazy.enable_notebook_display()

hll = HyperLogLog(precision=14)

for batch in range(10):
    # Add items
    hll.update([f"item_{batch}_{i}" for i in range(10000)])

    # Update display
    clear_output(wait=True)
    print(f"Batch {batch + 1}/10")
    display(hll)
    time.sleep(0.5)
```

## Example Notebook

Here's a complete example notebook:

```python
# Cell 1: Setup
import hazy
hazy.enable_notebook_display()

from hazy import (
    BloomFilter, HyperLogLog, CountMinSketch, TopK
)
```

```python
# Cell 2: Bloom Filter
bf = BloomFilter(expected_items=100000, false_positive_rate=0.01)
bf.update([f"user_{i}" for i in range(50000)])
bf
```

```python
# Cell 3: Test membership
test_items = ["user_100", "user_50000", "nonexistent_user"]
for item in test_items:
    result = "✓" if item in bf else "✗"
    print(f"{item}: {result}")
```

```python
# Cell 4: HyperLogLog for cardinality
hll = HyperLogLog(precision=14)
hll.update([f"visitor_{i}" for i in range(1_000_000)])
print(f"Estimated unique visitors: {hll.cardinality():,.0f}")
hll
```

```python
# Cell 5: Count-Min Sketch for frequencies
cms = CountMinSketch(width=10000, depth=5)

# Simulate click stream
import random
pages = ["/home", "/about", "/products", "/contact", "/blog"]
weights = [100, 20, 50, 10, 30]

for _ in range(10000):
    page = random.choices(pages, weights=weights)[0]
    cms.add(page)

print("Page view estimates:")
for page in pages:
    print(f"  {page}: {cms[page]}")

cms
```

```python
# Cell 6: TopK for trending
tk = TopK(k=20)

# Simulate hashtag stream
hashtags = ["#python", "#datascience", "#ml", "#ai", "#coding"]
weights = [50, 30, 25, 20, 15]

for _ in range(5000):
    tag = random.choices(hashtags, weights=weights)[0]
    tk.add(tag)

print("Trending hashtags:")
for tag, count in tk.top(5):
    print(f"  {tag}: {count}")

tk
```

```python
# Cell 7: Visualization
from hazy.viz import plot_bloom, plot_hll, plot_cms, plot_topk, show

plot_bloom(bf)
show()
```

## Disabling Rich Display

To revert to default repr:

```python
# Temporarily use default display
print(repr(bf))

# Or disable globally by not calling enable_notebook_display()
```

## Troubleshooting

### Display Not Working

1. Make sure you called `enable_notebook_display()`
2. Check that you're in a Jupyter environment
3. Restart the kernel and try again

### HTML Not Rendering

Some environments (like VS Code's notebook viewer) may need settings adjusted:

```python
from IPython.display import display, HTML

# Force HTML display
html = bf._repr_html_()
display(HTML(html))
```

### Performance with Large Structures

Rich display is optimized but very large structures may be slow:

```python
# For very large structures, use basic display
print(f"BloomFilter: {len(bf):,} items, {bf.fill_ratio:.1%} full")
```
