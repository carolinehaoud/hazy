# Visualization

Hazy includes built-in visualization tools for debugging, education, and analysis. This guide covers all available plotting functions with complete examples.

## Installation

Install hazy with visualization support:

```bash
pip install hazy[viz]
```

This includes `matplotlib` as a dependency.

## Quick Start

```python
from hazy import BloomFilter
from hazy.viz import plot_bloom, show

# Create and populate a filter
bf = BloomFilter(expected_items=10000)
bf.update([f"item_{i}" for i in range(5000)])

# Visualize it
plot_bloom(bf)
show()
```

## Bloom Filter Visualizations

### Bit Array Heatmap

Visualize the bit array as a 2D heatmap showing which bits are set:

```python
from hazy import BloomFilter
from hazy.viz import plot_bloom, show
import matplotlib.pyplot as plt

# Create filter with different fill levels
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

fill_levels = [0.25, 0.5, 0.75]
for ax, fill in zip(axes, fill_levels):
    bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)
    n_items = int(10000 * fill)
    bf.update([f"item_{i}" for i in range(n_items)])

    plt.sca(ax)
    plot_bloom(bf, title=f"Fill: {bf.fill_ratio:.0%} ({n_items:,} items)")

plt.tight_layout()
plt.savefig("bloom_fill_comparison.png", dpi=150)
plt.show()
```

**What to look for:**

- **Sparse (light)**: Filter has room for more items
- **Dense (dark)**: Filter is getting full, FPR increasing
- **Uniform distribution**: Hash functions are working well
- **Patterns or clusters**: Potential hash quality issues

### Fill Ratio and FPR Curves

Plot how fill ratio and false positive rate change as items are added:

```python
from hazy import BloomFilter
from hazy.viz import plot_bloom_fill_curve, show
import matplotlib.pyplot as plt

bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)

fig, ax = plt.subplots(figsize=(10, 6))
plot_bloom_fill_curve(bf, max_items=15000)
plt.title("Bloom Filter Performance vs Items Added")
plt.savefig("bloom_curves.png", dpi=150)
plt.show()
```

**Understanding the curves:**

- **Fill ratio** approaches 1.0 asymptotically
- **FPR** increases exponentially as filter fills
- The "knee" shows optimal operating range
- Beyond expected_items, FPR degrades rapidly

## HyperLogLog Visualization

### Register Histogram

Visualize the distribution of register values:

```python
from hazy import HyperLogLog
from hazy.viz import plot_hll, show
import matplotlib.pyplot as plt

# Compare different cardinalities
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

cardinalities = [1_000, 100_000, 10_000_000]
for ax, n in zip(axes, cardinalities):
    hll = HyperLogLog(precision=12)
    hll.update([f"item_{i}" for i in range(n)])

    plt.sca(ax)
    plot_hll(hll, title=f"Cardinality: {n:,}\nEstimate: {int(hll.cardinality()):,}")

plt.tight_layout()
plt.savefig("hll_comparison.png", dpi=150)
plt.show()
```

**What to look for:**

- **Peak position**: Shifts right as more items are added
- **Distribution width**: Related to estimation variance
- **Empty registers (0s)**: Indicate low cardinality
- **High values (>30)**: Indicate very high cardinality

## Count-Min Sketch Visualization

### Counter Heatmap

Visualize the 2D counter array:

```python
from hazy import CountMinSketch
from hazy.viz import plot_cms, show
import matplotlib.pyplot as plt

# Create sketch with known heavy hitters
cms = CountMinSketch(width=100, depth=5)

# Add items with different frequencies
items = (
    ["hot_item"] * 1000 +
    ["warm_item"] * 100 +
    ["cold_item"] * 10 +
    [f"rare_{i}" for i in range(500)]
)

for item in items:
    cms.add(item)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
plt.sca(axes[0])
plot_cms(cms, title="Linear Scale", log_scale=False)

# Log scale (better for skewed distributions)
plt.sca(axes[1])
plot_cms(cms, title="Log Scale", log_scale=True)

plt.tight_layout()
plt.savefig("cms_heatmap.png", dpi=150)
plt.show()
```

**What to look for:**

- **Bright spots**: Columns hit by heavy hitters
- **Uniform background**: Normal collision noise
- **Row patterns**: Each row is independent hash function
- **Use log scale** when data is highly skewed

## Top-K Visualization

### Bar Chart with Error Bounds

```python
from hazy import TopK
from hazy.viz import plot_topk, show
import matplotlib.pyplot as plt
import random

# Simulate word frequencies (Zipf distribution)
tk = TopK(k=20)

words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
         "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"]
weights = [1.0 / (i + 1) ** 0.8 for i in range(len(words))]

# Add words according to weights
for _ in range(100_000):
    word = random.choices(words, weights=weights)[0]
    tk.add(word)

fig, ax = plt.subplots(figsize=(10, 8))
plot_topk(tk, n=15, title="Most Frequent Words")
plt.tight_layout()
plt.savefig("topk_chart.png", dpi=150)
plt.show()

# Print with error bounds
print("\nTop 10 with error bounds:")
for word, count, error in tk.top_with_error(10):
    print(f"  {word}: {count:,} ± {error}")
```

## MinHash Comparison

### Signature Comparison

Compare two MinHash signatures side by side:

```python
from hazy import MinHash
from hazy.viz import plot_minhash_comparison, show
import matplotlib.pyplot as plt

# Create two documents with partial overlap
doc1_words = {"machine", "learning", "is", "a", "subset", "of", "artificial", "intelligence"}
doc2_words = {"deep", "learning", "is", "part", "of", "machine", "learning", "methods"}

mh1 = MinHash(num_hashes=64)
mh2 = MinHash(num_hashes=64)

mh1.update(doc1_words)
mh2.update(doc2_words)

fig, ax = plt.subplots(figsize=(12, 6))
plot_minhash_comparison(mh1, mh2, title="Document Similarity Comparison")

# Add similarity annotation
similarity = mh1.jaccard(mh2)
plt.figtext(0.5, 0.02, f"Estimated Jaccard Similarity: {similarity:.2%}",
            ha="center", fontsize=12, style="italic")

plt.tight_layout()
plt.savefig("minhash_comparison.png", dpi=150)
plt.show()
```

## Creating Dashboards

Combine multiple visualizations into a dashboard:

```python
from hazy import BloomFilter, HyperLogLog, CountMinSketch, TopK
from hazy.viz import plot_bloom, plot_hll, plot_cms, plot_topk
import matplotlib.pyplot as plt

def create_analytics_dashboard():
    """Create a comprehensive analytics dashboard."""

    # Generate sample data
    bf = BloomFilter(expected_items=10000)
    hll = HyperLogLog(precision=12)
    cms = CountMinSketch(width=1000, depth=5)
    tk = TopK(k=20)

    # Simulate events
    pages = ["/home", "/about", "/products", "/blog", "/contact"]
    weights = [100, 20, 50, 30, 10]

    for i in range(50000):
        user = f"user_{i % 5000}"
        page = __import__("random").choices(pages, weights=weights)[0]

        bf.add(user)
        hll.add(user)
        cms.add(page)
        tk.add(page)

    # Create dashboard
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Analytics Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # 1. Bloom Filter (visitor tracking)
    ax1 = fig.add_subplot(2, 2, 1)
    plt.sca(ax1)
    plot_bloom(bf, title=f"Visitor Filter\n({len(bf):,} users, {bf.fill_ratio:.0%} full)")

    # 2. HyperLogLog (unique count)
    ax2 = fig.add_subplot(2, 2, 2)
    plt.sca(ax2)
    plot_hll(hll, title=f"Unique Visitors\nEstimate: {int(hll.cardinality()):,}")

    # 3. Count-Min Sketch (page views)
    ax3 = fig.add_subplot(2, 2, 3)
    plt.sca(ax3)
    plot_cms(cms, title="Page View Distribution", log_scale=True)

    # 4. Top-K (popular pages)
    ax4 = fig.add_subplot(2, 2, 4)
    plt.sca(ax4)
    plot_topk(tk, n=5, title="Top Pages")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("analytics_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nDashboard saved to 'analytics_dashboard.png'")

create_analytics_dashboard()
```

## Customization Options

### Color Maps

```python
from hazy import BloomFilter
from hazy.viz import plot_bloom
import matplotlib.pyplot as plt

bf = BloomFilter(expected_items=5000)
bf.update([f"item_{i}" for i in range(2500)])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
cmaps = ["Blues", "Greens", "Reds", "viridis"]

for ax, cmap in zip(axes.flat, cmaps):
    plt.sca(ax)
    plot_bloom(bf, title=f"Colormap: {cmap}", cmap=cmap)

plt.tight_layout()
plt.savefig("bloom_colormaps.png", dpi=150)
plt.show()
```

### Figure Sizes

```python
# Small (for inline/thumbnail)
plt.figure(figsize=(6, 4))
plot_bloom(bf)

# Large (for presentations)
plt.figure(figsize=(14, 10))
plot_bloom(bf)

# Custom aspect ratio
plt.figure(figsize=(16, 6))
plot_bloom(bf)
```

### Saving Figures

```python
# PNG (raster, good for web)
plt.savefig("figure.png", dpi=150, bbox_inches="tight")

# PDF (vector, good for papers)
plt.savefig("figure.pdf", bbox_inches="tight")

# SVG (vector, good for web)
plt.savefig("figure.svg", bbox_inches="tight")

# High-resolution PNG
plt.savefig("figure_hires.png", dpi=300, bbox_inches="tight")
```

## Non-Interactive Mode

For scripts and servers without a display:

```python
import matplotlib
matplotlib.use("Agg")  # Must be before importing pyplot

import matplotlib.pyplot as plt
from hazy import BloomFilter
from hazy.viz import plot_bloom

bf = BloomFilter(expected_items=10000)
bf.update([f"item_{i}" for i in range(5000)])

plot_bloom(bf)
plt.savefig("output.png", dpi=150)
print("Saved to output.png")
```

## Complete Example Script

```python
#!/usr/bin/env python
"""
Generate all visualization examples.

Run with: python generate_visualizations.py
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from hazy import BloomFilter, HyperLogLog, CountMinSketch, TopK, MinHash
from hazy.viz import (
    plot_bloom, plot_bloom_fill_curve, plot_hll,
    plot_cms, plot_topk, plot_minhash_comparison
)
import random


def main():
    print("Generating visualizations...")

    # 1. Bloom Filter
    bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)
    bf.update([f"item_{i}" for i in range(5000)])

    plt.figure(figsize=(10, 6))
    plot_bloom(bf, title="Bloom Filter (50% full)")
    plt.savefig("viz_bloom.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  - viz_bloom.png")

    # 2. Bloom Filter curves
    plt.figure(figsize=(10, 6))
    plot_bloom_fill_curve(bf, max_items=15000)
    plt.savefig("viz_bloom_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  - viz_bloom_curves.png")

    # 3. HyperLogLog
    hll = HyperLogLog(precision=12)
    hll.update([f"user_{i}" for i in range(100000)])

    plt.figure(figsize=(10, 6))
    plot_hll(hll, title=f"HyperLogLog (est: {int(hll.cardinality()):,})")
    plt.savefig("viz_hll.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  - viz_hll.png")

    # 4. Count-Min Sketch
    cms = CountMinSketch(width=100, depth=5)
    for _ in range(10000):
        cms.add(random.choice(["a", "b", "c", "d", "e"]))

    plt.figure(figsize=(10, 6))
    plot_cms(cms, title="Count-Min Sketch", log_scale=True)
    plt.savefig("viz_cms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  - viz_cms.png")

    # 5. Top-K
    tk = TopK(k=20)
    words = ["python", "rust", "java", "go", "javascript"]
    weights = [50, 30, 25, 20, 15]
    for _ in range(10000):
        tk.add(random.choices(words, weights=weights)[0])

    plt.figure(figsize=(10, 6))
    plot_topk(tk, n=5, title="Top Programming Languages")
    plt.savefig("viz_topk.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  - viz_topk.png")

    # 6. MinHash comparison
    mh1 = MinHash(num_hashes=64)
    mh2 = MinHash(num_hashes=64)
    mh1.update(["a", "b", "c", "d", "e"])
    mh2.update(["c", "d", "e", "f", "g"])

    plt.figure(figsize=(12, 6))
    plot_minhash_comparison(mh1, mh2, title="MinHash Signature Comparison")
    plt.savefig("viz_minhash.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  - viz_minhash.png")

    print("\nDone! Generated 6 visualization files.")


if __name__ == "__main__":
    main()
```
