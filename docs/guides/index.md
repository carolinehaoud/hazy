# Guides

Practical guides for using hazy effectively in your projects.

<div class="grid cards" markdown>

-   :material-content-save:{ .lg .middle } **Serialization**

    ---

    Save and load data structures using binary, JSON, and file I/O.

    [:octicons-arrow-right-24: Learn more](serialization.md)

-   :material-chart-line:{ .lg .middle } **Visualization**

    ---

    Plot and visualize data structures for debugging and analysis.

    [:octicons-arrow-right-24: Learn more](visualization.md)

-   :material-notebook:{ .lg .middle } **Jupyter Notebooks**

    ---

    Rich HTML display and interactive exploration in notebooks.

    [:octicons-arrow-right-24: Learn more](jupyter.md)

</div>

## Quick Links

### Serialization

```python
# Binary serialization (compact)
data = bf.to_bytes()
bf = BloomFilter.from_bytes(data)

# JSON (human-readable)
json_str = bf.to_json()
bf = BloomFilter.from_json(json_str)

# File I/O
bf.save("filter.hazy")
bf = BloomFilter.load("filter.hazy")
```

### Visualization

```python
from hazy.viz import plot_bloom, plot_hll, show

plot_bloom(bf)  # Bit array heatmap
plot_hll(hll)   # Register histogram
show()
```

### Jupyter

```python
import hazy
hazy.enable_notebook_display()

bf = hazy.BloomFilter(expected_items=1000)
bf  # Rich HTML display
```
