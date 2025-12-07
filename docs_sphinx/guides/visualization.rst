Visualization
=============

Hazy includes built-in visualization tools for debugging, education, and analysis.

Installation
------------

Install with visualization support:

.. code-block:: bash

   pip install hazy[viz]

This includes ``matplotlib`` as a dependency.

Quick Start
-----------

.. code-block:: python

   from hazy import BloomFilter
   from hazy.viz import plot_bloom, show

   bf = BloomFilter(expected_items=10000)
   bf.update([f"item_{i}" for i in range(5000)])

   plot_bloom(bf)
   show()

Bloom Filter Visualizations
---------------------------

Bit Array Heatmap
~~~~~~~~~~~~~~~~~

Visualize the bit array as a 2D heatmap:

.. code-block:: python

   from hazy import BloomFilter
   from hazy.viz import plot_bloom
   import matplotlib.pyplot as plt

   # Compare different fill levels
   fig, axes = plt.subplots(1, 3, figsize=(15, 4))

   for ax, fill in zip(axes, [0.25, 0.5, 0.75]):
       bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)
       n_items = int(10000 * fill)
       bf.update([f"item_{i}" for i in range(n_items)])

       plt.sca(ax)
       plot_bloom(bf, title=f"Fill: {bf.fill_ratio:.0%}")

   plt.tight_layout()
   plt.savefig("bloom_comparison.png", dpi=150)
   plt.show()

What to look for:

- **Sparse (light)**: Filter has room for more items
- **Dense (dark)**: Filter is getting full, FPR increasing
- **Uniform distribution**: Hash functions working well
- **Patterns**: Potential hash quality issues

Fill Ratio Curves
~~~~~~~~~~~~~~~~~

Plot fill ratio and FPR as items are added:

.. code-block:: python

   from hazy import BloomFilter
   from hazy.viz import plot_bloom_fill_curve
   import matplotlib.pyplot as plt

   bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)

   fig, ax = plt.subplots(figsize=(10, 6))
   plot_bloom_fill_curve(bf, max_items=15000)
   plt.title("Bloom Filter Performance vs Items Added")
   plt.savefig("bloom_curves.png", dpi=150)
   plt.show()

HyperLogLog Visualization
-------------------------

Register Histogram
~~~~~~~~~~~~~~~~~~

Visualize the distribution of register values:

.. code-block:: python

   from hazy import HyperLogLog
   from hazy.viz import plot_hll
   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 3, figsize=(15, 4))

   cardinalities = [1_000, 100_000, 10_000_000]
   for ax, n in zip(axes, cardinalities):
       hll = HyperLogLog(precision=12)
       hll.update([f"item_{i}" for i in range(n)])

       plt.sca(ax)
       plot_hll(hll, title=f"n={n:,}\nest={int(hll.cardinality()):,}")

   plt.tight_layout()
   plt.show()

What to look for:

- **Peak position**: Shifts right with more items
- **Distribution width**: Related to estimation variance
- **Empty registers (0s)**: Low cardinality indicator
- **High values (>30)**: Very high cardinality

Count-Min Sketch Visualization
------------------------------

Counter Heatmap
~~~~~~~~~~~~~~~

Visualize the 2D counter array:

.. code-block:: python

   from hazy import CountMinSketch
   from hazy.viz import plot_cms
   import matplotlib.pyplot as plt

   cms = CountMinSketch(width=100, depth=5)

   # Add items with different frequencies
   items = ["hot"] * 1000 + ["warm"] * 100 + ["cold"] * 10
   for item in items:
       cms.add(item)

   fig, axes = plt.subplots(1, 2, figsize=(14, 5))

   plt.sca(axes[0])
   plot_cms(cms, title="Linear Scale", log_scale=False)

   plt.sca(axes[1])
   plot_cms(cms, title="Log Scale", log_scale=True)

   plt.tight_layout()
   plt.show()

TopK Visualization
------------------

Bar Chart
~~~~~~~~~

.. code-block:: python

   from hazy import TopK
   from hazy.viz import plot_topk
   import matplotlib.pyplot as plt
   import random

   tk = TopK(k=20)

   # Zipf distribution
   words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I"]
   weights = [1.0 / (i + 1) ** 0.8 for i in range(len(words))]

   for _ in range(100_000):
       word = random.choices(words, weights=weights)[0]
       tk.add(word)

   fig, ax = plt.subplots(figsize=(10, 6))
   plot_topk(tk, n=10, title="Most Frequent Words")
   plt.tight_layout()
   plt.show()

MinHash Comparison
------------------

Signature Comparison
~~~~~~~~~~~~~~~~~~~~

Compare two MinHash signatures:

.. code-block:: python

   from hazy import MinHash
   from hazy.viz import plot_minhash_comparison
   import matplotlib.pyplot as plt

   doc1_words = {"machine", "learning", "is", "a", "subset"}
   doc2_words = {"deep", "learning", "is", "part", "of", "machine"}

   mh1 = MinHash(num_hashes=64)
   mh2 = MinHash(num_hashes=64)

   mh1.update(doc1_words)
   mh2.update(doc2_words)

   fig, ax = plt.subplots(figsize=(12, 6))
   plot_minhash_comparison(mh1, mh2, title="Document Similarity")

   similarity = mh1.jaccard(mh2)
   plt.figtext(0.5, 0.02, f"Jaccard Similarity: {similarity:.2%}",
               ha="center", fontsize=12)

   plt.tight_layout()
   plt.show()

Creating Dashboards
-------------------

Combine multiple visualizations:

.. code-block:: python

   from hazy import BloomFilter, HyperLogLog, CountMinSketch, TopK
   from hazy.viz import plot_bloom, plot_hll, plot_cms, plot_topk
   import matplotlib.pyplot as plt
   import random

   # Generate sample data
   bf = BloomFilter(expected_items=10000)
   hll = HyperLogLog(precision=12)
   cms = CountMinSketch(width=1000, depth=5)
   tk = TopK(k=20)

   pages = ["/home", "/about", "/products", "/blog", "/contact"]
   weights = [100, 20, 50, 30, 10]

   for i in range(50000):
       user = f"user_{i % 5000}"
       page = random.choices(pages, weights=weights)[0]

       bf.add(user)
       hll.add(user)
       cms.add(page)
       tk.add(page)

   # Create dashboard
   fig = plt.figure(figsize=(16, 12))
   fig.suptitle("Analytics Dashboard", fontsize=16, fontweight="bold")

   ax1 = fig.add_subplot(2, 2, 1)
   plt.sca(ax1)
   plot_bloom(bf, title=f"Visitors ({len(bf):,} users)")

   ax2 = fig.add_subplot(2, 2, 2)
   plt.sca(ax2)
   plot_hll(hll, title=f"Unique: {int(hll.cardinality()):,}")

   ax3 = fig.add_subplot(2, 2, 3)
   plt.sca(ax3)
   plot_cms(cms, title="Page Views", log_scale=True)

   ax4 = fig.add_subplot(2, 2, 4)
   plt.sca(ax4)
   plot_topk(tk, n=5, title="Top Pages")

   plt.tight_layout()
   plt.savefig("dashboard.png", dpi=150)
   plt.show()

Customization
-------------

Color Maps
~~~~~~~~~~

.. code-block:: python

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
   plt.show()

Saving Figures
~~~~~~~~~~~~~~

.. code-block:: python

   # PNG (raster, good for web)
   plt.savefig("figure.png", dpi=150, bbox_inches="tight")

   # PDF (vector, good for papers)
   plt.savefig("figure.pdf", bbox_inches="tight")

   # SVG (vector, good for web)
   plt.savefig("figure.svg", bbox_inches="tight")

Non-Interactive Mode
--------------------

For scripts without a display:

.. code-block:: python

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

See Also
--------

- :doc:`serialization` - Save and load structures
