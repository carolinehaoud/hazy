Count-Min Sketch
================

Count-Min Sketch is a probabilistic data structure for frequency estimation in data streams.

Overview
--------

- **Purpose**: Estimate how many times each item appears
- **Memory**: Fixed size regardless of unique items
- **Trade-off**: May overestimate counts, never underestimates

When to Use
-----------

- Tracking word frequencies in documents
- Network flow monitoring
- Click/view counting
- Heavy hitter detection
- Any frequency estimation at scale

Basic Usage
-----------

.. code-block:: python

   from hazy import CountMinSketch

   # Create sketch with width and depth
   cms = CountMinSketch(width=10000, depth=5)

   # Add items (optionally with count)
   cms.add("apple")
   cms.add("banana")
   cms.add("apple")
   cms.add("apple", count=5)  # Add 5 more apples

   # Query frequency
   print(f"apple: {cms['apple']}")    # 7
   print(f"banana: {cms['banana']}")  # 1
   print(f"cherry: {cms['cherry']}")  # 0

   # Memory usage
   print(f"Memory: {cms.size_in_bytes / 1024:.1f} KB")

Parameters
----------

``width``
   Number of counters per row. Larger = more accurate.

``depth``
   Number of hash functions/rows. More = lower error probability.

Typical configurations:

- Light use: width=1000, depth=5 (~20 KB)
- Medium: width=10000, depth=5 (~200 KB)
- Heavy: width=100000, depth=7 (~2.8 MB)

Properties
----------

``cms[item]`` or ``cms.get(item)``
   Returns estimated count (may be overestimate).

``size_in_bytes``
   Memory usage.

``width``
   Width parameter.

``depth``
   Depth parameter.

Error Bounds
------------

The error is bounded by:

- With probability 1 - δ, error ≤ ε * N
- where ε = e/width, δ = e^(-depth), N = total count

.. code-block:: python

   import math

   def cms_error_bound(width: int, depth: int, total_count: int) -> tuple:
       """Calculate error bound and confidence."""
       epsilon = math.e / width
       delta = math.e ** (-depth)
       max_error = epsilon * total_count
       confidence = 1 - delta
       return max_error, confidence

   # Example: width=10000, depth=5, 1M total items
   error, conf = cms_error_bound(10000, 5, 1_000_000)
   print(f"Max error: {error:.0f} with {conf:.4%} confidence")

Overestimation
--------------

Count-Min Sketch never underestimates, but may overestimate:

.. code-block:: python

   cms = CountMinSketch(width=1000, depth=5)  # Small for demo

   # Add many items
   for i in range(100_000):
       cms.add(f"item_{i}")

   # Query items that exist
   existing_errors = []
   for i in range(1000):
       count = cms[f"item_{i}"]
       existing_errors.append(count - 1)

   # Query items that don't exist
   nonexistent_counts = []
   for i in range(1000):
       count = cms[f"fake_{i}"]
       nonexistent_counts.append(count)

   print(f"Existing items: avg overestimate = {sum(existing_errors)/len(existing_errors):.1f}")
   print(f"Non-existent: avg count = {sum(nonexistent_counts)/len(nonexistent_counts):.1f}")

Merging Sketches
----------------

Sketches can be merged for distributed counting:

.. code-block:: python

   # Same dimensions required
   cms1 = CountMinSketch(width=10000, depth=5)
   cms2 = CountMinSketch(width=10000, depth=5)

   # Count on different machines
   for i in range(10000):
       cms1.add(f"item_{i}")

   for i in range(5000, 15000):
       cms2.add(f"item_{i}")

   # Merge
   combined = cms1.merge(cms2)

   print(f"item_7500: {combined['item_7500']}")  # Should be 2

Serialization
-------------

.. code-block:: python

   # Binary
   data = cms.to_bytes()
   cms2 = CountMinSketch.from_bytes(data)

   # File
   cms.save("sketch.hazy")
   cms2 = CountMinSketch.load("sketch.hazy")

Use Cases
---------

**Heavy Hitters Detection**

.. code-block:: python

   from hazy import CountMinSketch, TopK

   # Use CMS for counts, TopK for tracking top items
   cms = CountMinSketch(width=10000, depth=5)
   topk = TopK(k=100)

   for word in document.split():
       cms.add(word)
       topk.add(word)

   # Get heavy hitters with their counts
   for word, count in topk.top(10):
       accurate_count = cms[word]
       print(f"{word}: ~{accurate_count}")

**Rate Limiting**

.. code-block:: python

   from hazy import CountMinSketch
   from time import time

   class RateLimiter:
       def __init__(self, max_requests: int, window_seconds: int):
           self.cms = CountMinSketch(width=10000, depth=5)
           self.max_requests = max_requests
           self.window = window_seconds
           self.window_start = int(time()) // window_seconds

       def allow(self, user_id: str) -> bool:
           # Reset if window passed
           current_window = int(time()) // self.window
           if current_window != self.window_start:
               self.cms = CountMinSketch(width=10000, depth=5)
               self.window_start = current_window

           # Check and update
           key = f"{user_id}:{self.window_start}"
           if self.cms[key] >= self.max_requests:
               return False

           self.cms.add(key)
           return True

Memory Comparison
-----------------

.. list-table::
   :header-rows: 1

   * - Approach
     - 1M unique items
   * - Python Counter
     - ~50 MB
   * - CMS (10K × 5)
     - 200 KB

See Also
--------

- :doc:`hyperloglog` - For cardinality (unique count)
- :doc:`topk` - For finding most frequent items
