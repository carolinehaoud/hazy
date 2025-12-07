TopK
====

TopK (also known as Space-Saving or Heavy Hitters) tracks the most frequent items in a stream.

Overview
--------

- **Purpose**: Find the k most frequent items
- **Memory**: O(k) - proportional to number of items tracked
- **Trade-off**: Approximate counts, may miss items near threshold

When to Use
-----------

- Finding trending topics
- Popular pages/products
- Top search queries
- Frequent error types
- Any "top N" ranking

Basic Usage
-----------

.. code-block:: python

   from hazy import TopK

   # Track top 10 items
   tk = TopK(k=10)

   # Add items
   for word in ["apple", "banana", "apple", "cherry", "apple", "banana"]:
       tk.add(word)

   # Get top items
   for item, count in tk.top(5):
       print(f"{item}: {count}")

   # Output:
   # apple: 3
   # banana: 2
   # cherry: 1

Parameters
----------

``k``
   Number of top items to track. Larger k = more memory but tracks more items.

Methods
-------

``add(item, count=1)``
   Add an item with optional count.

``top(n=None)``
   Get top n items (default: all k). Returns list of (item, count) tuples.

``top_with_error(n=None)``
   Get top n with error bounds. Returns list of (item, count, error) tuples.

Properties
----------

``len(tk)``
   Number of items currently tracked.

``size_in_bytes``
   Memory usage.

How It Works
------------

TopK uses the Space-Saving algorithm:

1. Maintains k counters for items
2. When a new item arrives:

   - If item is tracked, increment its counter
   - If there's space, add it with count 1
   - Otherwise, replace the minimum-count item

.. code-block:: python

   tk = TopK(k=3)  # Only track 3 items

   # Add items
   tk.add("a")  # a:1
   tk.add("b")  # a:1, b:1
   tk.add("c")  # a:1, b:1, c:1
   tk.add("d")  # d replaces one of the items at count 1
   tk.add("a")  # a:2

   print(tk.top())

Error Bounds
------------

The algorithm provides guaranteed error bounds:

.. code-block:: python

   tk = TopK(k=100)

   # Add items with varying frequencies
   for i in range(10):
       for j in range(1000 - i * 100):
           tk.add(f"item_{i}")

   # Check with error bounds
   print("Item        | Count  | Error")
   print("-" * 35)
   for item, count, error in tk.top_with_error(5):
       print(f"{item:<11} | {count:>6} | ±{error}")

The error bound is at most N/k where N is total count of all items.

Streaming Example
-----------------

.. code-block:: python

   from hazy import TopK
   import random

   # Simulate word stream with Zipf distribution
   tk = TopK(k=20)

   words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"]

   # Zipf weights (most common words appear much more often)
   weights = [1.0 / (i + 1) for i in range(len(words))]

   # Process stream
   for _ in range(1_000_000):
       word = random.choices(words, weights=weights)[0]
       tk.add(word)

   # Results
   print("Top 10 words:")
   for word, count in tk.top(10):
       print(f"  {word:<10} {count:>10,}")

Adding with Counts
------------------

For pre-aggregated data:

.. code-block:: python

   tk = TopK(k=100)

   # Add with counts
   tk.add("product_a", count=500)
   tk.add("product_b", count=300)
   tk.add("product_a", count=200)  # Now 700 total

   print(tk.top(2))  # [('product_a', 700), ('product_b', 300)]

Merging TopK
------------

TopK structures can be merged:

.. code-block:: python

   tk1 = TopK(k=10)
   tk2 = TopK(k=10)

   # Different servers track different data
   for i in range(1000):
       tk1.add(f"item_{i % 100}")

   for i in range(500, 1500):
       tk2.add(f"item_{i % 100}")

   # Merge
   combined = tk1.merge(tk2)
   print(combined.top(5))

Serialization
-------------

.. code-block:: python

   # Binary
   data = tk.to_bytes()
   tk2 = TopK.from_bytes(data)

   # File
   tk.save("topk.hazy")
   tk2 = TopK.load("topk.hazy")

Use Cases
---------

**Trending Hashtags**

.. code-block:: python

   from hazy import TopK

   tk = TopK(k=50)

   for tweet in tweet_stream:
       for hashtag in extract_hashtags(tweet):
           tk.add(hashtag)

   # Display trending
   for tag, count in tk.top(10):
       print(f"#{tag}: {count:,} mentions")

**Popular Products**

.. code-block:: python

   from hazy import TopK

   tk = TopK(k=100)

   for event in purchase_stream:
       tk.add(event.product_id, count=event.quantity)

   # Best sellers
   for product_id, sales in tk.top(10):
       print(f"{product_id}: {sales:,} units")

Choosing k
----------

.. list-table::
   :header-rows: 1

   * - Use Case
     - Suggested k
   * - Top 10 with confidence
     - 50-100
   * - Top 100
     - 500-1000
   * - Heavy hitter detection
     - 100-1000

Rule of thumb: k should be 5-10x larger than the number of items you want to report.

See Also
--------

- :doc:`count_min_sketch` - For frequency estimation of all items
- :doc:`hyperloglog` - For counting unique items
