Quickstart Guide
================

This guide will get you up and running with Hazy in just a few minutes.

Bloom Filters
-------------

Bloom filters answer the question: **"Have I seen this item before?"**

.. code-block:: python

   from hazy import BloomFilter

   # Create a filter expecting 1 million items with 1% false positive rate
   bf = BloomFilter(expected_items=1_000_000, false_positive_rate=0.01)

   # Add items
   bf.add("user_123")
   bf.add("user_456")

   # Check membership
   print("user_123" in bf)  # True
   print("user_789" in bf)  # False (probably)

   # Check stats
   print(f"Items added: {len(bf)}")
   print(f"Memory: {bf.size_in_bytes / 1024:.1f} KB")

.. note::

   False positives are possible (saying "yes" when the answer is "no"),
   but false negatives never occur (if it says "no", it's definitely "no").

HyperLogLog
-----------

HyperLogLog estimates **how many unique items** you've seen:

.. code-block:: python

   from hazy import HyperLogLog

   # Create with precision 14 (good balance of accuracy and memory)
   hll = HyperLogLog(precision=14)

   # Add items (duplicates are handled automatically)
   for i in range(1_000_000):
       hll.add(f"user_{i % 100_000}")  # Only 100k unique

   # Estimate cardinality
   print(f"Unique items: {hll.cardinality():,.0f}")  # ~100,000

   # Memory usage
   print(f"Memory: {hll.size_in_bytes / 1024:.1f} KB")  # ~16 KB

Count-Min Sketch
----------------

Count-Min Sketch tracks **item frequencies**:

.. code-block:: python

   from hazy import CountMinSketch

   # Create sketch
   cms = CountMinSketch(width=10000, depth=5)

   # Count occurrences
   words = ["the", "quick", "brown", "fox", "the", "the", "quick"]
   for word in words:
       cms.add(word)

   # Query frequencies
   print(f"'the' appears: {cms['the']} times")    # 3
   print(f"'fox' appears: {cms['fox']} times")    # 1
   print(f"'dog' appears: {cms['dog']} times")    # 0

MinHash
-------

MinHash estimates **similarity between sets**:

.. code-block:: python

   from hazy import MinHash

   # Create MinHash signatures
   mh1 = MinHash(num_hashes=128)
   mh2 = MinHash(num_hashes=128)

   # Add items to each set
   doc1_words = {"machine", "learning", "is", "fascinating"}
   doc2_words = {"deep", "learning", "is", "amazing"}

   mh1.update(doc1_words)
   mh2.update(doc2_words)

   # Estimate Jaccard similarity
   similarity = mh1.jaccard(mh2)
   print(f"Similarity: {similarity:.2%}")

TopK
----

TopK tracks the **most frequent items**:

.. code-block:: python

   from hazy import TopK

   # Track top 10 items
   tk = TopK(k=10)

   # Add items (with optional counts)
   events = ["click", "view", "click", "purchase", "click", "view"]
   for event in events:
       tk.add(event)

   # Get top items
   for item, count in tk.top(5):
       print(f"{item}: {count}")

Combining Structures
--------------------

These structures work well together:

.. code-block:: python

   from hazy import BloomFilter, HyperLogLog, CountMinSketch, TopK

   class Analytics:
       def __init__(self):
           self.seen_users = BloomFilter(expected_items=1_000_000)
           self.unique_count = HyperLogLog(precision=14)
           self.page_views = CountMinSketch(width=10000, depth=5)
           self.trending = TopK(k=100)

       def record(self, user_id: str, page: str):
           # Track unique users
           is_new = user_id not in self.seen_users
           self.seen_users.add(user_id)
           self.unique_count.add(user_id)

           # Track page popularity
           self.page_views.add(page)
           self.trending.add(page)

           return is_new

   # Use it
   analytics = Analytics()
   for i in range(100_000):
       analytics.record(f"user_{i % 10_000}", f"/page_{i % 50}")

   print(f"Unique visitors: {analytics.unique_count.cardinality():,.0f}")
   print(f"Top pages: {analytics.trending.top(5)}")

Next Steps
----------

- Explore the :doc:`../tutorials/web_analytics` tutorial for a complete example
- Learn about :doc:`../guides/serialization` to save and load structures
- Check out :doc:`../guides/visualization` for plotting and debugging
