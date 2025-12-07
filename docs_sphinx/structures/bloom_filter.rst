Bloom Filter
============

A Bloom filter is a space-efficient probabilistic data structure for testing set membership.

Overview
--------

- **Purpose**: Test if an item is in a set
- **Memory**: Much smaller than storing items directly
- **Trade-off**: May have false positives, but never false negatives

When to Use
-----------

- Checking if a URL has been crawled
- Detecting if a username is taken
- Filtering database queries (avoid expensive lookups)
- Spell checking (is this word in the dictionary?)

Basic Usage
-----------

.. code-block:: python

   from hazy import BloomFilter

   # Create filter expecting 1 million items with 1% false positive rate
   bf = BloomFilter(expected_items=1_000_000, false_positive_rate=0.01)

   # Add items
   bf.add("hello")
   bf.add("world")

   # Check membership
   print("hello" in bf)    # True
   print("goodbye" in bf)  # False (probably)

   # Bulk add
   bf.update(["apple", "banana", "cherry"])

   # Check statistics
   print(f"Items: {len(bf)}")
   print(f"Fill ratio: {bf.fill_ratio:.1%}")
   print(f"Memory: {bf.size_in_bytes:,} bytes")

Parameters
----------

``expected_items``
   Number of items you expect to add. Oversizing is better than undersizing.

``false_positive_rate``
   Target probability of false positives (default: 0.01 = 1%).
   Lower rates require more memory.

Properties
----------

``len(bf)``
   Number of items added (approximate after many additions).

``fill_ratio``
   Fraction of bits set (0.0 to 1.0). Above 0.5 means degraded performance.

``size_in_bytes``
   Memory usage in bytes.

``num_bits``
   Total number of bits in the filter.

``num_hashes``
   Number of hash functions used.

False Positive Rate
-------------------

The actual false positive rate increases as you add items:

.. code-block:: python

   bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)

   # Add items and check FPR
   for i in range(0, 15001, 5000):
       if i > 0:
           bf.update([f"item_{j}" for j in range(i - 5000, i)])

       # Test against items not in filter
       false_positives = sum(
           1 for j in range(100000, 101000)
           if f"test_{j}" in bf
       )
       print(f"Items: {i:>5}, FPR: {false_positives/1000:.1%}")

.. warning::

   Adding more items than ``expected_items`` significantly increases false positives.

Serialization
-------------

.. code-block:: python

   # Binary (compact)
   data = bf.to_bytes()
   bf2 = BloomFilter.from_bytes(data)

   # JSON (human-readable)
   json_str = bf.to_json()
   bf2 = BloomFilter.from_json(json_str)

   # File
   bf.save("filter.hazy")
   bf2 = BloomFilter.load("filter.hazy")

Set Operations
--------------

.. code-block:: python

   bf1 = BloomFilter(expected_items=1000)
   bf2 = BloomFilter(expected_items=1000)

   bf1.update(["a", "b", "c"])
   bf2.update(["c", "d", "e"])

   # Union (items in either filter)
   combined = bf1 | bf2
   # or: combined = bf1.union(bf2)

Memory vs Accuracy Trade-off
-----------------------------

.. list-table::
   :header-rows: 1

   * - FPR
     - Bits per item
     - 1M items
   * - 10%
     - 4.8
     - 600 KB
   * - 1%
     - 9.6
     - 1.2 MB
   * - 0.1%
     - 14.4
     - 1.8 MB
   * - 0.01%
     - 19.2
     - 2.4 MB

See Also
--------

- :doc:`counting_bloom_filter` - Supports deletion
- :doc:`scalable_bloom_filter` - Grows automatically
- :doc:`cuckoo_filter` - Alternative with deletion support
