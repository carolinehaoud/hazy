Cuckoo Filter
=============

A Cuckoo filter is a space-efficient probabilistic data structure for set membership that supports deletion.

Overview
--------

- **Purpose**: Membership testing with deletion support
- **Memory**: Often better than Counting Bloom filters
- **Trade-off**: Fixed capacity, insertion can fail when full

When to Use
-----------

- Need to remove items from a membership filter
- Better memory efficiency than Counting Bloom filter
- Situations where insertion failures are acceptable

Basic Usage
-----------

.. code-block:: python

   from hazy import CuckooFilter

   # Create filter with capacity
   cf = CuckooFilter(capacity=100000)

   # Add items
   cf.add("user_123")
   cf.add("user_456")

   # Check membership
   print("user_123" in cf)  # True

   # Remove items
   cf.remove("user_123")
   print("user_123" in cf)  # False

   # Check statistics
   print(f"Items: {len(cf)}")
   print(f"Load factor: {cf.load_factor:.1%}")
   print(f"Memory: {cf.size_in_bytes:,} bytes")

Parameters
----------

``capacity``
   Maximum number of items the filter can hold.

``fingerprint_size``
   Bits per fingerprint (default: 8). More bits = lower FPR.

``bucket_size``
   Items per bucket (default: 4). Affects load factor.

Properties
----------

``len(cf)``
   Number of items currently stored.

``load_factor``
   Fraction of capacity used (0.0 to ~0.95).

``size_in_bytes``
   Memory usage.

``capacity``
   Maximum items the filter can hold.

Insertion Failures
------------------

Unlike Bloom filters, Cuckoo filters can fail to insert:

.. code-block:: python

   cf = CuckooFilter(capacity=1000)

   # Fill the filter
   successes = 0
   for i in range(1500):
       if cf.add(f"item_{i}"):
           successes += 1

   print(f"Inserted: {successes}")  # Around 950-1000
   print(f"Load factor: {cf.load_factor:.1%}")

.. warning::

   Plan for ~95% capacity utilization. Size your filter accordingly.

How It Works
------------

Cuckoo filters store fingerprints in a cuckoo hash table:

1. Each item hashes to two possible bucket locations
2. Insertion tries both locations
3. If both full, existing items are "kicked" to alternate locations
4. After max kicks, insertion fails

.. code-block:: python

   cf = CuckooFilter(capacity=10000)

   # Items that hash to same buckets may cause relocations
   for i in range(8000):
       cf.add(f"item_{i}")

   print(f"Load: {cf.load_factor:.1%}")  # ~80%

Deletion Safety
---------------

Unlike Counting Bloom filters, Cuckoo filter deletions are safe:

.. code-block:: python

   cf = CuckooFilter(capacity=1000)

   cf.add("item_a")
   cf.add("item_b")

   # Deleting non-existent item is safe (returns False)
   deleted = cf.remove("item_c")  # False, nothing happened
   print(f"Deleted: {deleted}")

   # Original items still present
   print("item_a" in cf)  # True
   print("item_b" in cf)  # True

.. note::

   Deleting a non-existent item returns False and doesn't corrupt the filter.
   However, false positives can still cause issues if you delete based on
   membership checks.

Serialization
-------------

.. code-block:: python

   # Binary
   data = cf.to_bytes()
   cf2 = CuckooFilter.from_bytes(data)

   # File
   cf.save("cuckoo_filter.hazy")
   cf2 = CuckooFilter.load("cuckoo_filter.hazy")

Memory Comparison
-----------------

For 1% false positive rate:

.. list-table::
   :header-rows: 1

   * - Filter Type
     - Bits per item
     - 1M items
   * - BloomFilter
     - 9.6
     - 1.2 MB
   * - CountingBloomFilter
     - 38.4
     - 4.8 MB
   * - CuckooFilter
     - ~12
     - 1.5 MB

Cuckoo filters are more memory-efficient than Counting Bloom filters
while supporting deletion.

Fingerprint Size Trade-offs
---------------------------

.. list-table::
   :header-rows: 1

   * - Fingerprint bits
     - False positive rate
     - Memory
   * - 8
     - ~3%
     - 1 byte/item
   * - 12
     - ~0.2%
     - 1.5 bytes/item
   * - 16
     - ~0.01%
     - 2 bytes/item

See Also
--------

- :doc:`bloom_filter` - Standard Bloom filter (no deletion)
- :doc:`counting_bloom_filter` - Alternative with counters
