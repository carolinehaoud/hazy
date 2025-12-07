Scalable Bloom Filter
=====================

A Scalable Bloom filter automatically grows as items are added, maintaining a target false positive rate.

Overview
--------

- **Purpose**: Membership testing with unknown cardinality
- **Memory**: Grows automatically as needed
- **Trade-off**: Slightly higher memory per item than fixed-size filters

When to Use
-----------

- Streams with unknown size
- Long-running services where data accumulates
- When you can't predict capacity upfront

Basic Usage
-----------

.. code-block:: python

   from hazy import ScalableBloomFilter

   # Create with initial capacity and target FPR
   sbf = ScalableBloomFilter(
       initial_capacity=10000,
       false_positive_rate=0.01
   )

   # Add items (filter grows automatically)
   for i in range(100000):
       sbf.add(f"item_{i}")

   # Check membership
   print("item_50000" in sbf)  # True

   # Check statistics
   print(f"Items: {len(sbf):,}")
   print(f"Filters: {sbf.num_filters}")
   print(f"Memory: {sbf.size_in_bytes / 1024:.1f} KB")

Parameters
----------

``initial_capacity``
   Capacity of the first internal filter.

``false_positive_rate``
   Target overall false positive rate (default: 0.01).

``growth_factor``
   How much each new filter grows (default: 2).

``tightening_ratio``
   How much tighter each filter's FPR (default: 0.9).

How It Works
------------

The scalable filter maintains a series of standard Bloom filters:

1. Starts with one filter of ``initial_capacity``
2. When a filter fills up, creates a new larger one
3. Each new filter has tighter FPR to maintain overall target
4. Membership check queries all filters

.. code-block:: python

   sbf = ScalableBloomFilter(initial_capacity=1000, false_positive_rate=0.01)

   # Add items in batches, checking filter count
   for batch in range(5):
       for i in range(batch * 5000, (batch + 1) * 5000):
           sbf.add(f"item_{i}")
       print(f"Items: {len(sbf):>6}, Filters: {sbf.num_filters}, "
             f"Memory: {sbf.size_in_bytes/1024:.1f} KB")

Properties
----------

``len(sbf)``
   Total items added.

``num_filters``
   Number of internal Bloom filters.

``size_in_bytes``
   Total memory usage.

``fill_ratio``
   Fill ratio of the current (newest) filter.

Growth Patterns
---------------

With default settings (growth_factor=2, tightening_ratio=0.9):

.. list-table::
   :header-rows: 1

   * - Filter #
     - Capacity
     - Individual FPR
   * - 1
     - 1,000
     - 0.90%
   * - 2
     - 2,000
     - 0.81%
   * - 3
     - 4,000
     - 0.73%
   * - 4
     - 8,000
     - 0.66%

The combined FPR stays close to the target.

Serialization
-------------

.. code-block:: python

   # Binary
   data = sbf.to_bytes()
   sbf2 = ScalableBloomFilter.from_bytes(data)

   # File
   sbf.save("scalable_filter.hazy")
   sbf2 = ScalableBloomFilter.load("scalable_filter.hazy")

Performance Considerations
--------------------------

**Lookup time** increases linearly with number of filters:

.. code-block:: python

   import time

   sbf = ScalableBloomFilter(initial_capacity=100)

   # Add many items to create multiple filters
   for i in range(100000):
       sbf.add(f"item_{i}")

   # Lookup must check all filters
   start = time.time()
   for i in range(10000):
       _ = f"test_{i}" in sbf
   elapsed = time.time() - start

   print(f"Filters: {sbf.num_filters}")
   print(f"Lookups: {10000/elapsed:,.0f}/sec")

.. tip::

   Use a larger ``initial_capacity`` if you have a rough estimate
   to minimize the number of internal filters.

Memory vs Fixed Filter
----------------------

For unknown cardinality, scalable filters may use 20-40% more memory
than an optimally-sized fixed filter, but they never run out of capacity.

See Also
--------

- :doc:`bloom_filter` - Fixed-size Bloom filter
- :doc:`cuckoo_filter` - Alternative with deletion support
