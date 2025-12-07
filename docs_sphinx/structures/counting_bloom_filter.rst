Counting Bloom Filter
=====================

A Counting Bloom filter extends the standard Bloom filter to support deletion by using counters instead of single bits.

Overview
--------

- **Purpose**: Membership testing with deletion support
- **Memory**: 4x more than standard Bloom filter (4-bit counters)
- **Trade-off**: Counters can overflow, deletions can cause false negatives

When to Use
-----------

- Session tracking with logout
- Temporary bans/blocks
- Caching where items expire
- Any scenario requiring removal

Basic Usage
-----------

.. code-block:: python

   from hazy import CountingBloomFilter

   # Create filter
   cbf = CountingBloomFilter(expected_items=10000, false_positive_rate=0.01)

   # Add items
   cbf.add("user_123")
   cbf.add("user_456")

   # Check membership
   print("user_123" in cbf)  # True

   # Remove items
   cbf.remove("user_123")
   print("user_123" in cbf)  # False

   # Add same item multiple times
   cbf.add("repeated")
   cbf.add("repeated")
   cbf.add("repeated")

   # Need to remove same number of times
   cbf.remove("repeated")
   print("repeated" in cbf)  # True (counter is 2)
   cbf.remove("repeated")
   cbf.remove("repeated")
   print("repeated" in cbf)  # False (counter is 0)

Parameters
----------

``expected_items``
   Expected number of unique items.

``false_positive_rate``
   Target false positive rate (default: 0.01).

``counter_bits``
   Bits per counter (default: 4, max count: 15).

Properties
----------

``len(cbf)``
   Approximate number of items.

``size_in_bytes``
   Memory usage.

``fill_ratio``
   Average counter value relative to max.

Counter Overflow
----------------

With 4-bit counters, the maximum count is 15:

.. code-block:: python

   cbf = CountingBloomFilter(expected_items=100)

   # Adding same item many times
   for i in range(20):
       cbf.add("overflow_test")

   # Counter saturates at 15, won't overflow
   # But removing more than 15 times won't work correctly

.. warning::

   If an item is added more than 15 times (with 4-bit counters),
   the counter saturates. Removing it may leave residual counts.

False Negatives
---------------

Unlike standard Bloom filters, counting filters can have false negatives
if you remove an item that was never added:

.. code-block:: python

   cbf = CountingBloomFilter(expected_items=1000)

   cbf.add("real_item")

   # This decrements counters that "real_item" also uses!
   cbf.remove("fake_item")  # Never added, but shares hash positions

   # May now incorrectly say "real_item" is not present
   print("real_item" in cbf)  # Could be False!

.. danger::

   Only remove items you're certain were added. Removing non-existent
   items can corrupt the filter.

Serialization
-------------

.. code-block:: python

   # Binary
   data = cbf.to_bytes()
   cbf2 = CountingBloomFilter.from_bytes(data)

   # JSON
   json_str = cbf.to_json()
   cbf2 = CountingBloomFilter.from_json(json_str)

   # File
   cbf.save("counting_filter.hazy")
   cbf2 = CountingBloomFilter.load("counting_filter.hazy")

Memory Comparison
-----------------

.. list-table::
   :header-rows: 1

   * - Filter Type
     - 1M items, 1% FPR
   * - BloomFilter
     - 1.2 MB
   * - CountingBloomFilter (4-bit)
     - 4.8 MB
   * - CountingBloomFilter (8-bit)
     - 9.6 MB

See Also
--------

- :doc:`bloom_filter` - Standard Bloom filter (no deletion)
- :doc:`cuckoo_filter` - Alternative with deletion and better memory
