HyperLogLog
===========

HyperLogLog is a probabilistic algorithm for estimating the cardinality (count of unique elements) of a set.

Overview
--------

- **Purpose**: Count unique items in a stream
- **Memory**: O(1) - typically 16 KB for billions of items
- **Trade-off**: Approximate count with ~1% error

When to Use
-----------

- Counting unique visitors/users
- Estimating distinct values in databases
- Network traffic analysis (unique IPs)
- Any cardinality estimation at scale

Basic Usage
-----------

.. code-block:: python

   from hazy import HyperLogLog

   # Create HLL with precision 14 (16KB, ~0.8% error)
   hll = HyperLogLog(precision=14)

   # Add items (duplicates automatically handled)
   for i in range(1_000_000):
       hll.add(f"user_{i}")

   # Get cardinality estimate
   print(f"Unique items: {hll.cardinality():,.0f}")

   # Memory usage
   print(f"Memory: {hll.size_in_bytes / 1024:.1f} KB")

Parameters
----------

``precision``
   Number of bits for register addressing (4-18). Higher = more accurate but more memory.

   - 10: 1 KB, ~3.3% error
   - 12: 4 KB, ~1.6% error
   - 14: 16 KB, ~0.8% error (recommended)
   - 16: 64 KB, ~0.4% error

Properties
----------

``cardinality()``
   Returns estimated count of unique items.

``size_in_bytes``
   Memory usage.

``precision``
   The precision parameter.

Precision vs Accuracy
---------------------

.. code-block:: python

   import random

   for precision in [10, 12, 14, 16]:
       hll = HyperLogLog(precision=precision)

       # Add 100k unique items
       for i in range(100_000):
           hll.add(f"item_{i}")

       estimate = hll.cardinality()
       error = abs(estimate - 100_000) / 100_000

       print(f"p={precision}: estimate={estimate:,.0f}, "
             f"error={error:.2%}, memory={hll.size_in_bytes/1024:.0f}KB")

Expected output:

.. code-block:: text

   p=10: estimate=102,145, error=2.15%, memory=1KB
   p=12: estimate=99,234, error=0.77%, memory=4KB
   p=14: estimate=100,089, error=0.09%, memory=16KB
   p=16: estimate=99,978, error=0.02%, memory=64KB

Merging HyperLogLogs
--------------------

HLLs can be merged for distributed counting:

.. code-block:: python

   # Create HLLs on different servers
   hll_server1 = HyperLogLog(precision=14)
   hll_server2 = HyperLogLog(precision=14)

   # Each server counts its users
   for i in range(50_000):
       hll_server1.add(f"user_{i}")

   for i in range(30_000, 80_000):  # Overlapping range
       hll_server2.add(f"user_{i}")

   # Merge to get total unique count
   combined = hll_server1.merge(hll_server2)
   # or: combined = hll_server1 | hll_server2

   print(f"Server 1: {hll_server1.cardinality():,.0f}")
   print(f"Server 2: {hll_server2.cardinality():,.0f}")
   print(f"Combined: {combined.cardinality():,.0f}")  # ~80,000

.. note::

   Merging is lossless - the merged HLL is as accurate as if all
   items were added to a single HLL.

Serialization
-------------

.. code-block:: python

   # Binary
   data = hll.to_bytes()
   hll2 = HyperLogLog.from_bytes(data)

   # File
   hll.save("hll.hazy")
   hll2 = HyperLogLog.load("hll.hazy")

Low Cardinality Bias
--------------------

HLL has higher relative error for small cardinalities:

.. code-block:: python

   for n in [100, 1000, 10_000, 100_000, 1_000_000]:
       hll = HyperLogLog(precision=14)
       for i in range(n):
           hll.add(f"item_{i}")

       estimate = hll.cardinality()
       error = abs(estimate - n) / n
       print(f"n={n:>10,}: estimate={estimate:>12,.0f}, error={error:.2%}")

For very small counts (< 1000), consider using exact counting.

Use Cases
---------

**Unique Visitors**

.. code-block:: python

   hll = HyperLogLog(precision=14)

   # Process log file
   for line in open("access.log"):
       ip = line.split()[0]
       hll.add(ip)

   print(f"Unique IPs: {hll.cardinality():,.0f}")

**Database Distinct Counts**

.. code-block:: python

   # Instead of SELECT COUNT(DISTINCT user_id) FROM events
   hll = HyperLogLog(precision=14)
   for row in cursor.execute("SELECT user_id FROM events"):
       hll.add(str(row[0]))

   print(f"Distinct users: {hll.cardinality():,.0f}")

Memory Comparison
-----------------

.. list-table::
   :header-rows: 1

   * - Approach
     - 1M unique items
   * - Python set
     - ~50 MB
   * - HyperLogLog (p=14)
     - 16 KB

HyperLogLog uses **3000x less memory** for the same task.

See Also
--------

- :doc:`count_min_sketch` - For frequency estimation
- :doc:`topk` - For finding most frequent items
