Tutorial: Database Query Optimization
======================================

In this tutorial, we'll explore how databases use Bloom filters to avoid expensive disk lookups — and how you can apply the same technique in your applications.

The Problem
-----------

Database queries often involve checking whether a key exists:

1. **Key lookups** in key-value stores
2. **JOIN operations** checking if keys exist in another table
3. **Cache checks** before hitting the database
4. **Index lookups** in LSM-tree databases

The problem: disk I/O is slow. Reading from disk to check if a key exists (only to find it doesn't) wastes time.

How Databases Solve This
------------------------

Many databases use Bloom filters as a "negative cache":

- **Before** reading from disk, check the Bloom filter
- If the filter says "no" → the key definitely doesn't exist (skip disk read)
- If the filter says "yes" → the key might exist (do the disk read)

This works because:

- Bloom filters have **no false negatives** — if it says "no", it's definitely no
- False positives are rare and just mean occasional unnecessary disk reads

.. list-table:: Databases Using Bloom Filters
   :header-rows: 1
   :widths: 25 75

   * - Database
     - How It Uses Bloom Filters
   * - Apache Cassandra
     - One Bloom filter per SSTable to skip disk reads
   * - PostgreSQL
     - Bloom index type for multi-column filtering
   * - RocksDB/LevelDB
     - Filter blocks in each SSTable
   * - Apache HBase
     - Bloom filters for row and column lookups
   * - Redis
     - RedisBloom module for probabilistic queries

Implementation
--------------

Step 1: Simulating a Key-Value Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's build a simple key-value store with Bloom filter optimization:

.. code-block:: python

   from hazy import BloomFilter
   import time
   import random
   import string


   class SimulatedDisk:
       """Simulates slow disk storage."""

       def __init__(self, latency_ms: float = 1.0):
           self.data: dict[str, str] = {}
           self.latency_ms = latency_ms
           self.read_count = 0

       def write(self, key: str, value: str):
           self.data[key] = value

       def read(self, key: str) -> str | None:
           """Simulate a slow disk read."""
           self.read_count += 1
           time.sleep(self.latency_ms / 1000)  # Simulate latency
           return self.data.get(key)

       def exists(self, key: str) -> bool:
           """Check if key exists (still requires disk read)."""
           self.read_count += 1
           time.sleep(self.latency_ms / 1000)
           return key in self.data


   class KeyValueStore:
       """Key-value store WITHOUT Bloom filter optimization."""

       def __init__(self, disk_latency_ms: float = 1.0):
           self.disk = SimulatedDisk(latency_ms=disk_latency_ms)

       def put(self, key: str, value: str):
           self.disk.write(key, value)

       def get(self, key: str) -> str | None:
           return self.disk.read(key)

       def exists(self, key: str) -> bool:
           return self.disk.exists(key)

       @property
       def disk_reads(self) -> int:
           return self.disk.read_count


   class BloomOptimizedKVStore:
       """Key-value store WITH Bloom filter optimization."""

       def __init__(
           self,
           expected_keys: int = 100_000,
           false_positive_rate: float = 0.01,
           disk_latency_ms: float = 1.0
       ):
           self.disk = SimulatedDisk(latency_ms=disk_latency_ms)
           self.bloom = BloomFilter(
               expected_items=expected_keys,
               false_positive_rate=false_positive_rate
           )
           self.bloom_checks = 0
           self.bloom_hits = 0
           self.bloom_false_positives = 0

       def put(self, key: str, value: str):
           self.bloom.add(key)
           self.disk.write(key, value)

       def get(self, key: str) -> str | None:
           self.bloom_checks += 1

           # Check Bloom filter first
           if key not in self.bloom:
               # Definitely not in store — skip disk read!
               return None

           self.bloom_hits += 1
           result = self.disk.read(key)

           # Track false positives
           if result is None:
               self.bloom_false_positives += 1

           return result

       def exists(self, key: str) -> bool:
           self.bloom_checks += 1

           if key not in self.bloom:
               return False

           self.bloom_hits += 1
           result = self.disk.exists(key)

           if not result:
               self.bloom_false_positives += 1

           return result

       @property
       def disk_reads(self) -> int:
           return self.disk.read_count

       def stats(self) -> dict:
           return {
               "bloom_checks": self.bloom_checks,
               "bloom_hits": self.bloom_hits,
               "bloom_rejections": self.bloom_checks - self.bloom_hits,
               "bloom_false_positives": self.bloom_false_positives,
               "disk_reads": self.disk_reads,
               "bloom_memory_bytes": self.bloom.size_in_bytes,
           }

Step 2: Benchmark the Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def generate_keys(n: int) -> list[str]:
       """Generate random keys."""
       return [
           ''.join(random.choices(string.ascii_lowercase, k=16))
           for _ in range(n)
       ]


   def benchmark_stores(n_existing: int = 10_000, n_queries: int = 10_000):
       """Compare stores with and without Bloom filter."""

       print("=" * 60)
       print("DATABASE BLOOM FILTER OPTIMIZATION BENCHMARK")
       print("=" * 60)

       # Generate keys
       existing_keys = generate_keys(n_existing)
       nonexistent_keys = generate_keys(n_queries)

       # Mix of existing and non-existing queries (80% non-existing)
       # This is realistic — most cache checks are misses
       query_keys = (
           random.sample(existing_keys, n_queries // 5) +
           nonexistent_keys[:n_queries * 4 // 5]
       )
       random.shuffle(query_keys)

       # Test 1: Without Bloom filter
       print(f"\n1. WITHOUT Bloom Filter")
       print("-" * 40)

       store_basic = KeyValueStore(disk_latency_ms=0.1)

       # Populate store
       for key in existing_keys:
           store_basic.put(key, f"value_{key}")

       # Query
       start = time.time()
       for key in query_keys[:1000]:  # Limit for speed
           store_basic.exists(key)
       elapsed_basic = time.time() - start

       print(f"   Queries: 1,000")
       print(f"   Disk reads: {store_basic.disk_reads:,}")
       print(f"   Time: {elapsed_basic:.3f}s")

       # Test 2: With Bloom filter
       print(f"\n2. WITH Bloom Filter")
       print("-" * 40)

       store_bloom = BloomOptimizedKVStore(
           expected_keys=n_existing,
           false_positive_rate=0.01,
           disk_latency_ms=0.1
       )

       # Populate store
       for key in existing_keys:
           store_bloom.put(key, f"value_{key}")

       # Query
       start = time.time()
       for key in query_keys[:1000]:
           store_bloom.exists(key)
       elapsed_bloom = time.time() - start

       stats = store_bloom.stats()
       print(f"   Queries: 1,000")
       print(f"   Bloom rejections: {stats['bloom_rejections']:,} (avoided disk reads)")
       print(f"   Disk reads: {stats['disk_reads']:,}")
       print(f"   False positives: {stats['bloom_false_positives']}")
       print(f"   Time: {elapsed_bloom:.3f}s")
       print(f"   Bloom memory: {stats['bloom_memory_bytes']:,} bytes")

       # Summary
       print(f"\n{'=' * 60}")
       print("SUMMARY")
       print("=" * 60)
       speedup = elapsed_basic / elapsed_bloom if elapsed_bloom > 0 else float('inf')
       disk_reduction = (1 - stats['disk_reads'] / store_basic.disk_reads) * 100
       print(f"   Speedup: {speedup:.1f}x faster")
       print(f"   Disk reads reduced: {disk_reduction:.1f}%")
       print(f"   Memory cost: {stats['bloom_memory_bytes'] / 1024:.1f} KB")


   # Run benchmark
   benchmark_stores()

Step 3: Multi-Level Bloom Filters (LSM-Tree Style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real databases like RocksDB use multiple Bloom filters for different data levels:

.. code-block:: python

   class LSMStyleStore:
       """
       Simulates an LSM-tree style store with Bloom filters at each level.

       In an LSM-tree:
       - Level 0: Recent writes (in memory)
       - Level 1-N: Progressively older data on disk
       - Each level has its own Bloom filter
       """

       def __init__(self):
           # Simulating 3 levels
           self.levels = [
               {"bloom": BloomFilter(expected_items=1000, false_positive_rate=0.01),
                "data": {}},
               {"bloom": BloomFilter(expected_items=10000, false_positive_rate=0.01),
                "data": {}},
               {"bloom": BloomFilter(expected_items=100000, false_positive_rate=0.01),
                "data": {}},
           ]
           self.level_checks = [0, 0, 0]
           self.level_hits = [0, 0, 0]

       def put(self, key: str, value: str, level: int = 0):
           """Write to a specific level."""
           self.levels[level]["bloom"].add(key)
           self.levels[level]["data"][key] = value

       def get(self, key: str) -> str | None:
           """
           Search from newest to oldest level.
           Bloom filter at each level avoids unnecessary searches.
           """
           for i, level in enumerate(self.levels):
               self.level_checks[i] += 1

               # Check Bloom filter first
               if key not in level["bloom"]:
                   continue  # Definitely not here, check next level

               self.level_hits[i] += 1

               # Might be here, do the lookup
               if key in level["data"]:
                   return level["data"][key]

           return None

       def stats(self) -> dict:
           total_memory = sum(
               level["bloom"].size_in_bytes for level in self.levels
           )
           return {
               "level_checks": self.level_checks,
               "level_hits": self.level_hits,
               "total_bloom_memory": total_memory,
           }


   # Demonstrate multi-level lookup
   print("\n" + "=" * 60)
   print("LSM-TREE STYLE MULTI-LEVEL BLOOM FILTERS")
   print("=" * 60)

   store = LSMStyleStore()

   # Populate different levels (simulating compaction)
   for i in range(100):
       store.put(f"recent_{i}", f"value_{i}", level=0)

   for i in range(1000):
       store.put(f"medium_{i}", f"value_{i}", level=1)

   for i in range(10000):
       store.put(f"old_{i}", f"value_{i}", level=2)

   # Query mix
   queries = (
       [f"recent_{i}" for i in range(50)] +
       [f"medium_{i}" for i in range(50)] +
       [f"old_{i}" for i in range(50)] +
       [f"nonexistent_{i}" for i in range(50)]
   )
   random.shuffle(queries)

   for key in queries:
       store.get(key)

   stats = store.stats()
   print(f"\nLevel 0 (recent): {stats['level_checks'][0]} checks, {stats['level_hits'][0]} hits")
   print(f"Level 1 (medium): {stats['level_checks'][1]} checks, {stats['level_hits'][1]} hits")
   print(f"Level 2 (old):    {stats['level_checks'][2]} checks, {stats['level_hits'][2]} hits")
   print(f"\nTotal Bloom memory: {stats['total_bloom_memory']:,} bytes")

Step 4: Join Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

Bloom filters can also optimize JOIN operations:

.. code-block:: python

   class JoinOptimizer:
       """
       Use Bloom filters to optimize JOIN operations.

       Before: For each row in Table A, query Table B
       After: Build Bloom filter of Table B keys, skip rows that won't match
       """

       def __init__(self):
           pass

       def naive_join(
           self,
           table_a: list[dict],
           table_b: list[dict],
           key: str
       ) -> list[dict]:
           """Naive nested loop join."""
           results = []
           comparisons = 0

           b_index = {row[key]: row for row in table_b}

           for row_a in table_a:
               comparisons += 1
               if row_a[key] in b_index:
                   results.append({**row_a, **b_index[row_a[key]]})

           return results, comparisons

       def bloom_join(
           self,
           table_a: list[dict],
           table_b: list[dict],
           key: str
       ) -> list[dict]:
           """Bloom filter optimized join."""
           results = []
           bloom_checks = 0
           actual_lookups = 0

           # Build Bloom filter from smaller table
           bloom = BloomFilter(
               expected_items=len(table_b),
               false_positive_rate=0.01
           )
           for row in table_b:
               bloom.add(str(row[key]))

           # Build index for actual matches
           b_index = {row[key]: row for row in table_b}

           # Join with Bloom filter pre-check
           for row_a in table_a:
               bloom_checks += 1

               # Skip if definitely not in table B
               if str(row_a[key]) not in bloom:
                   continue

               actual_lookups += 1
               if row_a[key] in b_index:
                   results.append({**row_a, **b_index[row_a[key]]})

           return results, bloom_checks, actual_lookups


   # Demonstrate join optimization
   print("\n" + "=" * 60)
   print("JOIN OPTIMIZATION WITH BLOOM FILTERS")
   print("=" * 60)

   # Create tables with sparse overlap
   table_a = [{"id": i, "name": f"A_{i}"} for i in range(10000)]
   table_b = [{"id": i * 10, "value": f"B_{i}"} for i in range(1000)]  # Only 10% overlap

   optimizer = JoinOptimizer()

   # Naive join
   results_naive, comparisons = optimizer.naive_join(table_a, table_b, "id")
   print(f"\nNaive Join:")
   print(f"  Comparisons: {comparisons:,}")
   print(f"  Results: {len(results_naive)}")

   # Bloom optimized join
   results_bloom, checks, lookups = optimizer.bloom_join(table_a, table_b, "id")
   print(f"\nBloom Optimized Join:")
   print(f"  Bloom checks: {checks:,}")
   print(f"  Actual lookups: {lookups:,}")
   print(f"  Lookups avoided: {checks - lookups:,} ({(1 - lookups/checks)*100:.1f}%)")
   print(f"  Results: {len(results_bloom)}")

Visualizing Database Optimization
----------------------------------

Create a visualization comparing the performance with and without Bloom filter optimization:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Collect benchmark data (from the benchmark run)
   # Note: These values would come from actual benchmark results
   basic_disk_reads = 1000
   bloom_disk_reads = 215  # ~80% reduction with 0.01 FPR + 20% existing keys
   basic_time = 0.1  # seconds
   bloom_time = 0.025  # seconds
   bloom_memory = 120_000  # bytes

   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   fig.suptitle('Database Bloom Filter Optimization', fontsize=16, fontweight='bold')

   # 1. Disk Reads Comparison
   ax1 = axes[0, 0]
   methods = ['Without\nBloom Filter', 'With\nBloom Filter']
   disk_reads = [basic_disk_reads, bloom_disk_reads]
   colors = ['#e74c3c', '#2ecc71']
   bars = ax1.bar(methods, disk_reads, color=colors, width=0.6)
   ax1.set_ylabel('Disk Reads')
   ax1.set_title('Disk Read Comparison (1000 queries)')
   ax1.set_ylim(0, max(disk_reads) * 1.2)
   for bar, reads in zip(bars, disk_reads):
       ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{reads:,}', ha='center', fontsize=12, fontweight='bold')
   reduction = (1 - bloom_disk_reads / basic_disk_reads) * 100
   ax1.annotate(f'{reduction:.0f}% reduction',
                xy=(1, bloom_disk_reads), xytext=(1.3, basic_disk_reads/2),
                fontsize=11, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

   # 2. Query Time Comparison
   ax2 = axes[0, 1]
   times = [basic_time * 1000, bloom_time * 1000]  # Convert to ms
   bars = ax2.bar(methods, times, color=colors, width=0.6)
   ax2.set_ylabel('Time (ms)')
   ax2.set_title('Query Time Comparison')
   ax2.set_ylim(0, max(times) * 1.2)
   for bar, t in zip(bars, times):
       ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{t:.1f}ms', ha='center', fontsize=12, fontweight='bold')
   speedup = basic_time / bloom_time
   ax2.annotate(f'{speedup:.1f}x faster',
                xy=(1, bloom_time*1000), xytext=(1.3, basic_time*500),
                fontsize=11, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

   # 3. Bloom Filter Decision Flow (Sankey-like visualization)
   ax3 = axes[1, 0]
   ax3.set_xlim(0, 10)
   ax3.set_ylim(0, 10)

   # Draw flow diagram
   total = 1000
   bloom_no = 785  # Bloom says "definitely not here"
   bloom_maybe = 215  # Bloom says "might be here"
   actual_found = 200  # Actually found
   false_positives = 15  # Bloom false positives

   # Boxes
   ax3.add_patch(plt.Rectangle((0.5, 4), 2, 2, color='#3498db', alpha=0.8))
   ax3.text(1.5, 5, f'Queries\n{total}', ha='center', va='center', fontsize=10, color='white')

   ax3.add_patch(plt.Rectangle((4, 6.5), 2.5, 1.5, color='#2ecc71', alpha=0.8))
   ax3.text(5.25, 7.25, f'Bloom: No\n{bloom_no}', ha='center', va='center', fontsize=9, color='white')

   ax3.add_patch(plt.Rectangle((4, 2), 2.5, 1.5, color='#f39c12', alpha=0.8))
   ax3.text(5.25, 2.75, f'Bloom: Maybe\n{bloom_maybe}', ha='center', va='center', fontsize=9, color='white')

   ax3.add_patch(plt.Rectangle((7.5, 6.5), 2, 1.5, color='#27ae60', alpha=0.8))
   ax3.text(8.5, 7.25, f'Skip Disk\n{bloom_no}', ha='center', va='center', fontsize=9, color='white')

   ax3.add_patch(plt.Rectangle((7.5, 3), 2, 1.2, color='#2ecc71', alpha=0.8))
   ax3.text(8.5, 3.6, f'Found\n{actual_found}', ha='center', va='center', fontsize=9, color='white')

   ax3.add_patch(plt.Rectangle((7.5, 1.3), 2, 1.2, color='#e74c3c', alpha=0.8))
   ax3.text(8.5, 1.9, f'FP\n{false_positives}', ha='center', va='center', fontsize=9, color='white')

   # Arrows
   ax3.annotate('', xy=(4, 7.25), xytext=(2.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
   ax3.annotate('', xy=(4, 2.75), xytext=(2.5, 4.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
   ax3.annotate('', xy=(7.5, 7.25), xytext=(6.5, 7.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
   ax3.annotate('', xy=(7.5, 3.6), xytext=(6.5, 2.9),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
   ax3.annotate('', xy=(7.5, 1.9), xytext=(6.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

   ax3.set_title('Bloom Filter Query Flow')
   ax3.axis('off')

   # 4. Performance Summary
   ax4 = axes[1, 1]
   ax4.axis('off')
   summary_text = f"""
   BLOOM FILTER OPTIMIZATION RESULTS
   {'─' * 42}

   QUERY PERFORMANCE:
   Total Queries:              {total:>10,}
   Disk Reads (No Bloom):      {basic_disk_reads:>10,}
   Disk Reads (With Bloom):    {bloom_disk_reads:>10,}
   Disk Reads Avoided:         {basic_disk_reads - bloom_disk_reads:>10,}

   {'─' * 42}
   EFFICIENCY GAINS:
   Disk Read Reduction:        {reduction:>9.1f}%
   Speedup Factor:             {speedup:>9.1f}x
   False Positive Rate:              ~1.5%

   MEMORY COST:
   Bloom Filter Size:          {bloom_memory/1024:>9.1f} KB

   {'─' * 42}
   ROI: {bloom_memory/1024:.0f} KB memory saves {basic_disk_reads - bloom_disk_reads}
        disk reads per 1000 queries
   """
   ax4.text(0.05, 0.5, summary_text, fontsize=10, fontfamily='monospace',
            verticalalignment='center', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

   plt.tight_layout()
   plt.savefig('database_optimization.png', dpi=150, bbox_inches='tight')
   plt.show()

This visualization shows the disk read reduction, query speedup, the Bloom filter decision flow, and overall performance gains.

.. image:: /_static/images/database_optimization.png
   :alt: Database Bloom Filter Optimization
   :align: center
   :width: 100%

Key Takeaways
-------------

1. **Bloom filters eliminate unnecessary disk reads** — the key optimization
2. **No false negatives** means "definitely not here" is always correct
3. **Multi-level Bloom filters** work great for LSM-tree style storage
4. **JOIN optimization** reduces lookups when tables have sparse overlap
5. **Memory tradeoff** — a few MB of RAM saves many disk I/O operations

Real-World Applications
-----------------------

You can apply this pattern to:

- **API caching**: Check Bloom filter before hitting Redis/database
- **CDN routing**: Check if content exists at edge before origin fetch
- **Distributed systems**: Check local node before network request
- **Search indexing**: Skip documents that can't match query terms

Exercises
---------

1. **Counting filter**: Use CountingBloomFilter to support key deletion
2. **Time-based filters**: Rotate Bloom filters daily for temporal data
3. **Distributed filters**: Merge Bloom filters from multiple nodes
4. **Adaptive sizing**: Use ScalableBloomFilter when data size is unknown

Next Tutorial
-------------

Continue to :doc:`stream_processing` to learn how to process infinite event streams with bounded memory.
