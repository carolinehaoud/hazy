Serialization
=============

All Hazy data structures support serialization for persistence and transfer.

Serialization Formats
---------------------

Hazy supports three serialization formats:

.. list-table::
   :header-rows: 1

   * - Format
     - Use Case
     - Size
   * - Binary
     - Compact storage, network transfer
     - Smallest
   * - JSON
     - Human-readable, debugging, APIs
     - Larger
   * - File
     - Convenient save/load
     - Same as binary

Binary Serialization
--------------------

Most compact format:

.. code-block:: python

   from hazy import BloomFilter

   bf = BloomFilter(expected_items=10000)
   bf.update(["item1", "item2", "item3"])

   # Serialize to bytes
   data = bf.to_bytes()
   print(f"Size: {len(data):,} bytes")

   # Deserialize
   bf2 = BloomFilter.from_bytes(data)
   print("item1" in bf2)  # True

JSON Serialization
------------------

Human-readable format:

.. code-block:: python

   from hazy import BloomFilter

   bf = BloomFilter(expected_items=1000)
   bf.add("hello")

   # Serialize to JSON
   json_str = bf.to_json()
   print(json_str[:100] + "...")

   # Deserialize
   bf2 = BloomFilter.from_json(json_str)
   print("hello" in bf2)  # True

File I/O
--------

Save and load directly to files:

.. code-block:: python

   from hazy import BloomFilter, HyperLogLog

   # Save
   bf = BloomFilter(expected_items=10000)
   bf.update([f"item_{i}" for i in range(5000)])
   bf.save("filter.hazy")

   hll = HyperLogLog(precision=14)
   hll.update([f"user_{i}" for i in range(100000)])
   hll.save("hll.hazy")

   # Load
   bf2 = BloomFilter.load("filter.hazy")
   hll2 = HyperLogLog.load("hll.hazy")

   print(f"BloomFilter items: {len(bf2)}")
   print(f"HyperLogLog cardinality: {hll2.cardinality():,.0f}")

All Data Structures
-------------------

Every Hazy structure supports the same interface:

.. code-block:: python

   from hazy import (
       BloomFilter, CountingBloomFilter, ScalableBloomFilter,
       CuckooFilter, HyperLogLog, CountMinSketch, MinHash, TopK
   )

   structures = [
       BloomFilter(expected_items=1000),
       CountingBloomFilter(expected_items=1000),
       ScalableBloomFilter(initial_capacity=1000),
       CuckooFilter(capacity=1000),
       HyperLogLog(precision=14),
       CountMinSketch(width=1000, depth=5),
       MinHash(num_hashes=128),
       TopK(k=100),
   ]

   for struct in structures:
       # All support these methods:
       data = struct.to_bytes()
       json_str = struct.to_json()
       struct.save(f"{type(struct).__name__.lower()}.hazy")

Network Transfer
----------------

Send structures over the network:

.. code-block:: python

   import socket
   from hazy import HyperLogLog

   # Server side
   def receive_hll(sock):
       # Receive size first
       size = int.from_bytes(sock.recv(4), "big")
       data = sock.recv(size)
       return HyperLogLog.from_bytes(data)

   # Client side
   def send_hll(sock, hll):
       data = hll.to_bytes()
       sock.send(len(data).to_bytes(4, "big"))
       sock.send(data)

   # Example usage
   hll = HyperLogLog(precision=14)
   hll.update([f"user_{i}" for i in range(10000)])

   data = hll.to_bytes()
   print(f"Transfer size: {len(data):,} bytes")

Database Storage
----------------

Store in databases as binary:

.. code-block:: python

   import sqlite3
   from hazy import BloomFilter

   # Create table
   conn = sqlite3.connect("cache.db")
   conn.execute("""
       CREATE TABLE IF NOT EXISTS filters (
           id TEXT PRIMARY KEY,
           data BLOB
       )
   """)

   # Save filter
   bf = BloomFilter(expected_items=10000)
   bf.update(["a", "b", "c"])

   conn.execute(
       "INSERT OR REPLACE INTO filters VALUES (?, ?)",
       ("my_filter", bf.to_bytes())
   )
   conn.commit()

   # Load filter
   row = conn.execute(
       "SELECT data FROM filters WHERE id = ?",
       ("my_filter",)
   ).fetchone()

   bf2 = BloomFilter.from_bytes(row[0])
   print("a" in bf2)  # True

Redis Storage
-------------

Store in Redis:

.. code-block:: python

   import redis
   from hazy import HyperLogLog

   r = redis.Redis()

   # Save
   hll = HyperLogLog(precision=14)
   hll.update([f"user_{i}" for i in range(10000)])
   r.set("unique_users", hll.to_bytes())

   # Load
   data = r.get("unique_users")
   hll2 = HyperLogLog.from_bytes(data)
   print(f"Users: {hll2.cardinality():,.0f}")

Versioning Considerations
-------------------------

When upgrading Hazy, serialized data may need migration:

.. code-block:: python

   import json
   from hazy import BloomFilter

   # Check version in JSON
   bf = BloomFilter(expected_items=1000)
   bf.add("test")
   data = json.loads(bf.to_json())
   print(f"Format version: {data.get('version', 'unknown')}")

   # Binary format includes version header
   binary = bf.to_bytes()
   print(f"Binary header: {binary[:8]}")

Compression
-----------

For additional space savings:

.. code-block:: python

   import zlib
   from hazy import BloomFilter

   bf = BloomFilter(expected_items=100000)
   bf.update([f"item_{i}" for i in range(50000)])

   # Uncompressed
   data = bf.to_bytes()
   print(f"Uncompressed: {len(data):,} bytes")

   # Compressed
   compressed = zlib.compress(data)
   print(f"Compressed: {len(compressed):,} bytes")
   print(f"Ratio: {len(compressed)/len(data):.1%}")

   # Decompress and restore
   decompressed = zlib.decompress(compressed)
   bf2 = BloomFilter.from_bytes(decompressed)

Best Practices
--------------

1. **Use binary for storage/transfer** - smallest size
2. **Use JSON for debugging** - human readable
3. **Include metadata** - store creation time, parameters
4. **Version your formats** - for future compatibility
5. **Consider compression** - for large structures

See Also
--------

- :doc:`visualization` - Plot and debug structures
