API Reference
=============

This section provides detailed API documentation for all Hazy classes.

Data Structures
---------------

Membership Testing
~~~~~~~~~~~~~~~~~~

- :doc:`../structures/bloom_filter` - Basic probabilistic set membership
- :doc:`../structures/counting_bloom_filter` - Bloom filter with deletion support
- :doc:`../structures/scalable_bloom_filter` - Auto-growing Bloom filter
- :doc:`../structures/cuckoo_filter` - Alternative with deletion and better memory

Cardinality Estimation
~~~~~~~~~~~~~~~~~~~~~~

- :doc:`../structures/hyperloglog` - Count unique items with minimal memory

Frequency Estimation
~~~~~~~~~~~~~~~~~~~~

- :doc:`../structures/count_min_sketch` - Estimate item frequencies
- :doc:`../structures/topk` - Track most frequent items

Similarity
~~~~~~~~~~

- :doc:`../structures/minhash` - Estimate Jaccard similarity between sets

Common Interface
----------------

All structures share a common interface for serialization and basic operations.

Serialization
~~~~~~~~~~~~~

All structures support three serialization methods:

**Binary (most compact)**

.. code-block:: python

   # Serialize to bytes
   data = structure.to_bytes()

   # Deserialize from bytes
   structure = StructureClass.from_bytes(data)

**JSON (human-readable)**

.. code-block:: python

   # Serialize to JSON string
   json_str = structure.to_json()

   # Deserialize from JSON
   structure = StructureClass.from_json(json_str)

**File I/O**

.. code-block:: python

   # Save to file
   structure.save("filename.hazy")

   # Load from file
   structure = StructureClass.load("filename.hazy")

Common Properties
~~~~~~~~~~~~~~~~~

All structures have these properties:

``size_in_bytes``
   Memory usage of the structure in bytes.

``len(structure)``
   Number of items (where applicable). For Bloom filters, this is the count
   of items added. For TopK, it's the number of tracked items.

Set Operations
~~~~~~~~~~~~~~

Some structures support set operations:

**BloomFilter, HyperLogLog**

.. code-block:: python

   # Union (items in either)
   combined = structure1 | structure2
   # or
   combined = structure1.merge(structure2)

**CountMinSketch**

.. code-block:: python

   # Merge counts
   combined = cms1.merge(cms2)

**MinHash**

.. code-block:: python

   # Merge signatures (union of sets)
   combined = mh1 | mh2

   # Compare signatures
   similarity = mh1.jaccard(mh2)

Type Hints
~~~~~~~~~~

All public methods have type hints for better IDE support:

.. code-block:: python

   from hazy import BloomFilter

   bf: BloomFilter = BloomFilter(expected_items=1000)
   bf.add("item")  # str
   result: bool = "item" in bf
