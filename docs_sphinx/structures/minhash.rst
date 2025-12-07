MinHash
=======

MinHash is a technique for quickly estimating the Jaccard similarity between sets.

Overview
--------

- **Purpose**: Estimate similarity between sets
- **Memory**: Fixed-size signature per set
- **Trade-off**: Approximate similarity with controllable error

When to Use
-----------

- Near-duplicate detection
- Document similarity
- Clustering similar items
- Recommendation systems
- Plagiarism detection

Basic Usage
-----------

.. code-block:: python

   from hazy import MinHash

   # Create MinHash signatures
   mh1 = MinHash(num_hashes=128)
   mh2 = MinHash(num_hashes=128)

   # Add set elements
   set1 = {"apple", "banana", "cherry", "date"}
   set2 = {"banana", "cherry", "elderberry", "fig"}

   mh1.update(set1)
   mh2.update(set2)

   # Estimate Jaccard similarity
   similarity = mh1.jaccard(mh2)

   # Compare to actual
   actual = len(set1 & set2) / len(set1 | set2)

   print(f"Estimated: {similarity:.2%}")
   print(f"Actual:    {actual:.2%}")

Parameters
----------

``num_hashes``
   Number of hash functions. More = higher accuracy but more memory.

   - 64: ~12% error
   - 128: ~9% error
   - 256: ~6% error
   - 512: ~4% error

Properties
----------

``jaccard(other)``
   Returns estimated Jaccard similarity (0.0 to 1.0).

``signature()``
   Returns the MinHash signature array.

``size_in_bytes``
   Memory usage.

How It Works
------------

MinHash works by:

1. Applying k different hash functions to each set element
2. For each hash function, keeping the minimum hash value
3. The signature is the vector of k minimum values
4. Similar sets will have similar signatures

.. code-block:: python

   mh = MinHash(num_hashes=128)
   mh.update(["a", "b", "c"])

   sig = mh.signature()
   print(f"Signature length: {len(sig)}")
   print(f"First 5 values: {sig[:5]}")

Accuracy vs Num Hashes
----------------------

.. code-block:: python

   import random

   # Two sets with 50% overlap
   set1 = {f"item_{i}" for i in range(100)}
   set2 = {f"item_{i}" for i in range(50, 150)}

   actual = len(set1 & set2) / len(set1 | set2)  # 0.333...

   for num_hashes in [32, 64, 128, 256, 512]:
       errors = []
       for _ in range(100):
           mh1 = MinHash(num_hashes=num_hashes)
           mh2 = MinHash(num_hashes=num_hashes)
           mh1.update(set1)
           mh2.update(set2)
           est = mh1.jaccard(mh2)
           errors.append(abs(est - actual))

       avg_error = sum(errors) / len(errors)
       print(f"k={num_hashes:>3}: avg error = {avg_error:.3f}")

Document Similarity
-------------------

For text documents, convert to shingles (n-grams):

.. code-block:: python

   def shingle(text: str, k: int = 5) -> set:
       """Convert text to k-character shingles."""
       text = text.lower()
       return {text[i:i+k] for i in range(len(text) - k + 1)}


   doc1 = "The quick brown fox jumps over the lazy dog"
   doc2 = "The fast brown fox leaps over the sleepy dog"

   mh1 = MinHash(num_hashes=128)
   mh2 = MinHash(num_hashes=128)

   mh1.update(shingle(doc1))
   mh2.update(shingle(doc2))

   print(f"Similarity: {mh1.jaccard(mh2):.1%}")

Merging MinHash
---------------

MinHash signatures can be merged:

.. code-block:: python

   # Combine two sets' signatures
   mh1 = MinHash(num_hashes=128)
   mh2 = MinHash(num_hashes=128)

   mh1.update(["a", "b", "c"])
   mh2.update(["d", "e", "f"])

   # Merge creates signature for union
   combined = mh1.merge(mh2)
   # or: combined = mh1 | mh2

   # Equivalent to:
   mh_direct = MinHash(num_hashes=128)
   mh_direct.update(["a", "b", "c", "d", "e", "f"])

Serialization
-------------

.. code-block:: python

   # Binary
   data = mh.to_bytes()
   mh2 = MinHash.from_bytes(data)

   # File
   mh.save("minhash.hazy")
   mh2 = MinHash.load("minhash.hazy")

Batch Similarity
----------------

For finding similar items in a collection:

.. code-block:: python

   from hazy import MinHash

   # Create index of documents
   documents = {
       "doc1": "Machine learning is fascinating",
       "doc2": "Deep learning advances AI",
       "doc3": "The weather is nice today",
       "doc4": "AI and machine learning are related",
   }

   signatures = {}
   for doc_id, text in documents.items():
       mh = MinHash(num_hashes=128)
       mh.update(set(text.lower().split()))
       signatures[doc_id] = mh

   # Find similar pairs
   from itertools import combinations

   for (id1, mh1), (id2, mh2) in combinations(signatures.items(), 2):
       sim = mh1.jaccard(mh2)
       if sim > 0.2:  # Threshold
           print(f"{id1} <-> {id2}: {sim:.1%}")

Memory Comparison
-----------------

.. list-table::
   :header-rows: 1

   * - num_hashes
     - Bytes per set
     - Error
   * - 64
     - 512
     - ~12%
   * - 128
     - 1024
     - ~9%
   * - 256
     - 2048
     - ~6%

For 1 million documents with 128 hashes: ~1 GB for all signatures.

See Also
--------

- :doc:`../tutorials/similarity_search` - Complete LSH tutorial
- :doc:`bloom_filter` - For set membership
