Tutorial: Document Similarity Search
=====================================

Learn how to find similar documents efficiently using MinHash and Locality-Sensitive Hashing (LSH).

The Problem
-----------

You have a large collection of documents and need to:

1. **Find near-duplicates** (plagiarism detection, content deduplication)
2. **Find similar items** (recommendation systems)
3. **Cluster related content** (topic modeling, search)

Comparing every pair of documents is O(n²) — too slow for large collections.

Solution: MinHash + LSH
-----------------------

MinHash creates compact "signatures" that preserve Jaccard similarity. LSH enables sub-linear search by hashing similar items to the same buckets.

Basic MinHash Usage
-------------------

.. code-block:: python

   from hazy import MinHash

   # Create MinHash signatures for two documents
   mh1 = MinHash(num_hashes=128)
   mh2 = MinHash(num_hashes=128)

   # Tokenize documents into sets of words/shingles
   doc1 = "the quick brown fox jumps over the lazy dog"
   doc2 = "the fast brown fox leaps over the sleepy dog"

   words1 = set(doc1.lower().split())
   words2 = set(doc2.lower().split())

   mh1.update(words1)
   mh2.update(words2)

   # Estimate Jaccard similarity
   estimated = mh1.jaccard(mh2)
   actual = len(words1 & words2) / len(words1 | words2)

   print(f"Estimated similarity: {estimated:.2%}")
   print(f"Actual similarity:    {actual:.2%}")

Using Shingles for Better Accuracy
----------------------------------

Single words lose word order. Use n-gram shingles instead:

.. code-block:: python

   def get_shingles(text: str, k: int = 3) -> set:
       """Extract character k-shingles from text."""
       text = text.lower()
       return {text[i:i+k] for i in range(len(text) - k + 1)}


   def get_word_shingles(text: str, k: int = 2) -> set:
       """Extract word k-shingles from text."""
       words = text.lower().split()
       return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}


   # Compare with shingles
   doc1 = "The quick brown fox jumps over the lazy dog"
   doc2 = "The quick brown fox leaps over the lazy cat"

   shingles1 = get_shingles(doc1, k=5)
   shingles2 = get_shingles(doc2, k=5)

   mh1 = MinHash(num_hashes=128)
   mh2 = MinHash(num_hashes=128)

   mh1.update(shingles1)
   mh2.update(shingles2)

   print(f"Similarity: {mh1.jaccard(mh2):.2%}")

Building a Similarity Index
---------------------------

For searching a collection of documents:

.. code-block:: python

   from hazy import MinHash
   from typing import List, Tuple


   class SimilarityIndex:
       """Index for finding similar documents."""

       def __init__(self, num_hashes: int = 128, shingle_size: int = 5):
           self.num_hashes = num_hashes
           self.shingle_size = shingle_size
           self.signatures = {}  # doc_id -> MinHash

       def _shingle(self, text: str) -> set:
           """Convert text to shingles."""
           text = text.lower()
           k = self.shingle_size
           return {text[i:i+k] for i in range(len(text) - k + 1)}

       def add(self, doc_id: str, text: str):
           """Add a document to the index."""
           mh = MinHash(num_hashes=self.num_hashes)
           mh.update(self._shingle(text))
           self.signatures[doc_id] = mh

       def find_similar(
           self,
           text: str,
           threshold: float = 0.5,
           top_k: int = 10
       ) -> List[Tuple[str, float]]:
           """Find documents similar to the query."""
           query_mh = MinHash(num_hashes=self.num_hashes)
           query_mh.update(self._shingle(text))

           results = []
           for doc_id, doc_mh in self.signatures.items():
               sim = query_mh.jaccard(doc_mh)
               if sim >= threshold:
                   results.append((doc_id, sim))

           # Sort by similarity descending
           results.sort(key=lambda x: x[1], reverse=True)
           return results[:top_k]


   # Example usage
   index = SimilarityIndex(num_hashes=128)

   # Add documents
   documents = {
       "doc1": "Machine learning is a subset of artificial intelligence.",
       "doc2": "Deep learning is part of machine learning methods.",
       "doc3": "Natural language processing uses machine learning.",
       "doc4": "The weather today is sunny and warm.",
       "doc5": "Climate change affects global weather patterns.",
   }

   for doc_id, text in documents.items():
       index.add(doc_id, text)

   # Search
   query = "Machine learning and artificial intelligence are related."
   results = index.find_similar(query, threshold=0.1)

   print("Similar documents:")
   for doc_id, sim in results:
       print(f"  {doc_id}: {sim:.2%} - {documents[doc_id][:50]}...")

Locality-Sensitive Hashing (LSH)
--------------------------------

For large collections, comparing against every document is slow. LSH provides approximate nearest neighbor search:

.. code-block:: python

   from hazy import MinHash
   from collections import defaultdict
   from typing import List, Tuple, Set
   import hashlib


   class LSHIndex:
       """
       Locality-Sensitive Hashing index for fast similarity search.

       Uses banding technique: divides signature into bands,
       documents that share any band hash are candidates.
       """

       def __init__(
           self,
           num_hashes: int = 128,
           num_bands: int = 16,
           shingle_size: int = 5
       ):
           assert num_hashes % num_bands == 0
           self.num_hashes = num_hashes
           self.num_bands = num_bands
           self.rows_per_band = num_hashes // num_bands
           self.shingle_size = shingle_size

           self.signatures = {}  # doc_id -> MinHash
           self.buckets = [defaultdict(set) for _ in range(num_bands)]

       def _shingle(self, text: str) -> set:
           text = text.lower()
           k = self.shingle_size
           return {text[i:i+k] for i in range(len(text) - k + 1)}

       def _get_band_hashes(self, mh: MinHash) -> List[str]:
           """Get hash for each band of the signature."""
           sig = mh.signature()
           band_hashes = []
           for i in range(self.num_bands):
               start = i * self.rows_per_band
               end = start + self.rows_per_band
               band = tuple(sig[start:end])
               h = hashlib.md5(str(band).encode()).hexdigest()
               band_hashes.append(h)
           return band_hashes

       def add(self, doc_id: str, text: str):
           """Add a document to the LSH index."""
           mh = MinHash(num_hashes=self.num_hashes)
           mh.update(self._shingle(text))
           self.signatures[doc_id] = mh

           # Add to buckets
           for band_idx, band_hash in enumerate(self._get_band_hashes(mh)):
               self.buckets[band_idx][band_hash].add(doc_id)

       def find_candidates(self, text: str) -> Set[str]:
           """Find candidate documents (may include false positives)."""
           mh = MinHash(num_hashes=self.num_hashes)
           mh.update(self._shingle(text))

           candidates = set()
           for band_idx, band_hash in enumerate(self._get_band_hashes(mh)):
               candidates.update(self.buckets[band_idx][band_hash])

           return candidates

       def find_similar(
           self,
           text: str,
           threshold: float = 0.5,
           top_k: int = 10
       ) -> List[Tuple[str, float]]:
           """Find similar documents using LSH."""
           query_mh = MinHash(num_hashes=self.num_hashes)
           query_mh.update(self._shingle(text))

           # Get candidates from LSH
           candidates = self.find_candidates(text)

           # Compute actual similarities for candidates only
           results = []
           for doc_id in candidates:
               sim = query_mh.jaccard(self.signatures[doc_id])
               if sim >= threshold:
                   results.append((doc_id, sim))

           results.sort(key=lambda x: x[1], reverse=True)
           return results[:top_k]


   # Example with larger collection
   lsh = LSHIndex(num_hashes=128, num_bands=16)

   # Add many documents
   for i in range(1000):
       lsh.add(f"doc_{i}", f"Document number {i} with some content about topic {i % 10}")

   # Add some similar documents
   lsh.add("similar_a", "Machine learning is transforming the world of AI")
   lsh.add("similar_b", "Machine learning transforms the AI world")
   lsh.add("similar_c", "AI and machine learning are changing everything")

   # Search
   query = "Machine learning and AI are revolutionizing technology"
   candidates = lsh.find_candidates(query)
   print(f"LSH found {len(candidates)} candidates out of {len(lsh.signatures)} documents")

   results = lsh.find_similar(query, threshold=0.2)
   print("\nMost similar:")
   for doc_id, sim in results[:5]:
       print(f"  {doc_id}: {sim:.2%}")

Choosing LSH Parameters
-----------------------

The probability that two documents with similarity *s* become candidates:

.. math::

   P = 1 - (1 - s^r)^b

where *r* = rows per band, *b* = number of bands.

.. code-block:: python

   def lsh_probability(similarity: float, bands: int, rows: int) -> float:
       """Probability that two docs become LSH candidates."""
       return 1 - (1 - similarity ** rows) ** bands


   # Example: 128 hashes, 16 bands = 8 rows per band
   print("Candidate probability by similarity:")
   print("  Similarity | Probability")
   print("  -----------|------------")
   for s in [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]:
       p = lsh_probability(s, bands=16, rows=8)
       print(f"  {s:>10.0%} | {p:>10.1%}")

Practical Example: Near-Duplicate Detection
-------------------------------------------

.. code-block:: python

   from hazy import MinHash
   from itertools import combinations


   def find_near_duplicates(
       documents: dict,
       threshold: float = 0.8,
       num_hashes: int = 128
   ) -> List[Tuple[str, str, float]]:
       """Find all pairs of near-duplicate documents."""

       # Create signatures
       signatures = {}
       for doc_id, text in documents.items():
           mh = MinHash(num_hashes=num_hashes)
           words = set(text.lower().split())
           mh.update(words)
           signatures[doc_id] = mh

       # Find similar pairs
       duplicates = []
       for (id1, mh1), (id2, mh2) in combinations(signatures.items(), 2):
           sim = mh1.jaccard(mh2)
           if sim >= threshold:
               duplicates.append((id1, id2, sim))

       return sorted(duplicates, key=lambda x: x[2], reverse=True)


   # Test with some documents
   docs = {
       "article_1": "The quick brown fox jumps over the lazy dog",
       "article_2": "The quick brown fox leaps over the lazy dog",
       "article_3": "A fast brown fox jumps over a sleepy dog",
       "article_4": "The weather is nice today",
       "article_5": "The weather is beautiful today",
   }

   dupes = find_near_duplicates(docs, threshold=0.5)
   print("Near-duplicates found:")
   for id1, id2, sim in dupes:
       print(f"  {id1} <-> {id2}: {sim:.1%}")

Memory Considerations
---------------------

.. list-table::
   :header-rows: 1

   * - num_hashes
     - Memory per doc
     - Accuracy
   * - 64
     - 512 bytes
     - ±12%
   * - 128
     - 1 KB
     - ±9%
   * - 256
     - 2 KB
     - ±6%
   * - 512
     - 4 KB
     - ±4%

For 1 million documents with 128 hashes: ~128 MB for signatures.

Best Practices
--------------

1. **Choose appropriate shingle size**: 5-10 for short text, 3-5 for longer documents
2. **Use enough hash functions**: 128 is a good default, more for higher accuracy
3. **Tune LSH bands**: More bands = more candidates = higher recall but slower
4. **Normalize text**: Lowercase, remove punctuation, optionally stem
5. **Consider word n-grams**: For semantic similarity, use word-level shingles

Next Steps
----------

- Explore :doc:`../guides/serialization` to save MinHash signatures
- Check :doc:`../guides/visualization` for similarity heatmaps
