# MinHash

MinHash is a technique for estimating the **Jaccard similarity** between sets. It creates compact signatures that can quickly determine how similar two sets are.

## When to Use

- Near-duplicate detection (documents, images, etc.)
- Clustering similar items
- Recommendation systems (finding similar users/items)
- Plagiarism detection
- Locality-Sensitive Hashing (LSH) for nearest neighbor search

## Basic Usage

```python
from hazy import MinHash

# Create MinHash signatures for two sets
mh1 = MinHash(num_hashes=128)
mh2 = MinHash(num_hashes=128)

# Add elements
mh1.update(["apple", "banana", "cherry", "date"])
mh2.update(["banana", "cherry", "elderberry", "fig"])

# Estimate Jaccard similarity
similarity = mh1.jaccard(mh2)
print(f"Similarity: {similarity:.2%}")  # ~33%
```

## What is Jaccard Similarity?

Jaccard similarity measures set overlap:

\[
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
\]

- J = 1.0: Sets are identical
- J = 0.0: Sets have no common elements
- J = 0.5: Half of all elements are shared

### Example

```
A = {apple, banana, cherry}
B = {banana, cherry, date}

Intersection: {banana, cherry} → 2 items
Union: {apple, banana, cherry, date} → 4 items

J(A, B) = 2/4 = 0.5
```

## Construction Options

```python
# Specify number of hash functions
mh = MinHash(num_hashes=128)  # Good default

# More hashes = more accurate but larger signature
mh = MinHash(num_hashes=256)  # Higher accuracy

# Fewer hashes = faster but less accurate
mh = MinHash(num_hashes=64)   # For LSH applications
```

### Parameter Estimation

```python
from hazy import estimate_minhash_params

params = estimate_minhash_params(error_rate=0.05)
print(f"Hashes needed: {params.num_hashes}")
print(f"Memory: {params.memory_bytes} bytes")
```

### Accuracy vs Hash Count

| Hashes | Memory | Std Error (J=0.5) |
|--------|--------|-------------------|
| 64 | 512 B | 6.25% |
| 128 | 1 KB | 4.4% |
| 256 | 2 KB | 3.1% |
| 512 | 4 KB | 2.2% |

## Key Operations

### Adding Elements

```python
mh = MinHash(num_hashes=128)

# Single element
mh.add("word")

# Multiple elements
mh.update(["the", "quick", "brown", "fox"])

# Different types
mh.add(b"binary data")
mh.add(12345)
```

### Computing Similarity

```python
similarity = mh1.jaccard(mh2)

# Similarity is symmetric
assert mh1.jaccard(mh2) == mh2.jaccard(mh1)
```

### Merging (Union)

```python
mh1 = MinHash(num_hashes=128)
mh2 = MinHash(num_hashes=128)

mh1.update(["a", "b", "c"])
mh2.update(["c", "d", "e"])

# Merge creates signature for union of sets
merged = mh1 | mh2  # Or: mh1.merge(mh2)

# Now merged represents {a, b, c, d, e}
```

### Getting the Signature

```python
mh = MinHash(num_hashes=128)
mh.update(["a", "b", "c"])

signature = mh.signature()
print(f"Signature length: {len(signature)}")  # 128
print(f"First hash: {signature[0]}")
```

## Statistics

```python
mh = MinHash(num_hashes=128)
mh.update(["word"] * 100 + ["other"] * 50)

print(f"Number of hashes: {mh.num_hashes}")
print(f"Memory: {mh.size_in_bytes} bytes")
```

## Serialization

```python
# Binary
data = mh.to_bytes()
mh2 = MinHash.from_bytes(data)

# JSON
json_str = mh.to_json()
mh2 = MinHash.from_json(json_str)

# File I/O
mh.save("signature.hazy")
mh2 = MinHash.load("signature.hazy")
```

## How It Works

### The MinHash Property

For two sets A and B, if we apply a random hash function h and take the minimum hash value from each set:

\[
P(\min(h(A)) = \min(h(B))) = J(A, B)
\]

The probability that the minimum hashes match equals the Jaccard similarity!

### Algorithm

1. Initialize k hash functions
2. For each hash function, track the minimum hash value seen
3. The signature is the vector of k minimum values
4. Similarity = fraction of matching positions

```
Signature of A: [min_1(A), min_2(A), ..., min_k(A)]
Signature of B: [min_1(B), min_2(B), ..., min_k(B)]

Estimated J(A,B) = (# matching positions) / k
```

### Error Analysis

The estimate is the mean of k Bernoulli trials, so:

\[
\text{Std Error} = \sqrt{\frac{J(1-J)}{k}}
\]

Maximum error occurs at J=0.5.

## Use Cases

### 1. Document Similarity

```python
from hazy import MinHash

def document_signature(text, num_hashes=128):
    """Create MinHash signature from document."""
    mh = MinHash(num_hashes=num_hashes)

    # Use word n-grams (shingles)
    words = text.lower().split()
    for i in range(len(words) - 2):
        shingle = " ".join(words[i:i+3])
        mh.add(shingle)

    return mh

doc1 = document_signature("The quick brown fox jumps over the lazy dog")
doc2 = document_signature("The fast brown fox leaps over the lazy dog")

print(f"Similarity: {doc1.jaccard(doc2):.2%}")
```

### 2. Near-Duplicate Detection

```python
from hazy import MinHash

class DuplicateDetector:
    def __init__(self, threshold=0.8, num_hashes=128):
        self.signatures = {}
        self.threshold = threshold
        self.num_hashes = num_hashes

    def add(self, doc_id, content):
        mh = MinHash(num_hashes=self.num_hashes)
        mh.update(self._shingle(content))
        self.signatures[doc_id] = mh

    def find_duplicates(self, doc_id):
        mh = self.signatures[doc_id]
        duplicates = []
        for other_id, other_mh in self.signatures.items():
            if other_id != doc_id:
                sim = mh.jaccard(other_mh)
                if sim >= self.threshold:
                    duplicates.append((other_id, sim))
        return duplicates

    def _shingle(self, text, k=3):
        words = text.lower().split()
        return [" ".join(words[i:i+k]) for i in range(len(words)-k+1)]
```

### 3. Recommendation System

```python
from hazy import MinHash

class UserSimilarity:
    def __init__(self, num_hashes=256):
        self.profiles = {}
        self.num_hashes = num_hashes

    def record_interaction(self, user_id, item_id):
        if user_id not in self.profiles:
            self.profiles[user_id] = MinHash(num_hashes=self.num_hashes)
        self.profiles[user_id].add(item_id)

    def find_similar_users(self, user_id, top_k=10):
        if user_id not in self.profiles:
            return []

        user_mh = self.profiles[user_id]
        similarities = []

        for other_id, other_mh in self.profiles.items():
            if other_id != user_id:
                sim = user_mh.jaccard(other_mh)
                similarities.append((other_id, sim))

        return sorted(similarities, key=lambda x: -x[1])[:top_k]
```

### 4. Clustering with LSH

```python
from hazy import MinHash

def lsh_bucket(signature, band_size):
    """Divide signature into bands for LSH."""
    bands = []
    for i in range(0, len(signature), band_size):
        band = tuple(signature[i:i+band_size])
        bands.append(hash(band))
    return bands

def find_candidates(signatures, band_size=4):
    """Find candidate pairs using LSH."""
    buckets = {}

    for doc_id, sig in signatures.items():
        for band_idx, band_hash in enumerate(lsh_bucket(sig, band_size)):
            key = (band_idx, band_hash)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(doc_id)

    # Candidates are docs that share at least one bucket
    candidates = set()
    for docs in buckets.values():
        if len(docs) > 1:
            for i, d1 in enumerate(docs):
                for d2 in docs[i+1:]:
                    candidates.add((min(d1, d2), max(d1, d2)))

    return candidates
```

## Comparison with Other Approaches

| Method | Time | Space | Exact |
|--------|------|-------|-------|
| Exact Jaccard | O(n) | O(n) | Yes |
| MinHash | O(k) | O(k) | No |
| SimHash | O(k) | O(k) | No |

Where n is set size and k is signature size.

## Best Practices

1. **Choose hash count based on accuracy needs**:
   - Quick filtering: 64-128 hashes
   - Accurate similarity: 256+ hashes

2. **Use shingles for text**:
   - Character n-grams for typo tolerance
   - Word n-grams for semantic similarity

3. **Same num_hashes required for comparison**:
   ```python
   mh1 = MinHash(num_hashes=128)
   mh2 = MinHash(num_hashes=256)
   # mh1.jaccard(mh2)  # Error! Incompatible
   ```

4. **Consider LSH for large-scale search** - O(1) candidate retrieval instead of O(n) comparisons
