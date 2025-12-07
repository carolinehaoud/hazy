# Tutorial: Document Similarity Search

Learn how to find similar documents efficiently using MinHash signatures. This technique is used for near-duplicate detection, content recommendation, and plagiarism detection.

## The Problem

You have a collection of documents (articles, products, user profiles) and need to:

1. **Find similar items** quickly
2. **Detect near-duplicates** without exact matching
3. **Scale to millions of documents** without comparing every pair
4. **Handle minor variations** (typos, rewording, formatting)

Comparing every pair of N documents is O(N²) — impossible at scale. MinHash reduces this to near-linear time.

## Understanding Jaccard Similarity

Before diving into MinHash, let's understand what we're measuring:

```python
def jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Calculate exact Jaccard similarity between two sets.

    Jaccard = |A ∩ B| / |A ∪ B|

    - 1.0 = identical sets
    - 0.0 = no common elements
    """
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# Example
doc1_words = {"the", "quick", "brown", "fox", "jumps"}
doc2_words = {"the", "lazy", "brown", "dog", "sleeps"}

similarity = jaccard_similarity(doc1_words, doc2_words)
print(f"Exact Jaccard: {similarity:.3f}")  # 0.222 (2 common / 9 total)
```

## MinHash: Approximate Jaccard in Constant Time

MinHash creates a compact "signature" that lets us estimate Jaccard similarity without storing or comparing full sets:

```python
from hazy import MinHash


def create_document_signature(text: str, num_hashes: int = 128) -> MinHash:
    """
    Create a MinHash signature from document text.

    Uses word 3-grams (shingles) for better similarity detection.
    """
    mh = MinHash(num_hashes=num_hashes)

    # Tokenize into words
    words = text.lower().split()

    # Create 3-grams (shingles)
    for i in range(len(words) - 2):
        shingle = " ".join(words[i:i+3])
        mh.add(shingle)

    return mh


# Create signatures
text1 = "The quick brown fox jumps over the lazy dog"
text2 = "The fast brown fox leaps over the lazy dog"
text3 = "Python is a great programming language for data science"

sig1 = create_document_signature(text1)
sig2 = create_document_signature(text2)
sig3 = create_document_signature(text3)

# Compare similarities
print(f"Doc1 vs Doc2 (similar): {sig1.jaccard(sig2):.3f}")
print(f"Doc1 vs Doc3 (different): {sig1.jaccard(sig3):.3f}")
print(f"Doc2 vs Doc3 (different): {sig2.jaccard(sig3):.3f}")
```

## Building a Document Similarity Index

```python
from hazy import MinHash
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re


@dataclass
class Document:
    """A document with ID and content."""
    id: str
    title: str
    content: str


class SimilarityIndex:
    """
    Index for finding similar documents using MinHash.
    """

    def __init__(self, num_hashes: int = 128, shingle_size: int = 3):
        """
        Initialize the similarity index.

        Args:
            num_hashes: Number of hash functions (more = more accurate)
            shingle_size: Size of word n-grams
        """
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        self.signatures: dict[str, MinHash] = {}
        self.documents: dict[str, Document] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        # Remove punctuation and lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()

    def _create_shingles(self, words: List[str]) -> List[str]:
        """Create word n-grams from word list."""
        if len(words) < self.shingle_size:
            return [" ".join(words)] if words else []

        shingles = []
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i:i + self.shingle_size])
            shingles.append(shingle)
        return shingles

    def add_document(self, doc: Document):
        """Add a document to the index."""
        # Combine title and content
        full_text = f"{doc.title} {doc.content}"

        # Create shingles
        words = self._tokenize(full_text)
        shingles = self._create_shingles(words)

        # Create MinHash signature
        mh = MinHash(num_hashes=self.num_hashes)
        mh.update(shingles)

        # Store
        self.signatures[doc.id] = mh
        self.documents[doc.id] = doc

    def find_similar(
        self,
        doc_id: str,
        threshold: float = 0.5,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find documents similar to the given document.

        Args:
            doc_id: ID of the query document
            threshold: Minimum similarity threshold
            top_k: Maximum number of results

        Returns:
            List of (doc_id, similarity) tuples, sorted by similarity
        """
        if doc_id not in self.signatures:
            raise ValueError(f"Document {doc_id} not in index")

        query_sig = self.signatures[doc_id]
        results = []

        for other_id, other_sig in self.signatures.items():
            if other_id == doc_id:
                continue

            similarity = query_sig.jaccard(other_sig)
            if similarity >= threshold:
                results.append((other_id, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def find_similar_to_text(
        self,
        text: str,
        threshold: float = 0.5,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find documents similar to the given text (not in index).
        """
        words = self._tokenize(text)
        shingles = self._create_shingles(words)

        query_sig = MinHash(num_hashes=self.num_hashes)
        query_sig.update(shingles)

        results = []
        for doc_id, doc_sig in self.signatures.items():
            similarity = query_sig.jaccard(doc_sig)
            if similarity >= threshold:
                results.append((doc_id, similarity))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def find_near_duplicates(
        self,
        threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """
        Find all pairs of near-duplicate documents.

        Returns:
            List of (doc_id1, doc_id2, similarity) tuples
        """
        duplicates = []
        doc_ids = list(self.signatures.keys())

        for i, id1 in enumerate(doc_ids):
            for id2 in doc_ids[i+1:]:
                similarity = self.signatures[id1].jaccard(self.signatures[id2])
                if similarity >= threshold:
                    duplicates.append((id1, id2, similarity))

        return sorted(duplicates, key=lambda x: -x[2])
```

## Example: Article Similarity

```python
# Create sample articles
articles = [
    Document(
        id="article_1",
        title="Introduction to Machine Learning",
        content="""Machine learning is a subset of artificial intelligence that
        enables systems to learn and improve from experience. It focuses on
        developing algorithms that can access data and use it to learn for themselves."""
    ),
    Document(
        id="article_2",
        title="Getting Started with ML",
        content="""Machine learning is part of artificial intelligence where
        computers learn from experience. The focus is on creating algorithms
        that access data and learn autonomously."""
    ),
    Document(
        id="article_3",
        title="Deep Learning Fundamentals",
        content="""Deep learning is a subset of machine learning that uses
        neural networks with many layers. It excels at tasks like image
        recognition and natural language processing."""
    ),
    Document(
        id="article_4",
        title="Python for Data Science",
        content="""Python is the most popular language for data science.
        Libraries like pandas, numpy, and scikit-learn make it easy to
        analyze data and build machine learning models."""
    ),
    Document(
        id="article_5",
        title="Introduction to Machine Learning Concepts",
        content="""Machine learning is a branch of artificial intelligence
        that allows systems to learn from experience. It develops algorithms
        that can process data and improve automatically."""
    ),
]

# Build index
index = SimilarityIndex(num_hashes=128, shingle_size=3)

for article in articles:
    index.add_document(article)

print(f"Indexed {len(index.documents)} articles\n")

# Find similar to article_1
print("="*60)
print("DOCUMENTS SIMILAR TO 'Introduction to Machine Learning'")
print("="*60)

similar = index.find_similar("article_1", threshold=0.2)
for doc_id, sim in similar:
    doc = index.documents[doc_id]
    print(f"\n  [{sim:.1%}] {doc.title}")
    print(f"         {doc.content[:80]}...")

# Find near-duplicates
print("\n" + "="*60)
print("NEAR-DUPLICATE PAIRS (>50% similarity)")
print("="*60)

duplicates = index.find_near_duplicates(threshold=0.5)
for id1, id2, sim in duplicates:
    print(f"\n  [{sim:.1%}] '{index.documents[id1].title}'")
    print(f"         '{index.documents[id2].title}'")

# Search by new text
print("\n" + "="*60)
print("SEARCH: 'artificial intelligence learns from data'")
print("="*60)

results = index.find_similar_to_text(
    "artificial intelligence learns from data",
    threshold=0.1
)
for doc_id, sim in results:
    doc = index.documents[doc_id]
    print(f"\n  [{sim:.1%}] {doc.title}")
```

## Visualizing Similarity

```python
import matplotlib.pyplot as plt
import numpy as np
from hazy.viz import plot_minhash_comparison


def visualize_document_similarities(index: SimilarityIndex):
    """Create a heatmap of document similarities."""

    doc_ids = list(index.signatures.keys())
    n = len(doc_ids)

    # Compute similarity matrix
    sim_matrix = np.zeros((n, n))
    for i, id1 in enumerate(doc_ids):
        for j, id2 in enumerate(doc_ids):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = index.signatures[id1].jaccard(index.signatures[id2])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1)

    # Labels
    labels = [index.documents[id].title[:20] + "..." for id in doc_ids]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Add values
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                          ha="center", va="center", fontsize=9,
                          color="white" if sim_matrix[i, j] > 0.5 else "black")

    ax.set_title("Document Similarity Matrix")
    plt.colorbar(im, label="Jaccard Similarity")
    plt.tight_layout()
    plt.savefig("similarity_matrix.png", dpi=150)
    plt.show()

    print("\nSaved similarity matrix to 'similarity_matrix.png'")


# Visualize
visualize_document_similarities(index)
```

## Scaling with Locality-Sensitive Hashing (LSH)

For millions of documents, comparing every pair is still too slow. LSH provides approximate nearest neighbor search:

```python
from hazy import MinHash
from collections import defaultdict
from typing import Set


class LSHIndex:
    """
    Locality-Sensitive Hashing index for fast similarity search.

    Divides MinHash signatures into bands. Documents that share
    any band are candidates for similarity comparison.
    """

    def __init__(
        self,
        num_hashes: int = 128,
        num_bands: int = 16,
        shingle_size: int = 3
    ):
        """
        Initialize LSH index.

        Args:
            num_hashes: Total hash functions in MinHash
            num_bands: Number of bands (num_hashes must be divisible)
            shingle_size: Word n-gram size
        """
        if num_hashes % num_bands != 0:
            raise ValueError("num_hashes must be divisible by num_bands")

        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.shingle_size = shingle_size

        # Band -> hash -> set of doc_ids
        self.buckets: list[dict[int, set]] = [
            defaultdict(set) for _ in range(num_bands)
        ]

        self.signatures: dict[str, MinHash] = {}
        self.documents: dict[str, Document] = {}

    def _get_band_hashes(self, signature: list[int]) -> list[int]:
        """Split signature into band hashes."""
        band_hashes = []
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(signature[start:end])
            band_hashes.append(hash(band))
        return band_hashes

    def add_document(self, doc: Document):
        """Add document to the LSH index."""
        # Create signature
        text = f"{doc.title} {doc.content}"
        words = re.sub(r'[^\w\s]', '', text.lower()).split()

        mh = MinHash(num_hashes=self.num_hashes)
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i:i + self.shingle_size])
            mh.add(shingle)

        # Store signature
        self.signatures[doc.id] = mh
        self.documents[doc.id] = doc

        # Add to LSH buckets
        band_hashes = self._get_band_hashes(mh.signature())
        for band_idx, band_hash in enumerate(band_hashes):
            self.buckets[band_idx][band_hash].add(doc.id)

    def find_candidates(self, doc_id: str) -> Set[str]:
        """Find candidate similar documents using LSH."""
        if doc_id not in self.signatures:
            raise ValueError(f"Document {doc_id} not in index")

        signature = self.signatures[doc_id].signature()
        band_hashes = self._get_band_hashes(signature)

        candidates = set()
        for band_idx, band_hash in enumerate(band_hashes):
            candidates.update(self.buckets[band_idx][band_hash])

        candidates.discard(doc_id)
        return candidates

    def find_similar(
        self,
        doc_id: str,
        threshold: float = 0.5
    ) -> list[tuple[str, float]]:
        """
        Find similar documents using LSH + verification.

        Much faster than comparing all pairs!
        """
        # Step 1: Get candidates from LSH (fast)
        candidates = self.find_candidates(doc_id)

        # Step 2: Verify candidates with actual MinHash comparison
        query_sig = self.signatures[doc_id]
        results = []

        for candidate_id in candidates:
            similarity = query_sig.jaccard(self.signatures[candidate_id])
            if similarity >= threshold:
                results.append((candidate_id, similarity))

        return sorted(results, key=lambda x: -x[1])


# Example with LSH
print("\n" + "="*60)
print("LSH-ACCELERATED SIMILARITY SEARCH")
print("="*60)

lsh_index = LSHIndex(num_hashes=128, num_bands=16)

for article in articles:
    lsh_index.add_document(article)

# Find similar using LSH
candidates = lsh_index.find_candidates("article_1")
print(f"\nCandidates for article_1 (LSH): {len(candidates)}")
print(f"  vs checking all {len(articles)-1} documents")

similar = lsh_index.find_similar("article_1", threshold=0.3)
print(f"\nSimilar documents found:")
for doc_id, sim in similar:
    print(f"  [{sim:.1%}] {lsh_index.documents[doc_id].title}")
```

## Performance Analysis

```python
import time
import random
import string


def generate_random_documents(n: int, words_per_doc: int = 100) -> list[Document]:
    """Generate random documents for testing."""
    word_pool = [
        ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        for _ in range(1000)
    ]

    docs = []
    for i in range(n):
        words = random.choices(word_pool, k=words_per_doc)
        docs.append(Document(
            id=f"doc_{i}",
            title=f"Document {i}",
            content=" ".join(words)
        ))
    return docs


def benchmark_similarity_search(doc_counts: list[int] = [100, 500, 1000, 5000]):
    """Benchmark similarity search at different scales."""

    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    print(f"{'Documents':>10} {'Index Time':>12} {'Search Time':>12} {'Candidates':>12}")
    print("-"*60)

    for n in doc_counts:
        docs = generate_random_documents(n)

        # Build index
        lsh = LSHIndex(num_hashes=128, num_bands=16)
        start = time.time()
        for doc in docs:
            lsh.add_document(doc)
        index_time = time.time() - start

        # Search
        start = time.time()
        candidates = lsh.find_candidates("doc_0")
        search_time = time.time() - start

        print(f"{n:>10,} {index_time:>11.3f}s {search_time:>11.4f}s {len(candidates):>12,}")


benchmark_similarity_search([100, 500, 1000, 2000])
```

## Key Takeaways

1. **MinHash** approximates Jaccard similarity in O(k) time, where k is the number of hash functions

2. **Shingles** (word n-grams) capture local word patterns better than individual words

3. **LSH** reduces similarity search from O(N²) to near O(N) by only comparing candidates

4. **Trade-offs**:
   - More hashes → more accurate, more memory
   - More bands → more candidates, higher recall
   - Fewer bands → faster, but might miss similar documents

## Exercises

1. **Character shingles**: Implement character n-grams instead of word n-grams for typo tolerance

2. **Weighted shingles**: Give more weight to rare words/shingles

3. **LSH parameter tuning**: Experiment with different band/row configurations

4. **Multi-probe LSH**: Search adjacent buckets for better recall

## Complete Example Code

```python
"""
Complete Document Similarity Example

Run with: python similarity_search.py
"""

from hazy import MinHash
import re


def create_signature(text: str, num_hashes: int = 128) -> MinHash:
    """Create MinHash signature from text."""
    mh = MinHash(num_hashes=num_hashes)
    words = re.sub(r'[^\w\s]', '', text.lower()).split()

    for i in range(len(words) - 2):
        mh.add(" ".join(words[i:i+3]))

    return mh


def main():
    documents = {
        "ml_intro": "Machine learning is a subset of artificial intelligence...",
        "ml_basics": "Machine learning belongs to artificial intelligence...",
        "python": "Python is a programming language for data science...",
        "deep_learning": "Deep learning uses neural networks with many layers...",
    }

    # Create signatures
    signatures = {
        doc_id: create_signature(text)
        for doc_id, text in documents.items()
    }

    # Find similar pairs
    print("Document Similarities:")
    doc_ids = list(signatures.keys())
    for i, id1 in enumerate(doc_ids):
        for id2 in doc_ids[i+1:]:
            sim = signatures[id1].jaccard(signatures[id2])
            print(f"  {id1} <-> {id2}: {sim:.1%}")


if __name__ == "__main__":
    main()
```
