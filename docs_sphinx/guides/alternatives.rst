Alternatives and Comparisons
============================

The Python ecosystem has several excellent libraries for probabilistic data structures. This guide helps you choose the right tool for your needs.

Related Libraries
-----------------

**datasketch**
    A well-established pure Python library with MinHash, HyperLogLog, and LSH implementations. Great for similarity search and cardinality estimation. If you're primarily doing document similarity with LSH, datasketch's LSH index is mature and feature-rich.

**pybloom-live**
    A maintained fork of pybloom providing Bloom filter implementations in pure Python. Simple API for basic membership testing.

**pdsa**
    Probabilistic data structures in Python with a focus on educational clarity. Good for learning how these algorithms work.

**redis**
    Redis provides server-based HyperLogLog and Bloom filter (via RedisBloom module) implementations. Ideal when you need shared state across multiple processes or services.

**python-bloomfilter**
    Another pure Python Bloom filter implementation with a straightforward API.

When to Choose Hazy
-------------------

Hazy may be a good fit when you need:

- **Performance**: Rust-backed implementations are typically 5-20x faster than pure Python
- **Multiple structures**: All 8 data structures in one package with a consistent API
- **Serialization**: Built-in binary, JSON, and file I/O for all structures
- **Memory efficiency**: Optimized memory layout from Rust
- **Type safety**: Full type hints and IDE support

When to Choose Alternatives
---------------------------

Consider other libraries when:

- **Pure Python required**: Some environments can't use native extensions
- **Distributed systems**: Redis provides shared state across services
- **Specialized LSH**: datasketch has more advanced LSH features for similarity search
- **Minimal dependencies**: Simpler libraries may suit small projects
- **Learning**: Pure Python implementations are easier to study and modify

Feature Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Feature
     - Hazy
     - datasketch
     - pybloom-live
     - Redis
     - Notes
   * - Bloom Filter
     - ✓
     - ✗
     - ✓
     - ✓
     -
   * - Counting Bloom
     - ✓
     - ✗
     - ✗
     - ✓
     -
   * - Cuckoo Filter
     - ✓
     - ✗
     - ✗
     - ✓
     -
   * - HyperLogLog
     - ✓
     - ✓
     - ✗
     - ✓
     -
   * - Count-Min Sketch
     - ✓
     - ✗
     - ✗
     - ✓
     -
   * - MinHash
     - ✓
     - ✓
     - ✗
     - ✗
     - datasketch has LSH index
   * - Top-K
     - ✓
     - ✗
     - ✗
     - ✓
     -
   * - Pure Python
     - ✗
     - ✓
     - ✓
     - N/A
     -
   * - Distributed
     - ✗
     - ✗
     - ✗
     - ✓
     - Redis is server-based

Acknowledgments
---------------

Hazy is built on the shoulders of giants. The algorithms implemented here are based on foundational research:

- **Bloom Filter**: Burton Howard Bloom (1970)
- **HyperLogLog**: Flajolet, Fusy, Gandouet, Meunier (2007)
- **Count-Min Sketch**: Cormode and Muthukrishnan (2005)
- **MinHash**: Broder (1997)
- **Cuckoo Filter**: Fan, Andersen, Kaminsky, Mitzenmacher (2014)

We're grateful to the authors of datasketch, pybloom, and other libraries that have made probabilistic data structures accessible to the Python community.
