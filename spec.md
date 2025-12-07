# Product Specification: Probabilistic Data Structures Library for Python

**Version:** 1.0 Draft  
**Author:** Caroline  
**Date:** December 2025  
**Status:** Planning

---

## Executive Summary

A modern, well-documented Python library providing unified access to probabilistic data structures for approximate computing at scale. The library will offer Bloom filters, HyperLogLog, Count-Min Sketch, Cuckoo filters, and related structures under a consistent, Pythonic API with excellent performance characteristics.

---

## Problem Statement

### Current Pain Points

1. **Fragmentation**: Existing Python libraries for probabilistic data structures are scattered across multiple packages with inconsistent APIs, varying maintenance status, and different design philosophies.

2. **Abandonment**: Popular libraries like `pybloom` are effectively unmaintained, with open issues dating back years and no Python 3.10+ compatibility testing.

3. **Incomplete Coverage**: No single library provides all commonly-needed structures. Users must cobble together `pybloom` + `hyperloglog` + `countminsketch` from different authors.

4. **Performance Gaps**: Pure Python implementations are slow; Rust/C-backed implementations exist but often have limited features or rough APIs.

5. **Poor Documentation**: Most existing libraries have minimal documentation, few examples, and no guidance on parameter selection.

### Market Opportunity

| Metric | Evidence |
|--------|----------|
| PyPI downloads for `pybloom` | ~50k/month (despite being unmaintained) |
| GitHub stars for `pyprobables` | ~350 (active but limited adoption) |
| Stack Overflow questions | 500+ questions tagged [bloom-filter] + [python] |
| Julia's `DataStructures.jl` | Includes probabilistic structures; widely used |

---

## Target Users

### Primary Personas

**1. Backend Engineers**
- Building deduplication systems, caches, or rate limiters
- Need: Fast membership testing, approximate counting
- Pain: Currently using Redis probabilistic modules or rolling their own

**2. Data Engineers**
- Processing large-scale data pipelines (Spark, Dask, Ray)
- Need: Cardinality estimation, frequency counting, set operations
- Pain: Shipping data to external services for approximate queries

**3. ML/Data Scientists**
- Feature engineering, similarity detection, streaming analytics
- Need: MinHash for similarity, Count-Min for frequency features
- Pain: Implementing from scratch for each project

**4. Systems Programmers**
- Building distributed systems, databases, network tools
- Need: Space-efficient set representations, probabilistic routing
- Pain: No "batteries included" solution in Python

### Secondary Personas

- **Students**: Learning about probabilistic data structures
- **Researchers**: Prototyping algorithms before production implementation
- **DevOps**: Log analysis, anomaly detection, monitoring

---

## Competitive Analysis

| Library | Stars | Last Update | Structures | Performance | API Quality | Docs |
|---------|-------|-------------|------------|-------------|-------------|------|
| `pyprobables` | 350 | Active | Bloom, CMS, Cuckoo | Pure Python (slow) | Good | Okay |
| `pybloom` | 500 | 2017 | Bloom only | C-backed | Dated | Poor |
| `rbloom` | 200 | Active | Bloom only | Rust-backed (fast) | Good | Good |
| `hyperloglog` | 150 | 2019 | HLL only | Pure Python | Basic | Poor |
| `datasketch` | 2k | Active | MinHash, HLL, LSH | Pure Python | Good | Good |

### Gap Analysis

- **No unified library** covers Bloom + HLL + CMS + Cuckoo + MinHash
- **No library** combines Rust/C performance with comprehensive coverage
- **datasketch** is closest but focused on similarity/LSH, not general probabilistic structures
- **pyprobables** has breadth but lacks performance and adoption

---

## Core Features (MVP)

### Data Structures

| Structure | Use Case | Priority |
|-----------|----------|----------|
| **Bloom Filter** | Set membership testing | P0 |
| **Counting Bloom Filter** | Membership + deletion support | P0 |
| **Scalable Bloom Filter** | Unknown cardinality | P1 |
| **Cuckoo Filter** | Membership + deletion + better FPR | P0 |
| **HyperLogLog** | Cardinality estimation | P0 |
| **HyperLogLog++** | Improved HLL for small/large cardinalities | P1 |
| **Count-Min Sketch** | Frequency estimation | P0 |
| **Count-Mean-Min Sketch** | Improved frequency estimation | P2 |
| **MinHash** | Set similarity (Jaccard) | P1 |
| **Top-K (Space-Saving)** | Heavy hitters | P1 |

### Core Operations

```python
# Unified API pattern for all structures
structure.add(item)           # Add single item
structure.update(items)       # Add multiple items (batch)
structure.query(item)         # Query (meaning varies by structure)
structure.merge(other)        # Combine two structures
structure.copy()              # Deep copy
structure.clear()             # Reset to empty state

# Serialization
structure.to_bytes()          # Binary serialization
Structure.from_bytes(data)    # Deserialize

# Introspection
len(structure)                # Approximate count (where applicable)
structure.size_in_bytes       # Memory footprint
structure.false_positive_rate # Current/expected FPR
```

### Parameter Helpers

```python
# Help users choose parameters
from probstruct import estimate_bloom_params

params = estimate_bloom_params(
    expected_items=1_000_000,
    false_positive_rate=0.01
)
# Returns: {'num_bits': 9585059, 'num_hashes': 7, 'memory_mb': 1.14}

bloom = BloomFilter(**params)
```

---

## Technical Requirements

### Performance Targets

| Operation | Target | Benchmark Against |
|-----------|--------|-------------------|
| Bloom add | < 500ns | rbloom |
| Bloom query | < 300ns | rbloom |
| HLL add | < 1µs | datasketch |
| CMS add | < 1µs | pyprobables |
| Serialization | < 10ms for 1M items | - |

### Implementation Strategy

**Core (Rust with PyO3)**
- All hot paths implemented in Rust
- Memory-safe, zero-copy where possible
- SIMD optimizations for batch operations

**Python Layer**
- Type hints throughout
- Dataclass-based configuration
- Context managers for memory-mapped large structures

### Compatibility

- Python 3.9+ (drop 3.8 in 2025)
- Wheels for: Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x86_64)
- No runtime dependencies (numpy optional for batch operations)
- Optional integrations: pandas, polars, numpy

---

## API Design Principles

### 1. Pythonic Defaults
```python
# Works out of the box with sensible defaults
bloom = BloomFilter(expected_items=10000)
bloom.add("hello")
"hello" in bloom  # True
```

### 2. Progressive Disclosure
```python
# Simple usage
bloom = BloomFilter(expected_items=10000)

# Power user: full control
bloom = BloomFilter(
    num_bits=1_000_000,
    num_hashes=7,
    hash_function=custom_hash,
    seed=42
)
```

### 3. Familiar Patterns
```python
# Set-like interface
bloom.add(item)
item in bloom
bloom |= other_bloom  # Union
bloom &= other_bloom  # Intersection (where supported)

# Dict-like for frequency structures
cms[item] += 1
cms[item]  # Returns estimated count
```

### 4. Fail-Safe Behavior
```python
# Clear error messages
bloom = BloomFilter(expected_items=-1)
# ValueError: expected_items must be positive, got -1

# Warnings for suboptimal usage
bloom = BloomFilter(expected_items=100, false_positive_rate=0.5)
# UserWarning: FPR of 0.5 is very high; consider 0.01-0.1 for most use cases
```

---

## Documentation Requirements

### Must-Have

1. **Quickstart Guide** (< 5 min to first working code)
2. **API Reference** (auto-generated from docstrings)
3. **Parameter Selection Guide** (how to choose settings)
4. **Performance Benchmarks** (reproducible, vs competitors)
5. **Use Case Cookbook** (10+ real-world examples)

### Examples to Include

- URL deduplication in a web crawler
- Cardinality estimation for database query planning
- Frequency counting in streaming data
- Cache membership pre-check
- Similar document detection with MinHash
- Rate limiting with counting Bloom filter

---

## Success Metrics

### Adoption (6 months post-launch)

| Metric | Target |
|--------|--------|
| GitHub stars | 500+ |
| PyPI downloads/month | 10,000+ |
| Contributors | 5+ |
| Open issues resolved | > 80% |

### Quality

| Metric | Target |
|--------|--------|
| Test coverage | > 95% |
| Benchmark regression | < 5% between releases |
| Documentation coverage | 100% public API |

---

## Development Phases

### Phase 1: Foundation (Weeks 1-4)
- [ ] Project scaffolding (PyO3, maturin, CI/CD)
- [ ] Bloom filter implementation + tests
- [ ] Counting Bloom filter
- [ ] Basic documentation

### Phase 2: Core Structures (Weeks 5-8)
- [ ] HyperLogLog
- [ ] Count-Min Sketch
- [ ] Cuckoo filter
- [ ] Serialization for all structures

### Phase 3: Polish (Weeks 9-12)
- [ ] MinHash
- [ ] Top-K / Heavy hitters
- [ ] Parameter estimation helpers
- [ ] Comprehensive benchmarks
- [ ] Full documentation

### Phase 4: Launch (Week 13+)
- [ ] PyPI release
- [ ] Announcement (Reddit, HN, Twitter)
- [ ] Gather feedback, iterate

---

## Naming Candidates

### PyPI Availability (Verified December 2025)

| Name | Available | Rationale |
|------|-----------|-----------|
| **probds** | ✅ YES | Short for "probabilistic data structures"; clear, professional |
| **hazy** | ✅ YES | Evokes approximate/fuzzy nature; short, memorable, unique |
| **pysketches** | ✅ YES | References "sketch" structures; follows Python naming conventions |
| **countish** | ✅ YES | Playful, suggests approximate counting |
| **softset** | ✅ YES | Evokes probabilistic membership; technical but approachable |
| **fuzzycount** | ✅ YES | Clear intent, descriptive |
| **probstruct** | ✅ YES | Professional but less memorable |
| `probly` | ❌ taken | Was a great option |
| `sketchy` | ❌ taken | Would have been perfect |
| `approx` | ❌ taken | Too generic anyway |
| `ballpark` | ❌ taken | Unrelated package |
| `probably` | ❌ taken | Already a testing library |

### Top 3 Recommendations

**1. `hazy`** ⭐ RECOMMENDED
- Short (4 chars), memorable, unique
- Evokes the approximate/fuzzy nature perfectly
- Easy to type, easy to say
- Great for branding: "hazy estimates", "hazy counting"
- Googleable: won't conflict with other tech terms
- Usage: `from hazy import BloomFilter`

**2. `probds`**
- Clear abbreviation: probabilistic data structures
- Professional, technical audience will understand immediately
- Short (6 chars), easy to type
- Less "fun" but more self-documenting
- Usage: `from probds import HyperLogLog`

**3. `pysketches`**
- References the "sketch" family of data structures (CMS, HLL, etc.)
- Follows Python naming convention (py- prefix)
- Clear to anyone familiar with the domain
- Slightly longer (10 chars) but still reasonable
- Usage: `from pysketches import CountMinSketch`

### Naming Criteria Checklist

For final selection, verify:
- [x] Available on PyPI
- [ ] Available on GitHub (check github.com/<name>)
- [x] Short (< 10 characters)
- [x] Easy to type and remember
- [x] Not easily confused with existing packages
- [x] Googleable (unique enough to find)
- [ ] Domain available (optional: .dev, .io)

**Final Recommendation:** Go with **`hazy`** — it's short, memorable, evocative, and available. It's the kind of name people remember after hearing it once.

---

## Technical Decisions (Resolved)

### 1. Rust vs C Backend → **Rust (PyO3 + Maturin)**

**Decision:** Use Rust with PyO3 for the native backend.

**Rationale:**
- **Performance parity**: Benchmarks show PyO3/Rust achieves nearly identical performance to C extensions (~25-70ns call overhead vs Cython's ~40ns). For operations taking >100ns (all our data structure operations), the difference is negligible.
- **Memory safety**: Rust's ownership model prevents entire classes of bugs (buffer overflows, use-after-free) that plague C extensions. This matters for a library users will trust with production data.
- **Modern tooling**: Maturin provides excellent build/wheel generation. PyO3 has active development (v0.20+), great documentation, and strong community.
- **Maintenance burden**: Rust code is easier to maintain safely than C. No manual memory management = fewer CVEs.
- **Precedent**: `rbloom`, `pydantic-core`, `polars`, `cryptography` all use Rust successfully.

**Trade-off acknowledged**: Slightly higher call overhead than Cython for trivial functions, but our operations (hashing, bit manipulation) are substantial enough that this is immaterial.

---

### 2. NumPy Integration → **Optional Dependency with First-Class Support**

**Decision:** NumPy is an optional dependency, but when installed, provide optimized batch operations.

**Implementation:**
```toml
# pyproject.toml
[project.optional-dependencies]
numpy = ["numpy>=1.20"]
all = ["numpy>=1.20", "polars>=0.19"]
```

```python
# hazy/_compat.py
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# hazy/bloom.py
def add_many(self, items):
    """Add multiple items. Accepts list, tuple, or numpy array."""
    if HAS_NUMPY and isinstance(items, np.ndarray):
        return self._add_numpy_array(items)  # Optimized path
    return self._add_iterable(items)
```

**Rationale:**
- **Zero-dependency core**: Users who just want a Bloom filter shouldn't need numpy. Keep `pip install hazy` lightweight.
- **Power users get speed**: Data engineers working with numpy arrays get optimized batch operations without copying.
- **Ecosystem fit**: This is the standard pattern (pandas, scikit-learn, etc. all do this).
- **Type hints**: Use `TYPE_CHECKING` pattern to support numpy types in signatures without requiring it at runtime.

**APIs enhanced by numpy:**
- `bloom.add_many(np.array([...]))` — vectorized insertion
- `bloom.contains_many(np.array([...]))` — returns boolean array
- `hll.add_many(np.array([...]))` — batch cardinality updates
- `cms.query_many(np.array([...]))` — returns frequency array

---

### 3. Redis Compatibility → **No, Use Our Own Format**

**Decision:** Do not attempt Redis serialization format compatibility. Provide our own efficient binary format.

**Rationale:**
- **No documented format**: RedisBloom doesn't publish a serialization spec. Their internal format is implementation-dependent and could change.
- **Different use cases**: Redis Bloom filters live in Redis; our filters live in Python memory or files. Users wanting Redis integration should use `redisbloom` client directly.
- **Complexity vs value**: Reverse-engineering Redis's format adds maintenance burden with unclear benefit. Users needing Redis interop are already using Redis clients.
- **Our format can be better**: We can optimize for Python use cases (memory-mapped files, zero-copy deserialization with Rust).

**What we will provide:**
```python
# Binary serialization (fast, compact)
data = bloom.to_bytes()
bloom = BloomFilter.from_bytes(data)

# JSON serialization (human-readable, debuggable)
data = bloom.to_json()
bloom = BloomFilter.from_json(data)

# File I/O
bloom.save("filter.hazy")
bloom = BloomFilter.load("filter.hazy")

# Memory-mapped (for huge filters)
bloom = BloomFilter.mmap("filter.hazy")
```

**Future consideration:** If there's significant demand, we could add a `hazy[redis]` extra that provides a `RedisBloomFilter` class using RedisBloom protocol directly.

---

### 4. Async Support → **No (Not Needed)**

**Decision:** Do not provide async variants of the core API.

**Rationale:**
- **CPU-bound, not I/O-bound**: All probabilistic data structure operations are pure computation (hashing + bit manipulation). There's nothing to `await`.
- **No blocking**: Operations complete in microseconds. There's no benefit to yielding to the event loop.
- **Async file I/O is separate**: If users want async file loading, they can use `aiofiles` + our `from_bytes()`:
  ```python
  async with aiofiles.open("filter.hazy", "rb") as f:
      data = await f.read()
  bloom = BloomFilter.from_bytes(data)  # Sync, but fast
  ```
- **Complexity cost**: Async APIs double the API surface for no performance benefit.
- **Precedent**: No other probabilistic data structure library provides async APIs.

**Exception:** If we later add a `RedisBloomFilter` class that talks to Redis, *that* would have async variants since Redis I/O is async-friendly. But that's a future extension, not core functionality.

---

### Summary Table

| Question | Decision | Key Reason |
|----------|----------|------------|
| Rust vs C | **Rust (PyO3)** | Safety + modern tooling, negligible perf diff |
| NumPy | **Optional dep, first-class when present** | Zero-dep core, optimized batch ops for power users |
| Redis compat | **No** | No spec, different use case, maintenance burden |
| Async | **No** | CPU-bound ops, no benefit, complexity cost |

---

## Appendix: Reference Implementations

- **BoomFilters** (Go): github.com/tylertreat/BoomFilters
- **datasketches** (Java/C++): datasketches.apache.org  
- **pdsa** (Python): github.com/gakhov/pdsa
- **rust-hyperloglog**: github.com/jedisct1/rust-hyperloglog

---

*Last updated: December 2025*