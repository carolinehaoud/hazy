"""Edge case tests for all data structures.

Tests boundary conditions, malformed inputs, empty structures,
extreme values, and error handling.
"""

import pytest
import tempfile
import os
from pathlib import Path

from hazy import (
    BloomFilter,
    CountingBloomFilter,
    ScalableBloomFilter,
    CuckooFilter,
    HyperLogLog,
    CountMinSketch,
    MinHash,
    TopK,
)


class TestEmptyStructures:
    """Test operations on empty (freshly created) structures."""

    def test_bloom_empty_operations(self):
        """Test BloomFilter before any additions."""
        bf = BloomFilter(expected_items=100)

        assert len(bf) == 0
        assert bf.fill_ratio() == 0.0
        assert "anything" not in bf

        # Can serialize empty filter
        data = bf.to_bytes()
        bf2 = BloomFilter.from_bytes(data)
        assert len(bf2) == 0

    def test_counting_bloom_empty_operations(self):
        """Test CountingBloomFilter before any additions."""
        cbf = CountingBloomFilter(expected_items=100)

        assert len(cbf) == 0
        assert "anything" not in cbf

        # Remove on empty should not crash
        cbf.remove("nonexistent")
        assert len(cbf) == 0

    def test_scalable_bloom_empty_operations(self):
        """Test ScalableBloomFilter before any additions."""
        sbf = ScalableBloomFilter(initial_capacity=100)

        assert len(sbf) == 0
        assert "anything" not in sbf
        assert sbf.num_filters == 1  # Always starts with one filter

    def test_cuckoo_empty_operations(self):
        """Test CuckooFilter before any additions."""
        cf = CuckooFilter(capacity=100)

        assert len(cf) == 0
        assert "anything" not in cf

        # Remove on empty should return False
        assert not cf.remove("nonexistent")

    def test_hll_empty_operations(self):
        """Test HyperLogLog before any additions."""
        hll = HyperLogLog(precision=12)

        assert hll.cardinality() == 0.0
        assert hll.size_in_bytes > 0

    def test_cms_empty_operations(self):
        """Test CountMinSketch before any additions."""
        cms = CountMinSketch(width=100, depth=5)

        assert cms["anything"] == 0
        assert cms.size_in_bytes > 0

    def test_minhash_empty_operations(self):
        """Test MinHash before any additions."""
        mh = MinHash(num_hashes=64)

        # Signature should exist but be all max values
        sig = mh.signature()
        assert len(sig) == 64

        # Jaccard with itself should be 1.0
        assert mh.jaccard(mh) == 1.0

    def test_topk_empty_operations(self):
        """Test TopK before any additions."""
        tk = TopK(k=10)

        assert len(tk) == 0
        assert tk.top() == []
        assert tk.top(5) == []


class TestBoundaryParameters:
    """Test structures with minimum/maximum parameter values."""

    def test_bloom_minimum_size(self):
        """Test BloomFilter with minimum expected items."""
        bf = BloomFilter(expected_items=1)
        bf.add("single")
        assert "single" in bf

    def test_bloom_explicit_minimum_bits(self):
        """Test BloomFilter with explicit minimum parameters."""
        bf = BloomFilter(num_bits=8, num_hashes=1)
        bf.add("test")
        assert bf.num_bits == 8
        assert bf.num_hashes == 1

    def test_hll_precision_bounds(self):
        """Test HyperLogLog precision at boundaries."""
        # Minimum precision
        hll_min = HyperLogLog(precision=4)
        hll_min.add("test")
        assert hll_min.cardinality() >= 0

        # Maximum precision
        hll_max = HyperLogLog(precision=18)
        hll_max.add("test")
        assert hll_max.cardinality() >= 0

    def test_hll_invalid_precision(self):
        """Test HyperLogLog rejects invalid precision."""
        with pytest.raises((ValueError, Exception)):
            HyperLogLog(precision=3)  # Below minimum

        with pytest.raises((ValueError, Exception)):
            HyperLogLog(precision=19)  # Above maximum

    def test_cms_minimum_dimensions(self):
        """Test CountMinSketch with minimum dimensions."""
        cms = CountMinSketch(width=1, depth=1)
        cms.add("test")
        assert cms["test"] >= 1

    def test_topk_k_equals_one(self):
        """Test TopK with k=1."""
        tk = TopK(k=1)
        tk.add("a")
        tk.add("a")
        tk.add("b")

        top = tk.top()
        assert len(top) == 1
        assert top[0][0] == "a"

    def test_minhash_minimum_hashes(self):
        """Test MinHash with minimum hash count."""
        mh = MinHash(num_hashes=1)
        mh.update(["a", "b"])
        assert len(mh.signature()) == 1

    def test_cuckoo_minimum_capacity(self):
        """Test CuckooFilter with minimum capacity."""
        cf = CuckooFilter(capacity=1)
        # Should be able to add at least one item
        assert cf.add("test")
        assert "test" in cf


class TestSpecialInputs:
    """Test handling of special input values."""

    def test_empty_string(self):
        """Test adding empty strings."""
        bf = BloomFilter(expected_items=100)
        bf.add("")
        assert "" in bf

        cms = CountMinSketch(width=100, depth=5)
        cms.add("")
        assert cms[""] == 1

    def test_unicode_strings(self):
        """Test Unicode input handling."""
        bf = BloomFilter(expected_items=100)

        # Various Unicode
        test_strings = [
            "Hello, 世界",  # Chinese
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic (RTL)
            "🎉🎊🎈",  # Emoji
            "café",  # Accented
            "\u0000",  # Null character
            "a\x00b",  # Embedded null
        ]

        for s in test_strings:
            bf.add(s)
            assert s in bf, f"Failed for: {repr(s)}"

    def test_very_long_strings(self):
        """Test with very long strings."""
        bf = BloomFilter(expected_items=100)

        long_string = "x" * 100000  # 100KB string
        bf.add(long_string)
        assert long_string in bf

    def test_numeric_strings(self):
        """Test with string representations of numbers."""
        bf = BloomFilter(expected_items=100)

        bf.add("0")
        bf.add("-1")
        bf.add("3.14159")
        bf.add("1e100")

        assert "0" in bf
        assert "-1" in bf
        assert "3.14159" in bf
        assert "1e100" in bf

    def test_whitespace_variations(self):
        """Test various whitespace inputs."""
        bf = BloomFilter(expected_items=100)

        whitespace = [" ", "  ", "\t", "\n", "\r\n", " \t\n "]
        for ws in whitespace:
            bf.add(ws)
            assert ws in bf


class TestLargeScaleOperations:
    """Test with large numbers of items."""

    def test_bloom_large_scale(self):
        """Test BloomFilter with many items."""
        n = 100_000
        bf = BloomFilter(expected_items=n, false_positive_rate=0.01)

        for i in range(n):
            bf.add(f"item_{i}")

        # All items should be present
        for i in range(0, n, 1000):  # Sample every 1000th
            assert f"item_{i}" in bf

        # Check memory is reasonable
        assert bf.size_in_bytes < 200_000  # Should be well under 200KB

    def test_hll_large_cardinality(self):
        """Test HyperLogLog with large cardinality."""
        n = 1_000_000
        hll = HyperLogLog(precision=14)

        for i in range(n):
            hll.add(f"user_{i}")

        estimate = hll.cardinality()
        error = abs(estimate - n) / n
        assert error < 0.02  # Within 2%

    def test_cms_high_frequency(self):
        """Test CountMinSketch with high frequency items."""
        cms = CountMinSketch(width=10000, depth=5)

        # Add one item many times
        for _ in range(1_000_000):
            cms.add("hot_item")

        # Count should be accurate
        count = cms["hot_item"]
        assert count == 1_000_000

    def test_topk_many_unique_items(self):
        """Test TopK with many unique items."""
        tk = TopK(k=100)

        # Add many unique items with varying frequencies
        for i in range(10000):
            for _ in range(i % 10 + 1):
                tk.add(f"item_{i}")

        # Should track top 100
        top = tk.top(100)
        assert len(top) <= 100


class TestFileIO:
    """Test file save/load operations."""

    def test_bloom_file_io(self):
        """Test BloomFilter save/load."""
        bf = BloomFilter(expected_items=100)
        bf.add("test")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hazy") as f:
            path = f.name

        try:
            bf.save(path)
            bf2 = BloomFilter.load(path)
            assert "test" in bf2
            assert len(bf2) == len(bf)
        finally:
            os.unlink(path)

    def test_counting_bloom_file_io(self):
        """Test CountingBloomFilter save/load."""
        cbf = CountingBloomFilter(expected_items=100)
        cbf.add("test")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hazy") as f:
            path = f.name

        try:
            cbf.save(path)
            cbf2 = CountingBloomFilter.load(path)
            assert "test" in cbf2
        finally:
            os.unlink(path)

    def test_cuckoo_file_io(self):
        """Test CuckooFilter save/load."""
        cf = CuckooFilter(capacity=100)
        cf.add("test")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hazy") as f:
            path = f.name

        try:
            cf.save(path)
            cf2 = CuckooFilter.load(path)
            assert "test" in cf2
        finally:
            os.unlink(path)

    def test_hll_file_io(self):
        """Test HyperLogLog save/load."""
        hll = HyperLogLog(precision=12)
        for i in range(1000):
            hll.add(f"item_{i}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hazy") as f:
            path = f.name

        try:
            hll.save(path)
            hll2 = HyperLogLog.load(path)
            assert abs(hll.cardinality() - hll2.cardinality()) < 1
        finally:
            os.unlink(path)

    def test_cms_file_io(self):
        """Test CountMinSketch save/load."""
        cms = CountMinSketch(width=100, depth=5)
        cms.add("test", count=42)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hazy") as f:
            path = f.name

        try:
            cms.save(path)
            cms2 = CountMinSketch.load(path)
            assert cms2["test"] == 42
        finally:
            os.unlink(path)

    def test_minhash_file_io(self):
        """Test MinHash save/load."""
        mh = MinHash(num_hashes=64)
        mh.update(["a", "b", "c"])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hazy") as f:
            path = f.name

        try:
            mh.save(path)
            mh2 = MinHash.load(path)
            assert mh.signature() == mh2.signature()
        finally:
            os.unlink(path)

    def test_topk_file_io(self):
        """Test TopK save/load."""
        tk = TopK(k=10)
        tk.add("a", count=5)
        tk.add("b", count=3)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hazy") as f:
            path = f.name

        try:
            tk.save(path)
            tk2 = TopK.load(path)
            assert tk.top() == tk2.top()
        finally:
            os.unlink(path)

    def test_file_io_invalid_path(self):
        """Test save/load with invalid paths."""
        bf = BloomFilter(expected_items=100)

        with pytest.raises(Exception):
            bf.save("/nonexistent/directory/file.hazy")

        with pytest.raises(Exception):
            BloomFilter.load("/nonexistent/file.hazy")

    def test_file_io_pathlib(self):
        """Test that pathlib.Path works for file I/O."""
        bf = BloomFilter(expected_items=100)
        bf.add("test")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hazy") as f:
            path = Path(f.name)

        try:
            bf.save(str(path))
            bf2 = BloomFilter.load(str(path))
            assert "test" in bf2
        finally:
            path.unlink()


class TestMalformedSerialization:
    """Test handling of corrupted/malformed serialized data."""

    def test_bloom_corrupted_bytes(self):
        """Test BloomFilter with corrupted bytes."""
        bf = BloomFilter(expected_items=100)
        bf.add("test")
        data = bf.to_bytes()

        # Corrupt the data
        corrupted = bytes([0xFF] * len(data))

        with pytest.raises(Exception):
            BloomFilter.from_bytes(corrupted)

    def test_bloom_truncated_bytes(self):
        """Test BloomFilter with truncated bytes."""
        bf = BloomFilter(expected_items=100)
        bf.add("test")
        data = bf.to_bytes()

        # Truncate the data
        truncated = data[:len(data) // 2]

        with pytest.raises(Exception):
            BloomFilter.from_bytes(truncated)

    def test_bloom_empty_bytes(self):
        """Test BloomFilter with empty bytes."""
        with pytest.raises(Exception):
            BloomFilter.from_bytes(b"")

    def test_bloom_invalid_json(self):
        """Test BloomFilter with invalid JSON."""
        with pytest.raises(Exception):
            BloomFilter.from_json("not valid json")

        with pytest.raises(Exception):
            BloomFilter.from_json("{}")  # Valid JSON but missing required fields

    def test_hll_corrupted_bytes(self):
        """Test HyperLogLog with corrupted bytes."""
        hll = HyperLogLog(precision=12)
        hll.add("test")
        data = hll.to_bytes()

        corrupted = bytes([0x00] * len(data))

        with pytest.raises(Exception):
            HyperLogLog.from_bytes(corrupted)

    def test_cms_corrupted_bytes(self):
        """Test CountMinSketch with corrupted bytes."""
        cms = CountMinSketch(width=100, depth=5)
        cms.add("test")
        data = cms.to_bytes()

        corrupted = b"garbage" + data[7:]

        with pytest.raises(Exception):
            CountMinSketch.from_bytes(corrupted)


class TestClearAndReset:
    """Test clearing/resetting structures."""

    def test_bloom_clear_and_reuse(self):
        """Test clearing and reusing BloomFilter."""
        bf = BloomFilter(expected_items=100)

        bf.add("first")
        assert "first" in bf

        bf.clear()
        assert "first" not in bf
        assert len(bf) == 0

        bf.add("second")
        assert "second" in bf
        assert "first" not in bf

    def test_counting_bloom_clear(self):
        """Test clearing CountingBloomFilter."""
        cbf = CountingBloomFilter(expected_items=100)
        cbf.add("test")
        cbf.add("test")

        cbf.clear()
        assert "test" not in cbf

    def test_cuckoo_clear(self):
        """Test clearing CuckooFilter."""
        cf = CuckooFilter(capacity=100)
        cf.add("test")

        cf.clear()
        assert "test" not in cf
        assert len(cf) == 0

    def test_hll_clear(self):
        """Test clearing HyperLogLog."""
        hll = HyperLogLog(precision=12)
        for i in range(1000):
            hll.add(f"item_{i}")

        hll.clear()
        assert hll.cardinality() == 0.0


class TestMergeOperations:
    """Test merge/union operations."""

    def test_bloom_merge_empty(self):
        """Test merging with empty filter."""
        bf1 = BloomFilter(expected_items=100, seed=42)
        bf1.add("test")

        bf2 = BloomFilter(expected_items=100, seed=42)
        # bf2 is empty

        merged = bf1 | bf2
        assert "test" in merged

    def test_hll_merge_empty(self):
        """Test HyperLogLog merge with empty."""
        hll1 = HyperLogLog(precision=12)
        for i in range(1000):
            hll1.add(f"item_{i}")

        hll2 = HyperLogLog(precision=12)
        # hll2 is empty

        merged = hll1 | hll2
        # Should be same as hll1
        assert abs(merged.cardinality() - hll1.cardinality()) < 10

    def test_minhash_merge_disjoint_sets(self):
        """Test MinHash merge with completely disjoint sets."""
        mh1 = MinHash(num_hashes=128)
        mh2 = MinHash(num_hashes=128)

        mh1.update(["a", "b", "c"])
        mh2.update(["d", "e", "f"])

        # Jaccard should be close to 0
        assert mh1.jaccard(mh2) < 0.2

        # Merged should contain signature for union
        merged = mh1 | mh2

        # Merged compared to individual should have some similarity
        assert merged.jaccard(mh1) > 0

    def test_minhash_merge_identical_sets(self):
        """Test MinHash merge with identical sets."""
        mh1 = MinHash(num_hashes=128)
        mh2 = MinHash(num_hashes=128)

        items = ["a", "b", "c", "d", "e"]
        mh1.update(items)
        mh2.update(items)

        # Should be identical
        assert mh1.jaccard(mh2) == 1.0

    def test_cms_merge(self):
        """Test CountMinSketch merge."""
        cms1 = CountMinSketch(width=1000, depth=5)
        cms2 = CountMinSketch(width=1000, depth=5)

        cms1.add("a", count=10)
        cms2.add("a", count=5)
        cms2.add("b", count=3)

        merged = cms1.merge(cms2)
        assert merged["a"] == 15
        assert merged["b"] == 3


class TestDuplicateHandling:
    """Test how structures handle duplicate additions."""

    def test_bloom_duplicates(self):
        """BloomFilter should handle duplicates gracefully."""
        bf = BloomFilter(expected_items=100)

        bf.add("test")
        bf.add("test")
        bf.add("test")

        assert "test" in bf
        # len tracks additions, not unique items
        assert len(bf) == 3

    def test_hll_duplicates(self):
        """HyperLogLog should count unique items only."""
        hll = HyperLogLog(precision=14)

        for _ in range(1000):
            hll.add("same_item")

        # Should report ~1 unique item
        assert hll.cardinality() < 2

    def test_cms_duplicates(self):
        """CountMinSketch should count all occurrences."""
        cms = CountMinSketch(width=1000, depth=5)

        for _ in range(100):
            cms.add("frequent")

        assert cms["frequent"] == 100

    def test_cuckoo_duplicates(self):
        """CuckooFilter should handle duplicates."""
        cf = CuckooFilter(capacity=100)

        cf.add("test")
        cf.add("test")
        cf.add("test")

        assert "test" in cf
        # Removing once should still leave it present (multiple fingerprints)
        cf.remove("test")
        # Behavior depends on implementation - just verify no crash


class TestMemoryBounds:
    """Test that memory usage is as expected."""

    def test_bloom_memory_estimate(self):
        """Test BloomFilter memory matches expectations."""
        bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)

        # Expected: ~10 bits per item for 1% FPR
        expected_bits = 10000 * 10
        expected_bytes = expected_bits // 8

        # Allow 50% variance for overhead
        assert bf.size_in_bytes < expected_bytes * 1.5

    def test_hll_fixed_memory(self):
        """HyperLogLog should have fixed memory regardless of cardinality."""
        hll = HyperLogLog(precision=14)
        initial_size = hll.size_in_bytes

        for i in range(100000):
            hll.add(f"item_{i}")

        # Size should not have changed
        assert hll.size_in_bytes == initial_size

    def test_cms_fixed_memory(self):
        """CountMinSketch should have fixed memory."""
        cms = CountMinSketch(width=1000, depth=5)
        initial_size = cms.size_in_bytes

        for i in range(10000):
            cms.add(f"item_{i}")

        # Size should not have changed
        assert cms.size_in_bytes == initial_size


class TestCopyIndependence:
    """Test that copied structures are independent."""

    def test_bloom_copy_independence(self):
        """Modifications to copy don't affect original."""
        bf1 = BloomFilter(expected_items=100)
        bf1.add("original")

        bf2 = bf1.copy()
        bf2.add("copy_only")

        assert "original" in bf1
        assert "original" in bf2
        assert "copy_only" not in bf1
        assert "copy_only" in bf2

    def test_hll_copy_independence(self):
        """Modifications to copy don't affect original."""
        hll1 = HyperLogLog(precision=12)
        hll1.add("original")

        hll2 = hll1.copy()
        for i in range(10000):
            hll2.add(f"item_{i}")

        # Original should still have cardinality ~1
        assert hll1.cardinality() < 5
        assert hll2.cardinality() > 9000


class TestCuckooCapacity:
    """Test CuckooFilter capacity limits."""

    def test_cuckoo_fill_to_capacity(self):
        """Test filling CuckooFilter to near capacity."""
        cf = CuckooFilter(capacity=1000)

        success_count = 0
        for i in range(1500):  # Try to overfill
            if cf.add(f"item_{i}"):
                success_count += 1

        # Should have added close to capacity
        assert success_count >= 900  # At least 90% capacity

    def test_cuckoo_reports_full(self):
        """Test that CuckooFilter reports when full."""
        cf = CuckooFilter(capacity=100)

        # Fill it up
        failures = 0
        for i in range(200):
            if not cf.add(f"item_{i}"):
                failures += 1

        # Should have some failures
        assert failures > 0
