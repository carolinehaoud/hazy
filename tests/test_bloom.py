"""Tests for Bloom filter."""

import pytest
from hazy import BloomFilter, estimate_bloom_params


class TestBloomFilter:
    """Tests for BloomFilter."""

    def test_basic_usage(self):
        """Test basic add and contains operations."""
        bf = BloomFilter(expected_items=1000)

        bf.add("hello")
        bf.add("world")

        assert "hello" in bf
        assert "world" in bf
        assert "foo" not in bf

    def test_in_operator(self):
        """Test that 'in' operator works."""
        bf = BloomFilter(expected_items=100)
        bf.add("test")

        assert "test" in bf
        assert "missing" not in bf

    def test_update_batch(self):
        """Test adding multiple items at once."""
        bf = BloomFilter(expected_items=1000)
        items = ["item1", "item2", "item3"]
        bf.update(items)

        for item in items:
            assert item in bf

    def test_false_positive_rate(self):
        """Test that FPR is within expected bounds."""
        expected_items = 10000
        target_fpr = 0.01

        bf = BloomFilter(expected_items=expected_items, false_positive_rate=target_fpr)

        # Add items
        for i in range(expected_items):
            bf.add(f"item_{i}")

        # Test false positives
        fp_count = 0
        test_count = 10000
        for i in range(test_count):
            if f"not_item_{i}" in bf:
                fp_count += 1

        actual_fpr = fp_count / test_count
        # Allow 3x margin for statistical variance
        assert actual_fpr < target_fpr * 3

    def test_len(self):
        """Test len() returns count of added items."""
        bf = BloomFilter(expected_items=100)
        assert len(bf) == 0

        bf.add("a")
        assert len(bf) == 1

        bf.add("b")
        assert len(bf) == 2

    def test_clear(self):
        """Test clearing the filter."""
        bf = BloomFilter(expected_items=100)
        bf.add("test")
        assert "test" in bf

        bf.clear()
        assert "test" not in bf
        assert len(bf) == 0

    def test_copy(self):
        """Test deep copy."""
        bf1 = BloomFilter(expected_items=100)
        bf1.add("test")

        bf2 = bf1.copy()
        assert "test" in bf2

        bf2.add("new")
        assert "new" not in bf1

    def test_merge(self):
        """Test merging two filters."""
        bf1 = BloomFilter(expected_items=100, seed=42)
        bf1.add("a")

        bf2 = BloomFilter(expected_items=100, seed=42)
        bf2.add("b")

        bf1.merge(bf2)
        assert "a" in bf1
        assert "b" in bf1

    def test_union_operator(self):
        """Test | operator for union."""
        bf1 = BloomFilter(expected_items=100, seed=42)
        bf1.add("a")

        bf2 = BloomFilter(expected_items=100, seed=42)
        bf2.add("b")

        bf3 = bf1 | bf2
        assert "a" in bf3
        assert "b" in bf3

    def test_serialization_bytes(self):
        """Test to_bytes/from_bytes."""
        bf1 = BloomFilter(expected_items=100)
        bf1.add("test")

        data = bf1.to_bytes()
        bf2 = BloomFilter.from_bytes(data)

        assert "test" in bf2
        assert bf1.num_bits == bf2.num_bits

    def test_serialization_json(self):
        """Test to_json/from_json."""
        bf1 = BloomFilter(expected_items=100)
        bf1.add("test")

        data = bf1.to_json()
        bf2 = BloomFilter.from_json(data)

        assert "test" in bf2

    def test_properties(self):
        """Test property getters."""
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)

        assert bf.num_bits > 0
        assert bf.num_hashes > 0
        assert bf.size_in_bytes > 0
        assert bf.seed == 0

    def test_fill_ratio(self):
        """Test fill ratio calculation."""
        bf = BloomFilter(expected_items=100)
        assert bf.fill_ratio() == 0.0

        for i in range(50):
            bf.add(f"item_{i}")

        assert 0 < bf.fill_ratio() < 1

    def test_explicit_parameters(self):
        """Test creating with explicit num_bits and num_hashes."""
        bf = BloomFilter(num_bits=1000, num_hashes=5)
        assert bf.num_bits == 1000
        assert bf.num_hashes == 5

    def test_seed_affects_hashing(self):
        """Test that different seeds produce different results."""
        bf1 = BloomFilter(expected_items=100, seed=1)
        bf2 = BloomFilter(expected_items=100, seed=2)

        bf1.add("test")
        bf2.add("test")

        # Both should contain the item
        assert "test" in bf1
        assert "test" in bf2

        # But bit patterns should differ
        assert bf1.to_json() != bf2.to_json()

    def test_validation_errors(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            BloomFilter()  # No expected_items or num_bits

        with pytest.raises(ValueError):
            BloomFilter(expected_items=0)

        with pytest.raises(ValueError):
            BloomFilter(expected_items=100, false_positive_rate=0)

        with pytest.raises(ValueError):
            BloomFilter(expected_items=100, false_positive_rate=1)

    def test_merge_validation(self):
        """Test that merge validates parameters."""
        bf1 = BloomFilter(expected_items=100)
        bf2 = BloomFilter(expected_items=200)

        with pytest.raises(ValueError):
            bf1.merge(bf2)


class TestBloomParamsEstimation:
    """Tests for parameter estimation."""

    def test_estimate_bloom_params(self):
        """Test parameter estimation."""
        params = estimate_bloom_params(1_000_000, 0.01)

        assert params.num_bits > 0
        assert params.num_hashes > 0
        assert params.memory_bytes > 0
        assert params.memory_mb > 0
        assert 0 < params.expected_fpr < 0.02  # Should be close to target

    def test_lower_fpr_needs_more_memory(self):
        """Test that lower FPR requires more memory."""
        params_1pct = estimate_bloom_params(1000, 0.01)
        params_01pct = estimate_bloom_params(1000, 0.001)

        assert params_01pct.memory_bytes > params_1pct.memory_bytes

    def test_more_items_needs_more_memory(self):
        """Test that more items require more memory."""
        params_1k = estimate_bloom_params(1000, 0.01)
        params_10k = estimate_bloom_params(10000, 0.01)

        assert params_10k.memory_bytes > params_1k.memory_bytes
