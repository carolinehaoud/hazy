"""Tests for Scalable Bloom filter."""

import pytest
from hazy import ScalableBloomFilter


class TestScalableBloomFilter:
    """Tests for ScalableBloomFilter."""

    def test_basic_usage(self):
        """Test basic add and contains operations."""
        sbf = ScalableBloomFilter(initial_capacity=100)

        sbf.add("hello")
        sbf.add("world")

        assert "hello" in sbf
        assert "world" in sbf
        assert "foo" not in sbf

    def test_auto_scaling(self):
        """Test that the filter scales automatically."""
        sbf = ScalableBloomFilter(initial_capacity=100)

        # Add many more items than initial capacity
        for i in range(1000):
            sbf.add(f"item_{i}")

        # Should have created multiple slices
        assert sbf.num_slices > 1

        # All items should still be found
        for i in range(1000):
            assert f"item_{i}" in sbf

    def test_update_batch(self):
        """Test adding multiple items at once."""
        sbf = ScalableBloomFilter(initial_capacity=100)
        items = ["item1", "item2", "item3"]
        sbf.update(items)

        for item in items:
            assert item in sbf

    def test_len(self):
        """Test len() returns count of added items."""
        sbf = ScalableBloomFilter(initial_capacity=100)
        assert len(sbf) == 0

        sbf.add("a")
        assert len(sbf) == 1

        sbf.add("b")
        assert len(sbf) == 2

    def test_clear(self):
        """Test clearing the filter."""
        sbf = ScalableBloomFilter(initial_capacity=100)
        sbf.add("test")
        assert "test" in sbf

        sbf.clear()
        assert "test" not in sbf
        assert len(sbf) == 0
        assert sbf.num_slices == 0

    def test_copy(self):
        """Test deep copy."""
        sbf1 = ScalableBloomFilter(initial_capacity=100)
        sbf1.add("test")

        sbf2 = sbf1.copy()
        assert "test" in sbf2

        sbf2.add("new")
        assert "new" not in sbf1

    def test_serialization_bytes(self):
        """Test to_bytes/from_bytes."""
        sbf1 = ScalableBloomFilter(initial_capacity=100)
        for i in range(200):  # Force scaling
            sbf1.add(f"item_{i}")

        data = sbf1.to_bytes()
        sbf2 = ScalableBloomFilter.from_bytes(data)

        assert sbf2.num_slices == sbf1.num_slices
        for i in range(200):
            assert f"item_{i}" in sbf2

    def test_capacity(self):
        """Test capacity calculation."""
        sbf = ScalableBloomFilter(initial_capacity=100)
        sbf.add("test")

        cap = sbf.capacity()
        assert cap >= 100

    def test_false_positive_rate(self):
        """Test FPR calculation."""
        sbf = ScalableBloomFilter(initial_capacity=100, false_positive_rate=0.01)

        for i in range(100):
            sbf.add(f"item_{i}")

        fpr = sbf.false_positive_rate()
        assert 0 < fpr < 0.1

    def test_growth_ratio(self):
        """Test that growth ratio affects scaling."""
        sbf_slow = ScalableBloomFilter(initial_capacity=10, growth_ratio=1.5)
        sbf_fast = ScalableBloomFilter(initial_capacity=10, growth_ratio=4.0)

        for i in range(100):
            sbf_slow.add(f"item_{i}")
            sbf_fast.add(f"item_{i}")

        # Faster growth = fewer slices needed
        assert sbf_slow.num_slices > sbf_fast.num_slices

    def test_validation_errors(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            ScalableBloomFilter(initial_capacity=0)

        with pytest.raises(ValueError):
            ScalableBloomFilter(initial_capacity=100, false_positive_rate=0)

        with pytest.raises(ValueError):
            ScalableBloomFilter(initial_capacity=100, growth_ratio=0.5)

        with pytest.raises(ValueError):
            ScalableBloomFilter(initial_capacity=100, fpr_ratio=1.5)

    def test_no_duplicates(self):
        """Test that duplicates don't inflate count."""
        sbf = ScalableBloomFilter(initial_capacity=100)

        for _ in range(100):
            sbf.add("same")

        # Should only count once
        assert len(sbf) == 1

    def test_file_io(self):
        """Test save/load from file."""
        import tempfile
        import os

        sbf1 = ScalableBloomFilter(initial_capacity=100)
        sbf1.add("test")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hazy") as f:
            path = f.name

        try:
            sbf1.save(path)
            sbf2 = ScalableBloomFilter.load(path)
            assert "test" in sbf2
        finally:
            os.unlink(path)
