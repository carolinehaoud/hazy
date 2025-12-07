"""Tests for Counting Bloom filter."""

import pytest
from hazy import CountingBloomFilter


class TestCountingBloomFilter:
    """Tests for CountingBloomFilter."""

    def test_basic_usage(self):
        """Test basic add and contains operations."""
        cbf = CountingBloomFilter(expected_items=1000)

        cbf.add("hello")
        cbf.add("world")

        assert "hello" in cbf
        assert "world" in cbf
        assert "foo" not in cbf

    def test_remove(self):
        """Test removal of items."""
        cbf = CountingBloomFilter(expected_items=100)

        cbf.add("test")
        assert "test" in cbf

        result = cbf.remove("test")
        assert result is True
        assert "test" not in cbf

    def test_remove_nonexistent(self):
        """Test removing an item that doesn't exist."""
        cbf = CountingBloomFilter(expected_items=100)

        result = cbf.remove("never_added")
        assert result is False

    def test_multiple_adds(self):
        """Test that multiple adds work correctly."""
        cbf = CountingBloomFilter(expected_items=100)

        cbf.add("test")
        cbf.add("test")
        cbf.add("test")

        # Still contains after first removal
        cbf.remove("test")
        assert "test" in cbf

        cbf.remove("test")
        assert "test" in cbf

        cbf.remove("test")
        assert "test" not in cbf

    def test_update_batch(self):
        """Test adding multiple items at once."""
        cbf = CountingBloomFilter(expected_items=1000)
        items = ["item1", "item2", "item3"]
        cbf.update(items)

        for item in items:
            assert item in cbf

    def test_len(self):
        """Test len() returns count of added items."""
        cbf = CountingBloomFilter(expected_items=100)
        assert len(cbf) == 0

        cbf.add("a")
        assert len(cbf) == 1

        cbf.add("b")
        assert len(cbf) == 2

        cbf.remove("a")
        assert len(cbf) == 1

    def test_clear(self):
        """Test clearing the filter."""
        cbf = CountingBloomFilter(expected_items=100)
        cbf.add("test")
        assert "test" in cbf

        cbf.clear()
        assert "test" not in cbf
        assert len(cbf) == 0

    def test_copy(self):
        """Test deep copy."""
        cbf1 = CountingBloomFilter(expected_items=100)
        cbf1.add("test")

        cbf2 = cbf1.copy()
        assert "test" in cbf2

        cbf2.add("new")
        assert "new" not in cbf1

    def test_merge(self):
        """Test merging two filters."""
        cbf1 = CountingBloomFilter(expected_items=100, seed=42)
        cbf1.add("a")

        cbf2 = CountingBloomFilter(expected_items=100, seed=42)
        cbf2.add("b")

        cbf1.merge(cbf2)
        assert "a" in cbf1
        assert "b" in cbf1

    def test_serialization_bytes(self):
        """Test to_bytes/from_bytes."""
        cbf1 = CountingBloomFilter(expected_items=100)
        cbf1.add("test")

        data = cbf1.to_bytes()
        cbf2 = CountingBloomFilter.from_bytes(data)

        assert "test" in cbf2
        assert cbf1.num_counters == cbf2.num_counters

    def test_serialization_json(self):
        """Test to_json/from_json."""
        cbf1 = CountingBloomFilter(expected_items=100)
        cbf1.add("test")

        data = cbf1.to_json()
        cbf2 = CountingBloomFilter.from_json(data)

        assert "test" in cbf2

    def test_properties(self):
        """Test property getters."""
        cbf = CountingBloomFilter(expected_items=1000)

        assert cbf.num_counters > 0
        assert cbf.num_hashes > 0
        assert cbf.size_in_bytes > 0
        assert cbf.seed == 0

    def test_validation_errors(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            CountingBloomFilter()

        with pytest.raises(ValueError):
            CountingBloomFilter(expected_items=0)

        with pytest.raises(ValueError):
            CountingBloomFilter(expected_items=100, false_positive_rate=0)

    def test_merge_validation(self):
        """Test that merge validates parameters."""
        cbf1 = CountingBloomFilter(expected_items=100)
        cbf2 = CountingBloomFilter(expected_items=200)

        with pytest.raises(ValueError):
            cbf1.merge(cbf2)
