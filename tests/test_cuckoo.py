"""Tests for Cuckoo filter."""

import pytest
from hazy import CuckooFilter


class TestCuckooFilter:
    """Tests for CuckooFilter."""

    def test_basic_usage(self):
        """Test basic add and contains operations."""
        cf = CuckooFilter(capacity=1000)

        result = cf.add("hello")
        assert result is True
        cf.add("world")

        assert "hello" in cf
        assert "world" in cf
        assert "foo" not in cf

    def test_remove(self):
        """Test removal of items."""
        cf = CuckooFilter(capacity=100)

        cf.add("test")
        assert "test" in cf

        result = cf.remove("test")
        assert result is True
        assert "test" not in cf

    def test_remove_nonexistent(self):
        """Test removing an item that doesn't exist."""
        cf = CuckooFilter(capacity=100)

        result = cf.remove("never_added")
        assert result is False

    def test_update_batch(self):
        """Test adding multiple items at once."""
        cf = CuckooFilter(capacity=1000)
        items = ["item1", "item2", "item3"]
        added = cf.update(items)

        assert added == 3
        for item in items:
            assert item in cf

    def test_len(self):
        """Test len() returns count of added items."""
        cf = CuckooFilter(capacity=100)
        assert len(cf) == 0

        cf.add("a")
        assert len(cf) == 1

        cf.add("b")
        assert len(cf) == 2

        cf.remove("a")
        assert len(cf) == 1

    def test_clear(self):
        """Test clearing the filter."""
        cf = CuckooFilter(capacity=100)
        cf.add("test")
        assert "test" in cf

        cf.clear()
        assert "test" not in cf
        assert len(cf) == 0

    def test_copy(self):
        """Test deep copy."""
        cf1 = CuckooFilter(capacity=100)
        cf1.add("test")

        cf2 = cf1.copy()
        assert "test" in cf2

        cf2.add("new")
        assert "new" not in cf1

    def test_capacity(self):
        """Test capacity method."""
        cf = CuckooFilter(capacity=1000)
        # Actual capacity may be slightly higher due to bucket sizing
        assert cf.capacity() >= 1000

    def test_load_factor(self):
        """Test load factor calculation."""
        cf = CuckooFilter(capacity=100)
        assert cf.load_factor() == 0.0

        cf.add("test")
        assert 0 < cf.load_factor() < 1

    def test_serialization_bytes(self):
        """Test to_bytes/from_bytes."""
        cf1 = CuckooFilter(capacity=100)
        cf1.add("test")

        data = cf1.to_bytes()
        cf2 = CuckooFilter.from_bytes(data)

        assert "test" in cf2

    def test_serialization_json(self):
        """Test to_json/from_json."""
        cf1 = CuckooFilter(capacity=100)
        cf1.add("test")

        data = cf1.to_json()
        cf2 = CuckooFilter.from_json(data)

        assert "test" in cf2

    def test_properties(self):
        """Test property getters."""
        cf = CuckooFilter(capacity=1000)

        assert cf.size_in_bytes > 0
        assert cf.seed == 0

    def test_filter_full(self):
        """Test that filter returns False when full."""
        cf = CuckooFilter(capacity=10)

        # Try to add many items
        added_count = 0
        for i in range(100):
            if cf.add(f"item_{i}"):
                added_count += 1

        # Should have added some but not all
        assert 0 < added_count < 100

    def test_validation_errors(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            CuckooFilter(capacity=0)
