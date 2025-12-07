"""Tests for TopK (Space-Saving)."""

import pytest
from hazy import TopK


class TestTopK:
    """Tests for TopK."""

    def test_basic_usage(self):
        """Test basic add and query operations."""
        tk = TopK(k=10)

        tk.add("apple")
        tk.add("apple")
        tk.add("banana")

        assert tk.query("apple") == 2
        assert tk.query("banana") == 1
        assert tk.query("missing") == 0

    def test_add_count(self):
        """Test adding with specific count."""
        tk = TopK(k=10)

        tk.add_count("test", 100)
        assert tk.query("test") >= 100

    def test_contains(self):
        """Test contains method."""
        tk = TopK(k=10)

        tk.add("test")
        assert tk.contains("test")
        assert "test" in tk
        assert "missing" not in tk

    def test_getitem(self):
        """Test [] operator for querying."""
        tk = TopK(k=10)
        tk.add("test")

        assert tk["test"] >= 1
        assert tk["missing"] == 0

    def test_update_batch(self):
        """Test adding multiple items at once."""
        tk = TopK(k=10)
        items = ["a", "b", "c", "a", "a"]
        tk.update(items)

        assert tk.query("a") >= 3
        assert tk.query("b") >= 1
        assert tk.query("c") >= 1

    def test_len(self):
        """Test len() returns number of tracked items."""
        tk = TopK(k=5)

        tk.add("a")
        assert len(tk) == 1

        tk.add("b")
        assert len(tk) == 2

        # Add more than k items
        for i in range(10):
            tk.add(f"item_{i}")

        assert len(tk) == 5  # Limited by k

    def test_top(self):
        """Test getting top items."""
        tk = TopK(k=10)

        tk.add_count("first", 100)
        tk.add_count("second", 50)
        tk.add_count("third", 25)

        top = tk.top()
        assert len(top) == 3
        assert top[0][0] == "first"
        assert top[0][1] >= 100

        # Test limiting
        top2 = tk.top(2)
        assert len(top2) == 2

    def test_top_with_error(self):
        """Test getting top items with error bounds."""
        tk = TopK(k=10)

        tk.add_count("test", 50)
        top = tk.top_with_error()

        assert len(top) == 1
        item, count, error = top[0]
        assert item == "test"
        assert count >= 50
        assert error >= 0

    def test_min_count(self):
        """Test min_count method."""
        tk = TopK(k=3)

        tk.add_count("a", 100)
        tk.add_count("b", 50)
        tk.add_count("c", 25)

        assert tk.min_count() == 25

    def test_clear(self):
        """Test clearing the tracker."""
        tk = TopK(k=10)
        tk.add("test")
        assert len(tk) == 1

        tk.clear()
        assert len(tk) == 0
        assert tk.query("test") == 0

    def test_copy(self):
        """Test deep copy."""
        tk1 = TopK(k=10)
        tk1.add("test")

        tk2 = tk1.copy()
        assert tk2.query("test") >= 1

        tk2.add("new")
        assert tk1.query("new") == 0

    def test_merge(self):
        """Test merging two trackers."""
        tk1 = TopK(k=10)
        tk2 = TopK(k=10)

        tk1.add_count("a", 10)
        tk2.add_count("b", 20)

        tk1.merge(tk2)
        assert tk1.query("a") >= 10
        assert tk1.query("b") >= 20

    def test_serialization_bytes(self):
        """Test to_bytes/from_bytes."""
        tk1 = TopK(k=10)
        tk1.add_count("test", 50)

        data = tk1.to_bytes()
        tk2 = TopK.from_bytes(data)

        assert tk2.query("test") >= 50

    def test_serialization_json(self):
        """Test to_json/from_json."""
        tk1 = TopK(k=10)
        tk1.add("test")

        data = tk1.to_json()
        tk2 = TopK.from_json(data)

        assert tk2.query("test") >= 1

    def test_properties(self):
        """Test property getters."""
        tk = TopK(k=10)

        assert tk.k == 10
        assert tk.size_in_bytes >= 0

    def test_eviction(self):
        """Test that items are evicted when at capacity."""
        tk = TopK(k=3)

        # Add items with increasing counts
        tk.add_count("rare", 1)
        tk.add_count("common", 10)
        tk.add_count("very_common", 100)

        # Add a new item with low count
        tk.add("new_item")

        # Should have evicted the rare item
        top = tk.top()
        items = [t[0] for t in top]
        assert "very_common" in items
        assert "common" in items

    def test_frequency_tracking(self):
        """Test that frequent items are tracked correctly."""
        tk = TopK(k=5)

        # Simulate Zipf-like distribution
        for _ in range(1000):
            tk.add("most_common")
        for _ in range(100):
            tk.add("second")
        for _ in range(10):
            tk.add("third")
        for _ in range(1):
            tk.add("rare")

        top = tk.top(3)
        items = [t[0] for t in top]

        assert items[0] == "most_common"
        assert top[0][1] >= 1000

    def test_validation_errors(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            TopK(k=0)

    def test_heavy_hitters_guarantee(self):
        """Test the heavy hitter guarantee."""
        tk = TopK(k=10)
        n = 1000

        # Add items with known frequencies
        for _ in range(n // 2):
            tk.add("heavy")  # 50% of stream

        for i in range(n // 2):
            tk.add(f"light_{i}")  # Many unique items

        # Heavy hitter (>= n/k) should be tracked
        assert tk.contains("heavy")
        assert tk.query("heavy") >= n // 2
