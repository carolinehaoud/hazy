"""
Property-based tests for CuckooFilter using Hypothesis.

These tests verify mathematical invariants and properties that should always hold.
"""

import pytest
from hypothesis import given, settings, assume, note
from hypothesis import strategies as st

from hazy import CuckooFilter


# Strategies for generating test data
items_strategy = st.lists(
    st.text(min_size=1, max_size=50),
    min_size=0,
    max_size=100,
)


class TestCuckooFilterProperties:
    """Property-based tests for CuckooFilter."""

    @given(items=items_strategy)
    @settings(max_examples=50)
    def test_no_false_negatives(self, items):
        """A Cuckoo filter should never have false negatives.

        If we add an item successfully, it must always be found.
        """
        cf = CuckooFilter(capacity=max(len(items) * 2, 100))

        added_items = []
        for item in items:
            if cf.add(item):  # add() returns True if successful
                added_items.append(item)

        for item in added_items:
            assert item in cf, f"False negative: {item!r} not found after adding"

    @given(items=items_strategy)
    @settings(max_examples=50)
    def test_delete_removes_items(self, items):
        """Deleted items should not be found (with high probability)."""
        assume(len(items) > 0)

        cf = CuckooFilter(capacity=max(len(items) * 2, 100))

        # Add all items
        added = []
        for item in items:
            if cf.add(item):
                added.append(item)

        assume(len(added) > 0)

        # Delete all items
        for item in added:
            cf.remove(item)

        # Most items should not be found
        # (Some might still appear due to fingerprint collisions)
        still_found = sum(1 for item in added if item in cf)
        note(f"Added {len(added)}, still found after delete: {still_found}")

        # Allow some false positives, but most should be gone
        assert still_found <= len(added) * 0.2 + 5

    @given(items=items_strategy)
    @settings(max_examples=30)
    def test_serialization_roundtrip(self, items):
        """Serialization should preserve all state."""
        cf = CuckooFilter(capacity=max(len(items) * 2, 100))

        added = []
        for item in items:
            if cf.add(item):
                added.append(item)

        # Binary roundtrip
        data = cf.to_bytes()
        cf2 = CuckooFilter.from_bytes(data)

        # All items should still be present
        for item in added:
            assert item in cf2, f"Item {item!r} lost in binary serialization"

    @given(items=items_strategy)
    @settings(max_examples=30)
    def test_json_roundtrip(self, items):
        """JSON serialization should preserve all state."""
        cf = CuckooFilter(capacity=max(len(items) * 2, 100))

        added = []
        for item in items:
            if cf.add(item):
                added.append(item)

        # JSON roundtrip
        json_str = cf.to_json()
        cf2 = CuckooFilter.from_json(json_str)

        # All items should still be present
        for item in added:
            assert item in cf2, f"Item {item!r} lost in JSON serialization"

    @given(items=items_strategy)
    @settings(max_examples=30)
    def test_len_matches_added(self, items):
        """len() should match number of successfully added unique items."""
        cf = CuckooFilter(capacity=max(len(items) * 2, 100))

        added_count = 0
        seen = set()
        for item in items:
            if item not in seen:
                if cf.add(item):
                    added_count += 1
                    seen.add(item)

        assert len(cf) == added_count

    @given(st.integers(min_value=100, max_value=1000))
    @settings(max_examples=20)
    def test_empty_filter_has_no_items(self, capacity):
        """An empty filter should report len() of 0."""
        cf = CuckooFilter(capacity=capacity)
        assert len(cf) == 0

    @given(items=items_strategy)
    @settings(max_examples=30)
    def test_clear_removes_all(self, items):
        """After clear(), filter should be empty."""
        cf = CuckooFilter(capacity=max(len(items) * 2, 100))

        for item in items:
            cf.add(item)

        cf.clear()

        assert len(cf) == 0

    @given(items=items_strategy)
    @settings(max_examples=30)
    def test_copy_is_independent(self, items):
        """Copy should be independent of original."""
        cf1 = CuckooFilter(capacity=max(len(items) * 2, 100))

        for item in items:
            cf1.add(item)

        cf2 = cf1.copy()
        original_len = len(cf1)

        # Add new items only to cf2
        for i in range(50):
            cf2.add(f"unique_to_cf2_{i}")

        # Original should not be affected
        assert len(cf1) == original_len


class TestCuckooFilterCapacity:
    """Tests for Cuckoo filter capacity behavior."""

    @given(st.integers(min_value=10, max_value=100))
    @settings(max_examples=20)
    def test_capacity_limit(self, capacity):
        """Filter should stop accepting items when full."""
        cf = CuckooFilter(capacity=capacity)

        # Try to add way more items than capacity
        successful = 0
        for i in range(capacity * 10):
            if cf.add(f"item_{i}"):
                successful += 1

        note(f"Capacity: {capacity}, Successfully added: {successful}")

        # Should have added approximately capacity items
        # (could be slightly more due to cuckoo relocation)
        assert successful >= capacity * 0.5
        assert successful <= capacity * 2

    @given(st.integers(min_value=50, max_value=200))
    @settings(max_examples=10)
    def test_load_factor(self, capacity):
        """Load factor should be reasonable before filter fills."""
        cf = CuckooFilter(capacity=capacity)

        # Fill to about 80% capacity
        target = int(capacity * 0.8)
        for i in range(target):
            cf.add(f"item_{i}")

        load = cf.load_factor
        note(f"Target: {target}, Actual len: {len(cf)}, Load factor: {load}")

        # Load factor should be less than 1.0
        assert load <= 1.0
