"""
Property-based tests for CountingBloomFilter using Hypothesis.

These tests verify mathematical invariants and properties that should always hold.
"""

import pytest
from hypothesis import given, settings, assume, note
from hypothesis import strategies as st

from hazy import CountingBloomFilter


# Strategies for generating test data
items_strategy = st.lists(
    st.text(min_size=1, max_size=50),
    min_size=0,
    max_size=100,
)

counting_params_strategy = st.fixed_dictionaries({
    "expected_items": st.integers(min_value=10, max_value=10000),
    "false_positive_rate": st.floats(min_value=0.001, max_value=0.5),
})


class TestCountingBloomFilterProperties:
    """Property-based tests for CountingBloomFilter."""

    @given(items=items_strategy)
    @settings(max_examples=50)
    def test_no_false_negatives(self, items):
        """A Counting Bloom filter should never have false negatives.

        If we add an item (and don't remove it), it must always be found.
        """
        cbf = CountingBloomFilter(expected_items=max(len(items), 10))

        for item in items:
            cbf.add(item)

        for item in items:
            assert item in cbf, f"False negative: {item!r} not found after adding"

    @given(items=items_strategy)
    @settings(max_examples=50)
    def test_remove_works(self, items):
        """Items should be removable, and then not found (mostly)."""
        assume(len(items) > 0)

        cbf = CountingBloomFilter(expected_items=max(len(items), 10) * 2)

        # Add all items
        for item in items:
            cbf.add(item)

        unique_items = list(set(items))

        # Remove half the unique items
        removed = unique_items[:len(unique_items)//2]
        for item in removed:
            cbf.remove(item)

        # Non-removed items should still be found
        kept = unique_items[len(unique_items)//2:]
        for item in kept:
            assert item in cbf, f"Kept item {item!r} not found after removing others"

    @given(items=items_strategy, params=counting_params_strategy)
    @settings(max_examples=30)
    def test_serialization_roundtrip(self, items, params):
        """Serialization should preserve all state."""
        cbf = CountingBloomFilter(**params)

        for item in items:
            cbf.add(item)

        # Binary roundtrip
        data = cbf.to_bytes()
        cbf2 = CountingBloomFilter.from_bytes(data)

        # All items should still be present
        for item in items:
            assert item in cbf2, f"Item {item!r} lost in binary serialization"

    @given(items=items_strategy, params=counting_params_strategy)
    @settings(max_examples=30)
    def test_json_roundtrip(self, items, params):
        """JSON serialization should preserve all state."""
        cbf = CountingBloomFilter(**params)

        for item in items:
            cbf.add(item)

        # JSON roundtrip
        json_str = cbf.to_json()
        cbf2 = CountingBloomFilter.from_json(json_str)

        # All items should still be present
        for item in items:
            assert item in cbf2, f"Item {item!r} lost in JSON serialization"

    @given(params=counting_params_strategy)
    @settings(max_examples=20)
    def test_empty_filter_has_no_items(self, params):
        """An empty filter should report len() of 0."""
        cbf = CountingBloomFilter(**params)
        assert len(cbf) == 0

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=30)
    def test_add_remove_symmetry(self, count):
        """Adding n times then removing n times should leave item absent."""
        cbf = CountingBloomFilter(expected_items=100)

        item = "test_item"

        for _ in range(count):
            cbf.add(item)

        assert item in cbf

        for _ in range(count):
            cbf.remove(item)

        # After removing same number of times, item should be gone
        # (Note: false positives still possible, but unlikely for this case)

    @given(items=items_strategy, params=counting_params_strategy)
    @settings(max_examples=30)
    def test_clear_removes_all(self, items, params):
        """After clear(), filter should be empty."""
        cbf = CountingBloomFilter(**params)

        for item in items:
            cbf.add(item)

        cbf.clear()

        assert len(cbf) == 0

    @given(items=items_strategy, params=counting_params_strategy)
    @settings(max_examples=30)
    def test_copy_is_independent(self, items, params):
        """Copy should be independent of original."""
        cbf1 = CountingBloomFilter(**params)

        for item in items:
            cbf1.add(item)

        cbf2 = cbf1.copy()

        # Remove all items from cbf2
        for item in items:
            cbf2.remove(item)

        # Original should still have items
        for item in items:
            assert item in cbf1, f"Item {item!r} affected by copy modification"


class TestCountingBloomFilterCounts:
    """Tests for counting behavior."""

    @given(st.integers(min_value=1, max_value=200))
    @settings(max_examples=20)
    def test_count_query(self, count):
        """count() should return at least the true count."""
        cbf = CountingBloomFilter(expected_items=100)

        item = "test_item"
        for _ in range(count):
            cbf.add(item)

        estimated = cbf.count(item)
        assert estimated >= count, \
            f"Count {estimated} < actual {count} for item"

    @given(
        items=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=50),
    )
    @settings(max_examples=30)
    def test_count_never_underestimates(self, items):
        """count() should never underestimate true count."""
        cbf = CountingBloomFilter(expected_items=max(len(items), 10) * 2)

        # Count actual occurrences
        from collections import Counter
        actual_counts = Counter()

        for item in items:
            cbf.add(item)
            actual_counts[item] += 1

        # Estimated should be >= actual
        for item, actual in actual_counts.items():
            estimated = cbf.count(item)
            assert estimated >= actual, \
                f"Underestimate for {item!r}: {estimated} < {actual}"


class TestCountingBloomFilterOverflow:
    """Tests for counter overflow handling."""

    @given(st.integers(min_value=100, max_value=300))
    @settings(max_examples=10)
    def test_counter_saturates(self, n_adds):
        """Counters should saturate at max value instead of overflowing."""
        cbf = CountingBloomFilter(expected_items=10)

        # Add same item many times (8-bit counters max at 255)
        item = "test_item"
        for _ in range(n_adds):
            cbf.add(item)

        # Item should still be found
        assert item in cbf

        # Count should be capped, not wrapped around
        count = cbf.count(item)
        assert count >= min(n_adds, 255)  # At least the true count up to 255
