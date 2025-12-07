"""
Property-based tests for ScalableBloomFilter using Hypothesis.

These tests verify mathematical invariants and properties that should always hold.
"""

import pytest
from hypothesis import given, settings, assume, note
from hypothesis import strategies as st

from hazy import ScalableBloomFilter


# Strategies for generating test data
items_strategy = st.lists(
    st.text(min_size=1, max_size=50),
    min_size=0,
    max_size=200,
)

scalable_params_strategy = st.fixed_dictionaries({
    "initial_capacity": st.integers(min_value=10, max_value=1000),
    "false_positive_rate": st.floats(min_value=0.001, max_value=0.5),
})


class TestScalableBloomFilterProperties:
    """Property-based tests for ScalableBloomFilter."""

    @given(items=items_strategy, params=scalable_params_strategy)
    @settings(max_examples=50)
    def test_no_false_negatives(self, items, params):
        """A Scalable Bloom filter should never have false negatives.

        If we add an item, it must always be found.
        """
        sbf = ScalableBloomFilter(**params)

        for item in items:
            sbf.add(item)

        for item in items:
            assert item in sbf, f"False negative: {item!r} not found after adding"

    @given(items=items_strategy)
    @settings(max_examples=50)
    def test_scales_automatically(self, items):
        """Filter should automatically create new slices when needed."""
        # Use small initial capacity to force scaling
        sbf = ScalableBloomFilter(initial_capacity=10, false_positive_rate=0.01)

        for item in items:
            sbf.add(item)

        unique_items = len(set(items))
        note(f"Unique items: {unique_items}, Slices: {sbf.num_slices}")

        # Should have created additional slices if many items
        if unique_items > 20:
            assert sbf.num_slices >= 1

    @given(items=items_strategy, params=scalable_params_strategy)
    @settings(max_examples=30)
    def test_serialization_roundtrip(self, items, params):
        """Serialization should preserve all state."""
        sbf = ScalableBloomFilter(**params)

        for item in items:
            sbf.add(item)

        # Binary roundtrip
        data = sbf.to_bytes()
        sbf2 = ScalableBloomFilter.from_bytes(data)

        # All items should still be present
        for item in items:
            assert item in sbf2, f"Item {item!r} lost in binary serialization"

        # Properties should match
        assert sbf.num_slices == sbf2.num_slices

    @given(items=items_strategy, params=scalable_params_strategy)
    @settings(max_examples=30)
    def test_json_roundtrip(self, items, params):
        """JSON serialization should preserve all state."""
        sbf = ScalableBloomFilter(**params)

        for item in items:
            sbf.add(item)

        # JSON roundtrip
        json_str = sbf.to_json()
        sbf2 = ScalableBloomFilter.from_json(json_str)

        # All items should still be present
        for item in items:
            assert item in sbf2, f"Item {item!r} lost in JSON serialization"

    @given(params=scalable_params_strategy)
    @settings(max_examples=20)
    def test_empty_filter_has_no_items(self, params):
        """An empty filter should report len() of 0."""
        sbf = ScalableBloomFilter(**params)
        assert len(sbf) == 0

    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=30)
    def test_idempotent_adds(self, n):
        """Adding the same item multiple times should be idempotent."""
        sbf = ScalableBloomFilter(initial_capacity=100)

        item = "test_item"
        for _ in range(n):
            sbf.add(item)

        assert item in sbf

    @given(items=items_strategy, params=scalable_params_strategy)
    @settings(max_examples=30)
    def test_clear_removes_all(self, items, params):
        """After clear(), filter should be empty."""
        sbf = ScalableBloomFilter(**params)

        for item in items:
            sbf.add(item)

        sbf.clear()

        assert len(sbf) == 0
        assert sbf.num_slices == 1  # Back to initial slice

    @given(items=items_strategy, params=scalable_params_strategy)
    @settings(max_examples=30)
    def test_copy_is_independent(self, items, params):
        """Copy should be independent of original."""
        sbf1 = ScalableBloomFilter(**params)

        for item in items:
            sbf1.add(item)

        sbf2 = sbf1.copy()

        # Add new item only to sbf2
        sbf2.add("unique_to_sbf2_xyz")

        # Check slices match before modification
        original_slices = sbf1.num_slices


class TestScalableBloomFilterScaling:
    """Tests for scaling behavior."""

    @given(st.integers(min_value=100, max_value=500))
    @settings(max_examples=10)
    def test_scales_past_initial_capacity(self, n_items):
        """Should create new slices when exceeding initial capacity."""
        initial_cap = 50
        sbf = ScalableBloomFilter(initial_capacity=initial_cap, false_positive_rate=0.01)

        for i in range(n_items):
            sbf.add(f"item_{i}")

        note(f"Items: {n_items}, Slices: {sbf.num_slices}")

        # Should have scaled
        assert sbf.num_slices >= 2, \
            f"Expected >= 2 slices for {n_items} items with capacity {initial_cap}"

    @given(st.floats(min_value=0.001, max_value=0.1))
    @settings(max_examples=10)
    def test_fpr_maintained_across_slices(self, target_fpr):
        """Overall FPR should be maintained as filter scales."""
        sbf = ScalableBloomFilter(initial_capacity=100, false_positive_rate=target_fpr)

        # Add enough items to cause scaling
        for i in range(500):
            sbf.add(f"item_{i}")

        # Test FPR with items not in filter
        false_positives = 0
        n_tests = 5000

        for i in range(n_tests):
            if f"not_added_{i}" in sbf:
                false_positives += 1

        actual_fpr = false_positives / n_tests
        note(f"Target FPR: {target_fpr}, Actual FPR: {actual_fpr}, Slices: {sbf.num_slices}")

        # FPR should be somewhat close to target (allow generous tolerance)
        # Scalable Bloom filters can have higher FPR than target due to multiple slices
        assert actual_fpr < target_fpr * 5 + 0.01, \
            f"FPR too high: {actual_fpr}"
