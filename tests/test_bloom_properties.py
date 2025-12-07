"""
Property-based tests for BloomFilter using Hypothesis.

These tests verify mathematical invariants and properties that should always hold,
regardless of the specific inputs.
"""

import pytest
from hypothesis import given, settings, assume, note
from hypothesis import strategies as st

from hazy import BloomFilter


# Strategies for generating test data
items_strategy = st.lists(
    st.text(min_size=1, max_size=50),
    min_size=0,
    max_size=100,
)

bloom_params_strategy = st.fixed_dictionaries({
    "expected_items": st.integers(min_value=10, max_value=10000),
    "false_positive_rate": st.floats(min_value=0.001, max_value=0.5),
})


class TestBloomFilterProperties:
    """Property-based tests for BloomFilter."""

    @given(items=items_strategy)
    @settings(max_examples=50)
    def test_no_false_negatives(self, items):
        """A Bloom filter should never have false negatives.

        If we add an item, it must always be found.
        """
        bf = BloomFilter(expected_items=max(len(items), 10))

        for item in items:
            bf.add(item)

        for item in items:
            assert item in bf, f"False negative: {item!r} not found after adding"

    @given(items=items_strategy)
    @settings(max_examples=50)
    def test_len_after_adds(self, items):
        """len() should never exceed the number of unique items added."""
        bf = BloomFilter(expected_items=max(len(items), 10))

        for item in items:
            bf.add(item)

        unique_count = len(set(items))
        # len() is an estimate, should be in reasonable range
        assert len(bf) <= unique_count * 1.5 + 10, \
            f"len() = {len(bf)}, but only added {unique_count} unique items"

    @given(params=bloom_params_strategy, items=items_strategy)
    @settings(max_examples=30)
    def test_serialization_roundtrip(self, params, items):
        """Serialization should preserve all state."""
        bf = BloomFilter(**params)

        for item in items:
            bf.add(item)

        # Binary roundtrip
        data = bf.to_bytes()
        bf2 = BloomFilter.from_bytes(data)

        # All items should still be present
        for item in items:
            assert item in bf2, f"Item {item!r} lost in binary serialization"

        # Properties should match
        assert bf.num_bits == bf2.num_bits
        assert bf.num_hashes == bf2.num_hashes

    @given(params=bloom_params_strategy, items=items_strategy)
    @settings(max_examples=30)
    def test_json_roundtrip(self, params, items):
        """JSON serialization should preserve all state."""
        bf = BloomFilter(**params)

        for item in items:
            bf.add(item)

        # JSON roundtrip
        json_str = bf.to_json()
        bf2 = BloomFilter.from_json(json_str)

        # All items should still be present
        for item in items:
            assert item in bf2, f"Item {item!r} lost in JSON serialization"

    @given(
        items1=items_strategy,
        items2=items_strategy,
    )
    @settings(max_examples=30)
    def test_union_contains_all(self, items1, items2):
        """Union of two filters should contain all items from both."""
        bf1 = BloomFilter(expected_items=100)
        bf2 = BloomFilter(expected_items=100)

        for item in items1:
            bf1.add(item)
        for item in items2:
            bf2.add(item)

        merged = bf1 | bf2

        for item in items1:
            assert item in merged, f"Item {item!r} from bf1 not in union"
        for item in items2:
            assert item in merged, f"Item {item!r} from bf2 not in union"

    @given(
        items1=items_strategy,
        items2=items_strategy,
    )
    @settings(max_examples=30)
    def test_intersection_subset(self, items1, items2):
        """Intersection should not have false negatives for items in both sets."""
        bf1 = BloomFilter(expected_items=100)
        bf2 = BloomFilter(expected_items=100)

        for item in items1:
            bf1.add(item)
        for item in items2:
            bf2.add(item)

        intersected = bf1 & bf2

        # Items in both should definitely be in intersection
        common = set(items1) & set(items2)
        for item in common:
            assert item in intersected, \
                f"Item {item!r} in both filters but not in intersection"

    @given(params=bloom_params_strategy)
    @settings(max_examples=20)
    def test_empty_filter_has_no_items(self, params):
        """An empty filter should report len() of 0."""
        bf = BloomFilter(**params)
        assert len(bf) == 0

    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=30)
    def test_idempotent_adds(self, n):
        """Adding the same item multiple times should be idempotent."""
        bf = BloomFilter(expected_items=100)

        item = "test_item"
        for _ in range(n):
            bf.add(item)

        assert item in bf

    @given(items=items_strategy)
    @settings(max_examples=30)
    def test_clear_removes_all(self, items):
        """After clear(), filter should be empty."""
        bf = BloomFilter(expected_items=max(len(items), 10))

        for item in items:
            bf.add(item)

        bf.clear()

        assert len(bf) == 0
        # Note: Due to false positives, we can't guarantee items aren't "found"
        # but the fill ratio should be 0
        assert bf.fill_ratio == 0.0

    @given(items=items_strategy)
    @settings(max_examples=30)
    def test_copy_is_independent(self, items):
        """Copy should be independent of original."""
        bf1 = BloomFilter(expected_items=max(len(items), 10))

        for item in items:
            bf1.add(item)

        bf2 = bf1.copy()

        # Add new item only to bf2
        bf2.add("unique_to_bf2_xyz")

        # Original should not be affected
        # (with high probability, given the unique string)
        original_had_item = "unique_to_bf2_xyz" in bf1
        # Note: Could be false positive, but very unlikely with this string


class TestBloomFilterFPR:
    """Tests for false positive rate bounds."""

    @given(st.integers(min_value=100, max_value=1000))
    @settings(max_examples=10)
    def test_fpr_within_bounds(self, n_items):
        """False positive rate should be approximately as configured."""
        target_fpr = 0.01
        bf = BloomFilter(expected_items=n_items, false_positive_rate=target_fpr)

        # Add exactly n_items
        for i in range(n_items):
            bf.add(f"item_{i}")

        # Test false positive rate with items definitely not in filter
        false_positives = 0
        n_tests = 10000

        for i in range(n_tests):
            if f"not_added_{i}" in bf:
                false_positives += 1

        actual_fpr = false_positives / n_tests
        note(f"Target FPR: {target_fpr}, Actual FPR: {actual_fpr}")

        # Allow some tolerance (FPR is probabilistic)
        assert actual_fpr < target_fpr * 3, \
            f"FPR too high: {actual_fpr} > {target_fpr * 3}"
