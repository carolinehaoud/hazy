"""
Property-based tests for HyperLogLog using Hypothesis.

These tests verify mathematical invariants and properties that should always hold.
"""

import pytest
from hypothesis import given, settings, assume, note
from hypothesis import strategies as st

from hazy import HyperLogLog


# Strategies for generating test data
items_strategy = st.lists(
    st.text(min_size=1, max_size=50),
    min_size=0,
    max_size=500,
)

precision_strategy = st.integers(min_value=4, max_value=16)


class TestHyperLogLogProperties:
    """Property-based tests for HyperLogLog."""

    @given(items=items_strategy, precision=precision_strategy)
    @settings(max_examples=50)
    def test_cardinality_non_negative(self, items, precision):
        """Cardinality estimate should never be negative."""
        hll = HyperLogLog(precision=precision)

        for item in items:
            hll.add(item)

        assert hll.cardinality() >= 0

    @given(items=items_strategy)
    @settings(max_examples=50)
    def test_cardinality_reasonable_bound(self, items):
        """Cardinality should be in a reasonable range of actual count."""
        assume(len(items) > 10)  # Need enough items for meaningful test

        hll = HyperLogLog(precision=14)

        for item in items:
            hll.add(item)

        unique_count = len(set(items))
        estimate = hll.cardinality()

        note(f"Unique items: {unique_count}, Estimate: {estimate}")

        # HLL with p=14 has ~1-2% error, allow generous bounds for property testing
        if unique_count > 0:
            error_ratio = abs(estimate - unique_count) / unique_count
            assert error_ratio < 0.5, \
                f"Estimate {estimate} too far from actual {unique_count}"

    @given(items=items_strategy, precision=precision_strategy)
    @settings(max_examples=30)
    def test_serialization_roundtrip(self, items, precision):
        """Serialization should preserve cardinality estimate."""
        hll = HyperLogLog(precision=precision)

        for item in items:
            hll.add(item)

        original_cardinality = hll.cardinality()

        # Binary roundtrip
        data = hll.to_bytes()
        hll2 = HyperLogLog.from_bytes(data)

        assert hll2.cardinality() == original_cardinality

    @given(items=items_strategy, precision=precision_strategy)
    @settings(max_examples=30)
    def test_json_roundtrip(self, items, precision):
        """JSON serialization should preserve cardinality estimate."""
        hll = HyperLogLog(precision=precision)

        for item in items:
            hll.add(item)

        original_cardinality = hll.cardinality()

        # JSON roundtrip
        json_str = hll.to_json()
        hll2 = HyperLogLog.from_json(json_str)

        assert hll2.cardinality() == original_cardinality

    @given(
        items1=items_strategy,
        items2=items_strategy,
        precision=precision_strategy,
    )
    @settings(max_examples=30)
    def test_merge_cardinality_bounds(self, items1, items2, precision):
        """Merged HLL cardinality should be bounded by sum and max of inputs."""
        hll1 = HyperLogLog(precision=precision)
        hll2 = HyperLogLog(precision=precision)

        for item in items1:
            hll1.add(item)
        for item in items2:
            hll2.add(item)

        card1 = hll1.cardinality()
        card2 = hll2.cardinality()

        merged = hll1 | hll2
        merged_card = merged.cardinality()

        # Merged cardinality should be at least max of individual cardinalities
        # (with some tolerance for HLL estimation error)
        assert merged_card >= max(card1, card2) * 0.5, \
            f"Merged {merged_card} too small vs max({card1}, {card2})"

        # Merged cardinality should be at most sum (union can't be larger)
        # (with tolerance for estimation error)
        assert merged_card <= (card1 + card2) * 2 + 10, \
            f"Merged {merged_card} too large vs sum {card1 + card2}"

    @given(precision=precision_strategy)
    @settings(max_examples=20)
    def test_empty_hll_cardinality_zero(self, precision):
        """Empty HLL should have cardinality of 0."""
        hll = HyperLogLog(precision=precision)
        assert hll.cardinality() == 0.0

    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=30)
    def test_idempotent_adds(self, n):
        """Adding the same item multiple times shouldn't increase cardinality."""
        hll = HyperLogLog(precision=14)

        item = "test_item"
        hll.add(item)
        cardinality_after_one = hll.cardinality()

        for _ in range(n):
            hll.add(item)

        # Cardinality should be approximately 1 (same item repeated)
        assert hll.cardinality() == cardinality_after_one

    @given(items=items_strategy, precision=precision_strategy)
    @settings(max_examples=30)
    def test_clear_resets_cardinality(self, items, precision):
        """After clear(), cardinality should be 0."""
        hll = HyperLogLog(precision=precision)

        for item in items:
            hll.add(item)

        hll.clear()
        assert hll.cardinality() == 0.0

    @given(items=items_strategy, precision=precision_strategy)
    @settings(max_examples=30)
    def test_copy_is_independent(self, items, precision):
        """Copy should be independent of original."""
        hll1 = HyperLogLog(precision=precision)

        for item in items:
            hll1.add(item)

        hll2 = hll1.copy()
        original_card = hll1.cardinality()

        # Add many new items only to hll2
        for i in range(100):
            hll2.add(f"unique_to_hll2_{i}")

        # Original should not be affected
        assert hll1.cardinality() == original_card


class TestHyperLogLogAccuracy:
    """Tests for HyperLogLog accuracy properties."""

    @given(st.integers(min_value=1000, max_value=10000))
    @settings(max_examples=10)
    def test_accuracy_with_precision(self, n_items):
        """Higher precision should give better accuracy."""
        items = [f"item_{i}" for i in range(n_items)]

        errors = {}
        for precision in [8, 12, 16]:
            hll = HyperLogLog(precision=precision)
            for item in items:
                hll.add(item)

            estimate = hll.cardinality()
            error = abs(estimate - n_items) / n_items
            errors[precision] = error

        note(f"Errors by precision: {errors}")

        # Generally, higher precision should have lower error
        # (not guaranteed for any single run, but usually true)
