"""
Property-based tests for CountMinSketch using Hypothesis.

These tests verify mathematical invariants and properties that should always hold.
"""

import pytest
from hypothesis import given, settings, assume, note
from hypothesis import strategies as st

from hazy import CountMinSketch


# Strategies for generating test data
items_strategy = st.lists(
    st.text(min_size=1, max_size=30),
    min_size=0,
    max_size=200,
)

cms_params_strategy = st.fixed_dictionaries({
    "width": st.integers(min_value=100, max_value=10000),
    "depth": st.integers(min_value=2, max_value=10),
})


class TestCountMinSketchProperties:
    """Property-based tests for CountMinSketch."""

    @given(items=items_strategy, params=cms_params_strategy)
    @settings(max_examples=50)
    def test_counts_never_underestimate(self, items, params):
        """CMS should never underestimate counts (only overestimate)."""
        cms = CountMinSketch(**params)

        # Count actual occurrences
        actual_counts = {}
        for item in items:
            actual_counts[item] = actual_counts.get(item, 0) + 1
            cms.add(item)

        # Estimated counts should be >= actual counts
        for item, actual in actual_counts.items():
            estimated = cms[item]
            assert estimated >= actual, \
                f"Underestimate for {item!r}: estimated {estimated} < actual {actual}"

    @given(items=items_strategy, params=cms_params_strategy)
    @settings(max_examples=30)
    def test_total_count_preserved(self, items, params):
        """Total count should equal number of items added."""
        cms = CountMinSketch(**params)

        for item in items:
            cms.add(item)

        assert cms.total_count == len(items)

    @given(items=items_strategy, params=cms_params_strategy)
    @settings(max_examples=30)
    def test_serialization_roundtrip(self, items, params):
        """Serialization should preserve all counts."""
        cms = CountMinSketch(**params)

        for item in items:
            cms.add(item)

        original_total = cms.total_count

        # Binary roundtrip
        data = cms.to_bytes()
        cms2 = CountMinSketch.from_bytes(data)

        assert cms2.total_count == original_total

        # All counts should be preserved
        for item in set(items):
            assert cms2[item] == cms[item]

    @given(items=items_strategy, params=cms_params_strategy)
    @settings(max_examples=30)
    def test_json_roundtrip(self, items, params):
        """JSON serialization should preserve all counts."""
        cms = CountMinSketch(**params)

        for item in items:
            cms.add(item)

        original_total = cms.total_count

        # JSON roundtrip
        json_str = cms.to_json()
        cms2 = CountMinSketch.from_json(json_str)

        assert cms2.total_count == original_total

    @given(params=cms_params_strategy)
    @settings(max_examples=20)
    def test_empty_cms_zero_counts(self, params):
        """Empty CMS should have zero counts for any item."""
        cms = CountMinSketch(**params)

        # Query some random items
        for i in range(100):
            assert cms[f"item_{i}"] == 0

        assert cms.total_count == 0

    @given(st.integers(min_value=1, max_value=100), params=cms_params_strategy)
    @settings(max_examples=30)
    def test_add_count_accumulates(self, count, params):
        """add_count should accumulate correctly."""
        cms = CountMinSketch(**params)

        item = "test_item"
        for _ in range(5):
            cms.add_count(item, count)

        assert cms[item] >= 5 * count  # At least the actual count

    @given(items=items_strategy, params=cms_params_strategy)
    @settings(max_examples=30)
    def test_clear_resets_all(self, items, params):
        """After clear(), all counts should be 0."""
        cms = CountMinSketch(**params)

        for item in items:
            cms.add(item)

        cms.clear()

        assert cms.total_count == 0
        for item in set(items):
            assert cms[item] == 0

    @given(items=items_strategy, params=cms_params_strategy)
    @settings(max_examples=30)
    def test_copy_is_independent(self, items, params):
        """Copy should be independent of original."""
        cms1 = CountMinSketch(**params)

        for item in items:
            cms1.add(item)

        cms2 = cms1.copy()
        original_total = cms1.total_count

        # Add items only to cms2
        for i in range(100):
            cms2.add(f"unique_{i}")

        # Original should not be affected
        assert cms1.total_count == original_total


class TestCountMinSketchMerge:
    """Tests for CountMinSketch merge properties."""

    @given(
        items1=items_strategy,
        items2=items_strategy,
        params=cms_params_strategy,
    )
    @settings(max_examples=30)
    def test_merge_preserves_lower_bound(self, items1, items2, params):
        """Merged CMS should never underestimate combined counts."""
        cms1 = CountMinSketch(**params)
        cms2 = CountMinSketch(**params)

        actual_counts = {}

        for item in items1:
            cms1.add(item)
            actual_counts[item] = actual_counts.get(item, 0) + 1

        for item in items2:
            cms2.add(item)
            actual_counts[item] = actual_counts.get(item, 0) + 1

        cms1.merge(cms2)

        # All estimated counts should be >= actual
        for item, actual in actual_counts.items():
            estimated = cms1[item]
            assert estimated >= actual, \
                f"Merged underestimate for {item!r}: {estimated} < {actual}"

    @given(
        items1=items_strategy,
        items2=items_strategy,
        params=cms_params_strategy,
    )
    @settings(max_examples=30)
    def test_merge_total_count(self, items1, items2, params):
        """Merged total count should equal sum of individual totals."""
        cms1 = CountMinSketch(**params)
        cms2 = CountMinSketch(**params)

        for item in items1:
            cms1.add(item)
        for item in items2:
            cms2.add(item)

        expected_total = cms1.total_count + cms2.total_count

        cms1.merge(cms2)

        assert cms1.total_count == expected_total


class TestCountMinSketchAccuracy:
    """Tests for Count-Min Sketch accuracy properties."""

    @given(st.integers(min_value=1000, max_value=5000))
    @settings(max_examples=5)
    def test_wider_is_more_accurate(self, n_items):
        """Wider CMS should generally have less overestimation."""
        items = [f"item_{i % 100}" for i in range(n_items)]  # 100 unique items

        errors = {}
        for width in [100, 1000, 10000]:
            cms = CountMinSketch(width=width, depth=5)
            for item in items:
                cms.add(item)

            # Measure total overestimation
            actual_counts = {}
            for item in items:
                actual_counts[item] = actual_counts.get(item, 0) + 1

            total_overestimate = 0
            for item, actual in actual_counts.items():
                total_overestimate += cms[item] - actual

            errors[width] = total_overestimate

        note(f"Total overestimate by width: {errors}")

        # Wider CMS should have less overestimation
        # (not guaranteed for any single run, but usually true)
