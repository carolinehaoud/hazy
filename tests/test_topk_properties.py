"""
Property-based tests for TopK using Hypothesis.

These tests verify mathematical invariants and properties that should always hold.
"""

import pytest
from hypothesis import given, settings, assume, note
from hypothesis import strategies as st
from collections import Counter

from hazy import TopK


# Strategies for generating test data
items_strategy = st.lists(
    st.text(min_size=1, max_size=20),
    min_size=0,
    max_size=500,
)

k_strategy = st.integers(min_value=1, max_value=50)


class TestTopKProperties:
    """Property-based tests for TopK."""

    @given(items=items_strategy, k=k_strategy)
    @settings(max_examples=50)
    def test_top_never_exceeds_k(self, items, k):
        """top() should never return more than k items."""
        tk = TopK(k=k)

        for item in items:
            tk.add(item)

        results = tk.top(k)
        assert len(results) <= k

    @given(items=items_strategy, k=k_strategy)
    @settings(max_examples=50)
    def test_counts_are_positive(self, items, k):
        """All counts in top() should be positive."""
        tk = TopK(k=k)

        for item in items:
            tk.add(item)

        for item, count in tk.top(k):
            assert count > 0, f"Count for {item!r} is {count}, expected > 0"

    @given(items=items_strategy, k=k_strategy)
    @settings(max_examples=50)
    def test_results_sorted_by_count(self, items, k):
        """top() results should be sorted by count in descending order."""
        tk = TopK(k=k)

        for item in items:
            tk.add(item)

        results = tk.top(k)
        if len(results) > 1:
            counts = [count for _, count in results]
            assert counts == sorted(counts, reverse=True), \
                f"Results not sorted: {counts}"

    @given(items=items_strategy, k=k_strategy)
    @settings(max_examples=30)
    def test_serialization_roundtrip(self, items, k):
        """Serialization should preserve top items."""
        tk = TopK(k=k)

        for item in items:
            tk.add(item)

        original_top = tk.top(k)

        # Binary roundtrip
        data = tk.to_bytes()
        tk2 = TopK.from_bytes(data)

        assert tk2.top(k) == original_top

    @given(items=items_strategy, k=k_strategy)
    @settings(max_examples=30)
    def test_json_roundtrip(self, items, k):
        """JSON serialization should preserve top items."""
        tk = TopK(k=k)

        for item in items:
            tk.add(item)

        original_top = tk.top(k)

        # JSON roundtrip
        json_str = tk.to_json()
        tk2 = TopK.from_json(json_str)

        assert tk2.top(k) == original_top

    @given(k=k_strategy)
    @settings(max_examples=20)
    def test_empty_topk_returns_empty(self, k):
        """Empty TopK should return empty list."""
        tk = TopK(k=k)
        assert tk.top(k) == []

    @given(st.integers(min_value=1, max_value=100), k=k_strategy)
    @settings(max_examples=30)
    def test_single_item_repeated(self, n, k):
        """Single item repeated n times should have count n."""
        tk = TopK(k=k)

        item = "test_item"
        for _ in range(n):
            tk.add(item)

        results = tk.top(1)
        assert len(results) == 1
        assert results[0][0] == item
        assert results[0][1] == n

    @given(items=items_strategy, k=k_strategy)
    @settings(max_examples=30)
    def test_clear_empties_topk(self, items, k):
        """After clear(), TopK should be empty."""
        tk = TopK(k=k)

        for item in items:
            tk.add(item)

        tk.clear()
        assert tk.top(k) == []

    @given(items=items_strategy, k=k_strategy)
    @settings(max_examples=30)
    def test_copy_is_independent(self, items, k):
        """Copy should be independent of original."""
        tk1 = TopK(k=k)

        for item in items:
            tk1.add(item)

        tk2 = tk1.copy()
        original_top = tk1.top(k)

        # Add items only to tk2
        for i in range(100):
            tk2.add(f"unique_{i}")

        # Original should not be affected
        assert tk1.top(k) == original_top


class TestTopKAccuracy:
    """Tests for TopK accuracy properties."""

    @given(st.integers(min_value=100, max_value=1000))
    @settings(max_examples=10)
    def test_finds_most_frequent(self, n_items):
        """TopK should find the most frequent items."""
        tk = TopK(k=10)

        # Create items with known frequency distribution
        # item_0: n times, item_1: n-1 times, etc.
        items = []
        for i in range(10):
            items.extend([f"item_{i}"] * (n_items - i * 10))

        for item in items:
            tk.add(item)

        results = tk.top(10)
        top_items = [item for item, _ in results]

        note(f"Top items: {top_items}")

        # Most frequent item should definitely be in top
        assert "item_0" in top_items

        # Most frequent should be first
        assert results[0][0] == "item_0"

    @given(st.integers(min_value=5, max_value=20))
    @settings(max_examples=10)
    def test_exact_counts_for_small_k(self, k):
        """For small streams, counts should be exact."""
        tk = TopK(k=k)

        # Add fewer unique items than k
        actual_counts = Counter()
        for i in range(k - 1):
            count = (i + 1) * 10
            for _ in range(count):
                tk.add(f"item_{i}")
                actual_counts[f"item_{i}"] += 1

        results = dict(tk.top(k))

        note(f"Actual: {actual_counts}")
        note(f"TopK: {results}")

        # Counts should match exactly when under capacity
        for item, actual in actual_counts.items():
            assert results.get(item) == actual, \
                f"Count mismatch for {item}: expected {actual}, got {results.get(item)}"


class TestTopKWithError:
    """Tests for error bounds in TopK."""

    @given(items=items_strategy, k=k_strategy)
    @settings(max_examples=30)
    def test_error_bounds_non_negative(self, items, k):
        """Error bounds should always be non-negative."""
        tk = TopK(k=k)

        for item in items:
            tk.add(item)

        for item, count, error in tk.top_with_error(k):
            assert error >= 0, f"Negative error bound for {item}: {error}"

    @given(items=items_strategy, k=k_strategy)
    @settings(max_examples=30)
    def test_count_minus_error_is_lower_bound(self, items, k):
        """count - error should be a lower bound on true count."""
        assume(len(items) > 0)

        tk = TopK(k=k)

        # Count actual occurrences
        actual_counts = Counter(items)

        for item in items:
            tk.add(item)

        for item, count, error in tk.top_with_error(k):
            lower_bound = count - error
            actual = actual_counts[item]

            # Lower bound should not exceed actual count
            assert lower_bound <= actual, \
                f"Lower bound {lower_bound} > actual {actual} for {item}"
