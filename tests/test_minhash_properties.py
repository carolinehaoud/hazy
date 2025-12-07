"""
Property-based tests for MinHash using Hypothesis.

These tests verify mathematical invariants and properties that should always hold.
"""

import pytest
from hypothesis import given, settings, assume, note
from hypothesis import strategies as st

from hazy import MinHash


# Strategies for generating test data
items_strategy = st.lists(
    st.text(min_size=1, max_size=30),
    min_size=0,
    max_size=100,
)

num_hashes_strategy = st.integers(min_value=16, max_value=256)


class TestMinHashProperties:
    """Property-based tests for MinHash."""

    @given(items=items_strategy, num_hashes=num_hashes_strategy)
    @settings(max_examples=50)
    def test_jaccard_range(self, items, num_hashes):
        """Jaccard estimate should always be in [0, 1]."""
        mh1 = MinHash(num_hashes=num_hashes)
        mh2 = MinHash(num_hashes=num_hashes)

        for item in items:
            mh1.add(item)
            mh2.add(item + "_suffix")  # Different items

        jaccard = mh1.jaccard(mh2)
        assert 0.0 <= jaccard <= 1.0, f"Jaccard {jaccard} out of range [0, 1]"

    @given(items=items_strategy, num_hashes=num_hashes_strategy)
    @settings(max_examples=50)
    def test_jaccard_with_self_is_one(self, items, num_hashes):
        """Jaccard of a set with itself should be 1.0."""
        assume(len(items) > 0)  # Need at least one item

        mh1 = MinHash(num_hashes=num_hashes)
        mh2 = MinHash(num_hashes=num_hashes)

        for item in items:
            mh1.add(item)
            mh2.add(item)

        jaccard = mh1.jaccard(mh2)
        assert jaccard == 1.0, f"Jaccard with identical sets should be 1.0, got {jaccard}"

    @given(items=items_strategy, num_hashes=num_hashes_strategy)
    @settings(max_examples=30)
    def test_serialization_roundtrip(self, items, num_hashes):
        """Serialization should preserve signature."""
        mh = MinHash(num_hashes=num_hashes)

        for item in items:
            mh.add(item)

        # Binary roundtrip
        data = mh.to_bytes()
        mh2 = MinHash.from_bytes(data)

        # Jaccard with itself should be 1.0
        assert mh.jaccard(mh2) == 1.0

    @given(items=items_strategy, num_hashes=num_hashes_strategy)
    @settings(max_examples=30)
    def test_json_roundtrip(self, items, num_hashes):
        """JSON serialization should preserve signature."""
        mh = MinHash(num_hashes=num_hashes)

        for item in items:
            mh.add(item)

        # JSON roundtrip
        json_str = mh.to_json()
        mh2 = MinHash.from_json(json_str)

        # Jaccard with itself should be 1.0
        assert mh.jaccard(mh2) == 1.0

    @given(num_hashes=num_hashes_strategy)
    @settings(max_examples=20)
    def test_empty_minhash_jaccard_zero(self, num_hashes):
        """Empty MinHash should have Jaccard 0 with non-empty MinHash."""
        mh1 = MinHash(num_hashes=num_hashes)  # Empty
        mh2 = MinHash(num_hashes=num_hashes)
        mh2.add("item")

        # Empty vs non-empty should be 0
        jaccard = mh1.jaccard(mh2)
        assert jaccard == 0.0, f"Empty vs non-empty should be 0, got {jaccard}"

    @given(st.integers(min_value=1, max_value=100), num_hashes=num_hashes_strategy)
    @settings(max_examples=30)
    def test_idempotent_adds(self, n, num_hashes):
        """Adding the same item multiple times shouldn't change signature."""
        mh = MinHash(num_hashes=num_hashes)

        item = "test_item"
        mh.add(item)
        signature_after_one = mh.signature()

        for _ in range(n):
            mh.add(item)

        assert mh.signature() == signature_after_one

    @given(items=items_strategy, num_hashes=num_hashes_strategy)
    @settings(max_examples=30)
    def test_clear_resets_signature(self, items, num_hashes):
        """After clear(), MinHash should be empty."""
        mh = MinHash(num_hashes=num_hashes)

        for item in items:
            mh.add(item)

        mh.clear()

        # Empty signature - all max values
        sig = mh.signature()
        assert all(v == 0xFFFFFFFFFFFFFFFF for v in sig)

    @given(items=items_strategy, num_hashes=num_hashes_strategy)
    @settings(max_examples=30)
    def test_copy_is_independent(self, items, num_hashes):
        """Copy should be independent of original."""
        mh1 = MinHash(num_hashes=num_hashes)

        for item in items:
            mh1.add(item)

        mh2 = mh1.copy()
        original_sig = mh1.signature()

        # Add items only to mh2
        for i in range(100):
            mh2.add(f"unique_{i}")

        # Original should not be affected
        assert mh1.signature() == original_sig

    @given(items=items_strategy, num_hashes=num_hashes_strategy)
    @settings(max_examples=30)
    def test_jaccard_symmetric(self, items, num_hashes):
        """Jaccard should be symmetric: J(A,B) = J(B,A)."""
        mh1 = MinHash(num_hashes=num_hashes)
        mh2 = MinHash(num_hashes=num_hashes)

        # Add different items to each
        for i, item in enumerate(items):
            if i % 2 == 0:
                mh1.add(item)
            else:
                mh2.add(item)

        assert mh1.jaccard(mh2) == mh2.jaccard(mh1)


class TestMinHashMerge:
    """Tests for MinHash merge properties."""

    @given(
        items1=items_strategy,
        items2=items_strategy,
        num_hashes=num_hashes_strategy,
    )
    @settings(max_examples=30)
    def test_merge_is_union(self, items1, items2, num_hashes):
        """Merged MinHash should represent union of sets."""
        mh1 = MinHash(num_hashes=num_hashes)
        mh2 = MinHash(num_hashes=num_hashes)
        mh_union = MinHash(num_hashes=num_hashes)

        for item in items1:
            mh1.add(item)
            mh_union.add(item)

        for item in items2:
            mh2.add(item)
            mh_union.add(item)

        mh1.merge(mh2)

        # Merged should match union
        assert mh1.jaccard(mh_union) == 1.0


class TestMinHashAccuracy:
    """Tests for MinHash accuracy properties."""

    @given(st.floats(min_value=0.1, max_value=0.9))
    @settings(max_examples=10)
    def test_jaccard_estimation_accuracy(self, overlap_ratio):
        """Test Jaccard estimation accuracy for known overlap."""
        n_items = 1000
        n_overlap = int(n_items * overlap_ratio)

        mh1 = MinHash(num_hashes=256)
        mh2 = MinHash(num_hashes=256)

        # Items 0 to n_overlap-1 are shared
        for i in range(n_overlap):
            mh1.add(f"shared_{i}")
            mh2.add(f"shared_{i}")

        # Items n_overlap to n_items-1 are unique to each
        for i in range(n_overlap, n_items):
            mh1.add(f"unique1_{i}")
            mh2.add(f"unique2_{i}")

        # True Jaccard = n_overlap / (2*n_items - n_overlap)
        true_jaccard = n_overlap / (2 * n_items - n_overlap)
        estimated_jaccard = mh1.jaccard(mh2)

        note(f"Overlap: {overlap_ratio}, True Jaccard: {true_jaccard:.3f}, Estimated: {estimated_jaccard:.3f}")

        # Allow reasonable error margin
        error = abs(estimated_jaccard - true_jaccard)
        assert error < 0.15, \
            f"Jaccard error too high: {error:.3f} (true={true_jaccard:.3f}, est={estimated_jaccard:.3f})"
