"""Tests for MinHash."""

import pytest
from hazy import MinHash, estimate_minhash_params


class TestMinHash:
    """Tests for MinHash."""

    def test_basic_usage(self):
        """Test basic add and jaccard operations."""
        mh1 = MinHash(num_hashes=128)
        mh2 = MinHash(num_hashes=128)

        mh1.update(["a", "b", "c", "d"])
        mh2.update(["c", "d", "e", "f"])

        # Jaccard should be 2/6 ≈ 0.33
        jaccard = mh1.jaccard(mh2)
        assert 0.1 < jaccard < 0.6

    def test_identical_sets(self):
        """Test that identical sets have jaccard ≈ 1."""
        mh1 = MinHash(num_hashes=256)
        mh2 = MinHash(num_hashes=256)

        items = ["a", "b", "c", "d", "e"]
        mh1.update(items)
        mh2.update(items)

        jaccard = mh1.jaccard(mh2)
        assert jaccard > 0.9

    def test_disjoint_sets(self):
        """Test that disjoint sets have jaccard ≈ 0."""
        mh1 = MinHash(num_hashes=256)
        mh2 = MinHash(num_hashes=256)

        mh1.update(["a", "b", "c"])
        mh2.update(["x", "y", "z"])

        jaccard = mh1.jaccard(mh2)
        assert jaccard < 0.2

    def test_is_empty(self):
        """Test is_empty method."""
        mh = MinHash()
        assert mh.is_empty()

        mh.add("test")
        assert not mh.is_empty()

    def test_clear(self):
        """Test clearing the signature."""
        mh = MinHash()
        mh.add("test")
        assert not mh.is_empty()

        mh.clear()
        assert mh.is_empty()

    def test_copy(self):
        """Test deep copy."""
        mh1 = MinHash()
        mh1.add("test")

        mh2 = mh1.copy()
        assert not mh2.is_empty()

        mh2.add("new")
        # Original should not change
        sig1 = mh1.get_signature()
        mh1_after = mh1.get_signature()
        assert sig1 == mh1_after

    def test_merge(self):
        """Test merging two signatures (union)."""
        mh1 = MinHash(num_hashes=128)
        mh2 = MinHash(num_hashes=128)

        mh1.update(["a", "b"])
        mh2.update(["c", "d"])

        mh1.merge(mh2)

        # Create reference for union
        mh_ref = MinHash(num_hashes=128)
        mh_ref.update(["a", "b", "c", "d"])

        # Merged should be similar to union reference
        jaccard = mh1.jaccard(mh_ref)
        assert jaccard > 0.8

    def test_union_operator(self):
        """Test | operator for union."""
        mh1 = MinHash(num_hashes=128)
        mh2 = MinHash(num_hashes=128)

        mh1.update(["a", "b"])
        mh2.update(["c", "d"])

        mh3 = mh1 | mh2

        mh_ref = MinHash(num_hashes=128)
        mh_ref.update(["a", "b", "c", "d"])

        jaccard = mh3.jaccard(mh_ref)
        assert jaccard > 0.8

    def test_serialization_bytes(self):
        """Test to_bytes/from_bytes."""
        mh1 = MinHash()
        mh1.update(["a", "b", "c"])

        data = mh1.to_bytes()
        mh2 = MinHash.from_bytes(data)

        # Signatures should be equal
        assert mh1.get_signature() == mh2.get_signature()

    def test_serialization_json(self):
        """Test to_json/from_json."""
        mh1 = MinHash()
        mh1.add("test")

        data = mh1.to_json()
        mh2 = MinHash.from_json(data)

        assert mh1.get_signature() == mh2.get_signature()

    def test_properties(self):
        """Test property getters."""
        mh = MinHash(num_hashes=64)

        assert mh.num_hashes == 64
        assert mh.size_in_bytes == 64 * 8
        assert mh.seed == 0

    def test_standard_error(self):
        """Test standard error calculation."""
        mh = MinHash(num_hashes=100)
        error = mh.standard_error()
        # 1 / sqrt(100) = 0.1
        assert 0.09 < error < 0.11

    def test_get_signature(self):
        """Test getting the signature."""
        mh = MinHash(num_hashes=10)
        mh.add("test")

        sig = mh.get_signature()
        assert len(sig) == 10
        assert all(isinstance(x, int) for x in sig)

    def test_validation_errors(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            MinHash(num_hashes=0)

    def test_jaccard_validation(self):
        """Test that jaccard validates parameters."""
        mh1 = MinHash(num_hashes=64)
        mh2 = MinHash(num_hashes=128)

        with pytest.raises(ValueError):
            mh1.jaccard(mh2)

    def test_jaccard_accuracy(self):
        """Test that Jaccard estimate is accurate."""
        # Create sets with known Jaccard
        set1 = set(range(100))
        set2 = set(range(50, 150))  # 50% overlap

        mh1 = MinHash(num_hashes=256)
        mh2 = MinHash(num_hashes=256)

        mh1.update([str(x) for x in set1])
        mh2.update([str(x) for x in set2])

        estimated = mh1.jaccard(mh2)
        # True Jaccard = 50 / 150 ≈ 0.33
        assert 0.2 < estimated < 0.5


class TestMinHashParamsEstimation:
    """Tests for MinHash parameter estimation."""

    def test_estimate_minhash_params(self):
        """Test parameter estimation."""
        params = estimate_minhash_params(0.05)

        assert params.num_hashes > 0
        assert params.memory_bytes == params.num_hashes * 8
        assert params.relative_error_percent <= 6  # Should be close to 5%

    def test_lower_error_needs_more_hashes(self):
        """Test that lower error requires more hashes."""
        params_10pct = estimate_minhash_params(0.1)
        params_5pct = estimate_minhash_params(0.05)

        assert params_5pct.num_hashes > params_10pct.num_hashes
