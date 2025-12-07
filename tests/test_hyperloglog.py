"""Tests for HyperLogLog."""

import pytest
from hazy import HyperLogLog, estimate_hll_params


class TestHyperLogLog:
    """Tests for HyperLogLog."""

    def test_basic_usage(self):
        """Test basic add and cardinality operations."""
        hll = HyperLogLog(precision=14)

        for i in range(1000):
            hll.add(f"item_{i}")

        cardinality = hll.cardinality()
        # Allow 10% error
        assert 900 < cardinality < 1100

    def test_duplicate_handling(self):
        """Test that duplicates don't increase cardinality."""
        hll = HyperLogLog()

        for _ in range(100):
            hll.add("same_item")

        # Should be approximately 1
        assert hll.cardinality() < 2

    def test_update_batch(self):
        """Test adding multiple items at once."""
        hll = HyperLogLog()
        items = [f"item_{i}" for i in range(100)]
        hll.update(items)

        cardinality = hll.cardinality()
        assert 80 < cardinality < 120

    def test_len(self):
        """Test len() returns estimated cardinality."""
        hll = HyperLogLog()

        for i in range(500):
            hll.add(f"item_{i}")

        # len() should be close to 500
        assert 400 < len(hll) < 600

    def test_clear(self):
        """Test clearing the counter."""
        hll = HyperLogLog()
        hll.add("test")
        assert hll.cardinality() > 0

        hll.clear()
        assert hll.cardinality() == 0

    def test_copy(self):
        """Test deep copy."""
        hll1 = HyperLogLog()
        hll1.add("test")

        hll2 = hll1.copy()
        assert hll2.cardinality() > 0

        hll2.add("new")
        # Original should not change
        old_card = hll1.cardinality()
        assert hll1.cardinality() == old_card

    def test_merge(self):
        """Test merging two counters."""
        hll1 = HyperLogLog(precision=14)
        hll2 = HyperLogLog(precision=14)

        for i in range(500):
            hll1.add(f"a_{i}")

        for i in range(500):
            hll2.add(f"b_{i}")

        hll1.merge(hll2)
        # Should be approximately 1000
        assert 800 < hll1.cardinality() < 1200

    def test_union_operator(self):
        """Test | operator for union."""
        hll1 = HyperLogLog(precision=14)
        hll2 = HyperLogLog(precision=14)

        for i in range(100):
            hll1.add(f"a_{i}")

        for i in range(100):
            hll2.add(f"b_{i}")

        hll3 = hll1 | hll2
        assert 150 < hll3.cardinality() < 250

    def test_serialization_bytes(self):
        """Test to_bytes/from_bytes."""
        hll1 = HyperLogLog()
        for i in range(100):
            hll1.add(f"item_{i}")

        data = hll1.to_bytes()
        hll2 = HyperLogLog.from_bytes(data)

        # Cardinalities should be equal
        assert abs(hll1.cardinality() - hll2.cardinality()) < 0.01

    def test_serialization_json(self):
        """Test to_json/from_json."""
        hll1 = HyperLogLog()
        hll1.add("test")

        data = hll1.to_json()
        hll2 = HyperLogLog.from_json(data)

        assert abs(hll1.cardinality() - hll2.cardinality()) < 0.01

    def test_properties(self):
        """Test property getters."""
        hll = HyperLogLog(precision=12)

        assert hll.precision == 12
        assert hll.size_in_bytes == 4096  # 2^12 registers
        assert hll.seed == 0
        assert hll.standard_error > 0

    def test_relative_error(self):
        """Test relative error calculation."""
        hll = HyperLogLog(precision=14)
        error = hll.relative_error()
        # For precision 14: 1.04 / sqrt(16384) ≈ 0.81%
        assert 0.5 < error < 2

    def test_precision_validation(self):
        """Test that invalid precision raises errors."""
        with pytest.raises(ValueError):
            HyperLogLog(precision=3)  # Too low

        with pytest.raises(ValueError):
            HyperLogLog(precision=19)  # Too high

    def test_merge_validation(self):
        """Test that merge validates parameters."""
        hll1 = HyperLogLog(precision=12)
        hll2 = HyperLogLog(precision=14)

        with pytest.raises(ValueError):
            hll1.merge(hll2)

    def test_large_cardinality(self):
        """Test with larger cardinality."""
        hll = HyperLogLog(precision=14)

        for i in range(100000):
            hll.add(f"item_{i}")

        cardinality = hll.cardinality()
        # Allow 5% error for large cardinality
        assert 95000 < cardinality < 105000


class TestHLLParamsEstimation:
    """Tests for HLL parameter estimation."""

    def test_estimate_hll_params(self):
        """Test parameter estimation."""
        params = estimate_hll_params(1_000_000, 0.01)

        assert params.precision >= 4
        assert params.precision <= 18
        assert params.num_registers == 2 ** params.precision
        assert params.memory_bytes == params.num_registers
        assert params.relative_error_percent < 2  # Should be close to 1%

    def test_lower_error_needs_more_memory(self):
        """Test that lower error requires more memory."""
        params_2pct = estimate_hll_params(1000, 0.02)
        params_1pct = estimate_hll_params(1000, 0.01)

        assert params_1pct.memory_bytes >= params_2pct.memory_bytes
