"""Tests for Count-Min Sketch."""

import pytest
from hazy import CountMinSketch, estimate_cms_params


class TestCountMinSketch:
    """Tests for CountMinSketch."""

    def test_basic_usage(self):
        """Test basic add and query operations."""
        cms = CountMinSketch(width=1000, depth=5)

        cms.add("hello")
        cms.add("hello")
        cms.add("world")

        assert cms.query("hello") >= 2
        assert cms.query("world") >= 1
        assert cms.query("nonexistent") == 0

    def test_add_count(self):
        """Test adding with specific count."""
        cms = CountMinSketch(width=1000, depth=5)

        cms.add_count("test", 10)
        assert cms.query("test") >= 10

    def test_getitem(self):
        """Test [] operator for querying."""
        cms = CountMinSketch(width=1000, depth=5)
        cms.add("test")

        assert cms["test"] >= 1
        assert cms["missing"] == 0

    def test_update_batch(self):
        """Test adding multiple items at once."""
        cms = CountMinSketch(width=1000, depth=5)
        items = ["a", "b", "c", "a", "a"]
        cms.update(items)

        assert cms.query("a") >= 3
        assert cms.query("b") >= 1
        assert cms.query("c") >= 1

    def test_len_total_count(self):
        """Test len() and total_count property."""
        cms = CountMinSketch(width=100, depth=3)

        cms.add("a")
        cms.add("b")
        cms.add("a")

        assert len(cms) == 3
        assert cms.total_count == 3

    def test_clear(self):
        """Test clearing the sketch."""
        cms = CountMinSketch(width=100, depth=3)
        cms.add("test")
        assert cms.query("test") > 0

        cms.clear()
        assert cms.query("test") == 0
        assert len(cms) == 0

    def test_copy(self):
        """Test deep copy."""
        cms1 = CountMinSketch(width=100, depth=3)
        cms1.add("test")

        cms2 = cms1.copy()
        assert cms2.query("test") > 0

        cms2.add("new")
        assert cms1.query("new") == 0

    def test_merge(self):
        """Test merging two sketches."""
        cms1 = CountMinSketch(width=100, depth=3, seed=42)
        cms1.add("a")

        cms2 = CountMinSketch(width=100, depth=3, seed=42)
        cms2.add("b")

        cms1.merge(cms2)
        assert cms1.query("a") >= 1
        assert cms1.query("b") >= 1

    def test_serialization_bytes(self):
        """Test to_bytes/from_bytes."""
        cms1 = CountMinSketch(width=100, depth=3)
        cms1.add("test")
        cms1.add("test")

        data = cms1.to_bytes()
        cms2 = CountMinSketch.from_bytes(data)

        assert cms2.query("test") >= 2

    def test_serialization_json(self):
        """Test to_json/from_json."""
        cms1 = CountMinSketch(width=100, depth=3)
        cms1.add("test")

        data = cms1.to_json()
        cms2 = CountMinSketch.from_json(data)

        assert cms2.query("test") >= 1

    def test_properties(self):
        """Test property getters."""
        cms = CountMinSketch(width=100, depth=5)

        assert cms.width == 100
        assert cms.depth == 5
        assert cms.size_in_bytes == 100 * 5 * 8
        assert cms.seed == 0

    def test_error_rate_confidence(self):
        """Test error rate and confidence methods."""
        cms = CountMinSketch(width=100, depth=5)

        error = cms.error_rate()
        conf = cms.confidence()

        assert 0 < error < 0.1
        assert 0 < conf < 1

    def test_alternative_constructor(self):
        """Test constructor with error_rate and confidence."""
        cms = CountMinSketch(error_rate=0.01, confidence=0.99)

        assert cms.width > 0
        assert cms.depth > 0

    def test_inner_product(self):
        """Test inner product estimation."""
        cms1 = CountMinSketch(width=100, depth=3, seed=42)
        cms2 = CountMinSketch(width=100, depth=3, seed=42)

        cms1.add("a")
        cms1.add("a")
        cms2.add("a")
        cms2.add("b")

        product = cms1.inner_product(cms2)
        # Inner product should be at least 2 (from "a")
        assert product >= 2

    def test_frequency_accuracy(self):
        """Test that frequency estimates are accurate."""
        cms = CountMinSketch(width=10000, depth=7)

        # Add items with known frequencies
        for _ in range(100):
            cms.add("frequent")
        for _ in range(10):
            cms.add("moderate")
        cms.add("rare")

        # Estimates should be at least the true count (may overestimate)
        assert cms.query("frequent") >= 100
        assert cms.query("moderate") >= 10
        assert cms.query("rare") >= 1

    def test_validation_errors(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            CountMinSketch()  # No parameters

        with pytest.raises(ValueError):
            CountMinSketch(width=0, depth=5)

        with pytest.raises(ValueError):
            CountMinSketch(width=100, depth=0)

        with pytest.raises(ValueError):
            CountMinSketch(error_rate=0, confidence=0.99)

    def test_merge_validation(self):
        """Test that merge validates parameters."""
        cms1 = CountMinSketch(width=100, depth=3)
        cms2 = CountMinSketch(width=200, depth=3)

        with pytest.raises(ValueError):
            cms1.merge(cms2)


class TestCMSParamsEstimation:
    """Tests for CMS parameter estimation."""

    def test_estimate_cms_params(self):
        """Test parameter estimation."""
        params = estimate_cms_params(1_000_000, 0.001, 0.99)

        assert params.width > 0
        assert params.depth > 0
        assert params.memory_bytes > 0
        assert params.error_rate == 0.001
        assert params.confidence == 0.99

    def test_lower_error_needs_more_width(self):
        """Test that lower error requires more width."""
        params_1pct = estimate_cms_params(1000, 0.01, 0.99)
        params_01pct = estimate_cms_params(1000, 0.001, 0.99)

        assert params_01pct.width > params_1pct.width
