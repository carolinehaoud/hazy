"""Tests for visualization module.

These tests verify that visualization functions work correctly
without requiring a display (using matplotlib's Agg backend).
"""

import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from hazy import (
    BloomFilter,
    HyperLogLog,
    CountMinSketch,
    TopK,
    MinHash,
    ScalableBloomFilter,
)

# Import visualization functions
try:
    from hazy.viz import (
        plot_bloom,
        plot_bloom_fill_curve,
        plot_hll,
        plot_cms,
        plot_topk,
        plot_minhash_comparison,
        plot_scalable_bloom,
        show,
        HAS_MATPLOTLIB,
    )
    VIZ_AVAILABLE = HAS_MATPLOTLIB
except ImportError:
    VIZ_AVAILABLE = False


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')


@pytest.mark.skipif(not VIZ_AVAILABLE, reason="matplotlib not installed")
class TestPlotBloom:
    """Tests for plot_bloom function."""

    def test_basic_plot(self):
        """Test basic Bloom filter visualization."""
        bf = BloomFilter(expected_items=1000)
        bf.update([f"item_{i}" for i in range(500)])

        ax = plot_bloom(bf)
        assert ax is not None

    def test_empty_filter(self):
        """Test visualization of empty filter."""
        bf = BloomFilter(expected_items=100)
        ax = plot_bloom(bf)
        assert ax is not None

    def test_custom_title(self):
        """Test with custom title."""
        bf = BloomFilter(expected_items=100)
        bf.add("test")
        ax = plot_bloom(bf, title="Custom Title")
        assert ax.get_title() == "Custom Title"

    def test_custom_colormap(self):
        """Test with different colormaps."""
        bf = BloomFilter(expected_items=100)
        bf.add("test")

        for cmap in ["Blues", "Greens", "Reds", "viridis"]:
            ax = plot_bloom(bf, cmap=cmap)
            assert ax is not None
            plt.close()

    def test_hide_stats(self):
        """Test hiding statistics."""
        bf = BloomFilter(expected_items=100)
        bf.add("test")
        ax = plot_bloom(bf, show_stats=False)
        assert ax is not None

    def test_custom_axes(self):
        """Test plotting to existing axes."""
        fig, ax = plt.subplots()
        bf = BloomFilter(expected_items=100)
        bf.add("test")

        result = plot_bloom(bf, ax=ax)
        assert result is ax

    def test_custom_figsize(self):
        """Test custom figure size."""
        bf = BloomFilter(expected_items=100)
        ax = plot_bloom(bf, figsize=(8, 4))
        assert ax is not None


@pytest.mark.skipif(not VIZ_AVAILABLE, reason="matplotlib not installed")
class TestPlotBloomFillCurve:
    """Tests for plot_bloom_fill_curve function."""

    def test_basic_curve(self):
        """Test basic fill curve plotting."""
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        bf.update([f"item_{i}" for i in range(100)])

        result = plot_bloom_fill_curve(bf)
        assert result is not None

    def test_custom_max_items(self):
        """Test with custom max_items."""
        bf = BloomFilter(expected_items=1000)
        result = plot_bloom_fill_curve(bf, max_items=5000)
        assert result is not None


@pytest.mark.skipif(not VIZ_AVAILABLE, reason="matplotlib not installed")
class TestPlotHLL:
    """Tests for plot_hll function."""

    def test_basic_plot(self):
        """Test basic HLL visualization."""
        hll = HyperLogLog(precision=12)
        for i in range(10000):
            hll.add(f"user_{i}")

        ax = plot_hll(hll)
        assert ax is not None

    def test_empty_hll(self):
        """Test visualization of empty HLL."""
        hll = HyperLogLog(precision=12)
        ax = plot_hll(hll)
        assert ax is not None

    def test_custom_title(self):
        """Test with custom title."""
        hll = HyperLogLog(precision=12)
        hll.add("test")
        ax = plot_hll(hll, title="My HLL")
        assert ax.get_title() == "My HLL"

    def test_hide_stats(self):
        """Test hiding statistics."""
        hll = HyperLogLog(precision=12)
        hll.add("test")
        ax = plot_hll(hll, show_stats=False)
        assert ax is not None


@pytest.mark.skipif(not VIZ_AVAILABLE, reason="matplotlib not installed")
class TestPlotCMS:
    """Tests for plot_cms function."""

    def test_basic_plot(self):
        """Test basic CMS visualization."""
        cms = CountMinSketch(width=100, depth=5)
        for i in range(1000):
            cms.add(f"item_{i % 50}")

        ax = plot_cms(cms)
        assert ax is not None

    def test_empty_cms(self):
        """Test visualization of empty CMS."""
        cms = CountMinSketch(width=100, depth=5)
        ax = plot_cms(cms)
        assert ax is not None

    def test_log_scale(self):
        """Test with log scale."""
        cms = CountMinSketch(width=100, depth=5)
        cms.add("hot", count=1000)
        cms.add("cold", count=1)

        ax = plot_cms(cms, log_scale=True)
        assert ax is not None

    def test_linear_scale(self):
        """Test with linear scale."""
        cms = CountMinSketch(width=100, depth=5)
        cms.add("test", count=10)

        ax = plot_cms(cms, log_scale=False)
        assert ax is not None

    def test_custom_colormap(self):
        """Test with different colormaps."""
        cms = CountMinSketch(width=100, depth=5)
        cms.add("test")

        for cmap in ["YlOrRd", "Blues", "viridis"]:
            ax = plot_cms(cms, cmap=cmap)
            assert ax is not None
            plt.close()


@pytest.mark.skipif(not VIZ_AVAILABLE, reason="matplotlib not installed")
class TestPlotTopK:
    """Tests for plot_topk function."""

    def test_basic_plot(self):
        """Test basic TopK visualization."""
        tk = TopK(k=10)
        tk.add("apple", count=100)
        tk.add("banana", count=50)
        tk.add("cherry", count=25)

        ax = plot_topk(tk)
        assert ax is not None

    def test_empty_topk_raises(self):
        """Test that empty TopK raises error."""
        tk = TopK(k=10)

        with pytest.raises(ValueError, match="empty"):
            plot_topk(tk)

    def test_with_n(self):
        """Test showing specific number of items."""
        tk = TopK(k=20)
        for i in range(20):
            tk.add(f"item_{i}", count=20 - i)

        ax = plot_topk(tk, n=5)
        assert ax is not None

    def test_show_error_bars(self):
        """Test with error bars."""
        tk = TopK(k=10)
        tk.add("a", count=100)
        tk.add("b", count=50)

        ax = plot_topk(tk, show_error=True)
        assert ax is not None

    def test_hide_error_bars(self):
        """Test without error bars."""
        tk = TopK(k=10)
        tk.add("a", count=100)
        tk.add("b", count=50)

        ax = plot_topk(tk, show_error=False)
        assert ax is not None

    def test_vertical_bars(self):
        """Test vertical bar chart."""
        tk = TopK(k=10)
        tk.add("a", count=100)
        tk.add("b", count=50)

        ax = plot_topk(tk, horizontal=False)
        assert ax is not None


@pytest.mark.skipif(not VIZ_AVAILABLE, reason="matplotlib not installed")
class TestPlotMinHashComparison:
    """Tests for plot_minhash_comparison function."""

    def test_basic_comparison(self):
        """Test basic MinHash comparison visualization."""
        mh1 = MinHash(num_hashes=64)
        mh2 = MinHash(num_hashes=64)

        mh1.update(["a", "b", "c", "d"])
        mh2.update(["c", "d", "e", "f"])

        result = plot_minhash_comparison(mh1, mh2)
        assert result is not None

    def test_identical_sets(self):
        """Test comparison of identical sets."""
        mh1 = MinHash(num_hashes=64)
        mh2 = MinHash(num_hashes=64)

        items = ["a", "b", "c"]
        mh1.update(items)
        mh2.update(items)

        result = plot_minhash_comparison(mh1, mh2)
        assert result is not None

    def test_disjoint_sets(self):
        """Test comparison of completely different sets."""
        mh1 = MinHash(num_hashes=64)
        mh2 = MinHash(num_hashes=64)

        mh1.update(["a", "b", "c"])
        mh2.update(["d", "e", "f"])

        result = plot_minhash_comparison(mh1, mh2)
        assert result is not None

    def test_different_lengths_raises(self):
        """Test that different signature lengths raise error."""
        mh1 = MinHash(num_hashes=64)
        mh2 = MinHash(num_hashes=128)

        mh1.update(["a"])
        mh2.update(["a"])

        with pytest.raises(ValueError, match="same length"):
            plot_minhash_comparison(mh1, mh2)


@pytest.mark.skipif(not VIZ_AVAILABLE, reason="matplotlib not installed")
class TestPlotScalableBloom:
    """Tests for plot_scalable_bloom function."""

    def test_basic_plot(self):
        """Test basic ScalableBloomFilter visualization."""
        sbf = ScalableBloomFilter(initial_capacity=1000)
        for i in range(500):
            sbf.add(f"item_{i}")

        ax = plot_scalable_bloom(sbf)
        assert ax is not None

    def test_empty_filter(self):
        """Test visualization of empty filter."""
        sbf = ScalableBloomFilter(initial_capacity=100)
        ax = plot_scalable_bloom(sbf)
        assert ax is not None

    def test_multiple_slices(self):
        """Test visualization with multiple slices."""
        sbf = ScalableBloomFilter(initial_capacity=100)
        # Add enough items to trigger multiple slices
        for i in range(1000):
            sbf.add(f"item_{i}")

        ax = plot_scalable_bloom(sbf)
        assert ax is not None


@pytest.mark.skipif(not VIZ_AVAILABLE, reason="matplotlib not installed")
class TestShowFunction:
    """Tests for show() convenience function."""

    def test_show_doesnt_crash(self):
        """Test that show() works (non-interactive mode)."""
        bf = BloomFilter(expected_items=100)
        bf.add("test")
        plot_bloom(bf)

        # In Agg backend, show() should not block or crash
        show()


@pytest.mark.skipif(not VIZ_AVAILABLE, reason="matplotlib not installed")
class TestSaveFigure:
    """Tests for saving figures."""

    def test_save_bloom_png(self, tmp_path):
        """Test saving Bloom filter plot to PNG."""
        bf = BloomFilter(expected_items=100)
        bf.add("test")

        plot_bloom(bf)
        path = tmp_path / "bloom.png"
        plt.savefig(path, dpi=100)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_hll_png(self, tmp_path):
        """Test saving HLL plot to PNG."""
        hll = HyperLogLog(precision=12)
        for i in range(1000):
            hll.add(f"item_{i}")

        plot_hll(hll)
        path = tmp_path / "hll.png"
        plt.savefig(path, dpi=100)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_cms_png(self, tmp_path):
        """Test saving CMS plot to PNG."""
        cms = CountMinSketch(width=100, depth=5)
        cms.add("test", count=10)

        plot_cms(cms)
        path = tmp_path / "cms.png"
        plt.savefig(path, dpi=100)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_topk_png(self, tmp_path):
        """Test saving TopK plot to PNG."""
        tk = TopK(k=10)
        tk.add("a", count=100)
        tk.add("b", count=50)

        plot_topk(tk)
        path = tmp_path / "topk.png"
        plt.savefig(path, dpi=100)

        assert path.exists()
        assert path.stat().st_size > 0
