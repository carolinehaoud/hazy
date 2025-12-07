"""
Jupyter notebook integration for hazy data structures.

Provides rich HTML representations for display in Jupyter notebooks.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hazy import (
        BloomFilter,
        CountingBloomFilter,
        ScalableBloomFilter,
        HyperLogLog,
        CountMinSketch,
        TopK,
        MinHash,
    )

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def bloom_repr_html(bf: "BloomFilter") -> str:
    """Generate HTML representation for BloomFilter."""
    fill_pct = bf.fill_ratio() * 100
    fill_bar_width = min(100, fill_pct)

    html = f"""
    <div style="font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 5px; max-width: 500px;">
        <div style="font-weight: bold; margin-bottom: 8px;">BloomFilter</div>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 2px 8px;">Bits:</td><td style="padding: 2px 8px;">{bf.num_bits:,}</td></tr>
            <tr><td style="padding: 2px 8px;">Hash functions:</td><td style="padding: 2px 8px;">{bf.num_hashes}</td></tr>
            <tr><td style="padding: 2px 8px;">Items added:</td><td style="padding: 2px 8px;">{len(bf):,}</td></tr>
            <tr><td style="padding: 2px 8px;">Memory:</td><td style="padding: 2px 8px;">{bf.size_in_bytes:,} bytes</td></tr>
            <tr><td style="padding: 2px 8px;">FPR:</td><td style="padding: 2px 8px;">{bf.false_positive_rate:.4f}</td></tr>
        </table>
        <div style="margin-top: 8px;">
            <div style="font-size: 11px; margin-bottom: 2px;">Fill ratio: {fill_pct:.1f}%</div>
            <div style="background: #eee; border-radius: 3px; height: 12px; width: 100%;">
                <div style="background: #4a90d9; height: 100%; width: {fill_bar_width}%; border-radius: 3px;"></div>
            </div>
        </div>
    </div>
    """
    return html


def counting_bloom_repr_html(cbf: "CountingBloomFilter") -> str:
    """Generate HTML representation for CountingBloomFilter."""
    html = f"""
    <div style="font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 5px; max-width: 500px;">
        <div style="font-weight: bold; margin-bottom: 8px;">CountingBloomFilter</div>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 2px 8px;">Counters:</td><td style="padding: 2px 8px;">{cbf.num_counters:,}</td></tr>
            <tr><td style="padding: 2px 8px;">Hash functions:</td><td style="padding: 2px 8px;">{cbf.num_hashes}</td></tr>
            <tr><td style="padding: 2px 8px;">Items added:</td><td style="padding: 2px 8px;">{len(cbf):,}</td></tr>
            <tr><td style="padding: 2px 8px;">Memory:</td><td style="padding: 2px 8px;">{cbf.size_in_bytes:,} bytes</td></tr>
        </table>
    </div>
    """
    return html


def scalable_bloom_repr_html(sbf: "ScalableBloomFilter") -> str:
    """Generate HTML representation for ScalableBloomFilter."""
    capacity = sbf.capacity()
    items = len(sbf)
    fill_pct = (items / capacity * 100) if capacity > 0 else 0
    fill_bar_width = min(100, fill_pct)

    html = f"""
    <div style="font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 5px; max-width: 500px;">
        <div style="font-weight: bold; margin-bottom: 8px;">ScalableBloomFilter</div>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 2px 8px;">Slices:</td><td style="padding: 2px 8px;">{sbf.num_slices}</td></tr>
            <tr><td style="padding: 2px 8px;">Items:</td><td style="padding: 2px 8px;">{items:,}</td></tr>
            <tr><td style="padding: 2px 8px;">Capacity:</td><td style="padding: 2px 8px;">{capacity:,}</td></tr>
            <tr><td style="padding: 2px 8px;">Memory:</td><td style="padding: 2px 8px;">{sbf.size_in_bytes:,} bytes</td></tr>
            <tr><td style="padding: 2px 8px;">FPR:</td><td style="padding: 2px 8px;">{sbf.false_positive_rate():.4f}</td></tr>
        </table>
        <div style="margin-top: 8px;">
            <div style="font-size: 11px; margin-bottom: 2px;">Capacity used: {fill_pct:.1f}%</div>
            <div style="background: #eee; border-radius: 3px; height: 12px; width: 100%;">
                <div style="background: #4a90d9; height: 100%; width: {fill_bar_width}%; border-radius: 3px;"></div>
            </div>
        </div>
    </div>
    """
    return html


def hll_repr_html(hll: "HyperLogLog") -> str:
    """Generate HTML representation for HyperLogLog."""
    html = f"""
    <div style="font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 5px; max-width: 500px;">
        <div style="font-weight: bold; margin-bottom: 8px;">HyperLogLog</div>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 2px 8px;">Precision:</td><td style="padding: 2px 8px;">{hll.precision}</td></tr>
            <tr><td style="padding: 2px 8px;">Registers:</td><td style="padding: 2px 8px;">{2**hll.precision:,}</td></tr>
            <tr><td style="padding: 2px 8px;">Cardinality:</td><td style="padding: 2px 8px; font-weight: bold;">{hll.cardinality():,.0f}</td></tr>
            <tr><td style="padding: 2px 8px;">Memory:</td><td style="padding: 2px 8px;">{hll.size_in_bytes:,} bytes</td></tr>
            <tr><td style="padding: 2px 8px;">Std Error:</td><td style="padding: 2px 8px;">±{hll.standard_error:.2%}</td></tr>
        </table>
    </div>
    """
    return html


def cms_repr_html(cms: "CountMinSketch") -> str:
    """Generate HTML representation for CountMinSketch."""
    html = f"""
    <div style="font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 5px; max-width: 500px;">
        <div style="font-weight: bold; margin-bottom: 8px;">CountMinSketch</div>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 2px 8px;">Dimensions:</td><td style="padding: 2px 8px;">{cms.width} × {cms.depth}</td></tr>
            <tr><td style="padding: 2px 8px;">Total count:</td><td style="padding: 2px 8px;">{cms.total_count:,}</td></tr>
            <tr><td style="padding: 2px 8px;">Memory:</td><td style="padding: 2px 8px;">{cms.size_in_bytes:,} bytes</td></tr>
            <tr><td style="padding: 2px 8px;">Error rate:</td><td style="padding: 2px 8px;">ε = {cms.error_rate():.4f}</td></tr>
            <tr><td style="padding: 2px 8px;">Confidence:</td><td style="padding: 2px 8px;">{cms.confidence():.2%}</td></tr>
        </table>
    </div>
    """
    return html


def topk_repr_html(topk: "TopK", max_display: int = 10) -> str:
    """Generate HTML representation for TopK."""
    items = topk.top(max_display)

    if not items:
        items_html = "<tr><td colspan='2' style='padding: 4px 8px; color: #888;'>Empty</td></tr>"
    else:
        max_count = items[0][1] if items else 1
        items_html = ""
        for item, count in items:
            bar_width = (count / max_count * 100) if max_count > 0 else 0
            items_html += f"""
            <tr>
                <td style="padding: 2px 8px; max-width: 150px; overflow: hidden; text-overflow: ellipsis;">{item}</td>
                <td style="padding: 2px 8px; width: 60px; text-align: right;">{count:,}</td>
                <td style="padding: 2px 8px; width: 100px;">
                    <div style="background: #4a90d9; height: 10px; width: {bar_width}%; border-radius: 2px;"></div>
                </td>
            </tr>
            """

    html = f"""
    <div style="font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 5px; max-width: 500px;">
        <div style="font-weight: bold; margin-bottom: 8px;">TopK (k={topk.k}, tracked={len(topk)})</div>
        <table style="border-collapse: collapse; width: 100%;">
            <tr style="border-bottom: 1px solid #ddd;">
                <th style="padding: 4px 8px; text-align: left;">Item</th>
                <th style="padding: 4px 8px; text-align: right;">Count</th>
                <th style="padding: 4px 8px;"></th>
            </tr>
            {items_html}
        </table>
    </div>
    """
    return html


def minhash_repr_html(mh: "MinHash") -> str:
    """Generate HTML representation for MinHash."""
    html = f"""
    <div style="font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 5px; max-width: 500px;">
        <div style="font-weight: bold; margin-bottom: 8px;">MinHash</div>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 2px 8px;">Hash functions:</td><td style="padding: 2px 8px;">{mh.num_hashes}</td></tr>
            <tr><td style="padding: 2px 8px;">Memory:</td><td style="padding: 2px 8px;">{mh.size_in_bytes:,} bytes</td></tr>
            <tr><td style="padding: 2px 8px;">Empty:</td><td style="padding: 2px 8px;">{mh.is_empty()}</td></tr>
            <tr><td style="padding: 2px 8px;">Std Error:</td><td style="padding: 2px 8px;">±{mh.standard_error():.2%}</td></tr>
        </table>
    </div>
    """
    return html


def enable_jupyter_integration():
    """
    Enable rich HTML display for hazy types in Jupyter notebooks.

    Call this function once to enable _repr_html_ for all hazy types.

    Example:
        >>> from hazy._jupyter import enable_jupyter_integration
        >>> enable_jupyter_integration()
        >>> bf = BloomFilter(expected_items=1000)
        >>> bf  # Will display rich HTML in Jupyter
    """
    try:
        from hazy._hazy import (
            BloomFilter,
            CountingBloomFilter,
            ScalableBloomFilter,
            HyperLogLog,
            CountMinSketch,
            TopK,
            MinHash,
        )

        # Monkey-patch _repr_html_ methods
        BloomFilter._repr_html_ = lambda self: bloom_repr_html(self)
        CountingBloomFilter._repr_html_ = lambda self: counting_bloom_repr_html(self)
        ScalableBloomFilter._repr_html_ = lambda self: scalable_bloom_repr_html(self)
        HyperLogLog._repr_html_ = lambda self: hll_repr_html(self)
        CountMinSketch._repr_html_ = lambda self: cms_repr_html(self)
        TopK._repr_html_ = lambda self: topk_repr_html(self)
        MinHash._repr_html_ = lambda self: minhash_repr_html(self)

        return True
    except ImportError:
        return False
