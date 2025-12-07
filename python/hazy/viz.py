"""
Visualization utilities for probabilistic data structures.

Install with: pip install hazy[viz]

Example:
    >>> from hazy import BloomFilter
    >>> from hazy.viz import plot_bloom, plot_hll, plot_cms, plot_topk
    >>>
    >>> bf = BloomFilter(expected_items=1000)
    >>> bf.update([f"item_{i}" for i in range(500)])
    >>> plot_bloom(bf)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from hazy import (
        BloomFilter,
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
    plt = None


def _check_matplotlib():
    """Raise ImportError if matplotlib is not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Visualization requires matplotlib. "
            "Install with: pip install hazy[viz]"
        )


def plot_bloom(
    bf: "BloomFilter",
    figsize: Tuple[int, int] = (12, 4),
    cmap: str = "Blues",
    title: Optional[str] = None,
    show_stats: bool = True,
    ax: Optional[Any] = None,
) -> Any:
    """
    Visualize a Bloom filter as a bit array heatmap.

    Args:
        bf: BloomFilter instance to visualize
        figsize: Figure size (width, height) in inches
        cmap: Matplotlib colormap name
        title: Plot title (auto-generated if None)
        show_stats: Whether to show statistics annotation
        ax: Matplotlib axes to plot on (creates new figure if None)

    Returns:
        Matplotlib axes object

    Example:
        >>> bf = BloomFilter(expected_items=1000)
        >>> bf.update(["apple", "banana", "cherry"])
        >>> plot_bloom(bf)
    """
    _check_matplotlib()

    # Get the bit array data
    data = bf.to_json()
    import json
    bf_data = json.loads(data)
    bits = bf_data["bits"]
    num_bits = bf_data["num_bits"]

    # Convert u64 words to bit array
    bit_array = []
    for word in bits:
        for i in range(64):
            if len(bit_array) < num_bits:
                bit_array.append((word >> i) & 1)

    # Reshape into 2D grid for visualization
    cols = min(256, num_bits)
    rows = math.ceil(num_bits / cols)

    # Pad to fill grid
    bit_array.extend([0] * (rows * cols - len(bit_array)))
    grid = [bit_array[i * cols:(i + 1) * cols] for i in range(rows)]

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        pass  # ax provided externally

    im = ax.imshow(grid, cmap=cmap, aspect='auto', interpolation='nearest')

    # Title
    if title is None:
        title = f"Bloom Filter ({num_bits:,} bits, {bf.num_hashes} hashes)"
    ax.set_title(title)
    ax.set_xlabel("Bit Position (mod 256)")
    ax.set_ylabel("Row")

    # Stats annotation
    if show_stats:
        stats_text = (
            f"Items: {len(bf):,}\n"
            f"Fill: {bf.fill_ratio():.1%}\n"
            f"FPR: {bf.false_positive_rate:.4f}"
        )
        ax.text(
            1.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    plt.colorbar(im, ax=ax, label="Bit Set", shrink=0.8)
    plt.tight_layout()

    return ax


def plot_bloom_fill_curve(
    bf: "BloomFilter",
    max_items: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot theoretical fill ratio and FPR curves for a Bloom filter.

    Args:
        bf: BloomFilter instance (uses its parameters)
        max_items: Maximum items to plot (defaults to 2x expected capacity)
        figsize: Figure size
        ax: Matplotlib axes

    Returns:
        Matplotlib axes object
    """
    _check_matplotlib()

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        ax1, ax2 = ax

    m = bf.num_bits
    k = bf.num_hashes

    if max_items is None:
        # Estimate capacity where FPR ≈ 50%
        max_items = int(m / k * 2)

    items = list(range(0, max_items, max(1, max_items // 200)))

    # Calculate theoretical values
    fill_ratios = []
    fprs = []
    for n in items:
        # Fill ratio: 1 - (1 - 1/m)^(kn) ≈ 1 - e^(-kn/m)
        fill = 1 - math.exp(-k * n / m)
        fill_ratios.append(fill)

        # FPR: (1 - e^(-kn/m))^k
        fpr = fill ** k
        fprs.append(fpr)

    # Plot fill ratio
    ax1.plot(items, fill_ratios, 'b-', linewidth=2)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% fill')
    ax1.axvline(x=len(bf), color='g', linestyle='--', alpha=0.5, label=f'Current ({len(bf):,})')
    ax1.set_xlabel('Number of Items')
    ax1.set_ylabel('Fill Ratio')
    ax1.set_title('Bloom Filter Fill Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot FPR
    ax2.semilogy(items, fprs, 'b-', linewidth=2)
    ax2.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1% FPR')
    ax2.axhline(y=0.001, color='orange', linestyle='--', alpha=0.5, label='0.1% FPR')
    ax2.axvline(x=len(bf), color='g', linestyle='--', alpha=0.5, label=f'Current ({len(bf):,})')
    ax2.set_xlabel('Number of Items')
    ax2.set_ylabel('False Positive Rate (log scale)')
    ax2.set_title('Bloom Filter FPR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax1, ax2


def plot_hll(
    hll: "HyperLogLog",
    figsize: Tuple[int, int] = (10, 5),
    title: Optional[str] = None,
    show_stats: bool = True,
    ax: Optional[Any] = None,
) -> Any:
    """
    Visualize HyperLogLog register values as a histogram.

    Args:
        hll: HyperLogLog instance to visualize
        figsize: Figure size
        title: Plot title
        show_stats: Whether to show statistics
        ax: Matplotlib axes

    Returns:
        Matplotlib axes object

    Example:
        >>> hll = HyperLogLog(precision=12)
        >>> hll.update([f"user_{i}" for i in range(10000)])
        >>> plot_hll(hll)
    """
    _check_matplotlib()

    # Get register values from JSON
    import json
    data = json.loads(hll.to_json())
    registers = data["registers"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create histogram of register values
    max_val = max(registers) if registers else 1
    bins = list(range(0, max_val + 2))

    ax.hist(registers, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Register Value (leading zeros + 1)')
    ax.set_ylabel('Count')

    if title is None:
        title = f"HyperLogLog Registers (precision={hll.precision})"
    ax.set_title(title)

    # Stats annotation
    if show_stats:
        stats_text = (
            f"Registers: {len(registers):,}\n"
            f"Cardinality: {hll.cardinality():,.0f}\n"
            f"Std Error: {hll.standard_error:.2%}"
        )
        ax.text(
            0.98, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            fontfamily='monospace',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return ax


def plot_cms(
    cms: "CountMinSketch",
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = "YlOrRd",
    title: Optional[str] = None,
    show_stats: bool = True,
    log_scale: bool = True,
    ax: Optional[Any] = None,
) -> Any:
    """
    Visualize Count-Min Sketch as a 2D heatmap.

    Args:
        cms: CountMinSketch instance to visualize
        figsize: Figure size
        cmap: Matplotlib colormap
        title: Plot title
        show_stats: Whether to show statistics
        log_scale: Use logarithmic color scale
        ax: Matplotlib axes

    Returns:
        Matplotlib axes object

    Example:
        >>> cms = CountMinSketch(width=100, depth=5)
        >>> for word in ["apple"] * 50 + ["banana"] * 30:
        ...     cms.add(word)
        >>> plot_cms(cms)
    """
    _check_matplotlib()

    # Get table data from JSON
    import json
    data = json.loads(cms.to_json())
    table = data["table"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Apply log scale if requested
    import numpy as np
    table_array = np.array(table, dtype=float)
    if log_scale:
        table_array = np.log1p(table_array)  # log(1 + x) to handle zeros

    im = ax.imshow(table_array, cmap=cmap, aspect='auto', interpolation='nearest')

    if title is None:
        title = f"Count-Min Sketch ({cms.width}×{cms.depth})"
    ax.set_title(title)
    ax.set_xlabel('Width (hash buckets)')
    ax.set_ylabel('Depth (hash functions)')

    # Stats annotation
    if show_stats:
        stats_text = (
            f"Total: {cms.total_count:,}\n"
            f"Error: ε={cms.error_rate():.4f}\n"
            f"Conf: {cms.confidence():.2%}"
        )
        ax.text(
            1.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    label = "log(1 + count)" if log_scale else "Count"
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)
    plt.tight_layout()

    return ax


def plot_topk(
    topk: "TopK",
    n: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    show_error: bool = True,
    horizontal: bool = True,
    ax: Optional[Any] = None,
) -> Any:
    """
    Visualize Top-K items as a bar chart.

    Args:
        topk: TopK instance to visualize
        n: Number of items to show (defaults to k)
        figsize: Figure size
        title: Plot title
        show_error: Show error bars
        horizontal: Use horizontal bars
        ax: Matplotlib axes

    Returns:
        Matplotlib axes object

    Example:
        >>> tk = TopK(k=10)
        >>> for word in ["apple"] * 100 + ["banana"] * 50 + ["cherry"] * 25:
        ...     tk.add(word)
        >>> plot_topk(tk)
    """
    _check_matplotlib()

    if show_error:
        items = topk.top_with_error(n)
        labels = [item for item, count, error in items]
        counts = [count for item, count, error in items]
        errors = [error for item, count, error in items]
    else:
        items = topk.top(n)
        labels = [item for item, count in items]
        counts = [count for item, count in items]
        errors = None

    if not labels:
        raise ValueError("TopK is empty, nothing to plot")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        y_pos = range(len(labels))
        if show_error and errors:
            ax.barh(y_pos, counts, xerr=errors, alpha=0.7, color='steelblue',
                   error_kw=dict(ecolor='red', capsize=3))
        else:
            ax.barh(y_pos, counts, alpha=0.7, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('Count')
        ax.set_ylabel('Item')
    else:
        x_pos = range(len(labels))
        if show_error and errors:
            ax.bar(x_pos, counts, yerr=errors, alpha=0.7, color='steelblue',
                  error_kw=dict(ecolor='red', capsize=3))
        else:
            ax.bar(x_pos, counts, alpha=0.7, color='steelblue')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel('Item')
        ax.set_ylabel('Count')

    if title is None:
        title = f"Top-{len(labels)} Items"
    ax.set_title(title)

    ax.grid(True, alpha=0.3, axis='x' if horizontal else 'y')
    plt.tight_layout()

    return ax


def plot_minhash_comparison(
    mh1: "MinHash",
    mh2: "MinHash",
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Visualize MinHash signature comparison.

    Args:
        mh1: First MinHash
        mh2: Second MinHash
        figsize: Figure size
        title: Plot title
        ax: Matplotlib axes

    Returns:
        Matplotlib axes object
    """
    _check_matplotlib()

    sig1 = mh1.get_signature()
    sig2 = mh2.get_signature()

    if len(sig1) != len(sig2):
        raise ValueError("MinHash signatures must have same length")

    matches = [1 if s1 == s2 else 0 for s1, s2 in zip(sig1, sig2)]
    jaccard = sum(matches) / len(matches)

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        ax1, ax2 = ax

    # Plot signature comparison
    x = range(len(matches))
    colors = ['green' if m else 'red' for m in matches]
    ax1.bar(x, [1] * len(matches), color=colors, alpha=0.7, width=1.0)
    ax1.set_xlabel('Hash Function Index')
    ax1.set_ylabel('Match')
    ax1.set_title('Signature Comparison (green=match)')
    ax1.set_yticks([])

    # Plot Jaccard similarity
    ax2.barh(['Jaccard\nSimilarity'], [jaccard], color='steelblue', alpha=0.7)
    ax2.barh(['Jaccard\nSimilarity'], [1 - jaccard], left=[jaccard], color='lightgray', alpha=0.7)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Similarity')
    ax2.set_title(f'Estimated Jaccard: {jaccard:.3f}')

    # Add percentage label
    ax2.text(jaccard / 2, 0, f'{jaccard:.1%}', ha='center', va='center', fontweight='bold')

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    return ax1, ax2


def plot_scalable_bloom(
    sbf: "ScalableBloomFilter",
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Visualize Scalable Bloom Filter slices.

    Args:
        sbf: ScalableBloomFilter instance
        figsize: Figure size
        title: Plot title
        ax: Matplotlib axes

    Returns:
        Matplotlib axes object
    """
    _check_matplotlib()

    # This won't work with bincode, need to add a method to get slice info
    # For now, show basic stats

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create a simple visualization showing slice counts
    slices = list(range(1, sbf.num_slices + 1))
    if not slices:
        ax.text(0.5, 0.5, 'Empty filter', ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Show capacity growth
        capacity = sbf.capacity()
        items = len(sbf)

        ax.bar(['Items', 'Capacity'], [items, capacity], color=['steelblue', 'lightgray'], alpha=0.7)
        ax.set_ylabel('Count')

        ax.text(0, items + capacity * 0.02, f'{items:,}', ha='center', fontweight='bold')
        ax.text(1, capacity + capacity * 0.02, f'{capacity:,}', ha='center', fontweight='bold')

    if title is None:
        title = f"Scalable Bloom Filter ({sbf.num_slices} slices)"
    ax.set_title(title)

    plt.tight_layout()
    return ax


# Convenience function to show all
def show():
    """Display all pending matplotlib figures."""
    _check_matplotlib()
    plt.show()
