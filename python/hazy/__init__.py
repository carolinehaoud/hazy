"""
Hazy: A modern Python library for probabilistic data structures.

This library provides efficient implementations of probabilistic data structures
including Bloom filters, HyperLogLog, Count-Min Sketch, and more.

Example usage:
    >>> from hazy import BloomFilter
    >>> bf = BloomFilter(expected_items=10000)
    >>> bf.add("hello")
    >>> "hello" in bf
    True
    >>> "world" in bf
    False
"""

from hazy._hazy import (
    BloomFilter,
    CountingBloomFilter,
    ScalableBloomFilter,
    CuckooFilter,
    HyperLogLog,
    CountMinSketch,
    MinHash,
    TopK,
)
from hazy._helpers import (
    estimate_bloom_params,
    estimate_counting_bloom_params,
    estimate_cuckoo_params,
    estimate_hll_params,
    estimate_cms_params,
    estimate_minhash_params,
)

__version__ = "0.1.0"
__all__ = [
    # Data structures
    "BloomFilter",
    "CountingBloomFilter",
    "ScalableBloomFilter",
    "CuckooFilter",
    "HyperLogLog",
    "CountMinSketch",
    "MinHash",
    "TopK",
    # Parameter helpers
    "estimate_bloom_params",
    "estimate_counting_bloom_params",
    "estimate_cuckoo_params",
    "estimate_hll_params",
    "estimate_cms_params",
    "estimate_minhash_params",
    # Jupyter integration
    "enable_notebook_display",
]


def enable_notebook_display():
    """
    Enable rich HTML display for hazy types in Jupyter notebooks.

    Call this once at the start of your notebook:

        >>> import hazy
        >>> hazy.enable_notebook_display()
        >>> bf = hazy.BloomFilter(expected_items=1000)
        >>> bf  # Shows rich HTML representation
    """
    from hazy._jupyter import enable_jupyter_integration
    return enable_jupyter_integration()
