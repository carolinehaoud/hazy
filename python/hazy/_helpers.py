"""
Parameter estimation helpers for probabilistic data structures.

These functions help users choose optimal parameters for their use cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class BloomParams:
    """Parameters for a Bloom filter."""
    num_bits: int
    num_hashes: int
    memory_bytes: int
    memory_mb: float
    expected_fpr: float


@dataclass
class CountingBloomParams:
    """Parameters for a Counting Bloom filter."""
    num_counters: int
    num_hashes: int
    memory_bytes: int
    memory_mb: float
    expected_fpr: float


@dataclass
class CuckooParams:
    """Parameters for a Cuckoo filter."""
    capacity: int
    memory_bytes: int
    memory_mb: float
    expected_fpr: float


@dataclass
class HLLParams:
    """Parameters for a HyperLogLog counter."""
    precision: int
    num_registers: int
    memory_bytes: int
    standard_error: float
    relative_error_percent: float


@dataclass
class CMSParams:
    """Parameters for a Count-Min Sketch."""
    width: int
    depth: int
    memory_bytes: int
    memory_mb: float
    error_rate: float
    confidence: float


@dataclass
class MinHashParams:
    """Parameters for a MinHash signature."""
    num_hashes: int
    memory_bytes: int
    standard_error: float
    relative_error_percent: float


def estimate_bloom_params(
    expected_items: int,
    false_positive_rate: float = 0.01,
) -> BloomParams:
    """
    Estimate optimal Bloom filter parameters.

    Args:
        expected_items: Expected number of items to be added
        false_positive_rate: Target false positive rate (default: 0.01 = 1%)

    Returns:
        BloomParams with recommended settings

    Example:
        >>> params = estimate_bloom_params(1_000_000, 0.01)
        >>> print(f"Need {params.memory_mb:.1f} MB for 1M items at 1% FPR")
    """
    if expected_items <= 0:
        raise ValueError("expected_items must be positive")
    if false_positive_rate <= 0 or false_positive_rate >= 1:
        raise ValueError("false_positive_rate must be between 0 and 1 (exclusive)")

    # Optimal number of bits: m = -n * ln(p) / (ln(2)^2)
    ln2_squared = math.log(2) ** 2
    num_bits = int(math.ceil(-expected_items * math.log(false_positive_rate) / ln2_squared))

    # Optimal number of hashes: k = (m/n) * ln(2)
    num_hashes = max(1, int(round((num_bits / expected_items) * math.log(2))))

    # Calculate memory
    memory_bytes = (num_bits + 7) // 8
    memory_mb = memory_bytes / (1024 * 1024)

    # Expected FPR with these parameters
    expected_fpr = (1 - math.exp(-num_hashes * expected_items / num_bits)) ** num_hashes

    return BloomParams(
        num_bits=num_bits,
        num_hashes=num_hashes,
        memory_bytes=memory_bytes,
        memory_mb=memory_mb,
        expected_fpr=expected_fpr,
    )


def estimate_counting_bloom_params(
    expected_items: int,
    false_positive_rate: float = 0.01,
) -> CountingBloomParams:
    """
    Estimate optimal Counting Bloom filter parameters.

    Args:
        expected_items: Expected number of items to be added
        false_positive_rate: Target false positive rate (default: 0.01 = 1%)

    Returns:
        CountingBloomParams with recommended settings
    """
    if expected_items <= 0:
        raise ValueError("expected_items must be positive")
    if false_positive_rate <= 0 or false_positive_rate >= 1:
        raise ValueError("false_positive_rate must be between 0 and 1 (exclusive)")

    # Same calculation as Bloom filter
    ln2_squared = math.log(2) ** 2
    num_counters = int(math.ceil(-expected_items * math.log(false_positive_rate) / ln2_squared))
    num_hashes = max(1, int(round((num_counters / expected_items) * math.log(2))))

    # Each counter is 1 byte (8 bits)
    memory_bytes = num_counters
    memory_mb = memory_bytes / (1024 * 1024)

    expected_fpr = (1 - math.exp(-num_hashes * expected_items / num_counters)) ** num_hashes

    return CountingBloomParams(
        num_counters=num_counters,
        num_hashes=num_hashes,
        memory_bytes=memory_bytes,
        memory_mb=memory_mb,
        expected_fpr=expected_fpr,
    )


def estimate_cuckoo_params(
    expected_items: int,
    false_positive_rate: float = 0.01,
) -> CuckooParams:
    """
    Estimate optimal Cuckoo filter parameters.

    Args:
        expected_items: Expected number of items
        false_positive_rate: Target false positive rate (default: 0.01 = 1%)

    Returns:
        CuckooParams with recommended settings

    Note:
        Cuckoo filters are more space-efficient than Bloom filters for FPR < 3%.
    """
    if expected_items <= 0:
        raise ValueError("expected_items must be positive")
    if false_positive_rate <= 0 or false_positive_rate >= 1:
        raise ValueError("false_positive_rate must be between 0 and 1 (exclusive)")

    # Cuckoo filter uses 2 bytes per fingerprint, 4 entries per bucket
    # Capacity should be ~10% more than expected items for good performance
    capacity = int(expected_items * 1.1)

    # Memory: capacity entries * 2 bytes per fingerprint
    memory_bytes = capacity * 2
    memory_mb = memory_bytes / (1024 * 1024)

    # FPR depends on fingerprint size (16 bits = 2 bytes)
    # FPR ≈ 2 * bucket_size / 2^fingerprint_bits
    # With 16-bit fingerprints: FPR ≈ 2 * 4 / 65536 ≈ 0.00012
    expected_fpr = 8.0 / 65536.0

    return CuckooParams(
        capacity=capacity,
        memory_bytes=memory_bytes,
        memory_mb=memory_mb,
        expected_fpr=expected_fpr,
    )


def estimate_hll_params(
    expected_cardinality: int,
    error_rate: float = 0.02,
) -> HLLParams:
    """
    Estimate optimal HyperLogLog parameters.

    Args:
        expected_cardinality: Expected number of distinct items
        error_rate: Target relative error rate (default: 0.02 = 2%)

    Returns:
        HLLParams with recommended settings

    Example:
        >>> params = estimate_hll_params(1_000_000, 0.01)
        >>> print(f"Precision {params.precision} gives {params.relative_error_percent:.1f}% error")
    """
    if expected_cardinality <= 0:
        raise ValueError("expected_cardinality must be positive")
    if error_rate <= 0 or error_rate >= 1:
        raise ValueError("error_rate must be between 0 and 1 (exclusive)")

    # Standard error = 1.04 / sqrt(m) where m = 2^precision
    # So precision = log2((1.04 / error_rate)^2)
    m_required = (1.04 / error_rate) ** 2
    precision = max(4, min(18, int(math.ceil(math.log2(m_required)))))

    num_registers = 1 << precision
    memory_bytes = num_registers  # 1 byte per register

    standard_error = 1.04 / math.sqrt(num_registers)
    relative_error_percent = standard_error * 100

    return HLLParams(
        precision=precision,
        num_registers=num_registers,
        memory_bytes=memory_bytes,
        standard_error=standard_error,
        relative_error_percent=relative_error_percent,
    )


def estimate_cms_params(
    expected_total_count: int,
    error_rate: float = 0.001,
    confidence: float = 0.99,
) -> CMSParams:
    """
    Estimate optimal Count-Min Sketch parameters.

    Args:
        expected_total_count: Expected total count of all items
        error_rate: Maximum error as fraction of total count (default: 0.001 = 0.1%)
        confidence: Probability that error is within bounds (default: 0.99 = 99%)

    Returns:
        CMSParams with recommended settings

    Example:
        >>> params = estimate_cms_params(10_000_000, 0.0001, 0.99)
        >>> print(f"Width={params.width}, Depth={params.depth}")
    """
    if expected_total_count <= 0:
        raise ValueError("expected_total_count must be positive")
    if error_rate <= 0 or error_rate >= 1:
        raise ValueError("error_rate must be between 0 and 1 (exclusive)")
    if confidence <= 0 or confidence >= 1:
        raise ValueError("confidence must be between 0 and 1 (exclusive)")

    # Width = e / epsilon (e ≈ 2.718)
    width = int(math.ceil(math.e / error_rate))

    # Depth = ln(1 / (1 - confidence))
    delta = 1 - confidence
    depth = int(math.ceil(math.log(1 / delta)))

    memory_bytes = width * depth * 8  # 8 bytes per u64 counter
    memory_mb = memory_bytes / (1024 * 1024)

    return CMSParams(
        width=width,
        depth=depth,
        memory_bytes=memory_bytes,
        memory_mb=memory_mb,
        error_rate=error_rate,
        confidence=confidence,
    )


def estimate_minhash_params(
    target_error: float = 0.1,
) -> MinHashParams:
    """
    Estimate optimal MinHash parameters.

    Args:
        target_error: Target standard error for Jaccard estimate (default: 0.1 = 10%)

    Returns:
        MinHashParams with recommended settings

    Example:
        >>> params = estimate_minhash_params(0.05)
        >>> print(f"Need {params.num_hashes} hashes for 5% error")
    """
    if target_error <= 0 or target_error >= 1:
        raise ValueError("target_error must be between 0 and 1 (exclusive)")

    # Standard error = 1 / sqrt(num_hashes)
    # So num_hashes = 1 / error^2
    num_hashes = int(math.ceil(1 / (target_error ** 2)))

    memory_bytes = num_hashes * 8  # 8 bytes per u64

    standard_error = 1 / math.sqrt(num_hashes)
    relative_error_percent = standard_error * 100

    return MinHashParams(
        num_hashes=num_hashes,
        memory_bytes=memory_bytes,
        standard_error=standard_error,
        relative_error_percent=relative_error_percent,
    )
