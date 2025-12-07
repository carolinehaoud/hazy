"""
Benchmarks comparing hazy BloomFilter to other Python implementations.

Compares against:
- pyprobables (pure Python)
- pybloom-live (pure Python, if available)
- bloom-filter2 (pure Python, if available)

Run with: python benchmarks/bench_bloom.py
"""

import time
import sys
from typing import Callable, Any

# Number of items to test with
N_ITEMS = 100_000
N_QUERIES = 100_000


def benchmark(name: str, setup: Callable[[], Any], add_fn: Callable[[Any, str], None],
              query_fn: Callable[[Any, str], bool], n_items: int = N_ITEMS) -> dict:
    """Run benchmark for a Bloom filter implementation."""
    results = {"name": name}

    # Setup
    try:
        bf = setup()
    except Exception as e:
        return {"name": name, "error": str(e)}

    # Benchmark adds
    items = [f"item_{i}" for i in range(n_items)]
    start = time.perf_counter()
    for item in items:
        add_fn(bf, item)
    add_time = time.perf_counter() - start
    results["add_time"] = add_time
    results["add_ops_per_sec"] = n_items / add_time

    # Benchmark queries (mix of hits and misses)
    query_items = [f"item_{i}" for i in range(N_QUERIES // 2)]  # hits
    query_items += [f"missing_{i}" for i in range(N_QUERIES // 2)]  # misses

    start = time.perf_counter()
    for item in query_items:
        query_fn(bf, item)
    query_time = time.perf_counter() - start
    results["query_time"] = query_time
    results["query_ops_per_sec"] = N_QUERIES / query_time

    # Memory size if available
    try:
        results["size_bytes"] = bf.size_in_bytes
    except AttributeError:
        try:
            results["size_bytes"] = sys.getsizeof(bf)
        except:
            pass

    return results


def bench_hazy():
    """Benchmark hazy BloomFilter."""
    from hazy import BloomFilter

    return benchmark(
        name="hazy (Rust)",
        setup=lambda: BloomFilter(expected_items=N_ITEMS, false_positive_rate=0.01),
        add_fn=lambda bf, item: bf.add(item),
        query_fn=lambda bf, item: item in bf,
    )


def bench_pyprobables():
    """Benchmark pyprobables BloomFilter."""
    try:
        from probables import BloomFilter
    except ImportError:
        return {"name": "pyprobables", "error": "not installed (pip install pyprobables)"}

    return benchmark(
        name="pyprobables",
        setup=lambda: BloomFilter(est_elements=N_ITEMS, false_positive_rate=0.01),
        add_fn=lambda bf, item: bf.add(item),
        query_fn=lambda bf, item: bf.check(item),
    )


def bench_pybloom_live():
    """Benchmark pybloom-live BloomFilter."""
    try:
        from pybloom_live import BloomFilter
    except ImportError:
        return {"name": "pybloom-live", "error": "not installed (pip install pybloom-live)"}

    return benchmark(
        name="pybloom-live",
        setup=lambda: BloomFilter(capacity=N_ITEMS, error_rate=0.01),
        add_fn=lambda bf, item: bf.add(item),
        query_fn=lambda bf, item: item in bf,
    )


def bench_bloom_filter2():
    """Benchmark bloom-filter2."""
    try:
        from bloom_filter2 import BloomFilter
    except ImportError:
        return {"name": "bloom-filter2", "error": "not installed (pip install bloom-filter2)"}

    return benchmark(
        name="bloom-filter2",
        setup=lambda: BloomFilter(max_elements=N_ITEMS, error_rate=0.01),
        add_fn=lambda bf, item: bf.add(item),
        query_fn=lambda bf, item: item in bf,
    )


def format_number(n: float) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.2f}K"
    else:
        return f"{n:.2f}"


def main():
    print("=" * 70)
    print("Bloom Filter Benchmark")
    print(f"Items: {N_ITEMS:,}, Queries: {N_QUERIES:,}")
    print("=" * 70)
    print()

    benchmarks = [
        bench_hazy,
        bench_pyprobables,
        bench_pybloom_live,
        bench_bloom_filter2,
    ]

    results = []
    for bench_fn in benchmarks:
        print(f"Running {bench_fn.__name__}...")
        result = bench_fn()
        results.append(result)

        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Add: {format_number(result['add_ops_per_sec'])} ops/sec")
            print(f"  Query: {format_number(result['query_ops_per_sec'])} ops/sec")
        print()

    # Print comparison table
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Library':<20} {'Add (ops/s)':<15} {'Query (ops/s)':<15} {'Size (KB)':<12}")
    print("-" * 70)

    for result in results:
        if "error" in result:
            print(f"{result['name']:<20} {'N/A':<15} {'N/A':<15} {'N/A':<12}")
        else:
            add_ops = format_number(result["add_ops_per_sec"])
            query_ops = format_number(result["query_ops_per_sec"])
            size = result.get("size_bytes", 0) / 1024
            print(f"{result['name']:<20} {add_ops:<15} {query_ops:<15} {size:<12.1f}")

    # Calculate speedup vs other libraries
    hazy_result = results[0]
    if "error" not in hazy_result:
        print("\n" + "=" * 70)
        print("Speedup (hazy vs others)")
        print("=" * 70)
        for result in results[1:]:
            if "error" not in result:
                add_speedup = hazy_result["add_ops_per_sec"] / result["add_ops_per_sec"]
                query_speedup = hazy_result["query_ops_per_sec"] / result["query_ops_per_sec"]
                print(f"vs {result['name']:<18} Add: {add_speedup:>6.1f}x  Query: {query_speedup:>6.1f}x")


if __name__ == "__main__":
    main()
