"""
Benchmarks comparing hazy CountMinSketch to other Python implementations.

Compares against:
- datasketch (Cython optimized)
- pyprobables (pure Python)

Run with: python benchmarks/bench_cms.py
"""

import time
import sys
import random
from typing import Callable, Any

# Number of items to test with
N_ITEMS = 500_000
N_QUERIES = 100_000


def benchmark(name: str, setup: Callable[[], Any], add_fn: Callable[[Any, str], None],
              query_fn: Callable[[Any, str], int], n_items: int = N_ITEMS) -> dict:
    """Run benchmark for a Count-Min Sketch implementation."""
    results = {"name": name}

    # Setup
    try:
        cms = setup()
    except Exception as e:
        return {"name": name, "error": str(e)}

    # Create items with Zipf-like distribution
    items = []
    for i in range(n_items):
        # More frequent items have lower indices
        item_id = int(random.paretovariate(1.5)) % 10000
        items.append(f"item_{item_id}")

    # Benchmark adds
    start = time.perf_counter()
    for item in items:
        add_fn(cms, item)
    add_time = time.perf_counter() - start
    results["add_time"] = add_time
    results["add_ops_per_sec"] = n_items / add_time

    # Benchmark queries
    query_items = [f"item_{i % 10000}" for i in range(N_QUERIES)]

    start = time.perf_counter()
    for item in query_items:
        query_fn(cms, item)
    query_time = time.perf_counter() - start
    results["query_time"] = query_time
    results["query_ops_per_sec"] = N_QUERIES / query_time

    # Memory size if available
    try:
        results["size_bytes"] = cms.size_in_bytes
    except AttributeError:
        try:
            results["size_bytes"] = sys.getsizeof(cms)
        except:
            pass

    return results


def bench_hazy():
    """Benchmark hazy CountMinSketch."""
    from hazy import CountMinSketch

    return benchmark(
        name="hazy (Rust)",
        setup=lambda: CountMinSketch(width=10000, depth=5),
        add_fn=lambda cms, item: cms.add(item),
        query_fn=lambda cms, item: cms[item],
    )


def bench_datasketch():
    """Benchmark datasketch CountMinSketch."""
    try:
        from datasketch import MinHash  # datasketch doesn't have CMS
        return {"name": "datasketch", "error": "no CountMinSketch implementation"}
    except ImportError:
        return {"name": "datasketch", "error": "not installed (pip install datasketch)"}


def bench_pyprobables():
    """Benchmark pyprobables CountMinSketch."""
    try:
        from probables import CountMinSketch
    except ImportError:
        return {"name": "pyprobables", "error": "not installed (pip install pyprobables)"}

    return benchmark(
        name="pyprobables",
        setup=lambda: CountMinSketch(width=10000, depth=5),
        add_fn=lambda cms, item: cms.add(item),
        query_fn=lambda cms, item: cms.check(item),
    )


def bench_countminsketch():
    """Benchmark countminsketch package."""
    try:
        from countminsketch import CountMinSketch
    except ImportError:
        return {"name": "countminsketch", "error": "not installed (pip install countminsketch)"}

    return benchmark(
        name="countminsketch",
        setup=lambda: CountMinSketch(10000, 5),
        add_fn=lambda cms, item: cms.increment(item),
        query_fn=lambda cms, item: cms[item],
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
    print("Count-Min Sketch Benchmark")
    print(f"Items: {N_ITEMS:,}, Queries: {N_QUERIES:,}")
    print("=" * 70)
    print()

    benchmarks = [
        bench_hazy,
        bench_pyprobables,
        bench_countminsketch,
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
