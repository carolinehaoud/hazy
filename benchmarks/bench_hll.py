"""
Benchmarks comparing hazy HyperLogLog to other Python implementations.

Compares against:
- datasketch (Cython optimized)
- pyprobables (pure Python)

Run with: python benchmarks/bench_hll.py
"""

import time
import sys
from typing import Callable, Any

# Number of items to test with
N_ITEMS = 1_000_000


def benchmark(name: str, setup: Callable[[], Any], add_fn: Callable[[Any, str], None],
              cardinality_fn: Callable[[Any], float], n_items: int = N_ITEMS) -> dict:
    """Run benchmark for a HyperLogLog implementation."""
    results = {"name": name}

    # Setup
    try:
        hll = setup()
    except Exception as e:
        return {"name": name, "error": str(e)}

    # Benchmark adds
    items = [f"item_{i}" for i in range(n_items)]
    start = time.perf_counter()
    for item in items:
        add_fn(hll, item)
    add_time = time.perf_counter() - start
    results["add_time"] = add_time
    results["add_ops_per_sec"] = n_items / add_time

    # Benchmark cardinality
    start = time.perf_counter()
    for _ in range(1000):
        cardinality_fn(hll)
    card_time = time.perf_counter() - start
    results["cardinality_time"] = card_time / 1000
    results["cardinality_ops_per_sec"] = 1000 / card_time

    # Get estimated cardinality and error
    estimated = cardinality_fn(hll)
    results["estimated_cardinality"] = estimated
    results["error_pct"] = abs(estimated - n_items) / n_items * 100

    # Memory size if available
    try:
        results["size_bytes"] = hll.size_in_bytes
    except AttributeError:
        try:
            results["size_bytes"] = sys.getsizeof(hll)
        except:
            pass

    return results


def bench_hazy():
    """Benchmark hazy HyperLogLog."""
    from hazy import HyperLogLog

    return benchmark(
        name="hazy (Rust)",
        setup=lambda: HyperLogLog(precision=14),
        add_fn=lambda hll, item: hll.add(item),
        cardinality_fn=lambda hll: hll.cardinality(),
    )


def bench_datasketch():
    """Benchmark datasketch HyperLogLog."""
    try:
        from datasketch import HyperLogLog
    except ImportError:
        return {"name": "datasketch", "error": "not installed (pip install datasketch)"}

    return benchmark(
        name="datasketch",
        setup=lambda: HyperLogLog(p=14),
        add_fn=lambda hll, item: hll.update(item.encode()),
        cardinality_fn=lambda hll: hll.count(),
    )


def bench_pyprobables():
    """Benchmark pyprobables HyperLogLog."""
    try:
        from probables import HyperLogLog
    except ImportError:
        return {"name": "pyprobables", "error": "not installed (pip install pyprobables)"}

    return benchmark(
        name="pyprobables",
        setup=lambda: HyperLogLog(width=14),
        add_fn=lambda hll, item: hll.add(item),
        cardinality_fn=lambda hll: hll.cardinality(),
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
    print("HyperLogLog Benchmark")
    print(f"Items: {N_ITEMS:,}")
    print("=" * 70)
    print()

    benchmarks = [
        bench_hazy,
        bench_datasketch,
        bench_pyprobables,
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
            print(f"  Cardinality: {format_number(result['cardinality_ops_per_sec'])} ops/sec")
            print(f"  Estimated: {result['estimated_cardinality']:,.0f} (error: {result['error_pct']:.2f}%)")
        print()

    # Print comparison table
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Library':<20} {'Add (ops/s)':<15} {'Card (ops/s)':<15} {'Error %':<10}")
    print("-" * 70)

    for result in results:
        if "error" in result:
            print(f"{result['name']:<20} {'N/A':<15} {'N/A':<15} {'N/A':<10}")
        else:
            add_ops = format_number(result["add_ops_per_sec"])
            card_ops = format_number(result["cardinality_ops_per_sec"])
            error = f"{result['error_pct']:.2f}%"
            print(f"{result['name']:<20} {add_ops:<15} {card_ops:<15} {error:<10}")

    # Calculate speedup vs other libraries
    hazy_result = results[0]
    if "error" not in hazy_result:
        print("\n" + "=" * 70)
        print("Speedup (hazy vs others)")
        print("=" * 70)
        for result in results[1:]:
            if "error" not in result:
                add_speedup = hazy_result["add_ops_per_sec"] / result["add_ops_per_sec"]
                card_speedup = hazy_result["cardinality_ops_per_sec"] / result["cardinality_ops_per_sec"]
                print(f"vs {result['name']:<18} Add: {add_speedup:>6.1f}x  Cardinality: {card_speedup:>6.1f}x")


if __name__ == "__main__":
    main()
