"""
Benchmarks comparing hazy MinHash to other Python implementations.

Compares against:
- datasketch (Cython optimized)

Run with: python benchmarks/bench_minhash.py
"""

import time
import sys
from typing import Callable, Any

# Number of items to test with
N_ITEMS = 10_000
N_HASHES = 128


def benchmark(name: str, setup: Callable[[], Any], add_fn: Callable[[Any, str], None],
              jaccard_fn: Callable[[Any, Any], float], n_items: int = N_ITEMS) -> dict:
    """Run benchmark for a MinHash implementation."""
    results = {"name": name}

    # Setup
    try:
        mh1 = setup()
        mh2 = setup()
    except Exception as e:
        return {"name": name, "error": str(e)}

    # Create overlapping sets (50% overlap)
    items1 = [f"item_{i}" for i in range(n_items)]
    items2 = [f"item_{i}" for i in range(n_items // 2, n_items + n_items // 2)]

    # Benchmark adds for mh1
    start = time.perf_counter()
    for item in items1:
        add_fn(mh1, item)
    add_time = time.perf_counter() - start
    results["add_time"] = add_time
    results["add_ops_per_sec"] = n_items / add_time

    # Add items to mh2
    for item in items2:
        add_fn(mh2, item)

    # Benchmark Jaccard estimation
    start = time.perf_counter()
    for _ in range(1000):
        jaccard_fn(mh1, mh2)
    jaccard_time = time.perf_counter() - start
    results["jaccard_time"] = jaccard_time / 1000
    results["jaccard_ops_per_sec"] = 1000 / jaccard_time

    # Get estimated Jaccard and error
    estimated = jaccard_fn(mh1, mh2)
    # True Jaccard for 50% overlap: intersection / union = 5000 / 15000 = 0.333
    true_jaccard = 0.333
    results["estimated_jaccard"] = estimated
    results["error_pct"] = abs(estimated - true_jaccard) / true_jaccard * 100

    # Memory size if available
    try:
        results["size_bytes"] = mh1.size_in_bytes
    except AttributeError:
        try:
            results["size_bytes"] = sys.getsizeof(mh1)
        except:
            pass

    return results


def bench_hazy():
    """Benchmark hazy MinHash."""
    from hazy import MinHash

    return benchmark(
        name="hazy (Rust)",
        setup=lambda: MinHash(num_hashes=N_HASHES),
        add_fn=lambda mh, item: mh.add(item),
        jaccard_fn=lambda mh1, mh2: mh1.jaccard(mh2),
    )


def bench_datasketch():
    """Benchmark datasketch MinHash."""
    try:
        from datasketch import MinHash
    except ImportError:
        return {"name": "datasketch", "error": "not installed (pip install datasketch)"}

    return benchmark(
        name="datasketch",
        setup=lambda: MinHash(num_perm=N_HASHES),
        add_fn=lambda mh, item: mh.update(item.encode()),
        jaccard_fn=lambda mh1, mh2: mh1.jaccard(mh2),
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
    print("MinHash Benchmark")
    print(f"Items: {N_ITEMS:,}, Hashes: {N_HASHES}")
    print("=" * 70)
    print()

    benchmarks = [
        bench_hazy,
        bench_datasketch,
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
            print(f"  Jaccard: {format_number(result['jaccard_ops_per_sec'])} ops/sec")
            print(f"  Estimated Jaccard: {result['estimated_jaccard']:.3f} (error: {result['error_pct']:.1f}%)")
        print()

    # Print comparison table
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Library':<20} {'Add (ops/s)':<15} {'Jaccard (ops/s)':<15} {'Error %':<10}")
    print("-" * 70)

    for result in results:
        if "error" in result:
            print(f"{result['name']:<20} {'N/A':<15} {'N/A':<15} {'N/A':<10}")
        else:
            add_ops = format_number(result["add_ops_per_sec"])
            jaccard_ops = format_number(result["jaccard_ops_per_sec"])
            error = f"{result['error_pct']:.1f}%"
            print(f"{result['name']:<20} {add_ops:<15} {jaccard_ops:<15} {error:<10}")

    # Calculate speedup vs other libraries
    hazy_result = results[0]
    if "error" not in hazy_result:
        print("\n" + "=" * 70)
        print("Speedup (hazy vs others)")
        print("=" * 70)
        for result in results[1:]:
            if "error" not in result:
                add_speedup = hazy_result["add_ops_per_sec"] / result["add_ops_per_sec"]
                jaccard_speedup = hazy_result["jaccard_ops_per_sec"] / result["jaccard_ops_per_sec"]
                print(f"vs {result['name']:<18} Add: {add_speedup:>6.1f}x  Jaccard: {jaccard_speedup:>6.1f}x")


if __name__ == "__main__":
    main()
