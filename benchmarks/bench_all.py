#!/usr/bin/env python
"""
Run all hazy benchmarks and generate a summary report.

Usage:
    python benchmarks/bench_all.py

This will run benchmarks for:
- BloomFilter
- HyperLogLog
- CountMinSketch
- MinHash

And compare against other Python libraries where available.
"""

import subprocess
import sys
from pathlib import Path


def run_benchmark(script: str) -> None:
    """Run a benchmark script and print its output."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {script}")
    print("=" * 80 + "\n")

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print(f"Benchmark {script} failed with return code {result.returncode}")


def main():
    print("=" * 80)
    print("HAZY BENCHMARKS")
    print("Comparing hazy (Rust) vs other Python probabilistic data structures")
    print("=" * 80)

    # Find benchmark scripts
    bench_dir = Path(__file__).parent
    scripts = sorted(bench_dir.glob("bench_*.py"))
    scripts = [s for s in scripts if s.name != "bench_all.py"]

    if not scripts:
        print("No benchmark scripts found!")
        sys.exit(1)

    print(f"\nFound {len(scripts)} benchmark scripts:")
    for script in scripts:
        print(f"  - {script.name}")

    # Run each benchmark
    for script in scripts:
        run_benchmark(str(script))

    print("\n" + "=" * 80)
    print("BENCHMARKS COMPLETE")
    print("=" * 80)

    print("\nTo install comparison libraries:")
    print("  pip install pyprobables datasketch pybloom-live bloom-filter2 countminsketch")


if __name__ == "__main__":
    main()
