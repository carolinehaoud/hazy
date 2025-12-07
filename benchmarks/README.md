# Hazy Benchmarks

Benchmarks comparing hazy to other Python probabilistic data structure libraries.

## Running Benchmarks

```bash
# Run all benchmarks
python benchmarks/bench_all.py

# Run individual benchmarks
python benchmarks/bench_bloom.py
python benchmarks/bench_hll.py
python benchmarks/bench_cms.py
python benchmarks/bench_minhash.py
```

## Comparison Libraries

Install comparison libraries to see full benchmark results:

```bash
pip install pyprobables datasketch pybloom-live bloom-filter2 countminsketch
```

| Library | Structures | Implementation |
|---------|-----------|----------------|
| [pyprobables](https://github.com/barrust/pyprobables) | Bloom, HLL, CMS, Cuckoo | Pure Python |
| [datasketch](https://github.com/ekzhu/datasketch) | HLL, MinHash, LSH | Cython optimized |
| [pybloom-live](https://github.com/jaybaird/python-bloomfilter) | Bloom | Pure Python |
| [bloom-filter2](https://github.com/remram44/python-bloom-filter) | Bloom | Pure Python |
| [countminsketch](https://github.com/rafacarrascosa/countminsketch) | CMS | Pure Python |

## Typical Results

On a modern machine (M1/M2 Mac, recent Intel/AMD), hazy typically shows:

- **BloomFilter**: 5-20x faster than pure Python implementations
- **HyperLogLog**: 3-10x faster than pure Python, comparable to Cython
- **CountMinSketch**: 5-15x faster than pure Python
- **MinHash**: 2-5x faster than datasketch

Results vary based on:
- Python version
- CPU architecture
- Data characteristics
- Operation mix (add vs query heavy)

## Memory Usage

hazy uses binary serialization (bincode) which is typically 2-5x smaller than JSON or pickle serialization used by other libraries.

## Adding New Benchmarks

Each benchmark script follows a consistent pattern:

1. Define benchmark function that measures ops/sec
2. Implement adapter for each library's API
3. Run benchmarks and print comparison table

See existing scripts for examples.
