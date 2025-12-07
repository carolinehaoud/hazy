# Contributing

Thank you for your interest in contributing to Hazy! This guide will help you get started.

## Development Setup

### Prerequisites

- **Rust** 1.70+ ([install](https://rustup.rs/))
- **Python** 3.9+
- **maturin** for building the Rust extension

### Setup

```bash
# Clone the repository
git clone https://github.com/caroline/hazy.git
cd hazy

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e ".[dev]"

# Build the Rust extension in development mode
maturin develop

# Run tests to verify setup
pytest
```

## Project Structure

```
hazy/
├── src/                    # Rust source code
│   ├── lib.rs              # PyO3 module entry point
│   ├── bloom.rs            # BloomFilter implementation
│   ├── counting_bloom.rs   # CountingBloomFilter
│   ├── scalable_bloom.rs   # ScalableBloomFilter
│   ├── cuckoo.rs           # CuckooFilter
│   ├── hyperloglog.rs      # HyperLogLog
│   ├── count_min_sketch.rs # CountMinSketch
│   ├── minhash.rs          # MinHash
│   ├── topk.rs             # TopK
│   └── utils.rs            # Shared utilities
├── python/hazy/            # Python source code
│   ├── __init__.py         # Package exports
│   ├── _helpers.py         # Parameter estimation
│   ├── _compat.py          # NumPy compatibility
│   ├── _jupyter.py         # Jupyter integration
│   ├── _hazy.pyi           # Type stubs
│   └── viz.py              # Visualization
├── tests/                  # Python tests
├── docs/                   # Documentation (MkDocs)
├── benchmarks/             # Performance benchmarks
├── Cargo.toml              # Rust dependencies
└── pyproject.toml          # Python package config
```

## Making Changes

### Rust Code

1. All data structure implementations are in `src/`
2. Follow existing patterns for new structures:
   - Implement `#[pyclass]` with `#[pymethods]`
   - Add serialization with `Serialize`/`Deserialize`
   - Include `to_bytes()`, `from_bytes()`, `to_json()`, `from_json()`, `save()`, `load()`
   - Add `__len__`, `__contains__`, `__repr__` for Pythonic API
3. Use `xxhash` for hashing (see `utils.rs`)
4. Add the new class to `lib.rs`

### Python Code

1. Export new classes in `python/hazy/__init__.py`
2. Add type stubs in `python/hazy/_hazy.pyi`
3. Add parameter helpers in `python/hazy/_helpers.py`
4. Add visualization in `python/hazy/viz.py`
5. Add Jupyter HTML in `python/hazy/_jupyter.py`

### Tests

1. Add tests in `tests/test_<structure>.py`
2. Add property-based tests in `tests/test_<structure>_properties.py`
3. Cover: basic usage, edge cases, serialization, validation errors
4. Run tests: `pytest -v`
5. Check coverage: `pytest --cov=hazy`

### Documentation

1. Add/update pages in `docs/`
2. Follow existing structure and formatting
3. Include code examples that actually work
4. Build locally: `mkdocs serve`

## Code Style

### Rust

```bash
# Format code
cargo fmt

# Check for issues
cargo clippy

# Run before committing
cargo fmt && cargo clippy -- -D warnings
```

Guidelines:
- Use descriptive variable names
- Add doc comments for public functions
- Follow Rust naming conventions

### Python

```bash
# Check with ruff
ruff check python/

# Format with ruff
ruff format python/
```

Guidelines:
- Follow PEP 8
- Use type hints
- Add docstrings with examples

## Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_bloom.py

# With coverage
pytest --cov=hazy --cov-report=html

# Property-based tests only
pytest tests/test_*_properties.py

# Verbose output
pytest -v
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Make changes** following the guidelines above
4. **Add tests** for new functionality
5. **Run the test suite**: `pytest`
6. **Update documentation** if needed
7. **Commit** with a clear message
8. **Push** to your fork and open a Pull Request

### PR Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Rust code formatted (`cargo fmt`)
- [ ] No Clippy warnings (`cargo clippy`)
- [ ] Python code formatted (`ruff format python/`)
- [ ] No ruff warnings (`ruff check python/`)
- [ ] Type stubs updated if API changed
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated for notable changes

## Adding a New Data Structure

1. Create `src/newstructure.rs` with the implementation
2. Add `mod newstructure;` and `use` in `src/lib.rs`
3. Register with `m.add_class::<NewStructure>()?;`
4. Add to `python/hazy/__init__.py` exports
5. Add type stubs in `python/hazy/_hazy.pyi`
6. Add parameter helper in `python/hazy/_helpers.py` (if applicable)
7. Add visualization in `python/hazy/viz.py` (if applicable)
8. Add Jupyter HTML in `python/hazy/_jupyter.py`
9. Create `tests/test_newstructure.py`
10. Create `tests/test_newstructure_properties.py`
11. Add documentation in `docs/structures/`
12. Update README.md and CHANGELOG.md

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include Python version, OS, and hazy version
- For bugs, include a minimal reproducible example

## Questions?

Open a GitHub Discussion or Issue if you have questions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
