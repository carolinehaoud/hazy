# Installation

## From PyPI

The easiest way to install hazy is from PyPI:

```bash
pip install hazy
```

### Optional Dependencies

For visualization support (matplotlib-based plots):

```bash
pip install hazy[viz]
```

For NumPy integration:

```bash
pip install hazy[numpy]
```

For all optional dependencies:

```bash
pip install hazy[all]
```

## From Source

Building from source requires Rust 1.70+ and Python 3.9+.

### Prerequisites

1. **Install Rust** (if not already installed):

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

2. **Install maturin** (the Rust-Python build tool):

    ```bash
    pip install maturin
    ```

### Build and Install

```bash
# Clone the repository
git clone https://github.com/caroline/hazy.git
cd hazy

# Build and install in development mode
maturin develop

# Or build a release wheel
maturin build --release
pip install target/wheels/hazy-*.whl
```

## Verifying Installation

```python
import hazy
print(hazy.__version__)

# Quick test
bf = hazy.BloomFilter(expected_items=100)
bf.add("test")
print("test" in bf)  # True
```

## Platform Support

Hazy supports:

- **Python**: 3.9, 3.10, 3.11, 3.12, 3.13
- **OS**: Linux, macOS, Windows
- **Architecture**: x86_64, ARM64 (Apple Silicon)

Pre-built wheels are available for most platform combinations. If a wheel isn't available for your platform, pip will automatically build from source (requires Rust).

## Troubleshooting

### Rust Not Found

If you get an error about Rust not being found:

```
error: can't find Rust compiler
```

Install Rust using rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Build Fails on Windows

Make sure you have the Visual C++ Build Tools installed. You can get them from the [Visual Studio downloads page](https://visualstudio.microsoft.com/downloads/) (select "Build Tools for Visual Studio").

### Import Error

If you get an import error after installation:

```python
ImportError: cannot import name 'BloomFilter' from 'hazy'
```

Make sure you're not in the hazy source directory, as Python might try to import from the local `hazy/` folder instead of the installed package.
