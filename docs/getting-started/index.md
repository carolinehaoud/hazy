# Getting Started

Welcome to Hazy! This guide will help you get up and running quickly.

## What You'll Learn

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install hazy from PyPI or build from source

    [:octicons-arrow-right-24: Install now](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Learn the basics with hands-on examples

    [:octicons-arrow-right-24: Get started](quickstart.md)

-   :material-tune:{ .lg .middle } **Parameter Selection**

    ---

    Choose optimal parameters for your use case

    [:octicons-arrow-right-24: Learn more](parameters.md)

</div>

## Prerequisites

- **Python 3.9+** — Hazy supports Python 3.9 through 3.13
- **pip** — Python's package installer

That's it! Hazy comes with pre-built wheels for most platforms, so you don't need Rust installed.

## Quick Install

```bash
pip install hazy
```

## 30-Second Example

```python
from hazy import BloomFilter

# Create a filter for checking usernames
taken_usernames = BloomFilter(expected_items=100000, false_positive_rate=0.01)

# Add some usernames
taken_usernames.add("alice")
taken_usernames.add("bob")
taken_usernames.add("charlie")

# Check if a username is taken
def is_available(username):
    if username in taken_usernames:
        return False  # Probably taken
    return True  # Definitely available

print(is_available("alice"))    # False
print(is_available("newuser"))  # True
```

## Next Steps

1. [Install Hazy](installation.md) on your system
2. Follow the [Quick Start](quickstart.md) tutorial
3. Learn about [parameter selection](parameters.md) for optimal performance
4. Explore the [tutorials](../tutorial/index.md) for real-world examples
