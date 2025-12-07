# Tutorials

Learn how to use Hazy through practical, real-world examples. Each tutorial includes complete, runnable code.

## Available Tutorials

<div class="grid cards" markdown>

-   :material-google-analytics:{ .lg .middle } **Web Analytics Dashboard**

    ---

    Build a real-time analytics system that tracks:

    - Unique visitors per page
    - Page view frequencies
    - Trending content

    All using minimal memory with probabilistic data structures.

    [:octicons-arrow-right-24: Start tutorial](web-analytics.md)

-   :material-content-duplicate:{ .lg .middle } **Stream Deduplication**

    ---

    Process streaming data while detecting duplicates:

    - URL deduplication for web crawlers
    - Event deduplication in pipelines
    - Exactly-once processing patterns

    [:octicons-arrow-right-24: Start tutorial](deduplication.md)

-   :material-file-search:{ .lg .middle } **Document Similarity Search**

    ---

    Find similar documents efficiently:

    - Near-duplicate detection
    - Content recommendation
    - Plagiarism detection basics

    Using MinHash and Locality-Sensitive Hashing concepts.

    [:octicons-arrow-right-24: Start tutorial](similarity-search.md)

</div>

## Prerequisites

Before starting the tutorials, make sure you have:

1. **Hazy installed** with visualization support:
   ```bash
   pip install hazy[viz]
   ```

2. **Basic Python knowledge** — familiarity with classes, functions, and data structures

3. **A code editor** or Jupyter notebook environment

## Tutorial Format

Each tutorial follows a consistent structure:

1. **Problem Statement** — What we're trying to solve
2. **Solution Design** — Which data structures to use and why
3. **Implementation** — Step-by-step code with explanations
4. **Visualization** — Graphs and charts to understand the results
5. **Performance Analysis** — Memory and accuracy trade-offs
6. **Exercises** — Try it yourself challenges

## Which Tutorial Should I Start With?

| If you want to... | Start with |
|-------------------|------------|
| Learn the basics | [Web Analytics](web-analytics.md) |
| Process streaming data | [Deduplication](deduplication.md) |
| Find similar items | [Similarity Search](similarity-search.md) |

## Running the Examples

All code examples can be run directly. You can either:

**Option 1: Copy-paste into Python**
```python
# Copy any code block and run it
python my_script.py
```

**Option 2: Use Jupyter notebooks**
```python
import hazy
hazy.enable_notebook_display()  # For rich output
```

**Option 3: Interactive REPL**
```bash
python -i
>>> from hazy import *
```
