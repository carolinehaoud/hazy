.. Hazy documentation master file

.. image:: _static/logo-hero.svg
   :alt: Hazy
   :align: center
   :width: 150px

Welcome to Hazy's documentation!
================================

.. image:: https://img.shields.io/pypi/v/hazy?color=blue&label=PyPI
   :target: https://pypi.org/project/hazy/
   :alt: PyPI

.. image:: https://img.shields.io/badge/python-3.9+-blue
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/license-MIT-green
   :alt: MIT License

**Hazy** is a Python library providing fast, memory-efficient probabilistic data structures powered by Rust.

These data structures trade perfect accuracy for massive memory savings and speed improvements, making them ideal for big data applications.

Getting Started
---------------

If you're new to Hazy, start with the :doc:`getting_started/installation` guide, then try the :doc:`getting_started/quickstart` tutorial.

For a more comprehensive introduction, check out the :doc:`tutorials/web_analytics` tutorial which demonstrates how to build a complete analytics system.

Quick Example
-------------

.. code-block:: python

   from hazy import BloomFilter, HyperLogLog, CountMinSketch

   # Membership testing with Bloom filter
   bf = BloomFilter(expected_items=1_000_000, false_positive_rate=0.01)
   bf.add("hello")
   print("hello" in bf)  # True

   # Cardinality estimation with HyperLogLog
   hll = HyperLogLog(precision=14)
   for i in range(1_000_000):
       hll.add(f"user_{i}")
   print(f"Unique users: {hll.cardinality():,.0f}")  # ~1,000,000

   # Frequency estimation with Count-Min Sketch
   cms = CountMinSketch(width=10000, depth=5)
   for word in ["apple", "apple", "banana", "apple"]:
       cms.add(word)
   print(f"apple count: {cms['apple']}")  # 3

Features
--------

- **8 Data Structures**: BloomFilter, CountingBloomFilter, ScalableBloomFilter, CuckooFilter, HyperLogLog, CountMinSketch, MinHash, TopK
- **Rust Performance**: 5-20x faster than pure Python implementations
- **Memory Efficient**: Use megabytes instead of gigabytes
- **Easy to Use**: Pythonic API with type hints
- **Serialization**: Save/load to bytes, JSON, or files
- **Visualization**: Built-in matplotlib plotting functions

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/web_analytics
   tutorials/deduplication
   tutorials/similarity_search
   tutorials/fraud_detection
   tutorials/leaderboards
   tutorials/database_optimization
   tutorials/stream_processing

.. toctree::
   :maxdepth: 2
   :caption: Data Structures

   structures/bloom_filter
   structures/counting_bloom_filter
   structures/scalable_bloom_filter
   structures/cuckoo_filter
   structures/hyperloglog
   structures/count_min_sketch
   structures/minhash
   structures/topk

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/serialization
   guides/visualization
   guides/alternatives

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
