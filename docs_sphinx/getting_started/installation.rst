Installation Guide
==================

Requirements
------------

- Python 3.9 or higher
- pip (Python package manager)

Basic Installation
------------------

Install Hazy from PyPI:

.. code-block:: bash

   pip install hazy

This installs the core library with all data structures.

Optional Dependencies
---------------------

Visualization Support
~~~~~~~~~~~~~~~~~~~~~

To use the built-in visualization functions:

.. code-block:: bash

   pip install hazy[viz]

This includes ``matplotlib`` for plotting.

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development with all dependencies:

.. code-block:: bash

   pip install hazy[dev]

This includes:

- ``pytest`` for testing
- ``hypothesis`` for property-based testing
- ``matplotlib`` for visualization
- ``sphinx`` for documentation

Verifying Installation
----------------------

Verify your installation works:

.. code-block:: python

   import hazy
   print(hazy.__version__)

   # Quick test
   bf = hazy.BloomFilter(expected_items=1000)
   bf.add("hello")
   print("hello" in bf)  # True

Building from Source
--------------------

If you want to build from source (requires Rust):

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/hazy.git
   cd hazy

   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

   # Build and install
   pip install maturin
   maturin develop --release

Troubleshooting
---------------

**Import Error**: Make sure you're using Python 3.9+:

.. code-block:: bash

   python --version

**Build Errors**: If building from source, ensure Rust is installed:

.. code-block:: bash

   rustc --version
