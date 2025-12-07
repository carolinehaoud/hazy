# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the project root to the path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'Hazy'
copyright = '2024, Hazy Contributors'
author = 'Hazy Contributors'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

# MyST parser settings (allows markdown files)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Favicon only (no sidebar logo)
html_favicon = '_static/logo.svg'

html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom CSS
html_css_files = [
    'custom.css',
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
