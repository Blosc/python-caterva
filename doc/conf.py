# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

if 'READ_THE_DOCS' in os.environ:
    import subprocess
    # It is necessary for the readthedocs for finding cython docstrings
    subprocess.call('cd .. && python setup.py build_ext -i && cd ../doc', shell=True)

# -- Project information -----------------------------------------------------

project = 'cat4py'
copyright = '2019-2020, The Blosc Developers'
author = 'The Blosc Developers'

import cat4py
release = cat4py.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'nbsphinx',
    'numpydoc',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'

language = None

exclude_patterns = ['_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']

pygments_style = None

html_static_path = ["_static"]
html_theme = 'sphinx_rtd_theme'
html_logo = "_static/cat4py.svg"
html_favicon = "_static/caterva-logo.svg"
html_show_sourcelink = False

html_theme_options = {
    "logo_only": True,
}

autodoc_member_order = 'groupwise'
nbsphinx_exclude_output_prompt = False


def setup(app):
    app.add_css_file('custom.css')
