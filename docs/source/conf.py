import os
import sys

sys.path.insert(0, os.path.abspath('../casino'))

project = 'pycasino'
copyright = '2024, Konkov Vladimir'
release = version = '0.4.0'
author = 'Konkov Vladimir'

extensions = ['sphinx.ext.mathjax']

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
