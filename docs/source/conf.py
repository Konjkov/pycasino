import os
import sys

sys.path.insert(0, os.path.abspath('../casino'))

project = 'pycasino'
copyright = '2024, Konkov Vladimir'
release = version = '0.3.0'
author = 'Konkov Vladimir'

extensions = ['sphinx.ext.mathjax']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'classic'
html_static_path = ['_static']
