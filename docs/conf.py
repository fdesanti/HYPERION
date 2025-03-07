# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os 

project = 'HYPERION'
copyright = '2025, Federico De Santi'
author = 'Federico De Santi'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.mathjax', 'sphinx.ext.viewcode', "myst_parser"]

templates_path = ['_templates']

api_dir = 'hyperion'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', f'{api_dir}/config.rst', f'{api_dir}/modules.rst']

#exlude unnecessary files so that we don't see empty duplicates in the documentation
# Walk through the API directory recursively.
for root, dirs, files in os.walk(api_dir):
    for d in dirs:
        # Construct the expected rst file path corresponding to this directory.
        rst_path = os.path.join(root, f"{d}.rst")
        if os.path.exists(rst_path):
            exclude_patterns.append(rst_path)

def skip_member(app, what, name, obj, skip, options):
    if name == '__call__':
        return False  # Do not skip __call__
    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip_member)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
suppress_warnings = ['autodoc']
