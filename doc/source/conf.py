# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "spires-r0"
copyright = "2026, Niklas Griessbaum"
author = "Niklas Griessbaum"

# Version from the installed dist (setuptools_scm at build time); fall back to
# reading git tags directly in a source checkout.
try:
    from importlib.metadata import version

    release = version("spires-r0")
except Exception:
    try:
        from setuptools_scm import get_version

        release = get_version(root="../..", relative_to=__file__)
    except Exception:
        release = "unknown"

version = release

# -- General configuration ---------------------------------------------------
# This package is scaffolding; the docs are narrative-only (no autodoc yet).

extensions = [
    "myst_parser",             # Markdown
    "sphinx.ext.intersphinx",  # cross-link to the rest of the family
    "sphinx_markdown_tables",
]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "spires_contract": ("https://spires-contract.readthedocs.io/en/latest/", None),
}

suppress_warnings = ["myst.xref_missing"]

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
