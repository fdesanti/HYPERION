# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# # from the environment for the first two.
# SPHINXOPTS    ?=
# SPHINXBUILD   ?= sphinx-build
# SOURCEDIR     = .
# BUILDDIR      = _build

# # Put it first so that "make" without argument is like "make help".
# help:
# 	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# .PHONY: help Makefile

# # Catch-all target: route all unknown targets to Sphinx using the new
# # "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# apidoc:
# 	sphinx-apidoc -o api ../hyperion --separate

# organize:
# 	python organize_api.py

# html: apidoc organize
# 	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

SHELL := /bin/bash

SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
APIDIR        = api
PACKAGE_DIR   = ../hyperion  # Adjust to point to your package

.PHONY: clean clean_api apidoc indexes html docs

# Remove previous build files
clean:
	rm -rf $(BUILDDIR)/*

# Remove the generated API folder to force a fresh generation
clean_api:
	rm -rf $(APIDIR)

# Generate API .rst files using sphinx-apidoc; depends on clean_api to start fresh
apidoc: clean_api
	sphinx-apidoc -o $(APIDIR) $(PACKAGE_DIR) --separate

# Run the organization/index-generation script
indexes:
	python generate_api_indexes.py

# Build HTML documentation (depends on clean, apidoc, and indexes)
html: clean apidoc indexes
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

# One-shot target to build everything
docs: html
	@echo "Documentation build complete."