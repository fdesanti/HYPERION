# Minimal makefile for Sphinx documentation

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