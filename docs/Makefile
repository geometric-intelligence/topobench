# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean-api apidocs update-leaderboard

# Generate API documentation from source code
apidocs:
	@echo "Generating API documentation..."
	@bash generate_api_docs.sh

# Clean only the API documentation
clean-api:
	@echo "Cleaning API documentation..."
	@rm -rf api/
	@echo "API documentation cleaned."

# Refresh the TDL Challenge 2026 leaderboard data by fetching results.json
# files from PRs labeled track-1-gnn / track-2-tnn on the TopoBench repo.
# Honors the GITHUB_TOKEN env var for higher rate limits.
update-leaderboard:
	@echo "Refreshing leaderboard data from labeled PRs..."
	@python scripts/fetch_leaderboard_results.py

# Override html target to generate API docs first
html: apidocs
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
