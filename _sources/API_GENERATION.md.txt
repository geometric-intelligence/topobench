# Documentation Generation Guide

This guide explains how to automatically generate API documentation for TopoBench.

## Overview

The API documentation is automatically generated from the Python source code using Sphinx's `sphinx-apidoc` tool. This ensures that the documentation stays in sync with the codebase.

## Quick Start

### Generate and Build Documentation Locally

```bash
# Navigate to the docs directory
cd docs

# Generate API docs and build HTML
make html

# Or just generate API docs without building
make apidocs

# View the documentation
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

### Clean and Regenerate

```bash
# Clean everything
make clean

# Clean only API documentation
make clean-api

# Regenerate everything
make html
```

## How It Works

### 1. Automatic API Generation Script (`generate_api_docs.sh`)

This bash script:
- Removes old API documentation files
- Runs `sphinx-apidoc` to scan the `topobench/` package
- Generates `.rst` files for each module
- Creates an index file linking all modules

**Usage:**
```bash
cd docs
./generate_api_docs.sh
```

### 2. API Index Generator (`generate_api_index.py`)

This Python script:
- Scans the `api/` directory for generated `.rst` files
- Creates an `index.rst` file with a table of contents
- Organizes modules alphabetically

### 3. Makefile Integration

The Makefile has been updated with:
- `make apidocs`: Generate API documentation only
- `make clean-api`: Remove API documentation
- `make html`: Automatically runs `apidocs` before building

### 4. GitHub Actions Automation

The `.github/workflows/docs.yml` workflow:
- Automatically generates API docs on every push to main
- Builds and deploys documentation to GitHub Pages
- Ensures documentation is always up-to-date

## Configuration

### Sphinx Configuration (`conf.py`)

Key settings for API documentation:

```python
extensions = [
    "sphinx.ext.autodoc",      # Core autodoc functionality
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.napoleon",     # Support for NumPy/Google docstrings
    "numpydoc",               # NumPy documentation style
]

# Mock imports for dependencies not available during doc build
autodoc_mock_imports = [
    "torch", "torchvision", "torchaudio",
    "torch_geometric", "pyg_lib",
    "torch_sparse", "torch_scatter", "torch_cluster", "torch_spline_conv"
]
```

### sphinx-apidoc Options

The script uses these options:
- `--force`: Overwrite existing files
- `--separate`: Create separate pages for each module
- `--module-first`: Put module documentation before submodule docs
- `--no-toc`: Don't create a table of contents file
- `--maxdepth 4`: Maximum depth of submodules to show

## Customization

### Excluding Modules

To exclude specific modules from documentation, edit `generate_api_docs.sh`:

```bash
sphinx-apidoc \
    --force \
    --separate \
    -o "$API_DIR" \
    "$PROJECT_DIR/topobench" \
    "$PROJECT_DIR/topobench/__pycache__" \
    "$PROJECT_DIR/topobench/excluded_module"  # Add exclusions here
```

### Changing Documentation Structure

Modify `generate_api_index.py` to customize how the API index is organized:
- Group modules by functionality
- Add descriptions
- Create separate sections

### Adding Custom API Pages

1. Create `.rst` files in the `api/` directory
2. They will be automatically included in the index
3. Or manually edit `api/index.rst` for custom organization

## Troubleshooting

### Missing Modules

If modules are missing from the documentation:
1. Check that the module has a proper `__init__.py`
2. Ensure the module is not excluded in `generate_api_docs.sh`
3. Verify the module imports don't fail

### Import Errors During Build

Add problematic packages to `autodoc_mock_imports` in `conf.py`:

```python
autodoc_mock_imports = [
    "torch",
    "your_problematic_package",
]
```

### Regeneration Issues

```bash
# Complete clean and rebuild
cd docs
make clean
make clean-api
make html
```

## Best Practices

1. **Write Good Docstrings**: Follow NumPy documentation style
2. **Run Locally Before Pushing**: Test documentation builds before committing
3. **Review Generated Files**: Check that API docs look correct
4. **Keep Conf.py Updated**: Add new dependencies to `autodoc_mock_imports`

## CI/CD Integration

The documentation is automatically:
- **Generated**: On every commit
- **Built**: Using Sphinx
- **Deployed**: To GitHub Pages (on main branch)

No manual intervention required! 🎉

## Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [sphinx-apidoc Documentation](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html)
- [NumPy Documentation Guide](https://numpydoc.readthedocs.io/)
