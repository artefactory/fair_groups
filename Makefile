.PHONY: clean build install test lint format setup-dev install-dev venv

# Python environment
PYTHON := python3
UV := uv
VENV_DIR := .venv
VENV_ACTIVATE := $(VENV_DIR)/bin/activate

# Package information
PACKAGE_NAME := fair_partition
VERSION := $(shell grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

# Build directories
DIST_DIR := dist
BUILD_DIR := build

# Create virtual environment
venv:
	$(UV) venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"
	@echo "Please run: make install-dev"

# Install development dependencies
install-dev: venv
	. $(VENV_ACTIVATE) && $(UV) pip install --upgrade pip
	. $(VENV_ACTIVATE) && $(UV) pip install build hatchling
	. $(VENV_ACTIVATE) && $(UV) pip install -e ".[dev]"

# Clean build artifacts
clean:
	rm -rf $(DIST_DIR) $(BUILD_DIR) *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build the package
build: clean
	. $(VENV_ACTIVATE) && $(PYTHON) -m build

# Install the package from wheel
install-wheel:
	. $(VENV_ACTIVATE) && $(UV) pip install -i https://test.pypi.org/simple/ fair-partition

# Run tests
test:
	. $(VENV_ACTIVATE) && pytest tests/

# Run linting
lint:
	. $(VENV_ACTIVATE) && ruff check .
	. $(VENV_ACTIVATE) && mypy fair_partition/

# Format code
format:
	. $(VENV_ACTIVATE) && black .
	. $(VENV_ACTIVATE) && isort .

publish:
	. $(VENV_ACTIVATE) && $(UV) pip install build twine
	. $(VENV_ACTIVATE) && $(UV) run twine upload --repository testpypi dist/*

# Build and install in one command
build-and-publish: build publish install-wheel

# Help command
help:
	@echo "Available commands:"
	@echo "  make venv             - Create a new virtual environment"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make clean            - Remove build artifacts"
	@echo "  make build            - Build the package"
	@echo "  make install-wheel    - Install the package from wheel"
	@echo "  make test             - Run tests"
	@echo "  make lint             - Run linting"
	@echo "  make format           - Format code"