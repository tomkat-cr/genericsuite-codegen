.PHONY: help install dev build start test clean lint format

# Default target
help:
	@echo "Server Commands:"
	@echo "  install     - Install dependencies with Poetry"
	@echo "  dev         - Run development server"
	@echo "  build       - Build the application"
	@echo "  start       - Start production server"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean up build artifacts"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"

# Install dependencies
install:
	poetry install

# Run development server
dev:
	bash ./run-server.sh run

run: dev

# Build the application
build:
	poetry build

# Start production server
start: dev

# Start production server from root
start-from_root:
	cd server && bash ./run-server.sh run

# Run tests
test:
	poetry run pytest

# Clean up
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Run linting
lint:
	poetry run flake8 genericsuite_codegen/
	poetry run mypy genericsuite_codegen/

# Format code
format:
	poetry run black genericsuite_codegen/
	poetry run isort genericsuite_codegen/