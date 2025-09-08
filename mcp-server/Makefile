.PHONY: help install dev build start start-async test clean lint format check-env

# Default target
help:
	@echo "GenericSuite CodeGen MCP Server Commands:"
	@echo "  install     - Install dependencies with Poetry"
	@echo "  dev         - Run development server"
	@echo "  start       - Start production server"
	@echo "  start-async - Start server in async mode"
	@echo "  build       - Build the application"
	@echo "  test        - Run tests"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code with black and isort"
	@echo "  check-env   - Check environment configuration"
	@echo "  clean       - Clean up build artifacts"

# Install dependencies
install:
	poetry install

# Check environment configuration
check-env:
	@echo "Checking environment configuration..."
	@echo "MCP_SERVER_HOST: $${MCP_SERVER_HOST:-0.0.0.0}"
	@echo "MCP_SERVER_PORT: $${MCP_SERVER_PORT:-8070}"
	@echo "MCP_API_KEY: $${MCP_API_KEY:+Set}"
	@echo "MCP_DEBUG: $${MCP_DEBUG:-0}"
	@echo "OPENAI_API_KEY: $${OPENAI_API_KEY:+Set}"

# Run development server
dev: check-env
	poetry run python start_mcp_server.py

run: dev

# Start production server
start: check-env
	poetry run python start_mcp_server.py

# Start server in async mode
start-async: check-env
	poetry run python start_mcp_server.py --async

# Alternative way to run using module
start-module:
	poetry run python -m genericsuite_codegen

# Build the application
build:
	poetry build

# Run tests
test:
	poetry run pytest -v

# Run code linting
lint:
	poetry run flake8 genericsuite_codegen/
	poetry run black --check genericsuite_codegen/
	poetry run isort --check-only genericsuite_codegen/

# Format code
format:
	poetry run black genericsuite_codegen/
	poetry run isort genericsuite_codegen/

# Clean up
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -f mcp_server.log
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete