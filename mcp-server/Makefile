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
show-env-vars:
	@echo "Checking environment configuration..."
	@echo "MCP_SERVER_HOST: $${MCP_SERVER_HOST}"
	@echo "MCP_SERVER_PORT: $${MCP_SERVER_PORT}"
	@echo "MCP_API_KEY: $${MCP_API_KEY:+Set}"
	@echo "MCP_INSPECTOR: $${MCP_INSPECTOR}"
	@echo "OPENAI_API_KEY: $${OPENAI_API_KEY:+Set}"

check-env:
	@if [ -f ../.env ]; then \
		echo "Loading environment from ../.env"; \
		set -o allexport; \
		source ../.env; \
		set +o allexport; \
		make show-env-vars; \
	else \
		echo "No .env file found at ../.env"; \
		make show-env-vars; \
	fi

# Run development server
dev: check-env
	bash ./run_mcp_server.sh

run: dev

# Start production server
start: check-env
	bash ./run_mcp_server.sh

# Start server in async mode
start-async: check-env
	ADDITIONAL_ARGS="--async" bash ./run_mcp_server.sh

# Start server with MCP Inspector
start-inspector:
	MCP_INSPECTOR=1 bash ./run_mcp_server.sh

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