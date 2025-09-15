.PHONY: help install dev build preview clean lint format

# Default target
help:
	@echo "UI Commands:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Run development server"
	@echo "  build       - Build for production"
	@echo "  preview     - Preview production build"
	@echo "  clean       - Clean up build artifacts"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"

# Install dependencies
install:
	npm install

# Run development server
dev:
	bash ./run-client.sh run

run: dev

# Build for production
build:
	bash ./run-client.sh build

# Build from root directory
build_from_root:
	cd ui && bash ./run-client.sh build

# Preview production build
preview:
	bash ./run-client.sh preview

# Clean up
clean:
	npm run clean

# Run linting
lint:
	npm run lint

# Format code
format:
	npm run format