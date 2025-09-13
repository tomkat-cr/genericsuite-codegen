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
	npm run dev

run: dev

# Build for production
build:
	npm run build

# Build from root directory
build_from_root:
	cd ui && npm run build

# Preview production build
preview:
	npm run preview

# Clean up
clean:
	npm run clean

# Run linting
lint:
	npm run lint

# Format code
format:
	npm run format