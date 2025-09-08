#!/bin/bash
# deploy/docker_images/build_docker_images.sh

echo "Building Docker image for GenericSuite CodeGen..."

# Synchronize dependencies from pyproject.toml files to Dockerfile
echo "Synchronizing dependencies from pyproject.toml files..."
if ! ../dependency-sync/sync_dependencies.sh --defaults; then
    echo "Warning: Dependency synchronization failed, continuing with existing Dockerfile"
    echo "You may want to run the sync manually: deploy/dependency-sync/sync_dependencies.sh --defaults"
fi

# Build the main application image
docker build -t gscodegen_python -f Dockerfile .

if [ $? -eq 0 ]; then
    echo "Docker image built successfully: gscodegen_python"
else
    echo "Error building Docker image"
    exit 1
fi