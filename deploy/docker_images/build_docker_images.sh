#!/bin/bash
# deploy/docker_images/build_docker_images.sh

echo "Building Docker image for GenericSuite CodeGen..."

# Build the main application image
docker build -t gscodegen_python -f Dockerfile .

if [ $? -eq 0 ]; then
    echo "Docker image built successfully: gscodegen_python"
else
    echo "Error building Docker image"
    exit 1
fi