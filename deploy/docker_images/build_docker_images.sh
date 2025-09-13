#!/bin/bash
# deploy/docker_images/build_docker_images.sh

echo "Building Docker image for GenericSuite CodeGen..."

# Synchronize dependencies from pyproject.toml files to Dockerfile
echo "Synchronizing dependencies from pyproject.toml files..."
if ! ../dependency-sync/sync_dependencies.sh --defaults; then
    echo "Warning: Dependency synchronization failed, continuing with existing Dockerfile"
    echo "You may want to run the sync manually: deploy/dependency-sync/sync_dependencies.sh --defaults"
fi

build_image() {
    local image_name=$1
    local dockerfile=$2
    local src_path=$3

    echo "Building Docker image: $image_name"
    pwd
    docker build -t "$image_name" -f "$dockerfile" "$src_path"

    if [ $? -eq 0 ]; then
        echo "Docker image built successfully: $image_name"
    else
        echo "Error building Docker image: $image_name"
        exit 1
    fi
}

# Build the base image
build_image "gscodegen_python" "Dockerfile-Python" "."

# Build the MongoDB Atlas image
build_image "gscodegen_mongo_db_atlas" "Dockerfile-MongoDb-Atlas" "."

# Build the ollama image
# NOTE: This build has been disabled because it needs +30 Gb and lasts +16 min
#       If your PC/Mac disk has less than 80 Gb free space, it's preffered to use an
#       external service like OpenAI API.
#
# build_image "gscodegen_ollama" "Dockerfile-Ollama" "."

