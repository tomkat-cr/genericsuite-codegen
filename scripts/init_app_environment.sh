#!/bin/bash
# scripts/init_app_environment.sh
# This script copies the .env.example file to the .env file, as well as other example files to final/modifiable files.
# It is used to initialize the environment for the application the first time it is run.

copy_env_example() {
    local source_env_file_path="$1"
    local target_env_file_path="$2"
    echo ""
    if [ ! -f "$source_env_file_path" ]; then
        echo "Error: Source environment file not found: $source_env_file_path"
        exit 1
    fi
    if [ -f "$target_env_file_path" ]; then
        echo "Target environment file already exists: $target_env_file_path"
        echo "Skipping..."
    else
        echo "Copying $source_env_file_path to $target_env_file_path" ...
        if ! cp "$source_env_file_path" "$target_env_file_path"; then
            echo "Error: Failed to copy $source_env_file_path to $target_env_file_path"
            exit 1
        fi
        echo "Done"
    fi
}

check_directories() {
    local directory_name="$1"
    if [ ! -d "$BASE_DIR/$directory_name" ]; then
        echo "Error: $directory_name directory not found: $BASE_DIR/$directory_name"
        exit 1
    fi
}

SCRIPT_DIR=$(dirname "$0")
BASE_DIR="$SCRIPT_DIR/.."

check_directories "deploy"

copy_env_example "$BASE_DIR/.env.example" "$BASE_DIR/.env"
copy_env_example "$BASE_DIR/deploy/nginx.conf.example" "$BASE_DIR/deploy/nginx.conf"
copy_env_example "$BASE_DIR/deploy/docker-compose.yml.example" "$BASE_DIR/deploy/docker-compose.yml"