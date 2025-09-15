#!/bin/bash
# deploy/run-deploy.sh

load_envs() {
    if [ ! -f ../.env ]; then
        echo "Error: .env file not found in 'root' directory"
        exit 1
    fi
    echo "Loading environment variables from ../.env"
    set -o allexport; . ../.env; set +o allexport ;
}

start_up_function() {
    # Start up code
    echo "Starting up services..."
}

clean_up_function() {
    # Clean up code
    echo "Cleaning up services..."
    # Remove all files and directories except for the .env file
    rm -rf ../mcp-server/genericsuite_codegen
    rm -rf ../server/server-entrypoint.sh
    rm -rf ../server/local_repo_files
    rm -rf ../server/.env
    rm -rf ../server/mcp-server
    rm -rf ../deploy/local_repo_files
}

create_docker_images() {
    echo "Creating docker images..."
    cd docker_images
    if ! sh ./build_docker_images.sh
    then
        echo "Error creating docker images"
        exit 1
    fi
    cd ..
}

cleanup_when_specific_container_is_specified() {
    if [ "${CONTAINER_TO_RUN}" != "" ]; then
        echo "Running only ${CONTAINER_TO_RUN}"
        docker compose --project-name deploy down
        docker compose --project-name ${APP_NAME_LOWERCASE} down
    fi
}

load_envs

APP_NAME_LOWERCASE=$(echo "$APP_NAME" | tr '[:upper:]' '[:lower:]')
APP_NAME_LOWERCASE=$(echo "$APP_NAME_LOWERCASE" | tr '[:blank:]' '_')

if [ -z "$APP_NAME_LOWERCASE" ]; then
    echo "ERROR: APP_NAME environment variable not set"
    exit 1
fi

ACTION=$1
if [ -z "$ACTION" ]; then
    echo "Error: No action specified"
    exit 1
fi

# Specific container environment variables
export MONGODB_HOST_NAME=gscodegen-mongo
export MONGODB_HOST_PORT=27017
export MONGODB_USER=root
export MONGODB_PASSWORD=example
# export MONGODB_URI=mongodb://$MONGODB_USER:$MONGODB_PASSWORD@$MONGODB_HOST_NAME:$MONGODB_HOST_PORT
export MONGODB_URI=mongodb://$MONGODB_HOST_NAME:$MONGODB_HOST_PORT/?directConnection=true

export LOCAL_REPO_DIR=/var/local_repo_files

if [ "$ACTION" = "restart" ]; then
    echo "Restarting services..."
    docker compose --project-name ${APP_NAME_LOWERCASE} restart ${CONTAINER_TO_RUN}
    exit 0
elif [ "$ACTION" = "run" ]; then
    start_up_function
    create_docker_images
    echo "Starting services..."
    if ! docker network create my_shared_network 2>/dev/null; then
        echo "my_shared_network already exists"
    fi
    cleanup_when_specific_container_is_specified
    docker compose --project-name ${APP_NAME_LOWERCASE} up -d ${CONTAINER_TO_RUN}
    exit 0
elif [ "$ACTION" = "down" ]; then
    echo "Stopping services..."
    cleanup_when_specific_container_is_specified
    if ! docker compose --project-name ${APP_NAME_LOWERCASE} down
    then
        echo "Error stopping services... skipping clean up function"
    fi
    clean_up_function
    exit 0
elif [ "$ACTION" = "logs" ]; then
    echo "Showing logs..."
    docker compose --project-name ${APP_NAME_LOWERCASE} logs
    exit 0
elif [ "$ACTION" = "logs-f" ]; then
    echo "Showing logs..."
    docker compose --project-name ${APP_NAME_LOWERCASE} logs -f
    exit 0
elif [ "$ACTION" = "logs-f-server-client" ]; then
    echo "Showing logs..."
    docker compose --project-name ${APP_NAME_LOWERCASE} logs -f gscodegen-server gscodegen-client
    exit 0
else
    echo "Error: Invalid action specified"
    exit 1
fi