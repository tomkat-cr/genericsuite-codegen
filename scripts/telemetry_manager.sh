#!/bin/bash
# telemetry_manager.sh
# Telemetry Manager Script
# 2025-12-25 | CR

init_docker() {
    if [ "${DOCKER_CMD}" = "podman" ]; then
        if ! podman system connection list > /dev/null
        then
            echo "Podman machine not found. Creating one..."
            if ! podman machine create
            then
                echo "❌ Podman machine could not be created."
                exit 1
            fi
        else
            echo "Starting Podman machine..."
            if ! podman machine start
            then
                # is it already running?
                if podman machine info | grep "machinestate: Running" > /dev/null
                then
                    echo "Podman machine is already running."
                else
                    echo "❌ Podman machine could not be started."
                    exit 1
                fi
            fi
        fi
    fi
}

stop_existing_container() {
    init_docker
    if ! ${DOCKER_CMD} stop otel-tui
    then
        echo "Telemetry container is not running..."
    fi
}

start_container() {
    init_docker
    stop_existing_container
    echo "🚀 Starting telemetry..."
    ${DOCKER_CMD} run --rm -it -p 4318:4318 --name otel-tui ymtdzzz/otel-tui:latest
    echo "🚀 Telemetry started."
}

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "${SCRIPT_DIR}"

# Default values
DOCKER_CMD="podman"
TELEMETRY_URL="http://localhost:4318"

# Read .env file
if [ -f ../.env ]; then
    echo "🔍 Reading .env file..."
    set -o allexport; . ../.env; set +o allexport ;
else
    echo "❌ .env file not found. Please create one."
    exit 1
fi

if [ "$ACTION" == "" ]; then
    ACTION="$1"
fi

echo "🥗 Starting GenericSuite CodeGen Telemetry..."
echo "📂 Script directory: $SCRIPT_DIR"
echo "📂 Action: $ACTION"

if [ "$ACTION" == "run" ]; then
    start_container
elif [ "$ACTION" == "stop" ]; then
    stop_existing_container
elif [ "$ACTION" == "status" ]; then
    ${DOCKER_CMD} ps | grep otel-tui
elif [ "$ACTION" == "attach" ]; then
    ${DOCKER_CMD} attach otel-tui
else
    echo "❌ Invalid action. Please use 'start', 'stop', 'status' or 'attach'."
    exit 1
fi