#!/bin/bash
# run-client.sh
# 2025-09-14 | CR
#

# Script directory
SCRIPT_DIR=$(cd $(dirname $0); pwd)

# Change to script directory
cd "$SCRIPT_DIR"

if [ ! -f ../.env ]; then
    echo "Error: .env file not found in 'root' directory"
    exit 1
fi

echo "Loading environment variables"
set -o allexport; . ../.env; set +o allexport ;

ACTION=$1

if [ -z "$ACTION" ]; then
    echo "Error: No action specified"
    exit 1
fi

if [ "$ACTION" = "run" ]; then
    echo "Starting client in development mode..."
    npm run dev
elif [ "$ACTION" == "build" ]; then
    echo "Building client..."
    npm run build
elif [ "$ACTION" == "preview" ]; then
    echo "Previewing production build..."
    npm run preview
else
    echo "Error: Invalid action specified: $ACTION"
    exit 1
fi
