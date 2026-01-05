#!/bin/bash
# run-server.sh
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
	echo "poetry run uvicorn genericsuite_codegen.api.main:app --reload --host 0.0.0.0 --port ${SERVER_PORT:-8002}" 
	poetry run uvicorn \
        genericsuite_codegen.api.main:app \
        --reload \
        --host 0.0.0.0 \
        --port "${SERVER_PORT:-8002}"

elif [ "$ACTION" = "test" ]; then
    echo "Running tests..."
    export ALLOWED_HOSTS="*"
    export OPENAI_API_KEY="sk-proj-1234567890"
    export OPENAI_MODEL_NAME=gpt-4o-mini
    export LOCAL_REPO_DIR="../local_repo_files"
    poetry run pytest "${TEST_FILTER}"

else
    echo "Error: Invalid action specified: $ACTION"
    exit 1
fi
