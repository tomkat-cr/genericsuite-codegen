#!/bin/bash
# deploy/server-entrypoint.sh

load_envs() {
    set -e
    echo ""
    echo "Loading environment variables from /code/.env"
    if [ -f /code/load_envs.sh ]; then
        sh /code/load_envs.sh /code/.env
    fi
}

run_server() {
    cd /code && uvicorn genericsuite_codegen.api.main:app --host 0.0.0.0 --port 8000 --env-file /code/.env
    
}

run_mcp_server() {
    cd /code/mcp-server && python start_mcp_server.py --async
}

load_envs
run_server & run_mcp_server & wait