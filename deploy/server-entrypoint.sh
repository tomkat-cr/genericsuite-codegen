#!/bin/bash
# deploy/server-entrypoint.sh
# 2025-09-01 | CR

run_server() {
    cd /code/server && uvicorn genericsuite_codegen.api.main:app --host 0.0.0.0 --port 8000 --env-file /var/scripts/.env --reload
}

run_mcp_server() {
    cd /code/mcp-server && python start_mcp_server.py --async
}

run_server & run_mcp_server & wait