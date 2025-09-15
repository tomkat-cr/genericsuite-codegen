#!/bin/bash

# GenericSuite CodeGen MCP Server Startup Script

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

clean_up() {
    echo "🧹 Cleaning up..."
    rm -rf genericsuite_codegen
}

# Always execute the function clean_up when the script is terminated
trap clean_up EXIT

copy_lib() {
    echo "🔗 Linking common assets..."
    cp -r ../server/genericsuite_codegen .
}

# CLI Parameters
# MCP_INSPECTOR="${2:-0}"

# .env file read
if [ -f ../.env ]; then
    echo "🔍 Reading .env file..."
    set -o allexport; . ../.env; set +o allexport ;
else
    echo "❌ .env file not found. Please create one."
    exit 1
fi

echo "🥗 Starting GenericSuite CodeGen MCP Server..."
echo "📂 Server directory: $SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python not found. Please install Python 3.9 or later."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "🐍 Using Python: $PYTHON_CMD"

# Check if requirements are installed
echo "📦 Checking dependencies..."
if ! poetry run python -c "import fastmcp" &> /dev/null; then
    echo "📥 Installing dependencies..."
    poetry install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please check requirements.txt"
        exit 1
    fi
fi

echo "✅ Dependencies verified"
echo "🚀 Starting MCP server..."
echo ""

# Set PYTHONPATH to include the server directory
# export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Start the server

# Default values for environment variables

# Debug mode
if [ -z "$MCP_INSPECTOR" ]; then
    export MCP_INSPECTOR="0"
fi

# MCP server port
if [ -z "$MCP_SERVER_PORT" ]; then
    export MCP_SERVER_PORT=8070
fi

# MCP server host
if [ -z "$MCP_SERVER_HOST" ]; then
    export MCP_SERVER_HOST=0.0.0.0
fi

copy_lib

if [ "$MCP_INSPECTOR" = "1" ]; then
    npx @modelcontextprotocol/inspector \
        poetry \
        run \
        MCP_TRANSPORT=stdio $PYTHON_CMD start_mcp_server.py
else
    poetry run $PYTHON_CMD start_mcp_server.py
fi
