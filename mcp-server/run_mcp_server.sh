#!/bin/bash

# GenericSuite CodeGen MCP Server Startup Script

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

MCP_RUN_USING_POETRY=1

clean_up() {
    echo "🧹 Cleaning up..."
    rm -rf genericsuite_codegen
    if [ ! "$MCP_RUN_USING_POETRY" = "1" ]; then
        deactivate
    fi
    echo "🧹 Cleaning up... done"
}

# Always execute the function clean_up when the script is terminated
trap clean_up EXIT

copy_lib() {
    echo "🔗 Linking common assets..."
    # cp -r ../server/genericsuite_codegen .
    ln -s ../server/genericsuite_codegen .
}

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

if [ ! "$MCP_RUN_USING_POETRY" = "1" ]; then
    echo "🐍 Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."

if [ "$MCP_RUN_USING_POETRY" = "1" ]; then
    CHECKING_CMD_PREFIX="poetry run python"
    INSTALLING_CMD="poetry install"
else
    CHECKING_CMD_PREFIX="$PYTHON_CMD"
    INSTALLING_CMD="$PYTHON_CMD -m pip install --upgrade pip && $PYTHON_CMD -m pip install -r requirements.txt"
fi


if ! $CHECKING_CMD_PREFIX -c "import fastmcp" &> /dev/null; then
    echo "📥 Installing dependencies..."
    $INSTALLING_CMD
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please check requirements.txt"
        deactivate
        exit 1
    fi
fi

echo "✅ Dependencies verified"

# Default values for environment variables

# Debug mode
if [ -z "$MCP_INSPECTOR" ]; then
    export MCP_INSPECTOR="0"
fi

# MCP server port
if [ -z "$MCP_SERVER_PORT" ]; then
    export MCP_SERVER_PORT=8000
fi

# MCP server host
if [ -z "$MCP_SERVER_HOST" ]; then
    export MCP_SERVER_HOST=0.0.0.0
fi

if [ "$MCP_INSPECTOR" = "1" ]; then
    export MCP_TRANSPORT="stdio"
else
    export MCP_TRANSPORT="http"
fi

APP_RUN_ARGS="MCP_SERVER_PORT=$MCP_SERVER_PORT MCP_SERVER_HOST=$MCP_SERVER_HOST MCP_TRANSPORT=$MCP_TRANSPORT"

# Copy library
copy_lib

# Start the server
echo "🚀 Starting MCP server..."
echo ""

if [ "$MCP_INSPECTOR" = "1" ]; then
    if [ "$MCP_RUN_USING_POETRY" = "1" ]; then
        CLIENT_PORT=6274 SERVER_PORT=6277 npx @modelcontextprotocol/inspector \
            poetry run env $APP_RUN_ARGS $PYTHON_CMD start_mcp_server.py $ADDITIONAL_ARGS
    else
        CLIENT_PORT=6274 SERVER_PORT=6277 npx @modelcontextprotocol/inspector \
            env $APP_RUN_ARGS $PYTHON_CMD start_mcp_server.py $ADDITIONAL_ARGS
    fi
else
    if [ "$MCP_RUN_USING_POETRY" = "1" ]; then
        poetry run env $APP_RUN_ARGS $PYTHON_CMD start_mcp_server.py $ADDITIONAL_ARGS
    else
        env $APP_RUN_ARGS $PYTHON_CMD start_mcp_server.py $ADDITIONAL_ARGS
    fi
fi
