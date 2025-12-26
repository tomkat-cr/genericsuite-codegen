#!/bin/bash
# GenericSuite CodeGen Batch Server Startup Script
# 2025-12-24 | CR

print_debug() {
    if [ "${BATCH_SERVER_DEBUG}" = "1" ]; then
        echo "$1"
    fi
}

OPTION="${1}"
if [ -z "$OPTION" ]; then
    echo "❌ No option provided. Please provide an option."
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

export BATCH_SERVER_DEBUG=0
BATCH_RUN_USING_POETRY=1

# .env file read
if [ -f "${SCRIPT_DIR}/../.env" ]; then
    print_debug "🔍 Reading .env file..."
    set -o allexport; . "${SCRIPT_DIR}/../.env"; set +o allexport ;
else
    print_debug "❌ .env file not found. Please create one."
    exit 1
fi

print_debug "🥗 Starting GenericSuite CodeGen Batch Server..."
print_debug "📂 Script directory: $SCRIPT_DIR"

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
print_debug "🐍 Using Python: $PYTHON_CMD"

# Check if requirements are installed
print_debug "📦 Checking dependencies..."

if [ "$BATCH_RUN_USING_POETRY" = "1" ]; then
    CHECKING_CMD_PREFIX="poetry run python"
    INSTALLING_CMD="poetry install"
else
    CHECKING_CMD_PREFIX="$PYTHON_CMD"
    INSTALLING_CMD="$PYTHON_CMD -m pip install --upgrade pip && $PYTHON_CMD -m pip install -r requirements.txt"
fi

if ! $CHECKING_CMD_PREFIX -c "import pydantic" &> /dev/null; then
    print_debug "📥 Installing dependencies..."
    $INSTALLING_CMD
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please check requirements.txt"
        if [ "$BATCH_RUN_USING_POETRY" != "1" ]; then
            deactivate
        fi
        exit 1
    fi
fi

if [ ! "$BATCH_RUN_USING_POETRY" = "1" ]; then
    print_debug "🐍 Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
fi

# Start the server
print_debug "🚀 Starting Batch server..."
print_debug ""

if [ "$BATCH_RUN_USING_POETRY" = "1" ]; then
    print_debug "📦 Run using poetry..."
    CMD="poetry run python -m genericsuite_codegen.api.batch ${OPTION}"
else
    print_debug "📦 Run using pip..."
    CMD="$PYTHON_CMD -m genericsuite_codegen.api.batch ${OPTION}"
fi
print_debug "Running command:"
print_debug "$CMD"
print_debug ""
$CMD
