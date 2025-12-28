#!/bin/bash
# This script will run the update-knowledge-base endpoint every 30 seconds
# 2025-12-23 | CR
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export BATCH_SERVER_DEBUG=0
export BATCH_SERVER_SCRIPT_DEBUG=0

# .env file read
if [ -f "${SCRIPT_DIR}/../.env" ]; then
    set -o allexport; . "${SCRIPT_DIR}/../.env"; set +o allexport ;
else
    echo "❌ .env file not found. Please create one."
    exit 1
fi

CMD_POSTFIX=" > /dev/null 2>&1"
if [ "${BATCH_SERVER_DEBUG}" = "1" ]; then
    CMD_POSTFIX=""
fi

LOGGER_OPTIONS="silent"
if [ "${BATCH_SERVER_SCRIPT_DEBUG}" = "1" ]; then
    LOGGER_OPTIONS=""
fi
export LOGGER_OPTIONS

while true; do
    # curl http://localhost:8000/v1/update-knowledge-base > ${CMD_POSTFIX}
    bash ./server/run_batch_server.sh update-knowledge-base ${CMD_POSTFIX}
    sleep 30
done
