#!/bin/bash
# This script will run the update-knowledge-base endpoint every 30 seconds
# 2025-12-23 | CR
CMD_POSTFIX=" > /dev/null 2>&1"
LOGGER_OPTIONS="silent"
if [ "${BATCH_SERVER_DEBUG}" = "1" ]; then
    CMD_POSTFIX=""
    LOGGER_OPTIONS=""
fi
export LOGGER_OPTIONS
while true; do
    # curl http://localhost:8000/v1/update-knowledge-base > ${CMD_POSTFIX}
    bash ./server/run_batch_server.sh update-knowledge-base ${CMD_POSTFIX}
    sleep 30
done
