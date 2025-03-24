#!/bin/bash
set -e

echo "Starting run_all.py..."
/home/ubuntu/.venv/bin/python /home/ubuntu/run_all.py

echo "Waiting for 5 minutes..."
sleep 300

echo "Starting FastAPI server..."
exec /home/ubuntu/.venv/bin/python /home/ubuntu/server_fastapi.py