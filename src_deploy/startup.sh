#!/bin/bash
set -e

# Create a log directory if it doesn't exist
mkdir -p /home/ubuntu/logs

# Set PYTHONPATH to ensure imports work correctly
export PYTHONPATH="/home/ubuntu:$PYTHONPATH"

echo "Starting SGLang servers via entrypoint.py..."
# Start ONLY the SGLang servers with the --mode=sglang --no-wait flags
/home/ubuntu/.venv/bin/python /home/ubuntu/entrypoint.py --mode=sglang --no-wait > /home/ubuntu/logs/sglang.log 2>&1 &
SGLANG_PID=$!


#### Wait for SGLang servers to be ready ####
echo "Waiting for SGLang servers to initialize..."
# Wait for SGLang servers to be ready - look for the "ready to roll" message in logs
max_wait=480  #
start_time=$(date +%s)

llm_ready=false
embedding_ready=false

while true; do
  # Check if LLM server is ready
  if ! $llm_ready && grep -q "\[LLM\].*The server is fired up and ready to roll" /home/ubuntu/logs/sglang.log 2>/dev/null; then
    echo "✅ LLM server is ready!"
    llm_ready=true
  fi

  # Check if Embedding server is ready
  if ! $embedding_ready && grep -q "\[EMBEDDING\].*The server is fired up and ready to roll" /home/ubuntu/logs/sglang.log 2>/dev/null; then
    echo "✅ Embedding server is ready!"
    embedding_ready=true
  fi

  # If both are ready, we can proceed
  if $llm_ready && $embedding_ready; then
    echo "Both servers are ready! Proceeding to FastAPI server..."
    break
  fi

  # Check if entrypoint process is still running
  if ! kill -0 $SGLANG_PID 2>/dev/null; then
    echo "Entrypoint process has exited, checking logs for errors..."
    tail -n 50 /home/ubuntu/logs/sglang.log
    echo "Continuing to FastAPI server anyway..."
    break
  fi

  # Check time elapsed
  current_time=$(date +%s)
  elapsed=$((current_time - start_time))

  if [ $elapsed -ge $max_wait ]; then
    echo "Timed out waiting for SGLang servers, continuing anyway..."
    break
  fi

  # Print status update with more detail
  status="Waiting for SGLang servers to be ready... ($elapsed seconds elapsed)"
  if $llm_ready; then
    status="$status - LLM: ✅"
  else
    status="$status - LLM: ⏳"
  fi

  if $embedding_ready; then
    status="$status, Embedding: ✅"
  else
    status="$status, Embedding: ⏳"
  fi

  echo "$status"
  sleep 5
done

#### Start FastAPI server ####

echo "Starting FastAPI server via entrypoint.py..."
# Preserve PYTHONPATH when starting the FastAPI server
export PYTHONPATH="/home/ubuntu:$PYTHONPATH"
# Use the entrypoint with fastapi mode - this will run in the foreground and keep the container alive
exec /home/ubuntu/.venv/bin/python /home/ubuntu/entrypoint.py --mode=fastapi