#!/bin/bash

# frequently used commands before running the script
# bash /root/content-moderation/setup/remote/theta_env.sh
# source .env
# GCP_CREDENTIALS=$(jq . 'credentials.json')
# which env variables are must to run the script?
# HF_TOKEN (to download models from huggingface)
# GCP_CREDENTIALS (to use bigquery, gcs, etc)
# FLY_IO_DEPLOY_TOKEN (if you are deploying to fly.io, add this token to github secrets)
# rest of the variables are optional and are already in their optimal states do NOT change them
# there is a global config file in src_deploy/config.py that is used to set the default values of the variables
# if you need to change the values, you can do so in the config.py file or override them in the .env file
# if you are using 2 GPUs in theta you need to uncomment # os.environ["CUDA_VISIBLE_DEVICES"] = "0" in server_sglang.py file
# also uncomment # os.environ["CUDA_VISIBLE_DEVICES"] = "1" in server_sglang.py file
# this will allow you to use the 2 GPUs in theta for the SGLang servers
# you do not need this in production as we are using larger 1 GPU instances which has 24-48GB of memory
# this script has been tested on both A10 and L40S GPUs on fly.io
# as L40S(48GB) is cheaper than A10(24GB), we are using L40S in production
# https://fly.io/docs/about/pricing/#gpus-and-fly-machines

set -e

# Create a log directory if it doesn't exist
mkdir -p /root/content-moderation/src_deploy/logs

# Set PYTHONPATH to ensure imports work correctly
export PYTHONPATH="/home/ubuntu:$PYTHONPATH"

echo "Starting SGLang servers via entrypoint.py..."
# Start ONLY the SGLang servers with the --mode=sglang --no-wait flags
/root/content-moderation/.venv/bin/python /root/content-moderation/src_deploy/entrypoint.py --mode=sglang --no-wait > /root/content-moderation/src_deploy/logs/sglang.log 2>&1 &
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
  if ! $llm_ready && grep -q "\[LLM\].*The server is fired up and ready to roll" /root/content-moderation/src_deploy/logs/sglang.log 2>/dev/null; then
    echo "✅ LLM server is ready!"
    llm_ready=true
  fi

  # Check if Embedding server is ready
  if ! $embedding_ready && grep -q "\[EMBEDDING\].*The server is fired up and ready to roll" /root/content-moderation/src_deploy/logs/sglang.log 2>/dev/null; then
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
    tail -n 50 /root/content-moderation/src_deploy/logs/sglang.log
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
exec /root/content-moderation/.venv/bin/python /root/content-moderation/src_deploy/entrypoint.py --mode=fastapi