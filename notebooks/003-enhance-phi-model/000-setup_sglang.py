from sglang.utils import execute_shell_command, terminate_process
import requests
import time

# any random API key
API_KEY = "<add-your-api-key>"
INTERNAL_IP = "<add-your-internal-ip>"  # Your internal IP
EXTERNAL_IP = "<add-your-external-ip>"  # Your external IP
PORT = 8890


# Start the server
server_process = execute_shell_command(
    f"""
python3 -m sglang.launch_server \
    --model-path microsoft/Phi-3.5-mini-instruct \
    --port={PORT} \
    --attention-backend triton \
    --disable-cuda-graph \
    --mem-fraction-static 0.7 \
    --host 0.0.0.0 \
    --api-key {API_KEY}
"""
)

# Wait for server to start
time.sleep(10)

# Keep server running
print("Server is running. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping server...")
    terminate_process(server_process)
