# deployment logs


## monitoring logs
Monitor logs using this command:

```bash
# if you want to see the logs for a specific instance
fly logs -a stage-content-moderation-server-gpu -i <instance-id>

# else
fly logs -a stage-content-moderation-server-gpu
```

## Preview
Server logs should look like this:

```log
2025-03-25T10:56:14Z runner[3d8d7e96f15e58] ord [info]Pulling container image registry.fly.io/stage-content-moderation-server-gpu:deployment-01JQ6EPM720T04PVWMGYCJ7PAK
2025-03-25T10:58:58Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (40 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T10:59:03Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (45 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T10:59:08Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (50 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T10:59:13Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (55 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T10:59:18Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (60 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T10:59:23Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (65 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T10:59:28Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (70 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T10:59:33Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (75 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T10:59:38Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (80 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T10:59:43Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (85 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T10:59:48Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (90 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T10:59:53Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (95 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T10:59:58Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (100 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T11:00:03Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (105 seconds elapsed) - LLM: ⏳, Embedding: ⏳
2025-03-25T11:00:08Z app[3d8d7e96f15e58] ord [info]✅ LLM server is ready!
2025-03-25T11:00:08Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (110 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:00:13Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (115 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:00:18Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (120 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:00:23Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (125 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:00:28Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (130 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:00:33Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (135 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:00:38Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (140 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:00:43Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (145 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:00:48Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (150 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:00:53Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (155 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:00:58Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (160 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:01:03Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (165 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:01:08Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (170 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:01:13Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (175 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:01:18Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (180 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:01:23Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (185 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:01:28Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (190 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:01:33Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (195 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:01:38Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (200 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:01:43Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (205 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:01:48Z app[3d8d7e96f15e58] ord [info]Waiting for SGLang servers to be ready... (210 seconds elapsed) - LLM: ✅, Embedding: ⏳
2025-03-25T11:01:53Z app[3d8d7e96f15e58] ord [info]✅ Embedding server is ready!
2025-03-25T11:01:53Z app[3d8d7e96f15e58] ord [info]Both servers are ready! Proceeding to FastAPI server...
2025-03-25T11:01:53Z app[3d8d7e96f15e58] ord [info]Starting FastAPI server via entrypoint.py...
2025-03-25T11:01:53Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:53,925 - INFO - ================================================================================
2025-03-25T11:01:53Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:53,925 - INFO - CONTENT MODERATION SYSTEM STARTUP
2025-03-25T11:01:53Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:53,925 - INFO - ================================================================================
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:54,077 - INFO - nvidia-smi output:
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:54,077 - INFO - Tue Mar 25 11:01:53 2025
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]+-----------------------------------------------------------------------------------------+
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]|-----------------------------------------+------------------------+----------------------+
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]|                                         |                        |               MIG M. |
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]|=========================================+========================+======================|
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]|   0  NVIDIA L40S                    Off |   00000000:00:06.0 Off |                    0 |
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]| N/A   41C    P0             80W /  350W |   41684MiB /  46068MiB |      0%      Default |
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]|                                         |                        |                  N/A |
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]+-----------------------------------------+------------------------+----------------------+
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]+-----------------------------------------------------------------------------------------+
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]| Processes:                                                                              |
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]|        ID   ID                                                               Usage      |
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]|=========================================================================================|
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]|    0   N/A  N/A             822      C   sglang::scheduler_TP0                 32154MiB |
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]|    0   N/A  N/A            1500      C   sglang::scheduler_TP0                  9516MiB |
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]+-----------------------------------------------------------------------------------------+
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:54,079 - INFO - Checking for CUDA libraries:
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:54,079 - INFO - ✅ libcuda.so found
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:54,079 - INFO - ✅ libcudart.so found
2025-03-25T11:01:54Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:54,079 - INFO - ✅ libnvidia-ml.so found
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,525 - INFO - PyTorch CUDA available: True
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,526 - INFO - CUDA version: 12.4
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,541 - INFO - GPU device count: 1
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,544 - INFO - GPU 0: NVIDIA L40S
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,544 - INFO - GPU 0 memory: 47.68 GB
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,544 - INFO - -----------[start gpu checks]------------------
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,544 - INFO - nvidia_smi_ok: True
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,544 - INFO - cuda_libs_ok: True
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,544 - INFO - torch_cuda_ok: True
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,544 - INFO - -------------[end gpu checks]------------------
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,545 - INFO - --------------------------------
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,545 - INFO - GPU checks result: True
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,545 - INFO - --------------------------------
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:55,546 - INFO -
2025-03-25T11:01:55Z app[3d8d7e96f15e58] ord [info][STARTUP] Starting FastAPI server...
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:57,332 - INFO - Starting server on 0.0.0.0:8080
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]INFO:     Started server process [727]
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]INFO:     Waiting for application startup.
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:57,354 - INFO - Starting FastAPI server for content moderation
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:57,355 - INFO - Configuration: {'EMBEDDING_URL': 'http://127.0.0.1:8890/v1', 'LLM_URL': 'http://127.0.0.1:8899/v1', 'SGLANG_API_KEY': 'None', 'EMBEDDING_MODEL': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct', 'LLM_MODEL': 'microsoft/Phi-3.5-mini-instruct', 'TEMPERATURE': 0.0, 'MAX_NEW_TOKENS': 128, 'SERVER_HOST': '0.0.0.0', 'SERVER_PORT': 8080, 'GCS_BUCKET_NAME': 'test-ds-utility-bucket', 'GCS_PROMPT_PATH': 'project-artifacts-sagar/content-moderation/rag/moderation_prompts.yml', 'DATASET_ID': 'stage_test_tables', 'TABLE_ID': 'test_comment_mod_embeddings', 'GCP_CREDENTIALS': '**REDACTED**', 'PROMPT_PATH': '/app/data/prompts/moderation_prompts.yml'}
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:57,355 - INFO - Initializing moderation service...
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:57,355 - WARNING - Using default hardcoded prompts
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:57,357 - INFO - Initialized GCP credentials from JSON string
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:57,471 - INFO - Initialized embedding client with URL http://127.0.0.1:8890/v1
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:57,513 - INFO - Initialized LLM client with URL http://127.0.0.1:8899/v1
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:57,513 - INFO - Moderation service initialized successfully
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:01:57,513 - INFO - FastAPI server started
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]INFO:     Application startup complete.
2025-03-25T11:01:57Z app[3d8d7e96f15e58] ord [info]INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
2025-03-25T11:02:17Z app[3d8d7e96f15e58] ord [info]INFO:     172.16.1.138:43204 - "POST /moderate HTTP/1.1" 422 Unprocessable Entity
2025-03-25T11:02:35Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:02:35,619 - INFO - Processing moderation request of length 39 chars
2025-03-25T11:02:40Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:02:40,836 - INFO - BigQuery vector search returned 3 results
2025-03-25T11:02:42Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:02:42,539 - INFO - LLM classification result: hate_or_discrimination
2025-03-25T11:02:42Z app[3d8d7e96f15e58] ord [info]INFO:     172.16.1.138:53310 - "POST /moderate HTTP/1.1" 200 OK
2025-03-25T11:05:55Z proxy[3d8d7e96f15e58] ord [info]App stage-content-moderation-server-gpu has excess capacity, autostopping machine 3d8d7e96f15e58. 0 out of 1 machines left running (region=ord, process group=app)
2025-03-25T11:05:55Z app[3d8d7e96f15e58] ord [info] INFO Sending signal SIGINT to main child process w/ PID 727
2025-03-25T11:05:55Z app[3d8d7e96f15e58] ord [info]INFO:     Shutting down
2025-03-25T11:05:55Z app[3d8d7e96f15e58] ord [info]INFO:     Waiting for application shutdown.
2025-03-25T11:05:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:05:55,455 - INFO - Shutting down moderation service
2025-03-25T11:05:55Z app[3d8d7e96f15e58] ord [info]2025-03-25 11:05:55,455 - INFO - Closing HTTP session
2025-03-25T11:05:55Z app[3d8d7e96f15e58] ord [info]INFO:     Application shutdown complete.
2025-03-25T11:05:55Z app[3d8d7e96f15e58] ord [info]INFO:     Finished server process [727]
2025-03-25T11:05:57Z app[3d8d7e96f15e58] ord [info] INFO Main child exited normally with code: 0
2025-03-25T11:05:57Z app[3d8d7e96f15e58] ord [info] INFO Starting clean up.
2025-03-25T11:05:57Z app[3d8d7e96f15e58] ord [info] WARN could not unmount /rootfs: EINVAL: Invalid argument
2025-03-25T11:05:57Z app[3d8d7e96f15e58] ord [info][  462.517736] reboot: Power down
```