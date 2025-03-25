# Content Moderation Server

A FastAPI server for content moderation using BigQuery vector search and LLM classification.

## Overview

This server provides a REST API for classifying text content into moderation categories. It uses BigQuery vector search to find similar examples and constructs a RAG-enhanced prompt for an LLM to perform classification. The server supports both synchronous and asynchronous classification methods for better performance.

## Features

- REST API for content moderation
- OpenAI API-compatible integration with LLM and embedding models
- Configurable parameters through environment variables
- BigQuery vector search for finding similar examples
- Dynamic fallback to random embeddings if embedding service is unavailable
- Mock LLM responses if LLM service is unavailable
- Async support for better performance and concurrency
- GCS integration for embeddings and prompts storage
- Docker-ready deployment
- Graceful error handling and service degradation

## Requirements

- Python 3.8+
- Google Cloud credentials with access to BigQuery
- Access to the content moderation BigQuery table
- GCS bucket with embeddings JSONL file and prompts YAML file
- Optional: OpenAI API-compatible embedding service
- Optional: OpenAI API-compatible LLM service

## Installation

### Local Development

1. Clone the repository
2. Install dependencies:

```bash
pip install -r src_deploy/requirements.txt
```

3. Set up environment variables or create a `.env` file in the `src_deploy` directory (see Configuration)

### Docker Deployment

1. Build the Docker image:

```bash
docker build -t moderation-server -f src_deploy/Dockerfile .
```

2. Run the container:

```bash
docker run -p 8080:8080 \
  -e GCP_CREDENTIALS="$(cat /path/to/credentials.json)" \
  -e GCS_BUCKET=your-bucket-name \
  -e GCS_EMBEDDINGS_PATH=project-artifacts-sagar/content-moderation/rag/gcp-embeddings.jsonl \
  -e GCS_PROMPT_PATH=project-artifacts-sagar/content-moderation/rag/moderation_prompts.yml \
  moderation-server
```

## Configuration

The server is configured using environment variables, which can be set in several ways:

1. System environment variables
2. `.env` file in the `src_deploy` directory
3. Custom `.env` file specified with `--env-file`
4. Command-line arguments (override environment variables)
5. Docker environment variables

### Environment Variables

Core settings:
- `DATA_ROOT`: Directory for data files (default: `/app/data` in Docker)
- `GCP_CREDENTIALS`: GCP credentials JSON as a string (the entire JSON content, not a file path)
- `PROMPT_PATH`: Path to local prompts file (fallback if GCS prompts not available)

Server settings:
- `SERVER_HOST`: Host to bind the server to (default: "0.0.0.0")
- `SERVER_PORT`: Port to bind the server to (default: 8080)
- `DEBUG`: Enable debug mode (default: false)
- `RELOAD`: Enable auto-reload for development (default: false)

BigQuery settings:
- `BQ_PROJECT`: BigQuery project ID
- `BQ_DATASET`: BigQuery dataset ID
- `BQ_TABLE`: BigQuery table ID
- `BQ_TOP_K`: Number of examples to retrieve
- `BQ_DISTANCE_TYPE`: Distance metric for similarity search

GCS settings:
- `GCS_BUCKET`: GCS bucket name for embeddings and prompts
- `GCS_EMBEDDINGS_PATH`: Path to embeddings in GCS bucket
- `GCS_PROMPT_PATH`: Path to prompts YAML file in GCS bucket

LLM and Embedding API settings:
- `LLM_URL`: URL of the LLM API
- `EMBEDDING_URL`: URL of the embedding API
- `API_KEY`: API key for authentication

Application settings:
- `MAX_INPUT_LENGTH`: Maximum input length for moderation
- `MAX_NEW_TOKENS`: Maximum new tokens for LLM inference

### Example .env file

Create a file named `.env` in the `src_deploy` directory:

```
DATA_ROOT=/app/data
GCP_CREDENTIALS="{\"type\":\"service_account\",\"project_id\":\"your-project-id\",...}"
GCS_BUCKET=test-ds-utility-bucket
GCS_EMBEDDINGS_PATH=project-artifacts-sagar/content-moderation/rag/gcp-embeddings.jsonl
GCS_PROMPT_PATH=project-artifacts-sagar/content-moderation/rag/moderation_prompts.yml
DEBUG=true
RELOAD=false
BQ_PROJECT=project-id
BQ_DATASET=dataset-name
BQ_TABLE=table-name
BQ_TOP_K=5
BQ_DISTANCE_TYPE=COSINE
LLM_URL=http://localhost:8899/v1
EMBEDDING_URL=http://localhost:8890/v1
API_KEY=your-api-key
MAX_INPUT_LENGTH=2000
MAX_NEW_TOKENS=128
```

## Setting Up LLM and Embedding Services

The server is designed to work with OpenAI API-compatible services for both embedding generation and LLM inference. There are several ways to set up these services:

### Option 1: Using SGL.ai Server (recommended for production)

The SGL.ai server provides a high-performance inference server that's compatible with the OpenAI API.

1. Start the LLM server:

```bash
# Use the provided run_server.py script
python src_deploy/run_server.py --model-path microsoft/Phi-3.5-mini-instruct --port 8899
```

2. Start the embedding server (with a different model):

```bash
# Using a separate environment with the embedding model
python -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-1.5B-instruct --port 8890
```

### Option 2: Using vLLM or other OpenAI-compatible servers

You can use any OpenAI API-compatible server. For example, with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server --model microsoft/Phi-3.5-mini-instruct --port 8899
```

### Option 3: Using OpenAI's API

If you prefer to use the actual OpenAI API, you can set:

```
LLM_URL=https://api.openai.com/v1
EMBEDDING_URL=https://api.openai.com/v1
API_KEY=your-openai-api-key
```

And adjust the model names in the code to match available OpenAI models.

### Running Without LLM/Embedding Services

The server gracefully falls back to random embeddings and mock LLM responses if the respective services are not available. This allows you to test the API functionality even without the full AI stack.

## Usage

### Starting the Server Locally

To start the server with environment variables:

```bash
cd src_deploy
python run_server.py
```

Or with command-line arguments:

```bash
python run_server.py --port 8080 --reload --debug --bucket your-bucket-name \
  --gcs-embeddings-path project-artifacts-sagar/content-moderation/rag/gcp-embeddings.jsonl \
  --gcs-prompt-path project-artifacts-sagar/content-moderation/rag/moderation_prompts.yml
```

Using a custom .env file:

```bash
python run_server.py --env-file /path/to/custom.env
```

Loading GCP credentials from a file:

```bash
python run_server.py --gcp-credentials-file /path/to/credentials.json
```

### API Endpoints

#### POST /moderate

Moderates text content.

**Request:**

```bash
curl -X POST http://localhost:8080/moderate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Text to moderate",
    "num_examples": 3,
    "max_input_length": 2000
  }'
```

**Response:**

```json
{
  "query": "Text to moderate",
  "category": "clean",
  "raw_response": "Category: clean\nConfidence: MEDIUM\nExplanation: This content appears to be non-violating.",
  "similar_examples": [
    {
      "text": "Example text 1",
      "category": "clean",
      "distance": 0.123
    }
  ],
  "prompt": "Generated prompt with examples",
  "embedding_used": "random",
  "llm_used": false
}
```

#### GET /health

Returns the health status of the service.

**Request:**

```bash
curl http://localhost:8080/health
```

#### GET /config

Returns the current configuration (only available in debug mode).

**Request:**

```bash
curl http://localhost:8080/config
```

## Docker Deployment

### Building the Docker Image

```bash
docker build -t moderation-server -f src_deploy/Dockerfile .
```

### Running the Docker Container

```bash
# Basic run
docker run -p 8080:8080 \
  -e GCP_CREDENTIALS="$(cat /path/to/credentials.json)" \
  -e GCS_BUCKET=your-bucket-name \
  -e GCS_EMBEDDINGS_PATH=project-artifacts-sagar/content-moderation/rag/gcp-embeddings.jsonl \
  -e GCS_PROMPT_PATH=project-artifacts-sagar/content-moderation/rag/moderation_prompts.yml \
  moderation-server

# With additional configuration
docker run -p 8080:8080 \
  -e GCP_CREDENTIALS="$(cat /path/to/credentials.json)" \
  -e GCS_BUCKET=your-bucket-name \
  -e GCS_EMBEDDINGS_PATH=project-artifacts-sagar/content-moderation/rag/gcp-embeddings.jsonl \
  -e GCS_PROMPT_PATH=project-artifacts-sagar/content-moderation/rag/moderation_prompts.yml \
  -e BQ_DATASET=stage_test_tables \
  -e BQ_TABLE=test_comment_mod_embeddings \
  -e DEBUG=true \
  moderation-server
```

### Building the docker image
docker build -t mod-server -f src_deploy/Dockerfile .

### Running with docker
docker run -p 8080:8080 --env-file path/to/.env -t mod-server

### Testing with Docker

After starting the container, test the API:

```bash
# Health check
curl http://localhost:8080/health

# Moderation request
curl -X POST http://localhost:8080/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message", "num_examples": 3}'
```

## Future Improvements

1. Integration with real embedding model API
2. Integration with LLM API for classification
3. Support for streaming responses
4. Improved error handling and retries
5. Caching for better performance

## How It Works

The content moderation system uses a RAG (Retrieval-Augmented Generation) approach to classify text content:

1. **Text Embedding**:
   - The input text is converted to an embedding vector using the embedding service.
   - If the embedding service is unavailable, a random embedding from existing data is used.

2. **Similarity Search**:
   - The embedding vector is used to search for similar examples in the BigQuery vector table.
   - The search uses approximate nearest neighbor search to efficiently find matches.

3. **RAG Prompt Construction**:
   - The most similar examples are used to construct a few-shot prompt for the LLM.
   - The examples guide the LLM on how to classify the content.

4. **LLM Classification**:
   - The constructed prompt is sent to the LLM service for classification.
   - The LLM analyzes the text and categorizes it into one of the predefined categories.
   - If the LLM service is unavailable, a default classification is returned.

5. **Response Processing**:
   - The LLM's response is parsed to extract the final classification category.
   - The API returns a structured response with the classification and related metadata.

The system is designed with graceful degradation - if one component fails, it falls back to simpler alternatives rather than failing completely.

# Content Moderation Deployment

This folder contains scripts for deploying the content moderation system. The system consists of:

1. SGLang servers for LLM inference and embeddings
2. FastAPI server for moderation API
3. Integration with BigQuery for vector search

## Requirements

- Python 3.8+
- CUDA-compatible GPU
- Dependencies from requirements.txt

## Quick Start

To run the complete system with default settings:

```bash
python start_sglang_servers.py
```

This will:
1. Start the embedding server (port 8890)
2. Start the LLM server (port 8899)
3. Start the FastAPI moderation API (port 8080)

## Testing the Services

You can test if all services are working with:

```bash
python test_services.py --all
```

Or test individual components:

```bash
# Test just the embedding service
python test_services.py --embedding

# Test just the LLM service
python test_services.py --llm

# Test just the moderation API
python test_services.py --api
```

## Debugging

If you're having issues, use the debugging tool to check service connectivity:

```bash
python debug_urls.py
```

This will show:
- The current API URL settings in environment variables
- Connectivity status for each service
- Suggestions for fixing issues when a service is unreachable

You can also override URLs for testing:

```bash
python debug_urls.py --llm-url http://localhost:8899/v1 --embedding-url http://localhost:8890/v1
```

## Command-line Options

### `start_sglang_servers.py` Options

```
usage: start_sglang_servers.py [-h] [--host HOST] [--port PORT] [--reload] [--debug]
                  [--gcp-credentials-file GCP_CREDENTIALS_FILE]
                  [--bucket BUCKET]
                  [--gcs-embeddings-path GCS_EMBEDDINGS_PATH]
                  [--gcs-prompt-path GCS_PROMPT_PATH] [--prompt PROMPT]
                  [--llm-model LLM_MODEL] [--llm-host LLM_HOST]
                  [--llm-port LLM_PORT] [--embedding-model EMBEDDING_MODEL]
                  [--embedding-host EMBEDDING_HOST]
                  [--embedding-port EMBEDDING_PORT] [--api-key API_KEY]
                  [--llm-mem-fraction LLM_MEM_FRACTION]
                  [--embedding-mem-fraction EMBEDDING_MEM_FRACTION]
                  [--max-requests MAX_REQUESTS] [--llm-only] [--embedding-only]
```

### `run_server_fastapi.py` Options

Use this script if you want to start only the FastAPI server (assuming SGLang servers are already running):

```
usage: run_server_fastapi.py [-h] [--host HOST] [--port PORT] [--reload]
                            [--debug]
                            [--gcp-credentials-file GCP_CREDENTIALS_FILE]
                            [--prompt PROMPT] [--bucket BUCKET]
                            [--gcs-embeddings-path GCS_EMBEDDINGS_PATH]
                            [--gcs-prompt-path GCS_PROMPT_PATH]
                            [--llm-url LLM_URL] [--embedding-url EMBEDDING_URL]
                            [--api-key API_KEY]
```

## Environment Variables

The system can also be configured using environment variables:

### Server Settings
- `SERVER_HOST`: Host for the FastAPI server (default: `0.0.0.0`)
- `SERVER_PORT`: Port for the FastAPI server (default: `8080`)
- `RELOAD`: Enable auto-reload for development (default: `false`)
- `DEBUG`: Enable debug mode (default: `false`)

### LLM Server Settings
- `LLM_MODEL`: LLM model to use (default: `microsoft/Phi-3.5-mini-instruct`)
- `LLM_HOST`: Host for the LLM server (default: `127.0.0.1`)
- `LLM_PORT`: Port for the LLM server (default: `8899`)
- `LLM_MEM_FRACTION`: GPU memory fraction for LLM (default: `0.70`)
- `LLM_URL`: URL for the LLM server (default: `http://{LLM_HOST}:{LLM_PORT}/v1`)

### Embedding Server Settings
- `EMBEDDING_MODEL`: Embedding model to use (default: `Alibaba-NLP/gte-Qwen2-1.5B-instruct`)
- `EMBEDDING_HOST`: Host for the embedding server (default: `127.0.0.1`)
- `EMBEDDING_PORT`: Port for the embedding server (default: `8890`)
- `EMBEDDING_MEM_FRACTION`: GPU memory fraction for embedding (default: `0.70`)
- `EMBEDDING_URL`: URL for the embedding server (default: `http://{EMBEDDING_HOST}:{EMBEDDING_PORT}/v1`)

### General Settings
- `API_KEY`: API key for authentication (default: `None`)
- `MAX_REQUESTS`: Maximum concurrent requests for SGLang servers (default: `32`)
- `MAX_INPUT_LENGTH`: Maximum input text length (default: `2000`)
- `MAX_NEW_TOKENS`: Maximum number of new tokens to generate (default: `128`)

## Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ FastAPI Server  │────▶│  SGLang LLM       │     │  SGLang Embedding │
│ (Moderation API)│     │  Server (Port 8899)│     │  Server (Port 8890)│
└────────┬────────┘     └───────────────────┘     └─────────┬─────────┘
         │                                                   │
         │                                                   │
┌────────▼────────┐                                ┌─────────▼─────────┐
│   BigQuery      │◀───────────────────────────────│  Vector Database  │
│  (RAG Search)   │                                │   (Embeddings)    │
└─────────────────┘                                └───────────────────┘
```

## Troubleshooting

If you encounter issues with the FastAPI server not connecting to the SGLang servers, check:

1. That the SGLang servers are running (`ps aux | grep sglang`)
2. That the URLs are correctly set in the environment variables
3. Run the debug script: `python debug_urls.py`
4. Run the test script to diagnose issues: `python test_services.py --all`

### Common Issues

- **"Embedding client not initialized"**: This means the FastAPI server cannot connect to the embedding service. Check that the embedding server is running and the EMBEDDING_URL environment variable is set correctly.

- **Connection refused**: Check that the ports are not already in use by other processes.

- **Out of memory**: Reduce the memory fractions using `--llm-mem-fraction` and `--embedding-mem-fraction` options.

### Manual Testing with curl

Test embedding service:
```bash
curl -X POST http://localhost:8890/v1/embeddings \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer None" \
     -d '{
         "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
         "input": "This is a test sentence for embedding."
     }'
```

Test LLM service:
```bash
curl -X POST http://localhost:8899/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer None" \
    -d '{
        "model": "microsoft/Phi-3.5-mini-instruct",
        "messages": [{"role": "user", "content": "Who are you?"}],
        "max_tokens": 128
    }'
```

Test moderation API:
```bash
curl -X POST http://localhost:8080/moderate \
     -H "Content-Type: application/json" \
     -d '{
         "text": "This is a test sentence for moderation.",
         "num_examples": 3,
         "max_new_tokens": 128
     }'
```