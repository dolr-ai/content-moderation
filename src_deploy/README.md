# Content Moderation Server

A FastAPI server for content moderation using BigQuery vector search.

## Overview

This server provides a REST API for classifying text content into moderation categories. It uses BigQuery vector search to find similar examples and construct a system prompt for classification. In the current version, the server uses a simple majority voting from similar examples as a placeholder for LLM-based classification.

## Features

- REST API for content moderation
- Configurable parameters through environment variables
- BigQuery vector search for finding similar examples
- GCS integration for embeddings and prompts storage
- Docker-ready deployment
- Designed for easy integration with future LLM support

## Requirements

- Python 3.8+
- Google Cloud credentials with access to BigQuery
- Access to the content moderation BigQuery table
- GCS bucket with embeddings JSONL file and prompts YAML file

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

### Example .env file

Create a file named `.env` in the `src_deploy` directory:

```
DATA_ROOT=/app/data
GCP_CREDENTIALS="{\"type\":\"service_account\",\"project_id\":\"your-project-id\",...}"
GCS_BUCKET=test-ds-utility-bucket
GCS_EMBEDDINGS_PATH=project-artifacts-sagar/content-moderation/rag/gcp-embeddings.jsonl
GCS_PROMPT_PATH=project-artifacts-sagar/content-moderation/rag/moderation_prompts.yml
DEBUG=true
```

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