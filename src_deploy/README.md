# Content Moderation Server

A FastAPI server for content moderation using BigQuery vector search.

## Overview

This server provides a REST API for classifying text content into moderation categories. It uses BigQuery vector search to find similar examples and construct a system prompt for classification. In the current version, the server uses a simple majority voting from similar examples as a placeholder for LLM-based classification.

## Features

- REST API for content moderation
- Configurable parameters for classification
- BigQuery vector search for finding similar examples
- Designed for easy integration with future LLM support

## Requirements

- Python 3.8+
- Google Cloud credentials with access to BigQuery
- Access to the content moderation BigQuery table
- Embeddings JSONL file (local or in GCS)

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up Google Cloud credentials

## Configuration

The server can be configured in multiple ways:

1. Via configuration file (`dev_config.yml`)
2. Via environment variables
3. Via command-line arguments

### Configuration File

Create a YAML configuration file with the following structure:

```yaml
local:
  PROJECT_ROOT: /path/to/project
  DATA_ROOT: /path/to/data

secrets:
  GCP_CREDENTIALS_PATH: /path/to/credentials.json
```

### Environment Variables

- `CONFIG_PATH`: Path to config YAML file
- `GCP_CREDENTIALS_PATH`: Path to GCP credentials JSON file
- `PROMPT_PATH`: Path to prompts file
- `EMBEDDINGS_FILE`: Path to embeddings file

## Usage

### Starting the Server

To start the server with default configuration:

```bash
python -m src_deploy.run_server
```

Or with custom configuration:

```bash
python -m src_deploy.main --config /path/to/config.yml --port 8080
```

### API Endpoints

#### POST /moderate

Moderates text content.

**Request:**

```json
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

**Response:**

```json
{
  "status": "healthy",
  "embeddings_loaded": true,
  "version": "0.1.0",
  "config": {
    "dataset_id": "stage_test_tables",
    "table_id": "test_comment_mod_embeddings",
    "prompt_path": "/path/to/prompts.yml",
    "embeddings_file": "/path/to/embeddings.jsonl"
  }
}
```

## Future Improvements

1. Integration with real embedding model API
2. Integration with LLM API for classification
3. Support for streaming responses
4. Improved error handling and retries
5. Caching for better performance