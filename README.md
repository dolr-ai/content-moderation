# Content Moderation System

A production-ready content moderation system using BigQuery vector search, SGLang servers, and LLM classification.

## Overview

This system provides a RESTful API for content moderation that classifies text into the following categories:

- **hate_or_discrimination**: Content targeting protected characteristics with negative intent/prejudice
- **violence_or_threats**: Content that threatens, depicts, or promotes violence
- **offensive_language**: Hostile or inappropriate content WITHOUT targeting protected characteristics
- **nsfw_content**: Explicit sexual content or material intended to arouse
- **spam_or_scams**: Deceptive or unsolicited content designed to mislead
- **clean**: Content that is allowed and doesn't fall into above categories

The system uses a Retrieval Augmented Generation (RAG) approach with BigQuery vector search to find similar examples, then constructs a prompt for an LLM to classify the content.

## Features

- **RAG-Enhanced Moderation**: Uses similar examples to improve classification accuracy
- **Efficient Model Serving**: Optimized for GPUs using SGLang
- **Production-Ready API**: RESTful API with authentication and error handling
- **Vector Search**: Utilizes BigQuery for efficient vector similarity search
- **Flexible Configuration**: Easy to configure and adapt to different models
- **Performance Testing**: Tools for throughput and latency analysis

## Environment Setup

### Required Environment Variables

The following environment variables are **required** to run the system:

- **HF_TOKEN**: Hugging Face token for downloading models
- **GCP_CREDENTIALS**: GCP credentials JSON for BigQuery and GCS access
- **FLY_IO_DEPLOY_TOKEN**: Required only if deploying to fly.io (add to GitHub secrets)
- **API_KEY**: The API key for securing API endpoints (recommended for production use)

### Configuration Options

Configuration is managed through `config.py`, with the following key settings:

#### Server Settings

- `SERVER_HOST`: Host to bind server (default: "0.0.0.0")
- `SERVER_PORT`: Port to bind server (default: 8080)
- `DEBUG`: Enable debug mode (default: false)
- `API_KEY`: API key for securing endpoints (default: "None" - authentication disabled)

#### Model Settings

- `LLM_MODEL`: LLM model name (default: "microsoft/Phi-3.5-mini-instruct")
- `EMBEDDING_MODEL`: Embedding model name (default: "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
- `LLM_URL`: URL of LLM API (default: "http://localhost:8899/v1")
- `EMBEDDING_URL`: URL of embedding API (default: "http://localhost:8890/v1")

## Deployment Options

### Production Environment

In production, we use single larger GPU instances:

- L40S GPU (48GB) is recommended and more cost-efficient than A10 (24GB)
- The scripts have been tested on both A10 and L40S GPUs on fly.io

### Docker Setup

1. Build the Docker image:

```bash
docker build -t mod-server -f src_deploy/gpu.Dockerfile .
```

2. Run the container:

```bash
docker run -p 8080:8080 -t mod-server --env-file .env
```

## Security

The API requires API key-based authentication for all endpoints:

- All requests to API endpoints must include a valid API key in the `X-API-Key` header
- Requests without a valid API key will be rejected with a 401 or 403 status code
- The server will not start or will respond with a 500 error if no API key is configured

## API Reference

### Overview

The Content Moderation API provides a RESTful interface for classifying text content into predefined moderation categories. The system uses a Retrieval Augmented Generation (RAG) approach with BigQuery vector search to find similar examples, then uses an LLM to classify the content.

### Authentication

All API endpoints require authentication using an API key.

- Include the API key in the `X-API-Key` HTTP header with every request
- Requests without a valid API key will be rejected with a 401 or 403 status code
- The server will not start without a properly configured API key

### Base URL

**NOTE**: Get the live API URL from the API administrator. The BASE_URL mentioned below is for demo purposes.

```
https://content-moderation.fly.dev
```

### Endpoints

#### Health Check

Check if the API is operational.

```
GET /health
```

##### Request Headers

| Header    | Required | Description            |
| --------- | -------- | ---------------------- |
| X-API-Key | Yes      | Authentication API key |

##### Example Request

```bash
curl --location 'https://content-moderation.fly.dev/health' \
--header 'Content-Type: application/json' \
--header 'X-API-Key: api_key'
```

##### Response

Returns a JSON object with service status information.

| Field    | Description                                     |
| -------- | ----------------------------------------------- |
| status   | Overall system status (healthy, degraded, etc.) |
| version  | API version number                              |
| gcp      | GCP service configuration and status            |
| services | Status of dependent services (embedding, LLM)   |
| config   | Current system configuration                    |

Status codes:

- **200 OK**: API is operational
- **401 Unauthorized**: Invalid or missing API key

##### Example Response

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "gcp": {
    "credentials_configured": true,
    "bq_client_initialized": true,
    "storage_client_initialized": true,
    "dataset_id": "stage_test_tables",
    "table_id": "test_comment_mod_embeddings",
    "bq_pool_size": 40
  },
  "services": {
    "embedding": {
      "url": "http://127.0.0.1:8890/v1",
      "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
      "available": true
    },
    "llm": {
      "url": "http://127.0.0.1:8899/v1",
      "model": "microsoft/Phi-3.5-mini-instruct",
      "available": true
    }
  },
  "config": {
    "embedding_model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "llm_model": "microsoft/Phi-3.5-mini-instruct",
    "dataset_id": "stage_test_tables",
    "table_id": "test_comment_mod_embeddings"
  }
}
```

#### Moderate Content

Analyze and classify text content.

```
POST /moderate
```

##### Request Headers

| Header       | Required | Description              |
| ------------ | -------- | ------------------------ |
| X-API-Key    | Yes      | Authentication API key   |
| Content-Type | Yes      | Must be application/json |

##### Request Body Parameters

| Parameter        | Type    | Required | Description                                       |
| ---------------- | ------- | -------- | ------------------------------------------------- |
| text             | string  | Yes      | The text content to be moderated                  |
| num_examples     | integer | No       | Number of similar examples to return (default: 3) |
| max_input_length | integer | No       | Maximum input length to process (default: 2000)   |
| max_tokens       | integer | No       | Maximum tokens in LLM response (default: 128)     |

##### Response

Returns a JSON object with the following fields:

| Field            | Type    | Description                                                |
| ---------------- | ------- | ---------------------------------------------------------- |
| query            | string  | The original text that was submitted                       |
| category         | string  | Classification result (see categories in Overview section) |
| raw_response     | string  | Unprocessed response from the LLM                          |
| similar_examples | array   | Similar content examples used for classification           |
| prompt           | string  | The prompt sent to the LLM                                 |
| embedding_used   | string  | The embedding model used for vector search                 |
| llm_used         | boolean | Whether LLM was used for classification                    |
| timing           | object  | Performance metrics in milliseconds                        |

##### Example Request

```bash
curl --location 'https://content-moderation.fly.dev/moderate' \
--header 'Content-Type: application/json' \
--header 'X-API-Key: api_key' \
--data '{
    "text": "get 80% discount on your next purchase",
    "num_examples": 3,
    "max_tokens": 128,
    "max_input_length": 2000
}'
```

##### Example Response

```json
{
  "query": "WIN A 100% lottery on gifts worth 5000$!!!! WIN nowww!",
  "category": "spam_or_scams",
  "raw_response": "Category: spam_or_scams\nConfidence: HIGH\nExplanation: The text is promoting a lottery with a high monetary prize, which is a common characteristic of spam or scam messages.",
  "similar_examples": [
    {
      "text": "Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed £1000 cash or £5000 prize!",
      "category": "spam_or_scams",
      "distance": 0.19241615765565134
    },
    {
      "text": "Win the newest Harry Potter and the Order of the Phoenix (Book 5) reply HARRY, answer 5 questions - chance to be the first among readers!",
      "category": "spam_or_scams",
      "distance": 0.22700561769649286
    },
    {
      "text": "important information 4 orange user . today is your lucky day!2find out why log onto http://www.urawinner.com THERE'S A FANTASTIC SURPRISE AWAITING YOU!",
      "category": "spam_or_scams",
      "distance": 0.2426976852244851
    }
  ],
  "prompt": "Here are some example classifications:\n\nText: Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed £1000 cash or £5000 prize!\nCategory: spam_or_scams\n\nText: Win the newest Harry Potter and the Order of the Phoenix (Book 5) reply HARRY, answer 5 questions - chance to be the first among readers!\nCategory: spam_or_scams\n\nText: important information 4 orange user . today is your lucky day!2find out why log onto http://www.urawinner.com THERE'S A FANTASTIC SURPRISE AWAITING YOU!\nCategory: spam_or_scams\n\nNow, please classify this text:\nWIN A 100% lottery on gifts worth 5000$!!!! WIN nowww!",
  "embedding_used": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
  "llm_used": true,
  "timing": {
    "embedding_time_ms": 49.93391036987305,
    "llm_time_ms": 5889.359951019287,
    "bigquery_time_ms": 3168.850898742676,
    "total_time_ms": 9116.082906723022
  }
}
```

### Error Codes

| Status Code | Description                                                |
| ----------- | ---------------------------------------------------------- |
| 200         | Success                                                    |
| 400         | Bad Request - Invalid input or missing required parameters |
| 401         | Unauthorized - Missing API key                             |
| 403         | Forbidden - Invalid API key                                |
| 413         | Payload Too Large - Input text exceeds maximum length      |
| 500         | Internal Server Error - Something went wrong               |

### Rate Limits

The API implements rate limiting to ensure fair usage and system stability. Requests exceeding the rate limit will receive a 429 Too Many Requests response. Contact the API administrator for rate limit details and potential increases.

### Best Practices

1. Keep text inputs concise for faster processing
2. Cache results for identical content to reduce API calls
3. Implement retry logic with exponential backoff for transient errors
4. Include proper error handling in your client application
