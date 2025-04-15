# Content Moderation API Reference

## Overview

The Content Moderation API provides a RESTful interface for classifying text content into predefined moderation categories. The system uses a Retrieval Augmented Generation (RAG) approach with BigQuery vector search to find similar examples, then uses an LLM to classify the content.

## Authentication

All API endpoints require authentication using an API key.

- Include the API key in the `X-API-Key` HTTP header with every request
- Requests without a valid API key will be rejected with a 401 or 403 status code
- The server will not start without a properly configured API key

## Base URL

**`NOTE`: Get live API from the API administrator, the BASE_URL mentioned below is for demo purposes**

```
https://content-moderation.fly.dev
```

## Endpoints

### Health Check

Check if the API is operational.

```
GET /health
```

#### Request Headers

| Header     | Required | Description                              |
|------------|----------|------------------------------------------|
| X-API-Key  | Yes      | Authentication API key                   |

#### Response

- **200 OK**: API is operational, returns a JSON object with service status information

Sample response:
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

Response fields:
| Field | Description |
|-------|-------------|
| status | Overall system status (healthy, degraded, etc.) |
| version | API version number |
| gcp | GCP service configuration and status |
| services | Status of dependent services (embedding, LLM) |
| config | Current system configuration |

- **401 Unauthorized**: Invalid or missing API key

#### Example Request

```bash
curl --location 'https://content-moderation.fly.dev/health' \
--header 'Content-Type: application/json' \
--header 'X-API-Key: api_key'
```

### Moderate Content

Analyze and classify text content.

```
POST /moderate
```

#### Request Headers

| Header         | Required | Description                              |
|----------------|----------|------------------------------------------|
| X-API-Key      | Yes      | Authentication API key                   |
| Content-Type   | Yes      | Must be application/json                 |

#### Request Body Parameters

| Parameter         | Type    | Required | Description                                |
|-------------------|---------|----------|--------------------------------------------|
| text              | string  | Yes      | The text content to be moderated           |
| num_examples      | integer | No       | Number of similar examples to return (default: 3) |
| max_input_length  | integer | No       | Maximum input length to process (default: 2000) |
| max_tokens        | integer | No       | Maximum tokens in LLM response (default: 128) |

#### Response

Returns a JSON object with the following fields:

| Field             | Type    | Description                                          |
|-------------------|---------|------------------------------------------------------|
| query             | string  | The original text that was submitted                 |
| category          | string  | Classification result (see categories below)         |
| raw_response      | string  | Unprocessed response from the LLM                    |
| similar_examples  | array   | Similar content examples used for classification     |
| prompt            | string  | The prompt sent to the LLM                           |
| embedding_used    | string  | The embedding model used for vector search           |
| llm_used          | boolean | Whether LLM was used for classification              |
| timing            | object  | Performance metrics in milliseconds                  |

#### Classification Categories

- **hate_or_discrimination**: Content targeting protected characteristics with negative intent/prejudice
- **violence_or_threats**: Content that threatens, depicts, or promotes violence
- **offensive_language**: Hostile or inappropriate content WITHOUT targeting protected characteristics
- **nsfw_content**: Explicit sexual content or material intended to arouse
- **spam_or_scams**: Deceptive or unsolicited content designed to mislead
- **clean**: Content that is allowed and doesn't fall into above categories

#### Example Request

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

#### Example Response

```json
{
  "query": "WIN A 100% lottery on gifts worth 5000$!!!! WIN nowww!",
  "category": "spam_or_scams",
  "raw_response": "Category: spam_or_scams\nConfidence: HIGH\nExplanation: The text is promoting a lottery with a high monetary prize, which is a common characteristic of spam or scam messages.",
  "similar_examples": [
    {"text": "Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed £1000 cash or £5000 prize!", "category": "spam_or_scams", "distance": 0.19241615765565134},
    {"text": "Win the newest Harry Potter and the Order of the Phoenix (Book 5) reply HARRY, answer 5 questions - chance to be the first among readers!", "category": "spam_or_scams", "distance": 0.22700561769649286},
    {"text": "important information 4 orange user . today is your lucky day!2find out why log onto http://www.urawinner.com THERE'S A FANTASTIC SURPRISE AWAITING YOU!", "category": "spam_or_scams", "distance": 0.2426976852244851}
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

## Error Codes

| Status Code | Description                                                |
|-------------|------------------------------------------------------------|
| 200         | Success                                                    |
| 400         | Bad Request - Invalid input or missing required parameters |
| 401         | Unauthorized - Missing API key                             |
| 403         | Forbidden - Invalid API key                                |
| 413         | Payload Too Large - Input text exceeds maximum length      |
| 500         | Internal Server Error - Something went wrong               |

## Rate Limits

The API implements rate limiting to ensure fair usage and system stability. Requests exceeding the rate limit will receive a 429 Too Many Requests response. Contact the API administrator for rate limit details and potential increases.

## Best Practices

1. Keep text inputs concise for faster processing
2. Cache results for identical content to reduce API calls
3. Implement retry logic with exponential backoff for transient errors
4. Include proper error handling in your client application