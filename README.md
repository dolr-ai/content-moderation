# Content Moderation System

A RAG-enhanced content moderation system for classifying text into moderation categories.

## Overview

This system uses a combination of LLM inference and RAG (Retrieval-Augmented Generation) to classify text content into moderation categories. It leverages SGLang for efficient model serving and FAISS for vector similarity search.

## Features

- **RAG-Enhanced Moderation**: Uses similar examples to improve classification accuracy
- **Efficient Model Serving**: Optimized for T4 GPUs using SGLang
- **Flexible Configuration**: Easy to configure and adapt to different models
- **Batch Processing**: Support for processing multiple texts in parallel
- **Command-Line Interface**: Simple CLI for all operations
- **Performance Testing**: Tools for throughput and latency analysis

## Quick Start

1. Clone the repository
1. Run the bash script to setup the environment: [./setup/theta_env.sh](./setup/theta_env.sh)
```bash
bash ./setup/theta_env.sh
```
    - it will create a virtual environment and install the dependencies
1. Follow the instructions below to get the system running via single entrypoint.

### Starting the Servers

To run the system, you need to start both LLM and embedding servers.

You can start both servers together or individually:

```bash
# Start both LLM and embedding servers
python src/entrypoint.py server \
    --llm \
    --llm-port 8899 \
    --llm-model "microsoft/Phi-3.5-mini-instruct" \
    --mem-fraction-llm 0.80 \
    --embedding \
    --emb-port 8890 \
    --emb-model "Alibaba-NLP/Qwen2-1.5B-Instruct" \
    --mem-fraction-emb 0.25 \
    --max-requests 32

# Or start embedding server only
python src/entrypoint.py server --embedding --emb-port 8890 --emb-model "Alibaba-NLP/Qwen2-1.5B-Instruct"

# Or start LLM server only
python src/entrypoint.py server --llm --llm-port 8899 --llm-model "microsoft/Phi-3.5-mini-instruct"
```

### Setting Up the Vector Database

Once the LLM and embedding servers are running, you can create the vector database you will need `vector_db_text.jsonl` file for the same ask the owner of this project for the same:

```bash
python src/entrypoint.py vectordb \
    --create \
    --input-jsonl /path/to/vector_db_text.jsonl \
    --save-dir /path/to/faiss_vector_db \
    --prune-text-to-max-chars 2000 \
    --sample 5000
```

You can choose to sample the data or use the entire dataset. In case you don't want to sample the data, just remove the `--sample` flag.

### Moderating Content

Single text moderation:
```bash
python src/entrypoint.py moderate \
    --text "This is a test sentence for moderation." \
    --prompt-path /path/to/moderation_prompts.yml \
    --db-path /path/to/faiss_vector_db \
    --output /path/to/moderation_results.jsonl \
    --examples 3
```

The above command serves as a test to check if the system is working as expected:
  - The LLM and embedding servers are running.
  - The vector database is created.
  - Similar examples are retrieved from the vector database.
  - The final result is being returned and saved to the output file.

### Running as a Service

Once you have ensured single moderation is working as expected, you can start the moderation server:
```bash
python src/entrypoint.py moderation-server \
    --db-path /path/to/faiss_vector_db \
    --prompt-path /path/to/moderation_prompts.yml \
    --port 8000
```

Test the service:
```bash
curl -X POST http://localhost:8000/moderate \
     -H "Content-Type: application/json" \
     -d '{
         "text": "This is a test sentence for moderation.",
         "num_examples": 3
     }'
```

This marks the end of the setup process. You can now use the moderation server to moderate content.

### Performance Testing

The system includes a performance testing module to analyze throughput and latency of the moderation server:

#### Generating Test Data

First, generate test data for performance testing:

```bash
python src/performance/generate_test_data.py \
    --num-samples 1000 \
    --output-file data/test_data.jsonl
```

#### Running Performance Tests

Run a basic sequential test:

```bash
python src/entrypoint.py performance \
    --input-jsonl data/test_data.jsonl \
    --server-url http://localhost:8000 \
    --output-dir performance_results
```

Run a concurrent test with multiple concurrent requests:

```bash
python src/entrypoint.py performance \
    --input-jsonl data/test_data.jsonl \
    --test-type concurrent \
    --concurrency 20
```

Run a concurrency scaling test to find optimal throughput:

```bash
python src/entrypoint.py performance \
    --input-jsonl data/test_data.jsonl \
    --run-scaling-test \
    --concurrency-levels 1,2,4,8,16,32,64
```

The performance test generates a comprehensive report with throughput and latency analysis, along with visualizations and system recommendations.

## Moderation Categories

The system classifies text into the following categories:

1. `hate_or_discrimination`: Content targeting protected characteristics with negative intent/prejudice
2. `violence_or_threats`: Content that threatens, depicts, or promotes violence
3. `offensive_language`: Hostile or inappropriate content WITHOUT targeting protected characteristics
4. `nsfw_content`: Explicit sexual content or material intended to arouse
5. `spam_or_scams`: Deceptive or unsolicited content designed to mislead
6. `clean`: Content that is allowed and doesn't fall into above categories

