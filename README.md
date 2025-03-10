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
2. Run the bash script to setup the environment: [./setup/theta_env.sh](./setup/theta_env.sh)
```bash
bash ./setup/theta_env.sh
```
    - it will create a virtual environment and install the dependencies
3. Follow the instructions below to get the system running via single entrypoint.

## System Components

### 1. Servers

#### Starting Combined Servers

Start both LLM and embedding servers on separate GPUs:

```bash
python src/entrypoint.py server \
    --llm \
    --llm-port 8899 \
    --llm-model "microsoft/Phi-3.5-mini-instruct" \
    --mem-fraction-llm 0.95 \
    --llm-gpu-id 0 \
    --embedding \
    --emb-port 8890 \
    --emb-model "Alibaba-NLP/gte-Qwen2-1.5B-instruct" \
    --mem-fraction-emb 0.95 \
    --emb-gpu-id 1 \
    --max-requests 32 \
    --emb-timeout 60 \
    --llm-timeout 120
```

#### Starting Individual Servers

Start only the embedding server on GPU 1:

```bash
python src/entrypoint.py server \
    --embedding \
    --emb-port 8890 \
    --emb-model "Alibaba-NLP/gte-Qwen2-1.5B-instruct" \
    --mem-fraction-emb 0.95 \
    --emb-gpu-id 1
```

Start only the LLM server on GPU 0:

```bash
python src/entrypoint.py server \
    --llm \
    --llm-port 8899 \
    --llm-model "microsoft/Phi-3.5-mini-instruct" \
    --mem-fraction-llm 0.95 \
    --llm-gpu-id 0
```

#### Testing Server Connections

Test the embedding server:

```bash
curl -X POST http://localhost:8890/v1/embeddings \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer None" \
     -d '{
         "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
         "input": "This is a test sentence for embedding."
     }'
```

Test the LLM server:

```bash
curl -X POST http://localhost:8899/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer None" \
     -d '{
         "model": "microsoft/Phi-3.5-mini-instruct",
         "messages": [
             {
                 "role": "user",
                 "content": "Who are you?"
             }
         ]
     }'
```

### 2. Vector Database Setup

Create a complete vector database:

```bash
python src/entrypoint.py vectordb \
    --create \
    --input-jsonl /root/content-moderation/data/vector_db_text.jsonl \
    --save-dir /root/content-moderation/data/faiss_vector_db \
    --prune-text-to-max-chars 2000 \
    --sample 5000
```

You can choose to sample the data or use the entire dataset. In case you don't want to sample the data, just remove the `--sample` flag.

### 3. Content Moderation

#### Single Text Moderation

Run moderation on a single text input:

```bash
python src/entrypoint.py moderate \
    --text "This is a test sentence for moderation." \
    --prompt-path /root/content-moderation/prompts/moderation_prompts.yml \
    --db-path /root/content-moderation/data/rag/faiss_vector_db \
    --output /root/content-moderation/data/rag/moderation_results.jsonl \
    --max-input-length 2000 \
    --num-examples 3
```

The above command serves as a test to check if the system is working as expected:
  - The LLM and embedding servers are running.
  - The vector database is created.
  - Similar examples are retrieved from the vector database.
  - The final result is being returned and saved to the output file.

#### Running Moderation Server

Start the moderation server:

```bash
python src/entrypoint.py moderation-server \
    --db-path /root/content-moderation/data/rag/faiss_vector_db \
    --prompt-path /root/content-moderation/prompts/moderation_prompts.yml \
    --port 8000 \
    --max-input-length 2000
```

#### Testing Moderation Server

Test the Moderation Server with Curl:

```bash
curl -X POST http://localhost:8000/moderate \
     -H "Content-Type: application/json" \
     -d '{
         "text": "This is a test sentence for moderation.",
         "num_examples": 3
     }'
```

Health Check for Moderation Server:

```bash
curl http://localhost:8000/health
```

This marks the end of the setup process. You can now use the moderation server to moderate content.

### 4. Performance Testing

#### Sequential Testing

Run a basic sequential performance test:

```bash
python src/entrypoint.py performance \
    --input-jsonl /root/content-moderation/data/benchmark_v1.jsonl \
    --server-url http://localhost:8000 \
    --output-dir performance_results/sequential \
    --num-samples 100
```

Note: `--skip-visualization` can be added if you explicitly want to skip visualization generation. The default is to generate visualizations.

#### Visualization

Visualize the performance results:

```bash
python src/entrypoint.py visualize \
    --results-file performance_results/sequential/performance_results.json \
    --output-dir performance_results/sequential/visualizations
```

If you skip visualization during the performance test, you can still generate a report using the above command.

#### Concurrent Testing

Run a concurrent performance test:

```bash
python src/entrypoint.py performance \
    --input-jsonl /root/content-moderation/data/benchmark_v1.jsonl \
    --server-url http://localhost:8000 \
    --output-dir performance_results/concurrent \
    --test-type concurrent \
    --concurrency 10 \
    --num-samples 100
```

#### Concurrency Scaling

Run a concurrency scaling test:

```bash
python src/entrypoint.py performance \
    --input-jsonl /root/content-moderation/data/benchmark_v1.jsonl \
    --server-url http://localhost:8000 \
    --output-dir performance_results/scaling \
    --run-scaling-test \
    --test-type concurrent \
    --concurrency-levels 8,16,32,64 \
    --num-samples 1000
```

## Moderation Categories

The system classifies text into the following categories:

1. `hate_or_discrimination`: Content targeting protected characteristics with negative intent/prejudice
2. `violence_or_threats`: Content that threatens, depicts, or promotes violence
3. `offensive_language`: Hostile or inappropriate content WITHOUT targeting protected characteristics
4. `nsfw_content`: Explicit sexual content or material intended to arouse
5. `spam_or_scams`: Deceptive or unsolicited content designed to mislead
6. `clean`: Content that is allowed and doesn't fall into above categories