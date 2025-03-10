# 1. Servers

# Start both LLM and embedding servers on separate GPUs
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

# Start only the embedding server on GPU 1
python src/entrypoint.py server \
    --embedding \
    --emb-port 8890 \
    --emb-model "Alibaba-NLP/gte-Qwen2-1.5B-instruct" \
    --mem-fraction-emb 0.95 \
    --emb-gpu-id 1

# Start only the LLM server on GPU 0
python src/entrypoint.py server \
    --llm \
    --llm-port 8899 \
    --llm-model "microsoft/Phi-3.5-mini-instruct" \
    --mem-fraction-llm 0.95 \
    --llm-gpu-id 0

# Test the embedding server
curl -X POST http://localhost:8890/v1/embeddings \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer None" \
     -d '{
         "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
         "input": "This is a test sentence for embedding."
     }'

# Test the LLM server
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

# 2. Setup Vector Database

# Create a complete vector database
python src/entrypoint.py vectordb \
    --create \
    --input-jsonl /root/content-moderation/data/vector_db_text.jsonl \
    --save-dir /root/content-moderation/data/faiss_vector_db \
    --prune-text-to-max-chars 2000 \
    --sample 5000

# 3. Run Moderation (Single Text)
python src/entrypoint.py moderate \
    --text "This is a test sentence for moderation." \
    --prompt-path /root/content-moderation/prompts/moderation_prompts.yml \
    --db-path /root/content-moderation/data/rag/faiss_vector_db \
    --output /root/content-moderation/data/rag/moderation_results.jsonl \
    --max-input-length 2000 \
    --num-examples 3

# 4. Run Moderation Server
python src/entrypoint.py moderation-server \
    --db-path /root/content-moderation/data/rag/faiss_vector_db \
    --prompt-path /root/content-moderation/prompts/moderation_prompts.yml \
    --port 8000 \
    --max-input-length 2000

# Test the Moderation Server with Curl
curl -X POST http://localhost:8000/moderate \
     -H "Content-Type: application/json" \
     -d '{
         "text": "This is a test sentence for moderation.",
         "num_examples": 3
     }'

# Health Check for Moderation Server
curl http://localhost:8000/health

# 5. Performance Testing

# Run a basic sequential performance test
python src/entrypoint.py performance \
    --input-jsonl /root/content-moderation/data/benchmark_v1.jsonl \
    --server-url http://localhost:8000 \
    --output-dir performance_results/sequential \
    --num-samples 100

# --skip-visualization unless you explicitly specify it to be skipped default is to generate visualizations

# Visualize the performance results
python src/entrypoint.py visualize \
    --results-file performance_results/sequential/performance_results.json \
    --output-dir performance_results/sequential/visualizations

# if you skip visualization, you can still generate a report using above command

# Run a concurrent performance test
python src/entrypoint.py performance \
    --input-jsonl /root/content-moderation/data/benchmark_v1.jsonl \
    --server-url http://localhost:8000 \
    --output-dir performance_results/concurrent \
    --test-type concurrent \
    --concurrency 10 \
    --num-samples 100

# Run a concurrency scaling test
python src/entrypoint.py performance \
    --input-jsonl /root/content-moderation/data/benchmark_v1.jsonl \
    --server-url http://localhost:8000 \
    --output-dir performance_results/scaling \
    --run-scaling-test \
    --test-type concurrent \
    --concurrency-levels 8,16,32,64 \
    --num-samples 1000