# 1. Servers

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


# Start only the embedding server
python src/entrypoint.py server \
    --embedding \
    --emb-port 8890 \
    --emb-model "Alibaba-NLP/Qwen2-1.5B-Instruct"

# Start only the LLM server
python src/entrypoint.py server \
    --llm \
    --llm-port 8899 \
    --llm-model "microsoft/Phi-3.5-mini-instruct" \
    --mem-fraction-llm 0.80



# Test the embedding server
curl -X POST http://localhost:8890/v1/embeddings \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer None" \
     -d '{
         "model": "Alibaba-NLP/Qwen2-1.5B-Instruct",
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
python entrypoint.py moderate \
    --text "This is a test sentence for moderation." \
    --prompt-path /root/content-moderation/prompts/moderation_prompts.yml \
    --db-path /root/content-moderation/data/faiss_vector_db \
    --output ../data/moderation_results.jsonl \
    --examples 3

# 4. Run Moderation Server
python src/entrypoint.py moderation-server \
    --db-path /root/content-moderation/data/faiss_vector_db \
    --prompt-path /root/content-moderation/prompts/moderation_prompts.yml \
    --port 8000

# Test the Moderation Server with Curl
curl -X POST http://localhost:8000/moderate \
     -H "Content-Type: application/json" \
     -d '{
         "text": "This is a test sentence for moderation.",
         "num_examples": 3
     }'

# Health Check for Moderation Server
curl http://localhost:8000/health