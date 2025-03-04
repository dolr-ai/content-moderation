# 1. Servers

# Start both LLM and embedding servers
python src/entrypoint.py server --llm --embedding

# Start only the embedding server
python src/entrypoint.py server --embedding

# Start only the LLM server
python src/entrypoint.py server --llm

# Custom configuration
python src/entrypoint.py server --llm --embedding --llm-port 8899 --emb-port 8890


##### test curl commands

# test the embedding server
curl -X POST http://localhost:8890/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer None" \
  -d '{
    "model": "Alibaba-NLP/Qwen2-1.5B-Instruct",
    "input": "This is a test sentence for embedding."
  }'

# test the LLM server
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
  }'¯¯


# 2. Setup vector database
# Create a complete vector database
python src/entrypoint.py vectordb \
    --create \
    --input-jsonl /root/content-moderation/data/vector_db_text.jsonl \
    --save-dir /root/content-moderation/data/faiss_vector_db \
    --prune-text-to-max-chars 2000 \
    --sample 5000



# 3. Run moderation
python entrypoint.py moderate \
    --text "This is a test sentence for moderation." \
    --prompt-path /root/content-moderation/prompts/moderation_prompts.yml \
    --db-path /root/content-moderation/data/faiss_vector_db \
    --output ../data/moderation_results.jsonl \
    --examples 3


