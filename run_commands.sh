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
curl http://localhost:8890/v1/embeddings   -H "Content-Type: application/json"   -H "Authorization: Bearer None"   -d '{
    "model": "Alibaba-NLP/Qwen2-1.5B-Instruct",
    "input": "This is a test sentence for embedding."
  }'

# test the LLM server
curl http://localhost:8899/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer None"   -d '{
    "model": "microsoft/Phi-3.5-mini-instruct",
    "messages": [{"role": "user", "content": "Who are you?"}]
  }'


# 2. Setup vector database
# Create a complete vector database
python src/entrypoint.py vectordb --create --input-jsonl /root/content-moderation/data/vector_db_text.jsonl --save-dir /root/content-moderation/data/faiss_vector_db

# Create a sample vector database for testing
python src/entrypoint.py vectordb --create --input-jsonl data/rag/vector_db_text.jsonl --save-dir data/rag/faiss_vector_db --sample 1000



#################
cd src

# initialize the embedding server
python main.py server --embedding --emb-port 8890 --mem-fraction 0.75 --attention-backend "triton" --dtype "float16" --max-requests 64 --host "0.0.0.0" --chunked-prefill-size 1024 --schedule-policy "fcfs" --schedule-conservativeness 0.8 --disable-cuda-graph --log-level "info" --api-key "None"

# wait till you initialize the embedding server
# below line will be printed once the server is initialized
#2025-03-03 07:19:26,380 - INFO - Embedding server: [2025-03-03 07:19:26] INFO:     127.0.0.1:52850 - "GET /get_model_info HTTP/1.1" 200 OK

# test the embedding server
curl http://localhost:8890/v1/embeddings   -H "Content-Type: application/json"   -H "Authorization: Bearer None"   -d '{
    "model": "Alibaba-NLP/Qwen2-1.5B-Instruct",
    "input": "This is a test sentence for embedding."
  }'

# setup vector database this needs to be run only once takes around 6 mins for T4
python -m vector_db.setup_db --training-data /root/content-moderation/data/vector_db_text.jsonl --file-format jsonl --text-column text --category-column moderation_category --batch-size 64 --output-path ../data/vector_db --api-key "None" --max-characters 2000


python main.py setup-db --training-data /root/content-moderation/data/vector_db_text.jsonl --file-format jsonl --text-column text --category-column moderation_category --batch-size 64 --output-path ../data/vector_db --api-key "None" --max-characters 2000
# test vector database

python -m vector_db.test_db --query "This is a test sentence for vector database."

# test the vector database by moderating content
python -m main moderate "This is a test sentence for vector database."

# run the moderator
python -m main moderate "This is a test sentence for moderation."
