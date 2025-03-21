FROM python:3.10-slim

WORKDIR /app

# Install dependencies first to utilize Docker cache
COPY src_deploy/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service code
COPY src_deploy/ ./src_deploy/

# Create data directory for local fallback
RUN mkdir -p /app/data/prompts

# Copy local prompts file for fallback
COPY prompts/moderation_prompts.yml /app/data/prompts/moderation_prompts.yml

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_ROOT=/app/data
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8080
ENV GCS_EMBEDDINGS_PATH=project-artifacts-sagar/content-moderation/rag/gcp-embeddings.jsonl
ENV GCS_PROMPT_PATH=project-artifacts-sagar/content-moderation/rag/moderation_prompts.yml

# Expose the port
EXPOSE 8080

# Run the service
CMD ["python", "src_deploy/run_server.py"]