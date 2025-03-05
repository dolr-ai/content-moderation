"""
FastAPI server for content moderation

This module provides a FastAPI server that handles content moderation requests via HTTP.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

# Import the moderation system
from src.moderation.moderation_system import ModerationSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class ModerationRequest(BaseModel):
    text: str
    num_examples: int = Field(default=3, ge=1, le=10)


class ModerationResponse(BaseModel):
    query: str
    category: str
    raw_response: str
    similar_examples: List[Dict[str, Any]]


# Create FastAPI app
app = FastAPI(title="Content Moderation API", description="API for content moderation")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global moderation system instance
moderation_system = None
max_input_length = 2000


@app.post("/moderate", response_model=ModerationResponse)
async def moderate_text(request: ModerationRequest):
    """Moderate text content"""
    if moderation_system is None:
        raise HTTPException(status_code=500, detail="Moderation system not initialized")

    try:
        # Use the async version of classify_text
        result = await moderation_system.classify_text_async(
            request.text,
            num_examples=request.num_examples,
            max_input_length=max_input_length,
        )
        return result
    except Exception as e:
        logger.error(f"Error in moderation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if moderation_system is None:
        raise HTTPException(status_code=500, detail="Moderation system not initialized")
    return {
        "status": "healthy",
        "vector_db_loaded": moderation_system.index is not None,
    }


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler
    """
    if moderation_system is not None:
        logger.info("Shutting down moderation system")
        await moderation_system.close()


def run_server(
    vector_db_path: Union[str, Path],
    prompt_path: Union[str, Path],
    host: str = "0.0.0.0",
    port: int = 8000,
    embedding_url: str = "http://localhost:8890/v1",
    llm_url: str = "http://localhost:8899/v1",
    input_length: int = 2000,
    workers: int = 4,
):
    """
    Run the moderation server

    Args:
        vector_db_path: Path to vector database
        prompt_path: Path to prompt file
        host: Host to bind to
        port: Port to bind to
        embedding_url: URL for embedding API
        llm_url: URL for LLM API
        input_length: Maximum input length
        workers: Number of worker processes
    """
    global moderation_system
    global max_input_length

    # Set global max input length
    max_input_length = input_length

    # Check if APIs are available before starting server
    if not check_api_health(embedding_url, type="embedding"):
        logger.error(
            f"Embedding API at {embedding_url} is not responding. Please check if it's running."
        )
        sys.exit(1)

    if not check_api_health(llm_url, type="llm"):
        logger.error(
            f"LLM API at {llm_url} is not responding. Please check if it's running."
        )
        sys.exit(1)

    # Initialize the moderation system
    try:
        moderation_system = ModerationSystem(
            embedding_url=embedding_url,
            llm_url=llm_url,
            vector_db_path=vector_db_path,
            prompt_path=prompt_path,
        )
        logger.info(f"Moderation system initialized with vector DB: {vector_db_path}")
    except Exception as e:
        logger.error(f"Failed to initialize moderation system: {e}")
        raise

    # Run the server with uvicorn (with workers)
    import uvicorn

    logger.info(f"Starting moderation server on {host}:{port} with {workers} workers")
    uvicorn.run(app, host=host, port=port, workers=workers)


def check_api_health(api_url: str, type: str) -> bool:
    """
    Check if an API is healthy by sending a simple request

    Args:
        api_url: URL of the API to check
        type: Type of API ("embedding" or "llm")

    Returns:
        True if API is healthy, False otherwise
    """
    try:
        # Set default headers with API key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer None",  # Use the same API key pattern as in ModerationSystem
        }

        if type == "embedding":
            response = requests.post(
                f"{api_url}/embeddings",
                headers=headers,
                json={
                    "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                    "input": "This is a test sentence for embedding.",
                },
                timeout=30,  # Increased from 5 to 30 seconds
            )
        elif type == "llm":
            response = requests.post(
                f"{api_url}/chat/completions",
                headers=headers,
                json={
                    "model": "microsoft/Phi-3.5-mini-instruct",
                    "messages": [
                        {"role": "user", "content": "Hi"}
                    ],  # Simplified prompt for faster response
                },
                timeout=60,  # Increased from 5 to 60 seconds for LLM
            )
        else:
            logger.error(f"Unknown API type: {type}")
            return False

        # Check if response is successful
        if response.status_code == 200:
            return True
        else:
            logger.error(
                f"API {type} returned status code {response.status_code}: {response.text}"
            )
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to {type} API at {api_url}: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the moderation server")
    parser.add_argument(
        "--vector-db-path", required=True, help="Path to vector database"
    )
    parser.add_argument("--prompt-path", required=True, help="Path to prompt file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--embedding-url",
        default="http://localhost:8890/v1",
        help="URL for embedding API",
    )
    parser.add_argument(
        "--llm-url", default="http://localhost:8899/v1", help="URL for LLM API"
    )
    parser.add_argument(
        "--max-input-length", type=int, default=2000, help="Maximum input length"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes"
    )

    args = parser.parse_args()

    run_server(
        vector_db_path=args.vector_db_path,
        prompt_path=args.prompt_path,
        host=args.host,
        port=args.port,
        embedding_url=args.embedding_url,
        llm_url=args.llm_url,
        input_length=args.max_input_length,
        workers=args.workers,
    )
