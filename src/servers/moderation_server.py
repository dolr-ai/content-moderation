"""
FastAPI server for content moderation

This module provides a FastAPI server that handles content moderation requests via HTTP.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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


def run_server(
    vector_db_path: Union[str, Path],
    prompt_path: Union[str, Path],
    host: str = "0.0.0.0",
    port: int = 8000,
    embedding_url: str = "http://localhost:8890/v1",
    llm_url: str = "http://localhost:8899/v1",
    max_input_length: int = 2000,
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
    """
    global moderation_system

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

    # Run the server with uvicorn (no workers)
    import uvicorn

    logger.info(f"Starting moderation server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


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

    args = parser.parse_args()

    run_server(
        vector_db_path=args.vector_db_path,
        prompt_path=args.prompt_path,
        host=args.host,
        port=args.port,
        embedding_url=args.embedding_url,
        llm_url=args.llm_url,
        input_length=args.max_input_length,
    )
