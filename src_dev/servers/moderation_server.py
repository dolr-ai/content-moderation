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
    max_new_tokens: int = Field(default=128, ge=1, description="Maximum number of tokens to generate")


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
            max_new_tokens=request.max_new_tokens,
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


def create_app(
    vector_db_path: Union[str, Path],
    prompt_path: Union[str, Path],
    embedding_url: str = "http://localhost:8890/v1",
    llm_url: str = "http://localhost:8899/v1",
    input_length: int = 2000,
    max_new_tokens: int = 128,
) -> FastAPI:
    """
    Create and initialize the FastAPI application

    Args:
        vector_db_path: Path to vector database directory
        prompt_path: Path to prompts file
        embedding_url: URL for embedding server
        llm_url: URL for LLM server
        input_length: Maximum input length
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        FastAPI application
    """
    global moderation_system
    global max_input_length

    # Set global max input length
    max_input_length = input_length

    # Initialize the moderation system
    try:
        moderation_system = ModerationSystem(
            embedding_url=embedding_url,
            llm_url=llm_url,
            vector_db_path=vector_db_path,
            prompt_path=prompt_path,
            max_new_tokens=max_new_tokens,
        )

        logger.info(f"Moderation system initialized with vector DB: {vector_db_path}")
    except Exception as e:
        logger.error(f"Failed to initialize moderation system: {e}")
        raise

    return app


def run_server(
    vector_db_path: Union[str, Path],
    prompt_path: Union[str, Path],
    host: str = "0.0.0.0",
    port: int = 8000,
    embedding_url: str = "http://localhost:8890/v1",
    llm_url: str = "http://localhost:8899/v1",
    input_length: int = 2000,
    max_new_tokens: int = 128,
    reload: bool = False,
):
    """
    Run the moderation server

    Args:
        vector_db_path: Path to vector database directory
        prompt_path: Path to prompts file
        host: Host to bind to
        port: Port to bind to
        embedding_url: URL for embedding server
        llm_url: URL for LLM server
        input_length: Maximum input length
        max_new_tokens: Maximum number of tokens to generate
        reload: Whether to enable auto-reload for development
    """
    import uvicorn

    if reload:
        # For development with reload enabled
        uvicorn.run(
            "src.servers.moderation_server:create_app",
            host=host,
            port=port,
            reload=True,
            factory=True,
            kwargs={
                "vector_db_path": vector_db_path,
                "prompt_path": prompt_path,
                "embedding_url": embedding_url,
                "llm_url": llm_url,
                "input_length": input_length,
                "max_new_tokens": max_new_tokens,
            }
        )
    else:
        # For production
        app = create_app(
            vector_db_path=vector_db_path,
            prompt_path=prompt_path,
            embedding_url=embedding_url,
            llm_url=llm_url,
            input_length=input_length,
            max_new_tokens=max_new_tokens,
        )
        uvicorn.run(app, host=host, port=port)
