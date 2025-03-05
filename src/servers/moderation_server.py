"""
FastAPI server for content moderation

This module provides a FastAPI server that keeps the vector database loaded
and allows for moderation requests via HTTP.
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Import from parent package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.moderation.moderation_system import ModerationSystem

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


# Pydantic models for request/response
class ModerationRequest(BaseModel):
    text: str
    num_examples: int = Field(default=3, ge=1, le=10)


class ModerationResponse(BaseModel):
    query: str
    category: str
    raw_response: str
    similar_examples: List[Dict[str, Any]]


def get_moderation_system():
    """Get or initialize the moderation system"""
    global moderation_system
    if moderation_system is None:
        raise HTTPException(status_code=500, detail="Moderation system not initialized")
    return moderation_system


@app.post("/moderate", response_model=ModerationResponse)
async def moderate_text(
    request: ModerationRequest,
    system: ModerationSystem = Depends(get_moderation_system),
    max_input_length: int = 2000,
):
    """Moderate text content"""
    try:
        result = system.classify_text(
            # allow max 2000 characters for classification
            request.text,
            num_examples=request.num_examples,
            max_input_length=max_input_length,
        )
        return result
    except Exception as e:
        logger.error(f"Error in moderation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check(system: ModerationSystem = Depends(get_moderation_system)):
    """Health check endpoint"""
    return {"status": "healthy", "vector_db_loaded": system.index is not None}


def initialize_moderation_system(
    vector_db_path: Union[str, Path],
    prompt_path: Union[str, Path],
    embedding_url: str = "http://localhost:8890/v1",
    llm_url: str = "http://localhost:8899/v1",
):
    """Initialize the moderation system with the vector database"""
    global moderation_system

    try:
        moderation_system = ModerationSystem(
            embedding_url=embedding_url,
            llm_url=llm_url,
            vector_db_path=vector_db_path,
            prompt_path=prompt_path,
        )
        logger.info(f"Moderation system initialized with vector DB: {vector_db_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize moderation system: {e}")
        return False


def run_server(
    vector_db_path: Union[str, Path],
    prompt_path: Union[str, Path],
    host: str = "0.0.0.0",
    port: int = 8000,
    embedding_url: str = "http://localhost:8890/v1",
    llm_url: str = "http://localhost:8899/v1",
    max_input_length: int = 2000,
):
    """Run the FastAPI server"""
    import uvicorn

    # Initialize moderation system
    success = initialize_moderation_system(
        vector_db_path=vector_db_path,
        prompt_path=prompt_path,
        embedding_url=embedding_url,
        llm_url=llm_url,
    )

    if not success:
        logger.error("Failed to initialize moderation system. Exiting.")
        return False

    # Run server
    logger.info(f"Starting moderation server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
    return True
