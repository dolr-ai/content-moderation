"""
FastAPI server for content moderation

This module provides a FastAPI server that handles content moderation requests via HTTP.
It uses BigQuery vector search to find similar examples for classification.
"""

import logging
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import sys

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import configuration and models
from config import config
from models.api_models import (
    ModerationRequest,
    ModerationResponse,
    HealthCheckResponse,
)
from services.moderation_service import ModerationService


# Set up logging
logging_level = logging.DEBUG if config.debug else logging.INFO
logging.basicConfig(
    level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Content Moderation API",
    description="API for content moderation using BigQuery vector search",
    version="0.1.0",
    debug=config.debug,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global moderation service instance
moderation_service = None


def get_moderation_service() -> ModerationService:
    """
    Dependency to get moderation service instance
    Returns:
        ModerationService instance
    """
    if moderation_service is None:
        raise HTTPException(
            status_code=500, detail="Moderation service not initialized"
        )
    return moderation_service


@app.post("/moderate", response_model=ModerationResponse)
async def moderate_text(
    request: ModerationRequest,
    service: ModerationService = Depends(get_moderation_service),
):
    """
    Moderate text content
    Args:
        request: Moderation request with text to moderate
    Returns:
        Moderation response with classification
    """
    try:
        # Use the async version if the embedding and LLM URLs are configured
        if hasattr(service, "classify_text_async") and (
            service.embedding_url or service.llm_url
        ):
            logger.info("Using async classification method")
            result = await service.classify_text_async(
                query=request.text,
                num_examples=request.num_examples,
                max_input_length=request.max_input_length,
            )
        else:
            # Fallback to sync version
            logger.info("Using sync classification method")
            result = service.classify_text(
                query=request.text,
                num_examples=request.num_examples,
                max_input_length=request.max_input_length,
            )
        return result
    except Exception as e:
        logger.error(f"Error in moderation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthCheckResponse)
async def health_check(service: ModerationService = Depends(get_moderation_service)):
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "version": "0.1.0",
            "config": {
                "embedding_api_available": service.embedding_url is not None,
                "llm_api_available": service.llm_url is not None,
                "bigquery_enabled": service.gcp_utils.bq_client is not None,
                "gcs_enabled": service.gcp_utils.storage_client is not None,
            },
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Return the current configuration (for debugging)"""
    if config.debug:
        return {"config": config.to_dict()}
    else:
        return {"message": "Config endpoint only available in debug mode"}


@app.on_event("startup")
async def startup_event():
    """
    Startup event handler - initialize the moderation service
    """
    global moderation_service

    # Initialize the moderation service
    try:
        # Create moderation service
        moderation_service = ModerationService(
            gcp_credentials=config.gcp_credentials,
            prompt_path=config.prompt_path,
            bucket_name=config.gcs_bucket,
            gcs_prompt_path=config.gcs_prompt_path,
            dataset_id=config.bq_dataset,
            table_id=config.bq_table,
        )

        logger.info("Moderation service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize moderation service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler
    """
    global moderation_service
    if moderation_service is not None:
        try:
            # Close the HTTP session to clean up resources
            await moderation_service.close()
            logger.info("Moderation service resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing moderation service resources: {e}")
    logger.info("Server shutting down")


def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: Optional[bool] = None,
    debug: Optional[bool] = None,
):
    """
    Run the moderation server

    Args:
        host: Host to bind to (overrides config)
        port: Port to bind to (overrides config)
        reload: Whether to enable auto-reload (overrides config)
        debug: Whether to enable debug mode (overrides config)
    """
    # Use provided values or fall back to config
    server_host = host or config.host
    server_port = port or config.port
    server_reload = reload if reload is not None else config.reload

    logger.info(
        f"Starting server on {server_host}:{server_port} (reload={server_reload})"
    )

    if server_reload:
        # For development with reload enabled
        uvicorn.run(
            "main:app",
            host=server_host,
            port=server_port,
            reload=True,
            log_level="debug" if config.debug else "info",
        )
    else:
        # For production
        uvicorn.run(
            app,
            host=server_host,
            port=server_port,
            log_level="debug" if config.debug else "info",
        )


if __name__ == "__main__":
    # If running directly, use the run_server function
    run_server()
