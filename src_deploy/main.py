"""
FastAPI server for content moderation

This module provides a FastAPI server that handles content moderation requests via HTTP.
It uses BigQuery for similarity search to find similar examples for classification.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import sys

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import configuration and models
from src_deploy.config import config, init_config
from src_deploy.models.api_models import (
    ModerationRequest,
    ModerationResponse,
    HealthCheckResponse,
)
from src_deploy.services.moderation_service import ModerationService
from src_deploy import __version__

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Content Moderation API",
    description="API for content moderation using BigQuery vector search",
    version=__version__,
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
            "embeddings_loaded": service.embeddings_df is not None,
            "version": __version__,
            "config": {
                "dataset_id": service.gcp_utils.dataset_id,
                "table_id": service.gcp_utils.table_id,
                "prompt_path": str(service.prompt_path),
                "embeddings_file": (
                    str(service.embeddings_file_path)
                    if service.embeddings_file_path
                    else None
                ),
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """
    Startup event handler - initialize the moderation service
    """
    global moderation_service
    global config

    # Load configuration from environment variable if provided
    config_path = os.environ.get("CONFIG_PATH")
    if config_path:
        logger.info(f"Loading configuration from {config_path}")
        config = init_config(config_path)

    # Initialize the moderation service
    try:
        # Set credentials path from config
        credentials_path = config.gcp_credentials_path

        # Create moderation service
        moderation_service = ModerationService(
            gcp_credentials_path=credentials_path,
            prompt_path=config.prompt_path,
            # Default to loading embeddings from data directory if not specified
            embeddings_file=Path(config.data_root) / config.embeddings_file,
        )

        # Try loading embeddings from file or download from GCS
        try:
            # First, try to load from local file
            if moderation_service.embeddings_df is None:
                embeddings_path = Path(config.data_root) / config.embeddings_file
                if embeddings_path.exists():
                    moderation_service.load_embeddings(embeddings_path)
                    logger.info(f"Loaded embeddings from {embeddings_path}")
                else:
                    logger.warning(f"Embeddings file not found at {embeddings_path}")
                    # TODO: In future, can add code to download from GCS
                    # For now, we'll just log a warning as the bucket name is not configured
                    logger.warning(
                        "GCS download not configured. Please ensure embeddings file exists locally."
                    )
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            # Continue without embeddings, will need to be loaded manually

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
        logger.info("Shutting down moderation service")
        moderation_service = None


def create_app(
    config_path: Optional[str] = None,
    gcp_credentials_path: Optional[str] = None,
    prompt_path: Optional[str] = None,
    embeddings_file: Optional[str] = None,
) -> FastAPI:
    """
    Create and initialize the FastAPI application

    Args:
        config_path: Path to config YAML file
        gcp_credentials_path: Path to GCP credentials JSON file
        prompt_path: Path to prompts file
        embeddings_file: Path to embeddings file

    Returns:
        FastAPI application
    """
    global config

    # Load configuration
    if config_path:
        config = init_config(config_path)

    # Override configuration with function arguments
    if gcp_credentials_path:
        config.gcp_credentials_path = Path(gcp_credentials_path)
    if prompt_path:
        config.prompt_path = Path(prompt_path)
    if embeddings_file:
        config.embeddings_file = embeddings_file

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    config_path: Optional[str] = None,
    reload: bool = False,
    gcp_credentials_path: Optional[str] = None,
    prompt_path: Optional[str] = None,
    embeddings_file: Optional[str] = None,
):
    """
    Run the moderation server

    Args:
        host: Host to bind to
        port: Port to bind to
        config_path: Path to config YAML file
        reload: Whether to enable auto-reload for development
        gcp_credentials_path: Path to GCP credentials JSON file
        prompt_path: Path to prompts file
        embeddings_file: Path to embeddings file
    """
    if config_path:
        os.environ["CONFIG_PATH"] = config_path

    # Set configuration via environment variables
    if gcp_credentials_path:
        os.environ["GCP_CREDENTIALS_PATH"] = gcp_credentials_path
    if prompt_path:
        os.environ["PROMPT_PATH"] = prompt_path
    if embeddings_file:
        os.environ["EMBEDDINGS_FILE"] = embeddings_file

    if reload:
        # For development with reload enabled
        uvicorn.run(
            "src_deploy.main:app",
            host=host,
            port=port,
            reload=True,
        )
    else:
        # For production
        uvicorn.run(
            app,
            host=host,
            port=port,
        )


if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Start the moderation server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", help="Path to config YAML file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--gcp-credentials", help="Path to GCP credentials JSON file")
    parser.add_argument("--prompt", help="Path to prompts file")
    parser.add_argument("--embeddings", help="Path to embeddings file")

    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        config_path=args.config,
        reload=args.reload,
        gcp_credentials_path=args.gcp_credentials,
        prompt_path=args.prompt,
        embeddings_file=args.embeddings,
    )
