#!/usr/bin/env python3
"""
FastAPI server for content moderation

This module provides a FastAPI web server for content moderation using BigQuery for RAG.
"""

import os
import sys
import logging
import asyncio
import json
from typing import Dict, Any, List
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.api_models import (
    ModerationRequest,
    ModerationResponse,
    HealthCheckResponse,
)
from services.moderation_service import ModerationService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Content Moderation API",
    description="API for content moderation using LLMs and BigQuery RAG",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create service instance (will be initialized later)
moderation_service = None


async def get_service() -> ModerationService:
    """
    Dependency to get the moderation service

    Returns:
        Initialized moderation service
    """
    if moderation_service is None or not moderation_service.ready:
        raise HTTPException(status_code=503, detail="Service is not initialized yet")
    return moderation_service


def load_gcp_credentials() -> Dict[str, Any]:
    """
    Load GCP credentials from environment variables or file

    Returns:
        GCP credentials as a dictionary or None if not found
    """
    # Try to get from environment variable
    creds_json = os.environ.get("GCP_CREDENTIALS")
    if creds_json:
        try:
            return json.loads(creds_json)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in GCP_CREDENTIALS environment variable")

    # Try to load from file
    creds_file = os.environ.get("GCP_CREDENTIALS_FILE")
    if creds_file and Path(creds_file).exists():
        try:
            with open(creds_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading GCP credentials from file: {e}")

    # Let's get a list of credential paths from environment variable
    credential_paths = os.environ.get("GCP_CREDENTIAL_PATHS", "")
    if credential_paths:
        paths = [Path(p.strip()) for p in credential_paths.split(",") if p.strip()]
        for path in paths:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading GCP credentials from {path}: {e}")

    logger.warning("No GCP credentials found")
    return None


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global moderation_service

    try:
        logger.info("Starting FastAPI server for content moderation")

        # Get environment variables
        config = {
            # API endpoints
            "EMBEDDING_URL": os.environ.get(
                "EMBEDDING_URL", "http://localhost:8890/v1"
            ),
            "LLM_URL": os.environ.get("LLM_URL", "http://localhost:8899/v1"),
            "SGLANG_API_KEY": os.environ.get("SGLANG_API_KEY", "None"),
            # Model settings
            "EMBEDDING_MODEL": os.environ.get(
                "EMBEDDING_MODEL", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
            ),
            "LLM_MODEL": os.environ.get("LLM_MODEL", "microsoft/Phi-3.5-mini-instruct"),
            "TEMPERATURE": os.environ.get("TEMPERATURE", 0.0),
            "MAX_NEW_TOKENS": os.environ.get("MAX_NEW_TOKENS", 128),
            # Server settings
            "SERVER_HOST": os.environ.get("SERVER_HOST", "0.0.0.0"),
            "SERVER_PORT": (
                int(os.environ.get("SERVER_PORT"))
                if os.environ.get("SERVER_PORT")
                else 8080
            ),
            # Prompt settings
            "GCS_BUCKET_NAME": os.environ.get("GCS_BUCKET_NAME"),
            "GCS_PROMPT_PATH": os.environ.get("GCS_PROMPT_PATH"),
            # BigQuery settings
            "DATASET_ID": os.environ.get("DATASET_ID"),
            "TABLE_ID": os.environ.get("TABLE_ID"),
            # Load GCP credentials
            "GCP_CREDENTIALS": load_gcp_credentials(),
        }

        # Clean up config, removing None values
        config = {k: v for k, v in config.items() if v is not None}

        # Log configuration (without credentials)
        log_config = config.copy()
        if "GCP_CREDENTIALS" in log_config:
            log_config["GCP_CREDENTIALS"] = "**REDACTED**"
        logger.info(f"Configuration: {log_config}")

        # Create and initialize service
        moderation_service = ModerationService(config)

        # Initialize in the background to avoid blocking startup
        # but immediately run it to ensure we start initialization
        background_tasks = BackgroundTasks()
        background_tasks.add_task(moderation_service.initialize)
        await moderation_service.initialize()

        logger.info(f"FastAPI server started")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # We'll let the server start anyway, but the service won't be ready


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global moderation_service

    if moderation_service:
        logger.info("Shutting down moderation service")
        await moderation_service.shutdown()


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint

    Returns:
        Health status of the service
    """
    global moderation_service

    if moderation_service is None:
        return HealthCheckResponse(status="initializing", version="0.1.0", config={})

    health_data = await moderation_service.get_health()
    return HealthCheckResponse(**health_data)


@app.post("/moderate", response_model=ModerationResponse)
async def moderate_content(
    request: ModerationRequest, service: ModerationService = Depends(get_service)
):
    """
    Moderate content

    Args:
        request: Moderation request with text to moderate

    Returns:
        Moderation response with classification results
    """
    try:
        logger.info(
            f"Processing moderation request of length {len(request.text)} chars"
        )
        return await service.moderate_content(request)
    except Exception as e:
        logger.error(f"Error processing moderation request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server():
    """
    Run the FastAPI server using uvicorn
    """
    host = os.environ.get("SERVER_HOST")
    port_str = os.environ.get("SERVER_PORT")

    # Set defaults if not provided
    if not host:
        logger.warning("SERVER_HOST not specified, using 0.0.0.0")
        host = "0.0.0.0"

    if not port_str:
        logger.warning("SERVER_PORT not specified, using 8000")
        port = 8000
    else:
        try:
            port = int(port_str)
        except ValueError:
            logger.warning(f"Invalid SERVER_PORT value '{port_str}', using 8000")
            port = 8000

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "server_fastapi:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    run_server()
