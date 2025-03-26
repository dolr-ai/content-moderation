#!/usr/bin/env python3
"""
FastAPI server for content moderation with timing metrics
This module provides a FastAPI web server for content moderation using BigQuery for RAG.
"""

import os
import logging
import asyncio
import json
import time
from typing import Dict, Any, List
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

from models.api_models import (
    ModerationRequest,
    ModerationResponse,
    HealthCheckResponse,
)
from services.moderation_service import ModerationService
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instance (will be initialized during startup)
moderation_service = None


# Define lifespan context manager for app startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app
    Handles startup and shutdown events
    """
    global moderation_service

    # STARTUP
    try:
        logger.info("Starting FastAPI server for content moderation")

        # Get service configuration from centralized config
        service_config = config.get_moderation_service_config()

        # Log configuration (without credentials)
        log_config = service_config.copy()
        if "GCP_CREDENTIALS" in log_config:
            log_config["GCP_CREDENTIALS"] = "**REDACTED**"
        logger.info(f"Configuration: {log_config}")

        # Create and initialize service
        moderation_service = ModerationService(service_config)

        # Initialize the service
        await moderation_service.initialize()

        logger.info("FastAPI server started")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # We'll let the server start anyway, but the service won't be ready

    # Yield control back to FastAPI
    yield

    # SHUTDOWN
    if moderation_service:
        logger.info("Shutting down moderation service")
        await moderation_service.shutdown()


# Create FastAPI app with lifespan handler
app = FastAPI(
    title="Content Moderation API",
    description="API for content moderation using LLMs and BigQuery RAG",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add middleware to limit concurrent requests
# This helps prevent overwhelming BigQuery with too many concurrent requests
class ConcurrencyLimitMiddleware:
    def __init__(self, app, max_concurrent_requests=100):
        self.app = app
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.active_requests = 0
        self.lock = asyncio.Lock()
        self.max_concurrent = max_concurrent_requests
        logger.info(
            f"Initialized concurrency limiter with max {max_concurrent_requests} concurrent requests"
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async with self.semaphore:
            async with self.lock:
                self.active_requests += 1
                current = self.active_requests

            try:
                await self.app(scope, receive, send)
            finally:
                async with self.lock:
                    self.active_requests -= 1


# Calculate optimal concurrency based on configuration
bq_pool_size = int(config.get_moderation_service_config().get("BQ_POOL_SIZE", 20))
# Set server concurrency limit to 3x the BigQuery pool size for optimal throughput
max_concurrent_requests = bq_pool_size * 3
app.add_middleware(
    ConcurrencyLimitMiddleware, max_concurrent_requests=max_concurrent_requests
)


async def get_service() -> ModerationService:
    """
    Dependency to get the moderation service

    Returns:
        Initialized moderation service
    """
    if moderation_service is None or not moderation_service.ready:
        raise HTTPException(status_code=503, detail="Service is not initialized yet")
    return moderation_service


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

        response = await service.moderate_content(request)

        # Log timing information
        timing = response.timing
        logger.info(
            f"Moderation timing metrics: "
            f"embedding={timing.embedding_time_ms:.2f}ms, "
            f"llm={timing.llm_time_ms:.2f}ms, "
            f"bigquery={timing.bigquery_time_ms:.2f}ms, "
            f"total={timing.total_time_ms:.2f}ms"
        )

        return response
    except Exception as e:
        logger.error(f"Error processing moderation request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host=None, port=None, reload=None, debug=None):
    """
    Run the FastAPI server using uvicorn

    Args:
        host: Optional host override
        port: Optional port override
        reload: Optional reload flag override
        debug: Optional debug flag override
    """
    # Use values from config if not explicitly provided
    host = host or config.host
    port = port or config.port
    reload_flag = reload if reload is not None else config.reload

    # Use the correct module path
    module_path = "servers.server_fastapi:app"

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        module_path,
        host=host,
        port=port,
        log_level="info",
        reload=reload_flag,
    )


if __name__ == "__main__":
    run_server()
