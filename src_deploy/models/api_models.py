"""
API Models for the moderation server
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class ModerationRequest(BaseModel):
    """Request model for text moderation"""

    text: str = Field(..., description="Text content to moderate")
    num_examples: int = Field(
        default=3, ge=1, le=10, description="Number of examples to retrieve"
    )
    max_input_length: int = Field(
        default=2000, description="Maximum input length to process"
    )


class SimilarExample(BaseModel):
    """Model for similar examples returned from search"""

    text: str = Field(..., description="Example text content")
    category: str = Field(..., description="Moderation category")
    distance: float = Field(..., description="Distance/similarity score")


class TimingMetrics(BaseModel):
    """Model for timing metrics of the moderation process"""

    embedding_time_ms: float = Field(0, description="Time to generate embeddings in milliseconds")
    llm_time_ms: float = Field(0, description="Time for LLM classification in milliseconds")
    bigquery_time_ms: float = Field(0, description="Time for BigQuery vector search in milliseconds")
    total_time_ms: float = Field(0, description="Total processing time in milliseconds")


class ModerationResponse(BaseModel):
    """Response model for text moderation"""

    query: str = Field(..., description="Original query text")
    category: str = Field(..., description="Moderation category")
    raw_response: str = Field(..., description="Raw response from the system")
    similar_examples: List[SimilarExample] = Field(
        default_factory=list, description="Similar examples used for classification"
    )
    prompt: str = Field("", description="Generated prompt")
    embedding_used: str = Field("random", description="Type of embedding used")
    llm_used: bool = Field(False, description="Whether LLM was used for classification")
    timing: TimingMetrics = Field(default_factory=TimingMetrics, description="Performance timing metrics")


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint"""

    status: str = Field(..., description="Service status")
    version: str = Field("0.1.0", description="API version")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Service configuration"
    )
