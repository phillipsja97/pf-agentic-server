from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ResearchRequest(BaseModel):
    query: str
    depth: str = "standard"


class JobCreatedResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    id: str
    type: str
    status: JobStatus
    input: Optional[dict] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: str
    version: str
