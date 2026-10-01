from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="default", max_length=128)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    session_id: str


class PreferenceUpdate(BaseModel):
    tone: str | None = None
    language: str | None = None
    extra: dict[str, Any] | None = None


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    preferences: dict[str, Any]
    updated_at: datetime
