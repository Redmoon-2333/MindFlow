"""Typed request and response schemas for the public MindFlow API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class CollectorStatusResponse(BaseModel):
    status: str
    running: bool
    message: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    tools_used: list[str] = Field(default_factory=list)
    evidence_cited: bool = False
    degraded: bool = False


class PanelTranscriptEntry(BaseModel):
    role: str
    content: str
    round: int


class PanelMeta(BaseModel):
    degraded: bool


class PanelResponse(BaseModel):
    types: list[str]
    confidence: dict[str, float]
    technique: str | None
    rationale: str
    dissent: list[str]
    transcript: list[PanelTranscriptEntry]
    escalated: bool
    call_count: int
    degraded: bool
    meta: PanelMeta


InterventionIntensityValue = Literal["gentle", "standard", "strict"]
InterventionUserResponse = Literal["accepted", "ignored", "dismissed"]
InterventionFeedbackRating = Literal["helpful", "neutral", "annoying"]


class InterventionTriggerRequest(BaseModel):
    intensity: InterventionIntensityValue = "standard"


class InterventionResponseRequest(BaseModel):
    response: InterventionUserResponse
    latency_s: float = Field(default=0.0, ge=0.0)


class InterventionFeedbackRequest(BaseModel):
    rating: InterventionFeedbackRating
    comment: str | None = None


class InterventionPayload(BaseModel):
    id: str
    intervention_type: str
    title: str
    message: str
    dismissible: bool
    created_at: datetime


class InterventionTriggerResponse(BaseModel):
    intervention: InterventionPayload | None
    skipped: bool
    skip_reason: str | None = None


class InterventionCommandResponse(BaseModel):
    status: Literal["ok"] = "ok"
    intervention_id: str
    user_response: InterventionUserResponse | None = None
    feedback_rating: InterventionFeedbackRating | None = None


class InterventionHistoryResponse(BaseModel):
    items: list[dict[str, Any]]
    count: int
    has_more: bool = False
    next_cursor: str | None = None
