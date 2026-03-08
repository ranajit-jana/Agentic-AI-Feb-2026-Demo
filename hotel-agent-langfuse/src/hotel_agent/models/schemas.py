"""Pydantic models and LangGraph state definitions."""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# --- Enums ---

class Intent(str, Enum):
    BOOKING = "booking"
    AMENITIES = "amenities"
    BILLING = "billing"
    COMPLAINT = "complaint"
    GENERAL = "general"


class QueryStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


# --- LangGraph State ---

class AgentState(TypedDict):
    """State shared across all nodes in the LangGraph workflow."""
    messages: Annotated[list[BaseMessage], operator.add]
    intent: str
    confidence: float
    current_agent: str
    session_id: str
    user_id: str
    query_status: str
    metadata: dict[str, Any]
    review_passed: bool
    trace_id: str


# --- API Models ---

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="")
    user_id: str = Field(default="guest")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    response: str
    intent: str
    agent_used: str
    session_id: str
    trace_id: str
    query_status: str
    review_score: float | None = None


class HealthResponse(BaseModel):
    status: str
    langfuse_connected: bool
    chromadb_ready: bool
    langsmith_enabled: bool = False


# --- Tool Models ---

class RoomInfo(BaseModel):
    room_type: str
    price_per_night: float
    max_guests: int
    amenities: list[str]
    available: bool


class BookingRecord(BaseModel):
    booking_id: str
    guest_name: str
    room_type: str
    check_in: str
    check_out: str
    total_cost: float
    status: str


class BillItem(BaseModel):
    description: str
    amount: float
    date: str


class GuestBill(BaseModel):
    booking_id: str
    guest_name: str
    items: list[BillItem]
    total: float
    paid: bool


# --- Evaluation Models ---

class EvaluationScore(BaseModel):
    helpfulness: int = Field(ge=1, le=5)
    accuracy: int = Field(ge=1, le=5)
    tone: int = Field(ge=1, le=5)
    reasoning: str


class RouterClassification(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
