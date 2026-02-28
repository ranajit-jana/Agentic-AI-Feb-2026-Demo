"""
FastAPI server for the HR Management Agent.
Endpoints: GET /   |   POST /hr/chat
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from hr_agent import build_hr_graph

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(title="HR Management Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class HRRequest(BaseModel):
    message: str
    employee_id: str | None = None


class HRResponse(BaseModel):
    answer: str
    intent: str
    tools_used: list[str]
    employee_id: str | None

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "HR Management Agent"}


@app.post("/hr/chat", response_model=HRResponse)
async def chat(request: HRRequest) -> HRResponse:
    """Process an HR request and return a structured response."""
    graph = build_hr_graph()

    initial_state: dict[str, Any] = {
        "messages": [HumanMessage(content=request.message)],
        "intent": "",
        "employee_id": request.employee_id,
        "context": {},
    }

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Last AI message is the final answer
    messages = result["messages"]
    ai_messages = [
        m for m in messages
        if hasattr(m, "content") and not isinstance(m, HumanMessage)
        and not hasattr(m, "tool_call_id")  # exclude ToolMessages
    ]
    answer = ai_messages[-1].content if ai_messages else "I couldn't process your request."

    # Collect tool names actually called
    tools_used: list[str] = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tools_used.extend(tc["name"] for tc in msg.tool_calls)

    return HRResponse(
        answer=answer,
        intent=result.get("intent", "general"),
        tools_used=tools_used,
        employee_id=result.get("employee_id"),
    )


# ─────────────────────────────────────────────
# Run (development)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
