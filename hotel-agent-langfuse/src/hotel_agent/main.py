"""FastAPI application — HTTP API for the Hotel Customer Care Agent system."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

# LangSmith env vars must be set BEFORE importing LangChain/LangGraph modules
# so the tracing client is configured before any lazy initialization occurs.
from hotel_agent.config import settings

if settings.langsmith_tracing and settings.langsmith_api_key:
    # Set both legacy (LANGCHAIN_*) and current (LANGSMITH_*) env var names —
    # langsmith 0.3+ prefers LANGSMITH_* but langchain-core still reads LANGCHAIN_*.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import LangChainTracer

from hotel_agent.agents.db_agent import db_agent
from hotel_agent.agents.mcp_agent import mcp_agent, register_all_tools
from hotel_agent.graph.workflow import app_graph
from hotel_agent.models.schemas import AgentState, ChatRequest, ChatResponse, HealthResponse
from hotel_agent.observability.evaluation import evaluate_response
from hotel_agent.observability.metrics import (
    LatencyTimer,
    QueryMetrics,
    estimate_cost,
    get_performance_summary,
    record_query_metrics,
)
from hotel_agent.observability.tracing import (
    check_health as langfuse_health,
    create_trace,
    flush,
    score_trace,
)

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

if settings.langsmith_tracing and settings.langsmith_api_key:
    logger.info("LangSmith tracing enabled for project: %s", settings.langsmith_project)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Starting Hotel Customer Care Agent System...")
    register_all_tools()
    logger.info("MCP Agent: %s", mcp_agent.get_status())
    logger.info("DB Agent: %s", db_agent.check_health())
    yield
    logger.info("Shutting down — flushing Langfuse and LangSmith...")
    flush()


app = FastAPI(
    title="Hotel Customer Care Agent",
    description="Agentic AI system for Grand Horizon Hotel with full Langfuse observability",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a guest query through the multi-agent pipeline.

    Every request is fully traced in Langfuse for observability.
    """
    timer = LatencyTimer()
    timer.start()

    # Generate IDs
    session_id = request.session_id or str(uuid.uuid4())
    trace_id = str(uuid.uuid4())

    # Create Langfuse trace
    trace = create_trace(
        name="hotel_customer_care",
        session_id=session_id,
        user_id=request.user_id,
        input_data=request.message,
        metadata={"source": "api", **request.metadata},
    )
    trace_id = trace.id

    # Build initial state
    initial_state: AgentState = {
        "messages": [HumanMessage(content=request.message)],
        "intent": "",
        "confidence": 0.0,
        "current_agent": "",
        "session_id": session_id,
        "user_id": request.user_id,
        "query_status": "open",
        "metadata": {"_trace": trace, **request.metadata},
        "review_passed": False,
        "trace_id": trace_id,
    }

    try:
        # Build run config — LangGraph propagates callbacks to every node
        run_config: dict = {}
        if settings.langsmith_tracing and settings.langsmith_api_key:
            run_config["callbacks"] = [LangChainTracer(project_name=settings.langsmith_project)]

        # Run the LangGraph workflow
        final_state = await app_graph.ainvoke(initial_state, config=run_config)

        # Extract the final response
        ai_messages = [m for m in final_state["messages"] if hasattr(m, "type") and m.type == "ai"]
        response_text = ai_messages[-1].content if ai_messages else "I'm sorry, I couldn't process your request."

        # Update trace with output
        trace.update(output=response_text)

        # Record metrics
        latency = timer.elapsed_ms()
        metrics = QueryMetrics(
            trace_id=trace_id,
            session_id=session_id,
            intent=final_state.get("intent", "unknown"),
            agent_used=final_state.get("current_agent", "unknown"),
            latency_ms=latency,
            escalated=final_state.get("query_status") == "escalated",
        )
        record_query_metrics(metrics)

        # Async evaluation (fire-and-forget style — doesn't block response)
        # In production, run this in a background task
        review_score = None
        try:
            eval_score = await evaluate_response(
                query=request.message,
                response=response_text,
                trace_id=trace_id,
            )
            review_score = (eval_score.helpfulness + eval_score.accuracy + eval_score.tone) / 3
        except Exception as eval_exc:
            logger.warning("Evaluation failed: %s", eval_exc)

        return ChatResponse(
            response=response_text,
            intent=final_state.get("intent", "unknown"),
            agent_used=final_state.get("current_agent", "unknown"),
            session_id=session_id,
            trace_id=trace_id,
            query_status=final_state.get("query_status", "resolved"),
            review_score=review_score,
        )

    except Exception as exc:
        logger.error("Error processing query: %s", exc, exc_info=True)
        trace.update(output={"error": str(exc)})
        score_trace(trace_id, "error", 1.0, str(exc))
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    finally:
        flush()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """System health check including Langfuse, LangSmith, and ChromaDB status."""
    lf_ok = langfuse_health()
    db_health = db_agent.check_health()
    kb_ready = db_health.get("knowledge_base", {}).get("status") == "ready"
    ls_enabled = bool(os.environ.get("LANGSMITH_TRACING") == "true" and os.environ.get("LANGSMITH_API_KEY"))

    return HealthResponse(
        status="healthy" if lf_ok else "degraded",
        langfuse_connected=lf_ok,
        chromadb_ready=kb_ready,
        langsmith_enabled=ls_enabled,
    )


@app.get("/metrics")
async def metrics():
    """Get agent performance metrics summary."""
    return get_performance_summary()


@app.get("/tools")
async def tools():
    """List all registered tools via MCP agent."""
    return mcp_agent.get_tool_schemas()


@app.get("/tools/status")
async def tools_status():
    """Get MCP agent and tool usage status."""
    return mcp_agent.get_status()


@app.get("/db/status")
async def db_status():
    """Get database agent health status."""
    return db_agent.check_health()
