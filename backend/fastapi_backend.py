# IMPORT PACKAGES
import os
import json
import logging
import time
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Query, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
import uuid

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from fastapi.responses import JSONResponse, StreamingResponse

from .ai_agent import graph_builder
from utils import AgentState, ActiveFilters
from .auth import get_current_user_id
from .supabase_database import (
    insert_chat,
    get_chat_history,
    get_session_summaries,
    refresh_authentication,
)
from .conversation_module import conversation_module
from .config import settings
from .security import security_pipeline, metrics_collector, generate_request_id

load_dotenv()

# Structured JSON logging for Cloud Logging
class CloudLogFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return json.dumps(log_entry)

handler = logging.StreamHandler()
handler.setFormatter(CloudLogFormatter())
root_logger = logging.getLogger()
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Supervisor Multi-Agent API", version="2.0.0")

agent_app = None
_start_time = datetime.now(timezone.utc)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# CORS — restrict to frontend domain in production
# Note: Streamlit makes server-side requests, so CORS doesn't apply.
# If a browser-based frontend is added, restrict CORS_ORIGINS to that domain.
_cors_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "detail": "Too many requests. Please slow down."},
    )


# STARTUP: validate config + build agent graph
@app.on_event("startup")
async def startup():
    global agent_app

    required_vars = ["OPENROUTER_API_KEY", "SUPABASE_URL", "SUPABASE_JWT_SECRET", "SUPABASE_SERVICE_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")
    if not settings.supabase_db_url:
        logger.warning("SUPABASE_DB_URL not set — falling back to MemorySaver (not prod-safe)")

    agent_app = await graph_builder()
    logger.info("Agent graph built successfully")


# HOME ROUTE
@app.get("/")
async def home():
    return JSONResponse(
        content={
            "message": "Welcome to the Supervisor Multi-Agent API",
            "documentation": "Visit /docs for API documentation",
            "version": "2.0.0",
        },
        status_code=200,
    )


# HEALTH CHECK ROUTE
@app.get("/health")
async def health_check():
    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds()
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "sma-api",
            "version": "2.0.0",
            "uptime_seconds": uptime,
            "started_at": _start_time.isoformat(),
        },
        status_code=200,
    )


# --- AUTH ENDPOINTS ---

class RefreshRequest(BaseModel):
    refresh_token: str


@app.post("/api/auth/refresh")
def refresh_token(req: RefreshRequest):
    try:
        new_session = refresh_authentication(req.refresh_token)
        return {
            "access_token": new_session.session.access_token,
            "refresh_token": new_session.session.refresh_token,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token refresh failed: {str(e)}",
        )


# --- CHAT PERSISTENCE ENDPOINTS (backend-owned DB ops) ---

class ChatInsertRequest(BaseModel):
    session_id: str
    role: str
    content: str


@app.post("/api/sessions/chat")
def insert_chat_endpoint(req: ChatInsertRequest, user_id: str = Depends(get_current_user_id)):
    result = insert_chat(
        user_id=user_id, session_id=req.session_id, role=req.role, content=req.content
    )
    return {"status": "ok", "data": str(result)}


@app.get("/api/sessions")
def get_sessions_endpoint(user_id: str = Depends(get_current_user_id)):
    return get_session_summaries(user_id=user_id)


@app.get("/api/sessions/{session_id}")
def get_chat_history_endpoint(session_id: str, user_id: str = Depends(get_current_user_id)):
    return get_chat_history(user_id=user_id, session_id=session_id)


# --- CHAT STREAM ENDPOINT ---

class ChatStreamRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


async def generate_agent_response(message: str, thread_id: str, request_id: str, pii_notes: list[str]):
    memory_config = {"configurable": {"thread_id": thread_id}}
    start_time = time.time()

    # Conversation Module: load persistent state
    conv_state = conversation_module.load_state(thread_id)

    # Build working copy for GraphState
    recent_msgs = []
    for msg in conv_state.recent_messages:
        if msg.get("role") == "human":
            recent_msgs.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "ai":
            recent_msgs.append(AIMessage(content=msg["content"]))

    # Inject active filters from Conversation Module
    active_filters = ActiveFilters(**conv_state.active_filters) if conv_state.active_filters else ActiveFilters()

    # Clear filters if topic changed (from previous turn's query_analysis)
    snapshot = await agent_app.aget_state(config=memory_config)
    if snapshot and snapshot.values and "query_analysis" in snapshot.values:
        prev_qa = snapshot.values.get("query_analysis")
        if prev_qa and getattr(prev_qa, "topic_changed", False):
            conversation_module.clear_filters(thread_id)
            active_filters = ActiveFilters()

    events = agent_app.astream_events(
        input=AgentState(messages=recent_msgs + [HumanMessage(content=message)], active_filters=active_filters),
        version="v2",
        config=memory_config,
    )

    full_response = ""

    try:
        async for event in events:
            if event["event"] == "on_chat_model_stream":
                # Skip internal Query Analysis output — only stream agent responses
                metadata = event.get("metadata", {})
                if metadata.get("langgraph_node") == "query_analysis":
                    continue
                if isinstance(event["data"]["chunk"], AIMessageChunk):
                    event_content = event["data"]["chunk"].content
                    full_response += event_content
                    safe_content_json = {"type": "content", "content": event_content}
                    yield f"data: {json.dumps(safe_content_json)}\n\n"
    except Exception as e:
        logger.error(f"Agent streaming error: {e}", extra={"extra_data": {"request_id": request_id}})
        error_json = json.dumps({"type": "error", "content": str(e)})
        yield f"data: {error_json}\n\n"
    finally:
        # Conversation Module: persist updated state after graph execution
        updated_messages = conv_state.recent_messages + [
            {"role": "human", "content": message},
            {"role": "ai", "content": full_response},
        ]

        # Extract updated active filters from graph state
        final_snapshot = await agent_app.aget_state(config=memory_config)
        updated_filters = conv_state.active_filters
        if final_snapshot and final_snapshot.values and "active_filters" in final_snapshot.values:
            af = final_snapshot.values["active_filters"]
            if isinstance(af, dict):
                updated_filters = af
            elif hasattr(af, "model_dump"):
                updated_filters = af.model_dump()

        conversation_module.save_state(thread_id, updated_messages, updated_filters)

        # Output validation: PII redaction on full response
        ov_result = security_pipeline.validate_output(full_response)
        if ov_result.warnings:
            logger.info("Output validation warnings", extra={"extra_data": {"warnings": ov_result.warnings, "request_id": request_id}})

        # Record metrics
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        intent = ""
        if final_snapshot and final_snapshot.values and "query_analysis" in final_snapshot.values:
            qa = final_snapshot.values["query_analysis"]
            if qa:
                intent = getattr(qa, "intent", "")
        success = full_response != "" and "error" not in full_response.lower()
        metrics_collector.record(latency_ms=elapsed_ms, success=success, intent=intent)

        logger.info("Request completed", extra={"extra_data": {
            "request_id": request_id,
            "latency_ms": elapsed_ms,
            "intent": intent,
            "success": success,
            "pii_detected": bool(pii_notes),
            "output_warnings": len(ov_result.warnings),
        }})

        yield "data: {\"type\": \"end\"}\n\n"


@app.post("/chat_stream")
@limiter.limit(settings.rate_limit)
def chat_stream(
    request: Request,
    req: ChatStreamRequest,
    user_id: str = Depends(get_current_user_id),
):
    request_id = generate_request_id()

    # Security check: validate input
    is_allowed, cleaned_msg, security_notes = security_pipeline.validate_input(req.message)
    if not is_allowed:
        metrics_collector.record(latency_ms=0, success=False, blocked=True)
        logger.warning("Request blocked by security", extra={"extra_data": {
            "request_id": request_id,
            "reason": security_notes,
        }})
        raise HTTPException(status_code=400, detail=security_notes[0] if security_notes else "Message blocked")

    pii_notes = [n for n in security_notes if "PII" in n]

    return StreamingResponse(
        generate_agent_response(cleaned_msg, req.thread_id or user_id, request_id, pii_notes),
        media_type="text/event-stream",
    )


# --- METRICS ENDPOINT ---

@app.get("/api/metrics")
def get_metrics():
    return metrics_collector.summary