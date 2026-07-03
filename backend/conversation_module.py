import json
import logging
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConversationState(BaseModel):
    """
    Persistent conversation state owned by the Conversation Module.
    Not part of LangGraph GraphState — GraphState holds a working copy only.
    """
    thread_id: str
    recent_messages: list[dict] = Field(default_factory=list)
    active_filters: dict = Field(default_factory=dict)
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationModule:
    """
    Conversation Module — source of truth for conversation memory.

    Responsibilities:
    - load_state(): Retrieve persisted state for a thread
    - save_state(): Persist updated state after graph execution
    - clear_filters(): Reset active filters when topic changes

    For the POC, uses in-memory storage.
    Replace with Supabase/RDS PostgreSQL in production without touching LangGraph code.
    """

    def __init__(self):
        self._store: dict[str, ConversationState] = {}

    def load_state(self, thread_id: str) -> ConversationState:
        """
        Load conversation state for the given thread.
        Returns existing state or creates a fresh one.
        """
        if thread_id not in self._store:
            self._store[thread_id] = ConversationState(thread_id=thread_id)
            logger.info(f"ConversationModule: created fresh state for thread {thread_id}")
        else:
            logger.info(f"ConversationModule: loaded state for thread {thread_id}")
        return self._store[thread_id]

    def save_state(self, thread_id: str, recent_messages: list[dict], active_filters: dict) -> None:
        """
        Persist updated conversation state after graph execution.
        """
        state = self.load_state(thread_id)
        state.recent_messages = recent_messages[-6:]
        state.active_filters = active_filters
        state.updated_at = datetime.now(timezone.utc).isoformat()
        self._store[thread_id] = state
        logger.info(f"ConversationModule: saved state for thread {thread_id}")

    def clear_filters(self, thread_id: str) -> None:
        """
        Clear active filters when topic changes.
        """
        if thread_id in self._store:
            self._store[thread_id].active_filters = {}
            logger.info(f"ConversationModule: cleared filters for thread {thread_id}")

    def get_state_summary(self, thread_id: str) -> dict:
        """
        Return a summary of the conversation state (for debugging/observability).
        """
        state = self.load_state(thread_id)
        return {
            "thread_id": state.thread_id,
            "message_count": len(state.recent_messages),
            "active_filters": state.active_filters,
            "updated_at": state.updated_at,
        }


# Singleton instance for the POC
conversation_module = ConversationModule()
