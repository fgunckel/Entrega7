"""Supervisor agent: route between specialists using structured LLM output.

The supervisor owns the routing fields (`next_agent`, `routing_reason`,
`steps`, `task_completed`) of shared state. It never modifies specialist
artifacts. If the model fails, it falls back to a safe route instead of
entering a loop.
"""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from ada.config import get_settings
from ada.schemas import RoutingDecision

PROMPT_PATH = Path(__file__).parent / "prompts" / "supervisor.txt"


@lru_cache(maxsize=1)
def load_supervisor_prompt() -> str:
    """Load the versioned supervisor prompt from disk."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_supervisor_context(state: dict) -> str:
    """Summarize shared state into the minimal routing context."""
    has_research = bool(state.get("research_artifact"))
    has_analysis = bool(state.get("analysis_artifact"))

    validation_summary = "none"
    if isinstance(validation := state.get("validation_result"), dict):
        validation_summary = (
            f"is_valid={validation.get('is_valid')}, "
            f"errors={len(validation.get('errors', []))}, "
            f"requires_human_review={validation.get('requires_human_review')}"
        )

    return (
        f"User request: {state.get('user_request', '')}\n"
        f"Research artifact present: {has_research}\n"
        f"Analysis artifact present: {has_analysis}\n"
        f"Validation result: {validation_summary}\n"
        f"Final report exists: {bool(state.get('final_report'))}\n"
        f"Steps used: {state.get('steps', 0)}\n"
        f"Retries used: {state.get('retries', 0)}\n"
        f"Task completed flag: {state.get('task_completed', False)}"
    )


def _fallback_route(reason: str) -> dict:
    """Return a safe route when the model output cannot be parsed."""
    return RoutingDecision(
        next_agent="limited_finish",
        reason=f"Supervisor fallback: {reason}" if (reason := reason) else "Supervisor fallback.",
        task_completed=False,
    ).model_dump()


def route_next_agent(state: dict, chat_model) -> RoutingDecision:
    """Ask the model for the next agent and validate the structured output."""
    settings = get_settings()

    messages = [
        ("system", load_supervisor_prompt()),
        (
            "human",
            f"{build_supervisor_context(state)}\n\n"
            "Decide the next agent. Respond ONLY with the JSON object.",
        ),
    ]

    try:
        response = chat_model.invoke(messages)
    except Exception as error:
        return RoutingDecision(
            next_agent="limited_finish",
            reason=f"Chat model failed: {error}",
            task_completed=False,
        )

    try:
        content = content_of(response)
        parsed = json.loads(content)
        decision = RoutingDecision.model_validate(parsed)
    except (ValidationError, AttributeError, TypeError, ValueError) as error:
        return RoutingDecision(
            next_agent="limited_finish",
            reason=f"Invalid supervisor response: {error}",
            task_completed=False,
        )

    # Enforce execution limits deterministically.
    if state.get("steps", 0) >= settings.max_steps and decision.next_agent not in (
        "limited_finish",
        "END",
    ):
        return RoutingDecision(
            next_agent="limited_finish",
            reason="Step limit reached; stopping with a limited result.",
            task_completed=False,
        )

    return decision


def content_of(message) -> str:
    """Extract text content from a chat model response message."""
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)
