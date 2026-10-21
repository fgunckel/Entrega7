"""Human-in-the-loop: política de aprobación, pausa real y reanudación.

La pausa es un `interrupt()` real de LangGraph: el grafo guarda su
checkpoint en Redis y se detiene hasta que un humano envíe la decisión
por POST /tasks/{id}/approve. La reanudación usa Command(resume=...)
sobre el mismo thread_id, sin reiniciar los agentes ya ejecutados.

Política determinística (no depende del LLM): una tarea es crítica
cuando el cliente la marca como tal (`critical=true` en POST /tasks),
o cuando su costo estimado supera el umbral configurable
(ADA_HITL_COST_THRESHOLD), o cuando invoca herramientas con efectos
secundarios declarados.
"""

import logging
import os

from langgraph.types import interrupt

logger = logging.getLogger(__name__)

# Umbral de costo estimado (USD) que marca una tarea como crítica.
# Configurable por variable de entorno (ver .env.example).
DEFAULT_COST_THRESHOLD = 0.50


def get_cost_threshold() -> float:
    """Return the configured critical-cost threshold (USD)."""
    raw = os.getenv("ADA_HITL_COST_THRESHOLD", "").strip()
    try:
        return float(raw) if raw else DEFAULT_COST_THRESHOLD
    except ValueError:
        logger.warning(
            "ADA_HITL_COST_THRESHOLD inválido (%r); usando %s",
            raw,
            DEFAULT_COST_THRESHOLD,
        )
        return DEFAULT_COST_THRESHOLD


def is_critical_task(state: dict) -> bool:
    """Decide whether the task requires mandatory human approval.

    Deterministic rules (evaluated in order):
    - explicit flag `hitl_required` (set from `critical=true` in the API),
    - estimated cost >= configured threshold,
    - pending tools with side effects.
    Once approved (`hitl_approved`), the task is never critical again.
    """
    if state.get("hitl_approved"):
        return False

    if state.get("hitl_required"):
        return True

    try:
        estimated = float(state.get("estimated_cost_usd") or 0.0)
    except (TypeError, ValueError):
        estimated = 0.0
    if estimated >= get_cost_threshold():
        return True

    tools = state.get("pending_tools") or []
    return any(tool.get("side_effects") for tool in tools)


def make_hitl_node():
    """Build the human-approval node that interrupts the graph."""

    def hitl_approval_node(state: dict) -> dict:
        """Pause the graph until a human approves (or rejects) the task."""
        if not is_critical_task(state):
            return {}

        # Pausa real: el grafo guarda checkpoint y se detiene aquí hasta
        # que la API reanude con Command(resume={"approved": ...}).
        decision = interrupt(
            {
                "paused": True,
                "reason": state.get(
                    "hitl_reason", "Tarea crítica: requiere aprobación humana"
                ),
                "requires": "approve|reject",
            }
        )
        approved = bool(isinstance(decision, dict) and decision.get("approved"))
        if not approved:
            logger.info("HITL: tarea rechazada por el humano; finalizando de forma segura")
        return {
            "hitl_approved": approved,
            "hitl_rejected": not approved,
            "contributions": [
                {
                    "agent": "hitl_approval",
                    "artifact_type": "approval",
                    "summary": (
                        "Aprobación humana registrada"
                        if approved
                        else "Tarea rechazada por el humano"
                    ),
                    "step": 0,
                }
            ],
        }

    return hitl_approval_node
