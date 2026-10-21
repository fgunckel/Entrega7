"""LangGraph shared state.

Each node should update only the fields it owns, returning partial state updates.
"""

from typing import Literal, NotRequired

from langgraph.graph import MessagesState
from typing_extensions import TypedDict

AgentName = Literal[
    "researcher",
    "analyst",
    "validator",
    "synthesizer",
    "limited_finish",
    "END",
]


class AgentContribution(TypedDict):
    agent: str
    artifact_type: str
    summary: str
    step: int
    decision: NotRequired[str]
    reason: NotRequired[str]
    attempt: NotRequired[int]
    errors: NotRequired[list[str]]


class AgentState(MessagesState):
    user_request: str
    system_profile: dict
    criteria: list[dict]
    # Especificación estructurada de la decisión (decision.py). La fija la
    # CLI al construir el estado; los nodos la leen, nunca la inventan.
    decision_spec: NotRequired[dict | None]

    research_artifact: NotRequired[dict | None]
    analysis_artifact: NotRequired[dict | None]
    validation_result: NotRequired[dict | None]
    contributions: NotRequired[list[AgentContribution]]
    # Errores de validación pendientes, dirigidos al especialista responsable
    # para que corrija su propio artefacto. Se limpian al regenerar.
    validation_errors: NotRequired[list[str]]

    next_agent: AgentName
    routing_reason: str
    steps: int
    retries: int
    task_completed: bool
    final_report: NotRequired[dict | None]

    # --- Human-in-the-loop (Pre-entrega 7) ---
    # La fija la API cuando el cliente marca la tarea como crítica.
    hitl_required: NotRequired[bool]
    # Resultado de la aprobación humana (Command(resume=...)).
    hitl_approved: NotRequired[bool]
    # True cuando el humano rechazó la tarea: el grafo termina de forma segura.
    hitl_rejected: NotRequired[bool]
    # Motivo mostrado al humano cuando el grafo se pausa.
    hitl_reason: NotRequired[str]
    # Costo estimado (USD) y herramientas con efectos secundarios: entradas
    # de la política determinística de criticidad (app/hitl.py).
    estimated_cost_usd: NotRequired[float]
    pending_tools: NotRequired[list[dict]]
