"""ADA CLI: interactive architecture decisions with live graph execution.

Usage:
    uv run python -m ada
    uv run python -m ada "Decidir arquitectura para una API .NET"
    uv run python -m ada --request "..." [--verbose]

Interactive mode: type a decision request and watch the graph execute node
by node (real LangGraph streaming events). Exit with `quit`, `exit` or EOF
(Ctrl+D / Ctrl+Z). Requires GROQ_API_KEY in .env for live model calls;
without a key the CLI prints setup instructions without exposing secrets.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from ada.config import get_settings

EXIT_COMMANDS = {"quit", "exit"}
HELP_TEXT = (
    "Comandos:\n"
    "  help          mostrar esta ayuda\n"
    "  quit / exit   terminar la sesión\n"
    "Escribe una consulta de decisión de arquitectura y presiona Enter."
)

# Campos útiles por nodo para la demo (nunca el estado completo).
NODE_LABELS = {
    "supervisor": "Supervisor",
    "researcher": "Researcher",
    "analyst": "Analyst",
    "validator": "Validator",
    "synthesizer": "Synthesizer",
    "limited_finish": "Finalización segura",
}


def _print_header() -> None:
    print("=== ADA - Architecture Decision Assistant ===")
    print("Decisiones de arquitectura basadas en evidencia del corpus local.")
    print(HELP_TEXT)


def _route_summary(update: dict) -> str | None:
    """Human-readable routing decision from a supervisor update, if present."""
    next_agent = update.get("next_agent")
    if not next_agent:
        return None
    reason = update.get("routing_reason") or ""
    label = NODE_LABELS.get(next_agent, next_agent)
    return f"Enruta a: {label}" + (f" — {reason}" if reason else "")


def _validation_summary(update: dict) -> list[str]:
    """Blocking errors and attempt number from a validator update, if present."""
    lines: list[str] = []
    validation = update.get("validation_result") or {}
    if not validation:
        return lines
    is_valid = validation.get("is_valid")
    attempt = update.get("retries")
    header = "Validación: aprobada" if is_valid else "Validación: con errores"
    if attempt is not None:
        header += f" (intento {attempt})"
    lines.append(header)
    for error in validation.get("errors", []) or []:
        lines.append(f"  ERROR: {error}")
    for warning in validation.get("warnings", []) or []:
        lines.append(f"  AVISO: {warning}")
    if validation.get("requires_human_review"):
        lines.append("  Revisión humana pendiente.")
    return lines


def _node_event_lines(node: str, update: dict, verbose: bool) -> list[str]:
    """Useful demo fields emitted by a node update (never the full state)."""
    label = NODE_LABELS.get(node, node)
    lines = [f"[{label}]"]
    if node == "supervisor":
        route = _route_summary(update)
        if route:
            lines.append(f"  {route}")
    elif node == "researcher":
        evidence = (update.get("research_artifact") or {}).get("evidence", [])
        lines.append(f"  Evidencias recuperadas: {len(evidence)}")
    elif node == "analyst":
        ranking = (update.get("analysis_artifact") or {}).get("ranking", [])
        if ranking:
            lines.append(f"  Ranking: {' > '.join(ranking)}")
    elif node == "validator":
        lines.extend(f"  {line}" for line in _validation_summary(update))
    elif node == "synthesizer":
        report = update.get("final_report") or {}
        lines.append(f"  Informe generado (estado: {report.get('status', '?')})")
    elif node == "limited_finish":
        lines.append("  El flujo terminó sin recomendación definitiva.")
    if verbose:
        sensitive_keys = {"api_key", "groq_api_key", "token", "secret", "messages"}
        for key in sorted(update):
            if key in {"research_artifact", "analysis_artifact", "final_report"}:
                continue  # artefactos grandes: ya se resumen arriba
            if key.lower() in sensitive_keys or "key" in key.lower() or "token" in key.lower():
                continue  # nunca imprimir claves ni tokens
            value = update[key]
            text = str(value)
            if len(value if isinstance(value, str) else str(value)) > 200:
                text = text[:200] + "…"
            lines.append(f"  [{key}] {text}")
    return lines


def _print_final_summary(state: dict) -> None:
    """Final summary, clearly separated from the live events."""
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)

    report = state.get("final_report") or {}
    status = report.get("status", "?")
    if status == "complete":
        print("Resultado: decisión completada.")
    elif status == "needs_human_review":
        print("Resultado: requiere revisión humana (no es una decisión exitosa).")
    else:
        print("Resultado: finalización segura sin recomendación definitiva.")

    if report:
        print(f"\nEstado del informe: {status}")
        if report.get("executive_summary"):
            print(f"Resumen: {report['executive_summary']}")
        if report.get("recommendation"):
            print(f"Recomendación: {report['recommendation']}")
        considered = report.get("alternatives_considered") or []
        if considered and considered != ["N/A"]:
            print(f"Alternativas comparadas: {', '.join(considered)}")
        for risk in report.get("risks_and_limitations") or []:
            print(f"  Riesgo/Limitación: {risk}")

    analysis = state.get("analysis_artifact") or {}
    if analysis.get("alternatives"):
        print("\nPuntuaciones ponderadas:")
        for alternative in analysis["alternatives"]:
            print(f"  - {alternative.get('name')}: {alternative.get('weighted_score'):.3f}")
        ranking = analysis.get("ranking") or []
        if ranking:
            print(f"  Ranking: {' > '.join(ranking)}")

    research = state.get("research_artifact") or {}
    evidence = research.get("evidence") or []
    if evidence:
        print(f"\nFuentes ({len(evidence)}):")
        for item in evidence:
            print(f"  - [{item.get('document_id')}] {item.get('title')}")
            source = item.get("source_uri") or item.get("source") or "(fuente local)"
            print(f"    Origen: {source}")

    validation = state.get("validation_result") or {}
    if validation and not validation.get("is_valid"):
        print("\nErrores bloqueantes pendientes:")
        for error in validation.get("errors", []) or []:
            print(f"  - {error}")


async def _run_analysis(graph, user_request: str, verbose: bool) -> dict:
    """Stream the graph and print real LangGraph events as they arrive."""
    from ada.decision import build_criteria, extract_decision

    decision_spec = extract_decision(user_request).model_dump()
    # Los criterios extraídos de la consulta se convierten en Criterion
    # válidos (pesos normalizados a 1). Si la consulta no especifica
    # criterios, la lista queda vacía y los nodos aplican los
    # predeterminados.
    requested_criteria = decision_spec["requested_criteria"]
    criteria = [
        criterion.model_dump() for criterion in build_criteria(requested_criteria)
    ] if requested_criteria else []

    initial_state = {
        "user_request": user_request,
        "system_profile": {},
        "criteria": criteria,
        "decision_spec": decision_spec,
        "next_agent": "researcher",
        "routing_reason": "inicio",
        "steps": 0,
        "retries": 0,
        "task_completed": False,
    }

    final_state: dict = {}
    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        for node, update in (chunk or {}).items():
            if not isinstance(update, dict):
                continue
            final_state = {**final_state, **update}
            for line in _node_event_lines(node, update, verbose):
                print(line)
    return final_state


def _read_query() -> str | None:
    """Read one query from stdin. Returns None to exit the session."""
    try:
        raw = input("\nConsulta (quit para salir) > ")
    except EOFError:
        print()
        return None
    except KeyboardInterrupt:
        print()
        return None
    return (raw or "").strip()


def _run_session(chat_model, verbose: bool = False) -> int:
    """Interactive loop: one query per graph execution, repeatable.

    `verbose` se aplica a cada consulta de la sesión: por defecto la
    salida es concisa; con --verbose se añade el detalle de cada nodo
    (sin volcar el estado completo ni secretos).
    """
    from ada.graph import compile_graph

    graph = compile_graph(chat_model)
    _print_header()

    while True:
        query = _read_query()
        if query is None:  # EOF / Ctrl+C
            print("Sesión finalizada.")
            return 0
        if not query:
            print("Entrada vacía: escribe una consulta o 'quit' para salir.")
            continue
        if query.lower() in EXIT_COMMANDS:
            print("Sesión finalizada.")
            return 0
        if query.lower() == "help":
            print(HELP_TEXT)
            continue

        print("\n--- Ejecutando el grafo (eventos en vivo) ---")
        try:
            final_state = asyncio.run(_run_analysis(graph, query, verbose=verbose))
        except KeyboardInterrupt:
            print("\nEjecución interrumpida por el usuario.")
            continue
        except Exception as error:
            print(f"Error durante la ejecución del flujo: {error}")
            return 1
        _print_final_summary(final_state)


def main(argv: list[str] | None = None) -> int:
    """Entry point: `uv run python -m ada [query] [--verbose]`."""
    parser = argparse.ArgumentParser(description="ADA - Architecture Decision Assistant")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Consulta de decisión de arquitectura (opcional; si se omite, modo interactivo).",
    )
    parser.add_argument(
        "--request",
        dest="request",
        default=None,
        help="Alias de 'query' para compatibilidad con demos grabadas.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar detalles adicionales por nodo (sin secretos ni estado completo).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.groq_api_key or not settings.groq_api_key.get_secret_value().strip():
        print(
            "GROQ_API_KEY no está configurada.\n"
            "1. Copia .env.example a .env\n"
            "2. Añade tu clave de Groq en GROQ_API_KEY\n"
            "3. Vuelve a ejecutar: uv run python -m ada"
        )
        return 1

    from ada.config import get_chat_model

    try:
        chat_model = get_chat_model()
    except Exception as error:
        print(f"No se pudo inicializar el modelo: {error}")
        return 1

    user_request = args.query or args.request
    if user_request:
        # One-shot mode: run a single query and exit (useful for scripts/demos).
        from ada.graph import compile_graph

        graph = compile_graph(chat_model)
        print("=== ADA - Architecture Decision Assistant ===")
        print(f"Solicitud: {user_request}")
        print("\n--- Ejecutando el grafo (eventos en vivo) ---")
        try:
            final_state = asyncio.run(_run_analysis(graph, user_request, args.verbose))
        except KeyboardInterrupt:
            print("\nEjecución interrumpida por el usuario.")
            return 130
        except Exception as error:
            print(f"Error durante la ejecución del flujo: {error}")
            return 1
        _print_final_summary(final_state)
        return 0

    return _run_session(chat_model, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
