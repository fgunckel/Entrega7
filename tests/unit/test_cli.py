"""CLI and streaming tests for `python -m ada`.

All tests use a fake graph controlled by the test: no network, no Groq key,
no Hugging Face downloads. They verify that the CLI prints real node events
supplied by the graph, handles session commands cleanly, and never presents
a failed validation as a successful report.
"""

import pytest

from ada import __main__ as cli  # noqa: I001

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeGraph:
    """Graph fake that yields controlled update chunks (like stream_mode='updates')."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.calls: list[dict] = []

    def astream(self, initial_state, stream_mode=None):
        self.calls.append(initial_state)
        async def _gen():
            for chunk in self.chunks:
                yield chunk
        return _gen()


def make_graph_factory(chunks: list[dict]):
    graph = FakeGraph(chunks)
    return lambda *a, **k: graph, graph


@pytest.fixture
def run_session(monkeypatch):
    """Factory to run the interactive session with scripted stdin input."""
    def _run(inputs: list[str], chunks: list[dict] | None = None, verbose: bool = False):
        scripted = list(inputs)
        factory, graph = make_graph_factory(chunks or [])
        # __main__ importa compile_graph localmente: parchear el módulo origen.
        monkeypatch.setattr("ada.graph.compile_graph", factory)

        def scripted_input(*args, **kwargs):
            if not scripted:
                raise EOFError  # fin de la entrada simulada
            return scripted.pop(0)

        monkeypatch.setattr("builtins.input", scripted_input)
        outputs: list[str] = []
        write = outputs.append
        monkeypatch.setattr(
            "builtins.print",
            lambda *args, **kwargs: write(" ".join(str(arg) for arg in args)),
        )
        code = cli._run_session(object(), verbose=verbose)
        return code, outputs, graph
    return _run


# ---------------------------------------------------------------------------
# 1. Una consulta válida invoca el grafo y muestra eventos reales
# ---------------------------------------------------------------------------


def test_valid_query_invokes_graph_and_prints_node_events(run_session):
    chunks = [
        {"supervisor": {"next_agent": "researcher", "routing_reason": "sin evidencia"}},
        {"researcher": {"research_artifact": {"evidence": [{"document_id": "d1"}]}}},
        {"validator": {"validation_result": {"is_valid": True, "errors": [], "warnings": []}}},
        {"synthesizer": {"final_report": {"status": "complete"}}},
    ]
    code, outputs, graph = run_session(["Decidir arquitectura", "quit"], chunks)

    assert code == 0
    assert len(graph.calls) == 1  # el grafo se invocó exactamente una vez
    text = "\n".join(outputs)
    assert "[Supervisor]" in text
    assert "[Researcher]" in text
    assert "[Validator]" in text
    assert "[Synthesizer]" in text
    assert "Enruta a: Researcher" in text
    assert "Evidencias recuperadas: 1" in text
    assert "Validación: aprobada" in text


# ---------------------------------------------------------------------------
# 2. Entradas vacías no invocan el grafo y se vuelve a pedir consulta
# ---------------------------------------------------------------------------


def test_empty_input_does_not_invoke_graph(run_session):
    code, outputs, graph = run_session(["", "   ", "quit"])

    assert code == 0
    assert graph.calls == []
    text = "\n".join(outputs)
    assert "Entrada vacía" in text


# ---------------------------------------------------------------------------
# 3. quit / exit terminan sin traceback
# ---------------------------------------------------------------------------


def test_quit_terminates_cleanly(run_session):
    code, outputs, graph = run_session(["quit"])
    assert code == 0
    assert graph.calls == []
    assert "Sesión finalizada" in "\n".join(outputs)


def test_exit_terminates_cleanly(run_session):
    code, outputs, graph = run_session(["exit"])
    assert code == 0
    assert graph.calls == []
    assert "Sesión finalizada" in "\n".join(outputs)


# ---------------------------------------------------------------------------
# 4. EOF se maneja limpiamente
# ---------------------------------------------------------------------------


def test_eof_terminates_cleanly(run_session):
    # Exhausting the scripted input raises EOFError -> clean exit with code 0.
    code, outputs, graph = run_session([])
    assert code == 0
    assert graph.calls == []
    assert "Sesión finalizada" in "\n".join(outputs)


# ---------------------------------------------------------------------------
# 5. La salida muestra decisiones cuando existen
# ---------------------------------------------------------------------------


def test_route_decision_reason_is_shown(run_session):
    chunks = [
        {
            "supervisor": {
                "next_agent": "analyst",
                "routing_reason": "evidencia disponible",
            }
        }
    ]
    code, outputs, _ = run_session(["consulta", "quit"], chunks)
    assert "Enruta a: Analyst — evidencia disponible" in "\n".join(outputs)


# ---------------------------------------------------------------------------
# 6. Validación fallida o revisión humana no se presentan como completadas
# ---------------------------------------------------------------------------


def test_failed_validation_is_not_presented_as_complete(run_session):
    chunks = [
        {
            "validator": {
                "validation_result": {
                    "is_valid": False,
                    "errors": ["Weighted score mismatch for 'a'."],
                    "warnings": [],
                    "requires_human_review": True,
                }
            }
        },
        {"limited_finish": {"final_report": {"status": "limited"}}},
    ]
    code, outputs, _ = run_session(["consulta", "quit"], chunks)
    text = "\n".join(outputs)
    assert "Validación: con errores" in text
    assert "ERROR: Weighted score mismatch" in text
    assert "Revisión humana pendiente" in text
    assert "finalización segura sin recomendación definitiva" in text
    assert "decisión completada" not in text


# ---------------------------------------------------------------------------
# 7. Un informe válido se presenta al final con sus campos principales
# ---------------------------------------------------------------------------


def test_valid_report_is_presented_with_main_fields(run_session):
    chunks = [
        {
            "researcher": {
                "research_artifact": {
                    "evidence": [{"document_id": "d1", "title": "T", "source_uri": "u"}]
                }
            }
        },
        {
            "analyst": {
                "analysis_artifact": {
                    "alternatives": [{"name": "A", "weighted_score": 0.8}],
                    "ranking": ["a"],
                }
            }
        },
        {
            "synthesizer": {
                "final_report": {
                    "status": "complete",
                    "recommendation": "Recomendación: 'A'.",
                }
            }
        },
    ]
    code, outputs, _ = run_session(["consulta", "quit"], chunks)
    text = "\n".join(outputs)
    assert "decisión completada" in text
    assert "Recomendación: 'A'" in text
    assert "Puntuaciones ponderadas" in text
    assert "Fuentes (1)" in text


# ---------------------------------------------------------------------------
# 8. Errores de configuración sin secretos
# ---------------------------------------------------------------------------


def test_missing_key_message_contains_no_secrets(monkeypatch):
    class SettingsWithoutKey:
        groq_api_key = None

    monkeypatch.setattr(cli, "get_settings", lambda: SettingsWithoutKey())
    code = cli.main([])

    # main() prints via builtins.print; capture with capsys-like approach.
    import io
    from contextlib import redirect_stdout

    # Re-run capturing stdout (monkeypatched print is not active here).
    monkeypatch.setattr(cli, "get_settings", lambda: SettingsWithoutKey())
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cli.main([])
    assert code == 1
    output = buffer.getvalue()
    assert "GROQ_API_KEY no está configurada" in output
    assert "gsk_" not in output


# ---------------------------------------------------------------------------
# 9. Verbose no imprime claves ni estado sensible
# ---------------------------------------------------------------------------


def test_verbose_filters_large_artifacts_and_secrets():
    update = {
        "research_artifact": {"evidence": [{"document_id": "d1"}]},
        "user_request": "consulta",
        "groq_api_key": "gsk_secret_value",  # hipotético: nunca debe imprimirse
        "messages": [("system", "secreto")],
    }
    lines = cli._node_event_lines("researcher", update, verbose=True)
    text = "\n".join(lines)
    assert "gsk_secret_value" not in text
    assert "research_artifact" not in text  # artefactos grandes se omiten
    assert "[messages]" not in text  # historial de mensajes: nunca en verbose
    assert "[user_request] consulta" in text  # campos útiles sí se muestran


# ---------------------------------------------------------------------------
# 10. Resumen final separado de los eventos en vivo
# ---------------------------------------------------------------------------


def test_final_summary_is_separated_from_live_events(run_session):
    chunks = [{"synthesizer": {"final_report": {"status": "complete"}}}]
    code, outputs, _ = run_session(["consulta", "quit"], chunks)
    text = "\n".join(outputs)
    assert "RESUMEN FINAL" in text
    assert text.index("RESUMEN FINAL") > text.index("[Synthesizer]")


# ---------------------------------------------------------------------------
# 11. --verbose se propaga a la sesión interactiva, consulta por consulta
# ---------------------------------------------------------------------------


VERBOSE_CHUNKS = [
    {"supervisor": {"next_agent": "researcher", "routing_reason": "sin evidencia"}},
    {
        "researcher": {
            "research_artifact": {"evidence": [{"document_id": "d1"}]},
            "user_request": "consulta",
        }
    },
    {"validator": {"validation_result": {"is_valid": True, "errors": [], "warnings": []}}},
    {"synthesizer": {"final_report": {"status": "complete"}}},
]


def test_verbose_session_shows_extra_fields_per_query(run_session):
    code, outputs, graph = run_session(
        ["consulta", "quit"], VERBOSE_CHUNKS, verbose=True
    )

    assert code == 0
    assert len(graph.calls) == 1
    text = "\n".join(outputs)
    # El detalle verbose aparece en la consulta de la sesión.
    assert "[user_request] consulta" in text


def test_concise_session_without_verbose_has_no_extra_fields(run_session):
    code, outputs, _ = run_session(["consulta", "quit"], VERBOSE_CHUNKS, verbose=False)
    text = "\n".join(outputs)

    assert "[user_request]" not in text  # sin verbose: salida concisa
    assert "[Synthesizer]" in text  # los eventos de nodo sí se muestran


def test_verbose_applies_to_every_query_in_session(run_session):
    code, outputs, graph = run_session(
        ["primera", "segunda", "quit"], VERBOSE_CHUNKS, verbose=True
    )

    assert code == 0
    assert len(graph.calls) == 2  # cada consulta ejecutó el grafo
    text = "\n".join(outputs)
    # El detalle verbose aparece en ambas consultas.
    assert text.count("[user_request] consulta") == 2


def test_verbose_session_never_prints_secrets(run_session):
    chunks = [
        {
            "researcher": {
                "user_request": "consulta",
                "groq_api_key": "gsk_hypothetical_secret",
                "messages": [("system", "secreto")],
            }
        }
    ]
    code, outputs, _ = run_session(["consulta", "quit"], chunks, verbose=True)
    text = "\n".join(outputs)

    assert "gsk_hypothetical_secret" not in text
    assert "[messages]" not in text


def test_exit_command_works_with_verbose(run_session):
    code, outputs, graph = run_session(["exit"], VERBOSE_CHUNKS, verbose=True)

    assert code == 0
    assert len(graph.calls) == 0  # exit no ejecuta el grafo


def test_empty_input_does_not_invoke_graph_with_verbose(run_session):
    code, outputs, graph = run_session(["", "quit"], VERBOSE_CHUNKS, verbose=True)

    assert code == 0
    assert len(graph.calls) == 0  # entradas vacías no ejecutan el grafo
    assert any("Entrada vacía" in line for line in outputs)
