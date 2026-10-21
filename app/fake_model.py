"""Modelo de chat simulado para pruebas locales sin red ni cuota de Groq.

Uso: ADA_LLM_FAKE=1 en el worker/API. El enrutado del supervisor es
determinista (researcher -> analyst -> validator -> synthesizer) y no
depende de un proveedor externo. Solo para desarrollo y pruebas.
"""


class FakeChatModel:
    """Routes deterministically from the supervisor context text."""

    def invoke(self, messages):
        text = str(messages[-1][1] if isinstance(messages[-1], tuple) else messages[-1])
        if "Research artifact present: False" in text:
            nxt = "researcher"
        elif "Analysis artifact present: False" in text:
            nxt = "analyst"
        elif "Validation result: none" in text:
            nxt = "validator"
        elif "is_valid=True" in text:
            nxt = "synthesizer"
        else:
            nxt = "limited_finish"

        class Response:
            content = (
                f'{{"next_agent": "{nxt}", "reason": "fake routing", '
                f'"task_completed": false}}'
            )

        return Response()
