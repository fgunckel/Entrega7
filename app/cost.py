"""Registro de tokens y costo estimado por ejecución.

Captura el uso de tokens (entrada/salida) que el proveedor reporta en
`usage_metadata` de cada respuesta del LLM, usando el callback oficial de
langchain-core. Calcula el costo estimado en USD con tarifas configurables
por modelo (variables de entorno ADA_MODEL_RATES, formato JSON).

Si el proveedor no informa tokens, el dato se registra como no disponible:
nunca se inventan valores.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Tarifas por defecto (USD por millón de tokens). Configurables con
# ADA_MODEL_RATES='{"modelo": {"input": X, "output": Y}, ...}'.
# Verificar las vigentes en la página de precios del proveedor.
DEFAULT_MODEL_RATES: dict[str, dict[str, float]] = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "openai/gpt-oss-20b": {"input": 0.10, "output": 0.50},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.75},
}


def load_model_rates() -> dict[str, dict[str, float]]:
    """Load per-model rates from ADA_MODEL_RATES (JSON) or defaults."""
    raw = os.getenv("ADA_MODEL_RATES", "").strip()
    if not raw:
        return dict(DEFAULT_MODEL_RATES)
    try:
        rates = json.loads(raw)
        if isinstance(rates, dict) and rates:
            return rates
    except ValueError:
        logger.warning("ADA_MODEL_RATES inválido (%r); usando tarifas por defecto", raw)
    return dict(DEFAULT_MODEL_RATES)


def estimate_cost_usd(
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    rates: dict[str, dict[str, float]] | None = None,
) -> float | None:
    """Estimate the execution cost in USD, or None when data is missing.

    Rates are USD per million tokens. Returns None (dato no disponible)
    when the model is unknown or tokens were not reported.
    """
    if input_tokens is None or output_tokens is None:
        return None
    rates = rates or load_model_rates()
    rate = rates.get(model or "")
    if rate is None:
        # Modelo sin tarifa configurada: no inventar el costo.
        return None
    cost = (
        input_tokens * rate.get("input", 0.0) + output_tokens * rate.get("output", 0.0)
    ) / 1_000_000
    return round(cost, 6)


def extract_usage(usage: Any) -> tuple[int | None, int | None]:
    """Extract (input_tokens, output_tokens) from a usage_metadata dict."""
    if not isinstance(usage, dict):
        return None, None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None, None
    return input_tokens, output_tokens


class UsageRecorder:
    """Accumulate token usage across all LLM calls of one job.

    Designed to be used with langchain-core's get_usage_metadata_callback
    inside the worker thread that runs the graph.
    """

    def __init__(self) -> None:
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.llm_calls: int = 0

    def add(self, usage: Any) -> None:
        """Accumulate one usage_metadata report."""
        input_tokens, output_tokens = extract_usage(usage)
        if input_tokens is None:
            return
        self.llm_calls += 1
        self.input_tokens = (self.input_tokens or 0) + input_tokens
        self.output_tokens = (self.output_tokens or 0) + output_tokens

    @property
    def available(self) -> bool:
        """True when the provider reported token usage."""
        return self.input_tokens is not None

    def summary(self, model: str | None = None) -> dict[str, Any]:
        """Job-level summary persisted in Redis (hash ada:job:{id})."""
        if not self.available:
            return {
                "tokens_available": False,
                "cost_note": "El proveedor no informó uso de tokens; costo no disponible",
            }
        rates = load_model_rates()
        cost = estimate_cost_usd(model, self.input_tokens, self.output_tokens, rates)
        return {
            "tokens_available": True,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": (self.input_tokens or 0) + (self.output_tokens or 0),
            "llm_calls": self.llm_calls,
            "model": model,
            "estimated_cost_usd": cost,
            "cost_note": (
                "Costo estimado con tarifas configurables (ADA_MODEL_RATES); "
                "verificar vigencia en la página de precios del proveedor"
                if cost is not None
                else f"Sin tarifa configurada para el modelo {model!r}; "
                "costo no disponible (configurá ADA_MODEL_RATES)"
            ),
        }
