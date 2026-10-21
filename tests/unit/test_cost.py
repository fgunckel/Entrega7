"""Unit tests for the token/cost recorder (app/cost.py)."""

from app.cost import (
    DEFAULT_MODEL_RATES,
    UsageRecorder,
    estimate_cost_usd,
    extract_usage,
    load_model_rates,
)


class TestEstimateCost:
    def test_known_model_returns_cost(self):
        cost = estimate_cost_usd("llama-3.1-8b-instant", 1000, 500)
        expected = (1000 * 0.05 + 500 * 0.08) / 1_000_000
        assert cost == round(expected, 6)

    def test_unknown_model_returns_none(self):
        # Modelo sin tarifa: no inventar el costo.
        assert estimate_cost_usd("modelo-desconocido", 1000, 500) is None

    def test_missing_tokens_returns_none(self):
        assert estimate_cost_usd("llama-3.1-8b-instant", None, 500) is None
        assert estimate_cost_usd("llama-3.1-8b-instant", 1000, None) is None

    def test_custom_rates(self):
        rates = {"m": {"input": 1.0, "output": 2.0}}
        cost = estimate_cost_usd("m", 1_000_000, 1_000_000, rates=rates)
        assert cost == 3.0


class TestExtractUsage:
    def test_valid_usage(self):
        assert extract_usage({"input_tokens": 10, "output_tokens": 5}) == (10, 5)

    def test_invalid_usage(self):
        assert extract_usage(None) == (None, None)
        assert extract_usage({"input_tokens": "x"}) == (None, None)
        assert extract_usage("no-dict") == (None, None)


class TestUsageRecorder:
    def test_accumulates_multiple_calls(self):
        recorder = UsageRecorder()
        recorder.add({"input_tokens": 100, "output_tokens": 50})
        recorder.add({"input_tokens": 200, "output_tokens": 70})
        assert recorder.input_tokens == 300
        assert recorder.output_tokens == 120
        assert recorder.llm_calls == 2
        assert recorder.available

    def test_unavailable_when_no_usage(self):
        recorder = UsageRecorder()
        summary = recorder.summary(model="llama-3.1-8b-instant")
        assert not summary["tokens_available"]
        assert "no informó" in summary["cost_note"]
        assert "estimated_cost_usd" not in summary

    def test_summary_with_cost(self):
        recorder = UsageRecorder()
        recorder.add({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
        summary = recorder.summary(model="llama-3.1-8b-instant")
        assert summary["tokens_available"]
        assert summary["estimated_cost_usd"] == round(0.05 + 0.08, 6)

    def test_summary_unknown_model_no_cost(self):
        recorder = UsageRecorder()
        recorder.add({"input_tokens": 10, "output_tokens": 10})
        summary = recorder.summary(model="modelo-desconocido")
        assert summary["tokens_available"]
        assert summary["estimated_cost_usd"] is None
        assert "ADA_MODEL_RATES" in summary["cost_note"]


class TestLoadRates:
    def test_defaults_when_unset(self):
        assert load_model_rates() == DEFAULT_MODEL_RATES

    def test_custom_rates_from_env(self, monkeypatch):
        monkeypatch.setenv("ADA_MODEL_RATES", '{"mi-modelo": {"input": 1.0, "output": 2.0}}')
        rates = load_model_rates()
        assert rates == {"mi-modelo": {"input": 1.0, "output": 2.0}}

    def test_invalid_json_falls_back(self, monkeypatch):
        monkeypatch.setenv("ADA_MODEL_RATES", "no-json")
        assert load_model_rates() == DEFAULT_MODEL_RATES
