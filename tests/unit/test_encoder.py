"""Tests for the sentence-transformers encoder (mocked, no network)."""

import numpy as np
import pytest

from ada.retrieval.encoder import SentenceTransformerEncoder


class FakeSTModel:
    """Minimal stand-in for a sentence-transformers model."""

    def __init__(self, dim: int = 3) -> None:
        self._dim = dim
        self.encode_calls: list[list[str]] = []

    def encode(
        self,
        sentences: list[str],
        convert_to_numpy: bool = False,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.encode_calls.append(list(sentences))
        vectors = []
        for index, text in enumerate(sentences):
            vector = np.zeros(self._dim, dtype=np.float32)
            vector[index % self._dim] = float(len(text) % 7 + 1)
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def test_encoder_is_lazy_and_defers_model_load():
    encoder = SentenceTransformerEncoder(model_name="fake/model")

    assert encoder.is_loaded is False
    assert encoder.model_name == "fake/model"


def test_encoder_loads_model_on_first_call_and_encodes(monkeypatch):
    created = []

    def fake_loader(model_name: str):
        model = FakeSTModel(dim=4)
        model.model_name = model_name
        created.append(model)
        return model

    monkeypatch.setattr(
        "ada.retrieval.encoder._load_model",
        fake_loader,
    )

    encoder = SentenceTransformerEncoder(model_name="fake/model")
    vectors = encoder(["hola", "mundo"])

    assert len(created) == 1
    assert vectors.shape == (2, 4)
    assert vectors.dtype == np.float32
    assert created[0].encode_calls == [["hola", "mundo"]]


def test_encoder_reuses_loaded_model(monkeypatch):
    load_count = {"count": 0}

    def counting_loader(model_name: str):
        load_count["count"] += 1
        model = FakeSTModel(dim=2)
        model.model_name = model_name
        return model

    monkeypatch.setattr(
        "ada.retrieval.encoder._load_model",
        counting_loader,
    )

    encoder = SentenceTransformerEncoder(model_name="fake/model")
    encoder(["uno"])
    encoder(["dos"])

    assert load_count["count"] == 1
    assert encoder.is_loaded is True


def test_encoder_rejects_empty_text_list():
    encoder = SentenceTransformerEncoder(model_name="fake/model")

    with pytest.raises(ValueError, match="At least one text"):
        encoder([])


def test_encoder_rejects_empty_model_name():
    with pytest.raises(ValueError, match="must not be empty"):
        SentenceTransformerEncoder(model_name="   ")


def test_encoder_wraps_model_load_errors(monkeypatch):
    def failing_loader(model_name: str):
        raise OSError("network unavailable")

    monkeypatch.setattr(
        "ada.retrieval.encoder._load_model",
        failing_loader,
    )

    encoder = SentenceTransformerEncoder(model_name="fake/model")

    try:
        encoder(["texto"])
    except RuntimeError as error:
        assert "Failed to load embedding model" in str(error)
    else:
        raise AssertionError("Expected RuntimeError for model load failure.")
