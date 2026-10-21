"""Concrete embedding encoder backed by sentence-transformers.

The model is loaded lazily: importing this module never downloads a model
nor performs any network work. The download happens only on first use.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ada.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> "SentenceTransformer":
    """Load and cache a sentence-transformers model (lazy, on first use)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SentenceTransformerEncoder:
    """Encode texts into normalized float32 embeddings.

    Wraps sentence-transformers with lazy model loading so that importing
    the package stays cheap and offline-safe.
    """

    def __init__(self, model_name: str | None = None) -> None:
        if model_name is None:
            model_name = get_settings().embedding_model

        if not model_name.strip():
            raise ValueError("Embedding model name must not be empty.")

        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying model has been loaded already."""
        return self._model is not None

    def _ensure_model(self) -> "SentenceTransformer":
        """Load the model on first use."""
        if self._model is None:
            try:
                self._model = _load_model(self._model_name)
            except Exception as error:
                raise RuntimeError(
                    f"Failed to load embedding model '{self._model_name}'. "
                    "Check your Internet connection or set ADA_EMBEDDING_MODEL "
                    "to a locally available model."
                ) from error
        return self._model

    def __call__(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode a list of texts into a 2D float32 array."""
        if not texts:
            raise ValueError("At least one text is required to encode.")

        model = self._ensure_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
