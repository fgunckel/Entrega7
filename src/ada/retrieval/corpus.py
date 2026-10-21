"""Load the curated local corpus from data/corpus/."""

import json
from functools import lru_cache
from pathlib import Path

from ada.retrieval.documents import TechnicalDocument

DEFAULT_CORPUS_PATH = Path("data/corpus/corpus.json")


def load_corpus(corpus_path: Path | str | None = None) -> list[TechnicalDocument]:
    """Load and validate the technical corpus from disk.

    Raises clear errors for missing files, malformed JSON, duplicate IDs
    or documents that do not satisfy the `TechnicalDocument` contract.
    """
    path = Path(corpus_path) if corpus_path is not None else DEFAULT_CORPUS_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus file not found at '{path}'. "
            "Create data/corpus/corpus.json with the technical documents."
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Corpus file '{path}' is not valid JSON: {error}") from error

    entries = raw.get("documents") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"Corpus file '{path}' must contain a non-empty 'documents' list."
        )

    documents: list[TechnicalDocument] = []
    seen_ids: set[str] = set()

    for index, entry in enumerate(entries):
        try:
            document = TechnicalDocument.model_validate(entry)
        except Exception as error:
            raise ValueError(
                f"Invalid document at position {index} in '{path}': {error}"
            ) from error

        if document.document_id in seen_ids:
            raise ValueError(
                f"Duplicate document_id '{document.document_id}' in corpus."
            )
        documents.append(document)
        seen_ids.add(document.document_id)

    return documents


@lru_cache(maxsize=1)
def get_default_corpus() -> tuple[TechnicalDocument, ...]:
    """Load the default corpus once and cache it (lazy, on first use)."""
    return tuple(load_corpus())
