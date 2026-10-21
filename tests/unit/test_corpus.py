"""Tests for the corpus loader (local files only, no network)."""

from pathlib import Path

import pytest

from ada.retrieval.corpus import load_corpus


def test_load_corpus_from_default_path():
    documents = load_corpus()

    assert len(documents) >= 5
    ids = [document.document_id for document in documents]
    assert len(ids) == len(set(ids))
    assert all(document.source for document in documents)
    assert all(document.content for document in documents)


def test_load_corpus_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Corpus file not found"):
        load_corpus(tmp_path / "missing.json")


def test_load_corpus_invalid_json(tmp_path: Path):
    path = tmp_path / "corpus.json"
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_corpus(path)


def test_load_corpus_empty_documents_list(tmp_path: Path):
    path = tmp_path / "corpus.json"
    path.write_text('{"documents": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty 'documents'"):
        load_corpus(path)


def test_load_corpus_duplicate_ids(tmp_path: Path):
    path = tmp_path / "corpus.json"
    path.write_text(
        """
        {
          "documents": [
            {
              "document_id": "doc-1",
              "title": "T",
              "content": "C",
              "source": "s",
              "topic": "t"
            },
            {
              "document_id": "doc-1",
              "title": "T2",
              "content": "C2",
              "source": "s2",
              "topic": "t2"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate document_id"):
        load_corpus(path)


def test_load_corpus_invalid_document_contract(tmp_path: Path):
    path = tmp_path / "corpus.json"
    path.write_text(
        '{"documents": [{"document_id": "doc-1"}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid document"):
        load_corpus(path)
