"""
Post-retrieval filtering for multi-document RAG.

Sorts merged chunks by relevance score, caps how many chunks per source file,
and limits how many chunks proceed to the token limiter / LLM.
"""

import os
import re
import unicodedata
from typing import Optional

from haystack import Document, component

from davi.components.retrieval_query_expander import expand_retrieval_query


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(ch) != "Mn"
    )


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_text(text))
        if len(token) >= 3
    }


def _document_score(doc: Document) -> float:
    """Best-effort relevance score for ordering (higher = more relevant)."""
    if doc.score is not None:
        try:
            return float(doc.score)
        except (TypeError, ValueError):
            pass
    meta = doc.meta or {}
    raw = meta.get("score")
    if raw is None:
        return float("-inf")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("-inf")


def _file_id_key(doc: Document) -> str:
    meta = doc.meta or {}
    fid = meta.get("file_id")
    if fid:
        return str(fid)
    path = meta.get("file_path") or meta.get("file_name") or meta.get("original_file_path")
    if path:
        return str(path)
    return str(doc.id or "unknown")


def _source_label(doc: Document) -> str:
    meta = doc.meta or {}
    file_id = _file_id_key(doc)
    basename = file_id.split("--", 1)[-1] if "--" in file_id else file_id
    title = str(meta.get("source_title") or meta.get("file_name") or "")
    return _normalize_text(f"{basename} {title}")


def _filename_boost(query: str, doc: Document, boost_weight: float) -> float:
    if not query or boost_weight <= 0:
        return 0.0

    expanded_query = expand_retrieval_query(query)
    query_terms = _tokenize(expanded_query)
    if not query_terms:
        return 0.0

    source_text = _source_label(doc)
    matches = sum(1 for term in query_terms if term in source_text)
    return matches * boost_weight


@component
class RetrievalPostprocessor:
    """
    After BM25 + dense merge (and optional reranker):

    1. Re-rank by retrieval score plus optional filename/title boost from the query.
    2. Keep at most ``max_chunks_per_file_id`` chunks per ``file_id``.
    3. Keep at most ``max_total_chunks`` documents overall.
    """

    def __init__(
        self,
        max_chunks_per_file_id: int = 3,
        max_total_chunks: Optional[int] = 20,
        filename_boost_weight: float = 2.0,
    ):
        if max_chunks_per_file_id < 1:
            raise ValueError("max_chunks_per_file_id must be >= 1")
        if max_total_chunks is not None and max_total_chunks < 1:
            raise ValueError("max_total_chunks must be >= 1 when set")
        self.max_chunks_per_file_id = max_chunks_per_file_id
        self.max_total_chunks = max_total_chunks
        self.filename_boost_weight = filename_boost_weight

    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        query: str = "",
    ) -> dict[str, list[Document]]:
        if not documents:
            return {"documents": []}

        boost_weight = self.filename_boost_weight
        env_boost = os.environ.get("RAG_FILENAME_BOOST_WEIGHT")
        if env_boost is not None and str(env_boost).strip() != "":
            boost_weight = float(env_boost)

        def combined_score(doc: Document) -> float:
            return _document_score(doc) + _filename_boost(query, doc, boost_weight)

        sorted_docs = sorted(documents, key=combined_score, reverse=True)

        per_file_counts: dict[str, int] = {}
        chosen: list[Document] = []

        for doc in sorted_docs:
            if self.max_total_chunks is not None and len(chosen) >= self.max_total_chunks:
                break

            key = _file_id_key(doc)
            count = per_file_counts.get(key, 0)
            if count >= self.max_chunks_per_file_id:
                continue

            chosen.append(doc)
            per_file_counts[key] = count + 1

        return {"documents": chosen}
