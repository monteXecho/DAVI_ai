"""
Remove indexed document chunks from the document store by ``file_id`` metadata.

Used when DAVI deletes a source/document from MongoDB but OpenSearch chunks must
also be removed to avoid duplicates on re-upload.

Supports:
- Current Haystack OpenSearch integrations (``delete_by_filter`` / ``delete_all_documents``)
- Older ``OpenSearchDocumentStore`` builds via direct ``delete_by_query`` on the client
- ``InMemoryDocumentStore`` via filter + ``delete_documents``
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def _index_name(document_store) -> Optional[str]:
    return getattr(document_store, "index", None) or getattr(document_store, "_index", None)


def _opensearch_client(document_store):
    return getattr(document_store, "_client", None) or getattr(document_store, "client", None)


def _ensure_store_ready(document_store) -> None:
    init_fn = getattr(document_store, "_ensure_initialized", None)
    if callable(init_fn):
        init_fn()


def _filters_to_opensearch_body(filters: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenSearch delete_by_query body from Haystack metadata filters."""
    try:
        from haystack_integrations.document_stores.opensearch.filters import normalize_filters

        normalized = normalize_filters(filters)
        return {"query": {"bool": {"filter": normalized}}}
    except ImportError:
        pass

    field = str(filters.get("field", "")).strip()
    operator = str(filters.get("operator", "")).strip()
    value = filters.get("value")

    if operator == "in" and field:
        meta_field = field[5:] if field.startswith("meta.") else field
        values = list(value or [])
        # Metadata may be flattened (file_id) or nested (meta.file_id) depending on store version.
        return {
            "query": {
                "bool": {
                    "should": [
                        {"terms": {meta_field: values}},
                        {"terms": {f"meta.{meta_field}": values}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        }

    if operator == "==" and field:
        meta_field = field[5:] if field.startswith("meta.") else field
        return {
            "query": {
                "bool": {
                    "should": [
                        {"term": {meta_field: value}},
                        {"term": {f"meta.{meta_field}": value}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        }

    raise ValueError(f"Unsupported metadata filter for OpenSearch fallback: {filters!r}")


def _logical_name_from_file_id(file_id: str) -> Optional[str]:
    fid = str(file_id or "").strip()
    if "--" in fid:
        return fid.split("--", 1)[1]
    return fid or None


def _build_file_id_delete_body(file_ids: List[str]) -> dict[str, Any]:
    """
    OpenSearch delete_by_query body matching ``file_id`` in all known layouts.

    Chunks may store the id on ``file_id``, nested ``meta.file_id``, or only the
    logical basename (suffix after ``--``) depending on indexer / store version.
    """
    full_ids = list(dict.fromkeys(str(fid).strip() for fid in file_ids if fid and str(fid).strip()))
    if not full_ids:
        return {"query": {"match_none": {}}}

    logical_names = list(
        dict.fromkeys(
            ln for ln in (_logical_name_from_file_id(fid) for fid in full_ids) if ln
        )
    )

    should: list[dict[str, Any]] = []
    for field in ("file_id", "meta.file_id"):
        should.append({"terms": {field: full_ids}})
        if logical_names:
            should.append({"terms": {field: logical_names}})

    for fid in full_ids:
        # Legacy / partial values (keyword wildcard; safe on small QR-Chat indexes).
        should.append({"wildcard": {"file_id": {"value": f"*{fid}"}}})
        should.append({"wildcard": {"meta.file_id": {"value": f"*{fid}"}}})
        logical = _logical_name_from_file_id(fid)
        if logical:
            should.append({"wildcard": {"file_id": {"value": f"*--{logical}"}}})
            should.append({"wildcard": {"meta.file_id": {"value": f"*--{logical}"}}})

    return {"query": {"bool": {"should": should, "minimum_should_match": 1}}}


def _delete_by_filter(document_store, filters: dict[str, Any], *, refresh: bool = True) -> int:
    if hasattr(document_store, "delete_by_filter"):
        return int(document_store.delete_by_filter(filters=filters, refresh=refresh))

    client = _opensearch_client(document_store)
    index = _index_name(document_store)
    if client is not None and index:
        _ensure_store_ready(document_store)
        body = _filters_to_opensearch_body(filters)
        result = client.delete_by_query(
            index=index,
            body=body,
            refresh=refresh,
            wait_for_completion=True,
        )
        return int(result.get("deleted", 0))

    if hasattr(document_store, "filter_documents") and hasattr(document_store, "delete_documents"):
        docs = document_store.filter_documents(filters=filters)
        doc_ids = [doc.id for doc in docs if getattr(doc, "id", None)]
        if doc_ids:
            document_store.delete_documents(document_ids=doc_ids)
        return len(doc_ids)

    raise AttributeError(
        f"{type(document_store).__name__} does not support delete_by_filter "
        "and no OpenSearch client fallback is available"
    )


def _delete_all(document_store, *, refresh: bool = True) -> int:
    deleted_chunks = 0
    try:
        if hasattr(document_store, "count_documents"):
            deleted_chunks = int(document_store.count_documents())
    except Exception:
        logger.debug("count_documents unavailable before delete_all", exc_info=True)

    if hasattr(document_store, "delete_all_documents"):
        document_store.delete_all_documents(refresh=refresh)
        return deleted_chunks

    client = _opensearch_client(document_store)
    index = _index_name(document_store)
    if client is not None and index:
        _ensure_store_ready(document_store)
        result = client.delete_by_query(
            index=index,
            body={"query": {"match_all": {}}},
            refresh=refresh,
            wait_for_completion=True,
        )
        return int(result.get("deleted", deleted_chunks))

    if hasattr(document_store, "filter_documents") and hasattr(document_store, "delete_documents"):
        docs = document_store.filter_documents(filters={})
        doc_ids = [doc.id for doc in docs if getattr(doc, "id", None)]
        if doc_ids:
            document_store.delete_documents(document_ids=doc_ids)
        return len(doc_ids)

    raise AttributeError(
        f"{type(document_store).__name__} does not support delete_all_documents "
        "and no OpenSearch client fallback is available"
    )


def delete_indexed_documents_by_file_ids(document_store, file_ids: List[str]) -> int:
    """
    Delete all chunks whose ``file_id`` metadata matches one of ``file_ids``.

    Returns the number of chunks removed.
    """
    unique_ids = list(dict.fromkeys(fid.strip() for fid in file_ids if fid and str(fid).strip()))
    if not unique_ids:
        return 0

    index_name = _index_name(document_store)
    deleted_chunks = 0

    # Primary path: Haystack metadata filter (works when fields are flattened).
    filters = {
        "field": "file_id",
        "operator": "in",
        "value": unique_ids,
    }
    try:
        deleted_chunks = _delete_by_filter(document_store, filters, refresh=True)
    except Exception:
        logger.debug("Standard file_id filter delete failed", exc_info=True)

    # Expanded OpenSearch query when nothing was removed (nested meta.file_id, legacy basenames).
    if deleted_chunks == 0:
        client = _opensearch_client(document_store)
        if client is not None and index_name:
            _ensure_store_ready(document_store)
            body = _build_file_id_delete_body(unique_ids)
            result = client.delete_by_query(
                index=index_name,
                body=body,
                refresh=True,
                wait_for_completion=True,
            )
            deleted_chunks = int(result.get("deleted", 0))

    logger.info(
        "Removed indexed chunks: index=%s file_ids=%s deleted_chunks=%s",
        index_name,
        unique_ids,
        deleted_chunks,
    )
    return deleted_chunks


def delete_all_indexed_documents(document_store) -> int:
    """
    Remove every chunk in the OpenSearch index (used on chat rename / admin / company purge).
    """
    deleted_chunks = _delete_all(document_store, refresh=True)

    index_name = _index_name(document_store)
    logger.info(
        "Cleared entire index: index=%s deleted_chunks=%s",
        index_name,
        deleted_chunks,
    )
    return deleted_chunks
