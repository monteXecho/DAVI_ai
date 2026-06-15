"""
Remove indexed document chunks from the document store by ``file_id`` metadata.

Used when DAVI deletes a source/document from MongoDB but OpenSearch chunks must
also be removed to avoid duplicates on re-upload.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def delete_indexed_documents_by_file_ids(document_store, file_ids: List[str]) -> int:
    """
    Delete all chunks whose ``file_id`` metadata matches one of ``file_ids``.

    Returns the number of chunks removed.
    """
    unique_ids = list(dict.fromkeys(fid.strip() for fid in file_ids if fid and str(fid).strip()))
    if not unique_ids:
        return 0

    filters = {
        "field": "file_id",
        "operator": "in",
        "value": unique_ids,
    }

    deleted_chunks = document_store.delete_by_filter(filters=filters, refresh=True)

    index_name = getattr(document_store, "index", None)
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
    deleted_chunks = 0
    try:
        deleted_chunks = document_store.count_documents()
    except Exception:
        logger.debug("count_documents unavailable before delete_all")

    document_store.delete_all_documents(refresh=True)

    index_name = getattr(document_store, "index", None)
    logger.info(
        "Cleared entire index: index=%s deleted_chunks=%s",
        index_name,
        deleted_chunks,
    )
    return deleted_chunks
