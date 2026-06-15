import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from typing import List, Optional

from hayhooks import BasePipelineWrapper

from davi.document_deletion import (
    delete_all_indexed_documents,
    delete_indexed_documents_by_file_ids,
)
from davi.document_store import get_document_store, main_doc_store

logger = logging.getLogger(__name__)


class PipelineWrapper(BasePipelineWrapper):

    def setup(self) -> None:
        pass

    def run_api(
        self,
        index_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        delete_entire_index: bool = False,
    ) -> dict:
        """
        Remove indexed chunks from OpenSearch.

        - ``delete_entire_index=true``: wipe all chunks in ``index_id`` (chat rename, admin/company purge).
        - Otherwise ``file_ids`` must list full ids to remove, e.g. ``{index_id}--{filename}``.
        """
        try:
            doc_store = get_document_store(index_name=index_id) if index_id else main_doc_store

            if delete_entire_index:
                if not index_id:
                    return {"message": "Error: index_id required when delete_entire_index=true"}
                deleted_chunks = delete_all_indexed_documents(doc_store)
                return {
                    "message": f"Cleared entire index {index_id}",
                    "deleted_chunks": deleted_chunks,
                    "index_id": index_id,
                }

            if not file_ids:
                return {"message": "Error: file_ids required (or set delete_entire_index=true)"}

            deleted_chunks = delete_indexed_documents_by_file_ids(doc_store, file_ids)

            return {
                "message": f"Removed indexed documents for {len(file_ids)} file_id(s)",
                "deleted_chunks": deleted_chunks,
                "file_ids": file_ids,
            }
        except Exception as e:
            logger.exception("Failed to remove indexed documents")
            return {"message": f"Error removing indexed documents: {e}"}
