import os

from pathlib import Path

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

DOC_STORE_TYPE = os.environ.get("DOC_STORE_TYPE", "InMemory")
_VALID_DOC_STORE_TYPES = {"InMemory", "OpenSearch"}
if DOC_STORE_TYPE not in _VALID_DOC_STORE_TYPES:
    raise ValueError(f"{DOC_STORE_TYPE=} not in {_VALID_DOC_STORE_TYPES}")


INDEX_NAME = os.environ.get("INDEX_NAME", "davi-base")
INDEX_PATH = Path(os.environ.get("INDEX_PATH", f"{INDEX_NAME}.json"))


def get_document_store(index_name, index_path=None):
    if DOC_STORE_TYPE == "InMemory":
        main_doc_store = InMemoryDocumentStore(index=INDEX_NAME)
        if index_path is None:
            index_path = Path(f"{index_name}.json")
        if index_path.exists():
            main_doc_store = InMemoryDocumentStore.load_from_disk(index_path)
    elif DOC_STORE_TYPE == "OpenSearch":
        _OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost:9200")
        _OPENSEARCH_USE_SSL = os.environ.get("OPENSEARCH_USE_SSL", "TRUE") == "TRUE"
        _OPENSEARCH_VERIFY_CERTS = os.environ.get("OPENSEARCH_VERIFY_CERTS", "TRUE") == "TRUE"
        _OPENSEARCH_CA_CERTS = os.environ.get("OPENSEARCH_CA_CERTS", None)
        _OPENSEARCH_USERNAME = os.environ.get("OPENSEARCH_USERNAME", None)
        _OPENSEARCH_PASSWORD = os.environ.get("OPENSEARCH_PASSWORD", None)
        _OPENSEARCH_EMBEDDING_DIM = os.environ.get("OPENSEARCH_EMBEDDING_DIM", None)
        main_doc_store = OpenSearchDocumentStore(
            index=index_name,
            hosts=_OPENSEARCH_HOST,
            use_ssl=_OPENSEARCH_USE_SSL,
            verify_certs=_OPENSEARCH_VERIFY_CERTS,
            ca_certs=_OPENSEARCH_CA_CERTS,
            http_auth=(_OPENSEARCH_USERNAME, _OPENSEARCH_PASSWORD),
            embedding_dim=_OPENSEARCH_EMBEDDING_DIM,
        )
    return main_doc_store


main_doc_store = get_document_store(INDEX_NAME, INDEX_PATH)