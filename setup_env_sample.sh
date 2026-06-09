export HAYHOOKS_PIPELINES_DIR="$(pwd)/hayhooks_wrapper"
export HAYHOOKS_ADDITIONAL_PYTHONPATH="$(pwd)"
export PYTHONPATH="$(pwd):$PYTHONPATH"
# as we switch index per request, can set any default
# say: "davi-base"
export INDEX_NAME="<fill>" 
export DOC_STORE_TYPE=OpenSearch
export OPENSEARCH_HOST="<fill>" # Likely: "https://werkh-2.vm.bit.nl:9200"
export OPENSEARCH_USE_SSL=TRUE
export OPENSEARCH_CA_CERTS=<fill> # Likely: /path/to/http-ca.pem
# See: Opensearch-Setup
export OPENSEARCH_USERNAME=<fill> # See: Opensearch-Setup or ask admin
export OPENSEARCH_PASSWORD=<fill> # See: Opensearch-Setup or ask admin
export OPENSEARCH_EMBEDDING_DIM=768
export OPENAI_API_KEY="..." # Empty is fine
export LLM_BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export LLM_BASE_URL="http://0.0.0.0:4400"
export SERPERDEV_API_KEY="<fill>" # Ask admin
export HF_TOKEN="<fill>" # Ask admin

# Query pipeline: rag_pipeline_v2c (legacy) | rag_pipeline_v2d (multi-source recall)
# On Hayhooks start, davi_query logs the active pipeline and settings (see log_rag_pipeline_startup).
export RAG_QUERY_PIPELINE_NAME="rag_pipeline_v2d"
# BM25 + dense retrieval depth (OpenSearch; not GPU VRAM)
export RAG_BASE_TOP_K=30
# Scale top_k when many sources: min(max(BASE, N*PER_SOURCE), MAX)
export RAG_TOP_K_PER_SOURCE=3
export RAG_MAX_TOP_K=100
# After merge: max chunks per file_id, then max chunks to token limiter / LLM
export RAG_MAX_CHUNKS_PER_FILE=3
export RAG_MAX_TOTAL_CHUNKS=20
# Boost chunks whose filename/title matches expanded query terms (e.g. schooltijden)
export RAG_FILENAME_BOOST_WEIGHT=2.0