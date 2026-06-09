import argparse
import json
import os

import pandas as pd

from pathlib import Path

from loguru import logger

from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.joiners import BranchJoiner
from haystack.components.converters import OutputAdapter
from haystack.dataclasses import ChatMessage
from haystack.components.routers import ConditionalRouter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchBM25Retriever,
)
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchEmbeddingRetriever,
)
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import (
    MetaFieldGroupingRanker,
    SentenceTransformersSimilarityRanker,
)

import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.DEBUG)

from davi.components.token_handler import TokenLimiter
from davi.components.retrieval_postprocessor import RetrievalPostprocessor
from davi.components.retrieval_query_expander import RetrievalQueryExpander
from davi.document_store import main_doc_store, DOC_STORE_TYPE
from davi.llm import create_generator
from davi.prompts import QA_PROMPT, CHAT_SUMMARY_PROMPT


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _retrieval_postprocessor_from_env() -> RetrievalPostprocessor:
    return RetrievalPostprocessor(
        max_chunks_per_file_id=_env_int("RAG_MAX_CHUNKS_PER_FILE", 3),
        max_total_chunks=_env_int("RAG_MAX_TOTAL_CHUNKS", 20),
    )


def count_filtered_sources(filters: dict | None) -> int:
    """Count how many source file_ids are in a Haystack/OpenSearch filter."""
    if not filters:
        return 0
    if filters.get("field") != "file_id" or filters.get("operator") != "in":
        return 0
    value = filters.get("value")
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str):
        return len([part for part in value.split(",") if part.strip()])
    return 0


def compute_query_top_k(num_filtered_sources: int) -> int:
    """
    Scale BM25 + dense top_k with the number of sources in the query filter.

    With one source, base_top_k is enough. With many sources, retrieval must dig
    deeper globally before the per-file cap in RetrievalPostprocessor.
    """
    base_top_k = _env_int("RAG_BASE_TOP_K", 30)
    per_source = _env_int("RAG_TOP_K_PER_SOURCE", 3)
    max_top_k = _env_int("RAG_MAX_TOP_K", 100)
    if num_filtered_sources <= 1:
        return base_top_k
    return min(max(base_top_k, num_filtered_sources * per_source), max_top_k)


def _rag_pipeline_v1(document_store, generator_creation_fn):
    if DOC_STORE_TYPE == "InMemory":
        bm25_retriever = InMemoryBM25Retriever(document_store=document_store, top_k=8)
    else:
        bm25_retriever = OpenSearchBM25Retriever(document_store=document_store, top_k=8)

    text_embedder = SentenceTransformersTextEmbedder(
        model="paraphrase-multilingual-mpnet-base-v2"
    )

    if DOC_STORE_TYPE == "InMemory":
        embedding_retriever = InMemoryEmbeddingRetriever(
            document_store=document_store, top_k=5
        )
    else:
        embedding_retriever = OpenSearchEmbeddingRetriever(
            document_store=document_store, top_k=5
        )

    document_joiner = DocumentJoiner(join_mode="merge")

    reranker = SentenceTransformersSimilarityRanker(
        model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", top_k=5
    )

    template = (
        "Gebruik de informatie in de context om een uitgebreid antwoord te geven op de vraag. "
        "Als het antwoord niet uit de context kan worden afgeleid, geef dan geen antwoord. "
        "Context: {{ documents|map(attribute='content')|join(';')|replace('\n', ' ') }} "
        "Vraag: {{ query }} "
        "Antwoord:"
    )
    prompt_builder = PromptBuilder(
        template,
        required_variables=['documents', 'query']
    )
    generator = generator_creation_fn()
    answer_builder = AnswerBuilder()

    query_pipeline = Pipeline()

    # Add components
    query_pipeline.add_component("bm25_retriever", bm25_retriever)
    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component("embedding_retriever", embedding_retriever)
    query_pipeline.add_component("document_joiner", document_joiner)
    query_pipeline.add_component("reranker", reranker)
    query_pipeline.add_component("prompt_builder", prompt_builder)
    query_pipeline.add_component("llm", generator)
    query_pipeline.add_component("answer_builder", answer_builder)

    # Connect components
    query_pipeline.connect(
        "text_embedder.embedding", "embedding_retriever.query_embedding"
    )
    query_pipeline.connect("bm25_retriever.documents", "document_joiner.documents")
    query_pipeline.connect("embedding_retriever.documents", "document_joiner.documents")
    query_pipeline.connect("document_joiner.documents", "reranker.documents")
    query_pipeline.connect("reranker.documents", "prompt_builder.documents")
    query_pipeline.connect("reranker.documents", "answer_builder.documents")
    query_pipeline.connect("prompt_builder.prompt", "llm.prompt")
    query_pipeline.connect("llm.replies", "answer_builder.replies")

    return query_pipeline


def _rag_pipeline_v2(
    document_store,
    generator_creation_fn,
    embedder_model,
    reranker_model=None,
    base_top_k=8,
    retrieval_postprocessor: RetrievalPostprocessor | None = None,
    use_grouping_ranker: bool = True,
):
    if DOC_STORE_TYPE == "InMemory":
        bm25_retriever = InMemoryBM25Retriever(document_store=document_store, top_k=base_top_k)
    else:
        bm25_retriever = OpenSearchBM25Retriever(document_store=document_store, top_k=base_top_k)

    text_embedder = SentenceTransformersTextEmbedder(
        model=embedder_model, trust_remote_code=True,
        token=Secret.from_env_var("HF_TOKEN"),
    )

    if DOC_STORE_TYPE == "InMemory":
        embedding_retriever = InMemoryEmbeddingRetriever(
            document_store=document_store, top_k=base_top_k
        )
    else:
        embedding_retriever = OpenSearchEmbeddingRetriever(
            document_store=document_store, top_k=base_top_k
        )

    document_joiner = DocumentJoiner(join_mode="merge")

    prompt_builder = PromptBuilder(
        QA_PROMPT,
        required_variables=['documents', 'query']
    )
    chat_summary_generator = generator_creation_fn()
    generator = generator_creation_fn()
    answer_builder = AnswerBuilder()

    query_pipeline = Pipeline()

    query_pipeline.add_component(
        "interaction_router",
        ConditionalRouter(
            routes=[
                {
                    "condition": '{{ input_messages | length > 1}}',
                    "output": "{{ input_messages }}",
                    "output_name": "multi_turn_messages",
                    "output_type": list[ChatMessage]
                },
                {
                    "condition": '{{ True }}',
                    "output": "{{ input_messages[0].text }}",
                    "output_name": "first_question",
                    "output_type": str
                },
            ]
        )
    )

    query_pipeline.add_component(
        "chat_summary_prompt_builder",
        PromptBuilder(
            template=CHAT_SUMMARY_PROMPT,
            required_variables=["messages"]
        )
    )
    query_pipeline.add_component(
        "chat_summary_llm",
        chat_summary_generator
    )
    query_pipeline.add_component(
        "chat_summary_query",
        OutputAdapter(template="{{replies[0]}}", output_type=str)
    )
    query_pipeline.add_component(
        "retrieval_query",
        BranchJoiner(str)
    )

    query_pipeline.add_component("bm25_retriever", bm25_retriever)
    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component("embedding_retriever", embedding_retriever)
    query_pipeline.add_component("document_joiner", document_joiner)
    if reranker_model is not None:
        reranker = SentenceTransformersSimilarityRanker(
            model="BAAI/bge-reranker-v2-m3", top_k=8
        )
        query_pipeline.add_component("reranker", reranker)
    if retrieval_postprocessor is not None:
        query_pipeline.add_component("retrieval_query_expander", RetrievalQueryExpander())
        query_pipeline.add_component(
            "retrieval_postprocessor", retrieval_postprocessor
        )
    query_pipeline.add_component(
        "token_limiter", TokenLimiter(max_token_limit=11_400, model_name="Qwen/Qwen3-4B")
    )
    if use_grouping_ranker:
        query_pipeline.add_component(
            'meta_field_grouping_ranker',
            MetaFieldGroupingRanker(
                group_by="file_id",
                subgroup_by=None,
                sort_docs_by="split_id",
            )
        )
    query_pipeline.add_component("prompt_builder", prompt_builder)
    query_pipeline.add_component("llm", generator)
    query_pipeline.add_component("answer_builder", answer_builder)

    # Connect components
    query_pipeline.connect(
        "interaction_router.first_question", "retrieval_query"
    )
    query_pipeline.connect(
        "interaction_router.multi_turn_messages", "chat_summary_prompt_builder.messages"
    )
    query_pipeline.connect(
        "chat_summary_prompt_builder.prompt", "chat_summary_llm"
    )
    query_pipeline.connect(
        "chat_summary_llm.replies", "chat_summary_query.replies"
    )
    query_pipeline.connect(
        "chat_summary_query.output", "retrieval_query"
    )
    if retrieval_postprocessor is not None:
        query_pipeline.connect("retrieval_query", "retrieval_query_expander.query")
        query_pipeline.connect(
            "retrieval_query_expander.expanded_query", "text_embedder.text"
        )
        query_pipeline.connect(
            "retrieval_query_expander.expanded_query", "bm25_retriever.query"
        )
        query_pipeline.connect("retrieval_query", "prompt_builder.query")
        query_pipeline.connect("retrieval_query", "answer_builder.query")
        query_pipeline.connect("retrieval_query", "retrieval_postprocessor.query")
    else:
        query_pipeline.connect("retrieval_query", "text_embedder.text")
        query_pipeline.connect("retrieval_query", "bm25_retriever.query")
        query_pipeline.connect("retrieval_query", "prompt_builder.query")
        query_pipeline.connect("retrieval_query", "answer_builder.query")
    query_pipeline.connect(
        "text_embedder.embedding", "embedding_retriever.query_embedding"
    )
    query_pipeline.connect("bm25_retriever.documents", "document_joiner.documents")
    query_pipeline.connect("embedding_retriever.documents", "document_joiner.documents")
    if reranker_model is not None:
        if retrieval_postprocessor is not None:
            query_pipeline.connect(
                "retrieval_query_expander.expanded_query", "reranker.query"
            )
        else:
            query_pipeline.connect("retrieval_query", "reranker.query")
        query_pipeline.connect("document_joiner.documents", "reranker.documents")
        if retrieval_postprocessor is not None:
            query_pipeline.connect(
                "reranker.documents", "retrieval_postprocessor.documents"
            )
            query_pipeline.connect(
                "retrieval_postprocessor.documents", "token_limiter.documents"
            )
        else:
            query_pipeline.connect("reranker.documents", "token_limiter.documents")
    else:
        if retrieval_postprocessor is not None:
            query_pipeline.connect(
                "document_joiner.documents", "retrieval_postprocessor.documents"
            )
            query_pipeline.connect(
                "retrieval_postprocessor.documents", "token_limiter.documents"
            )
        else:
            query_pipeline.connect(
                "document_joiner.documents", "token_limiter.documents"
            )
    if use_grouping_ranker:
        query_pipeline.connect(
            "token_limiter.documents", "meta_field_grouping_ranker.documents"
        )
        query_pipeline.connect(
            "meta_field_grouping_ranker.documents", "prompt_builder.documents"
        )
        query_pipeline.connect(
            "meta_field_grouping_ranker.documents", "answer_builder.documents"
        )
    else:
        query_pipeline.connect("token_limiter.documents", "prompt_builder.documents")
        query_pipeline.connect("token_limiter.documents", "answer_builder.documents")
    query_pipeline.connect("prompt_builder.prompt", "llm.prompt")
    query_pipeline.connect("llm.replies", "answer_builder.replies")

    return query_pipeline


def create_rag_pipeline(document_store, generator_creation_fn, pipeline_name):
    if pipeline_name == "rag_pipeline_v1":
        return _rag_pipeline_v1(
            document_store=document_store, generator_creation_fn=generator_creation_fn
        )
    if pipeline_name == "rag_pipeline_v2":
        return _rag_pipeline_v2(
            document_store=document_store, generator_creation_fn=generator_creation_fn,
            embedder_model="BAAI/bge-m3",
            base_top_k=8,
            reranker_model="BAAI/bge-reranker-v2-m3",
        )
    if pipeline_name == "rag_pipeline_v2b":
        return _rag_pipeline_v2(
            document_store=document_store, generator_creation_fn=generator_creation_fn,
            embedder_model="intfloat/multilingual-e5-large",
            base_top_k=4,
            reranker_model=None,
        )
    if pipeline_name == "rag_pipeline_v2c":
        return _rag_pipeline_v2(
            document_store=document_store, generator_creation_fn=generator_creation_fn,
            # embedder_model="Alibaba-NLP/gte-multilingual-base",
            embedder_model="google/embeddinggemma-300m",
            base_top_k=5,
            reranker_model=None,
        )
    if pipeline_name == "rag_pipeline_v2d":
        return _rag_pipeline_v2(
            document_store=document_store,
            generator_creation_fn=generator_creation_fn,
            embedder_model="google/embeddinggemma-300m",
            base_top_k=_env_int("RAG_BASE_TOP_K", 30),
            reranker_model=None,
            retrieval_postprocessor=_retrieval_postprocessor_from_env(),
            use_grouping_ranker=False,
        )
    raise ValueError(f"Unknown rag pipeline {pipeline_name}")


def _env_source(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return f"{default} (default)"
    return raw


def _component_init_param(component, name: str, default: str = "n/a") -> str:
    """Read a Haystack component init value across versions (attr vs init_parameters)."""
    value = getattr(component, name, None)
    if value is not None:
        return str(value)
    init_params = getattr(component, "init_parameters", None)
    if isinstance(init_params, dict) and name in init_params:
        return str(init_params[name])
    if hasattr(component, "to_dict"):
        try:
            serialized = component.to_dict()
            init_params = serialized.get("init_parameters", {})
            if name in init_params:
                return str(init_params[name])
        except Exception:
            pass
    return default


def log_rag_pipeline_startup(pipeline_name: str, pipeline: Pipeline) -> None:
    """Log the active RAG query pipeline and effective settings at service startup."""
    try:
        lines = [
            "DAVI RAG query service — pipeline configuration",
            f"  pipeline: {pipeline_name}",
            f"  RAG_QUERY_PIPELINE_NAME: {_env_source('RAG_QUERY_PIPELINE_NAME', 'rag_pipeline_v2d')}",
        ]

        try:
            bm25 = pipeline.get_component("bm25_retriever")
            lines.append(f"  BM25 top_k: {_component_init_param(bm25, 'top_k')}")
        except ValueError:
            lines.append("  BM25 top_k: n/a")

        try:
            dense = pipeline.get_component("embedding_retriever")
            lines.append(f"  Dense top_k: {_component_init_param(dense, 'top_k')}")
        except ValueError:
            lines.append("  Dense top_k: n/a")

        try:
            embedder = pipeline.get_component("text_embedder")
            lines.append(f"  Embedder: {_component_init_param(embedder, 'model')}")
        except ValueError:
            lines.append("  Embedder: n/a")

        try:
            pipeline.get_component("reranker")
            lines.append("  Reranker: enabled")
        except ValueError:
            lines.append("  Reranker: disabled")

        try:
            postprocessor = pipeline.get_component("retrieval_postprocessor")
            lines.append(
                "  RetrievalPostprocessor: "
                f"max {postprocessor.max_chunks_per_file_id} chunks/file_id, "
                f"max {postprocessor.max_total_chunks} chunks total"
            )
            lines.append(
                f"  RAG_MAX_CHUNKS_PER_FILE: {_env_source('RAG_MAX_CHUNKS_PER_FILE', '3')}"
            )
            lines.append(
                f"  RAG_MAX_TOTAL_CHUNKS: {_env_source('RAG_MAX_TOTAL_CHUNKS', '20')}"
            )
        except ValueError:
            lines.append("  RetrievalPostprocessor: not used")

        if pipeline_name == "rag_pipeline_v2d":
            lines.append(f"  RAG_BASE_TOP_K: {_env_source('RAG_BASE_TOP_K', '30')}")
            lines.append(
                f"  RAG_TOP_K_PER_SOURCE: {_env_source('RAG_TOP_K_PER_SOURCE', '3')}"
            )
            lines.append(f"  RAG_MAX_TOP_K: {_env_source('RAG_MAX_TOP_K', '100')}")
            lines.append("  Grouping ranker: disabled (score order preserved)")
            lines.append(
                f"  RAG_FILENAME_BOOST_WEIGHT: {_env_source('RAG_FILENAME_BOOST_WEIGHT', '2.0')}"
            )
            lines.append("  Retrieval query expander: enabled (Dutch retrieval terms)")

        logger.info("\n".join(lines))
    except Exception as exc:
        logger.warning(f"Could not log full RAG pipeline config: {exc}")
        logger.info(f"DAVI RAG query service — pipeline: {pipeline_name}")


def create_and_run_pipeline(queries, pipeline_name):
    pipeline = create_rag_pipeline(
        document_store=main_doc_store,
        generator_creation_fn=create_generator,
        pipeline_name=pipeline_name,
    )
    answers = []
    for query in queries:
        if 'v2' in pipeline_name:
            chat_messages = [ChatMessage.from_openai_dict_format({'role': 'user', 'content': query})]

            # Consider Alternative to setting the index at each call
            pipeline_input = {
                "interaction_router": {"input_messages": chat_messages}
            }
            result = pipeline.run(
                pipeline_input,
                include_outputs_from={'llm', 'answer_builder', 'prompt_builder'},
            )
        else:
            pipeline_input = {
                "bm25_retriever": {"query": query},
                "text_embedder": {"text": query},
                "prompt_builder": {"query": query},
                "answer_builder": {"query": query},
            }

            try:
                pipeline.get_component("reranker")
                pipeline_input['reranker'] = {
                    "query": query
                }
            except ValueError:
                ...
            result = pipeline.run(
                pipeline_input,
                include_outputs_from={'sw_retriever', 'llm', 'answer_builder', 'prompt_builder'},
            )

        # Get answer
        answer = result["answer_builder"]["answers"][0]
        logger.info(f'Query: {query}')
        logger.info(f'num-docs: {len(answer.documents)}')
        logger.info(f'Answer: {answer.data}')
        answers.append(answer)
    return answers


def main(args):
    logger.info(args)
    if ".csv" in args.query_input:
        queries = pd.read_csv(args.query_input)[args.query_csv_column]
    else:
        queries = [args.query_input]
    answers = create_and_run_pipeline(queries, pipeline_name=args.pipeline_name)
    output = []
    for query, answer in zip(queries, answers):
        logger.info(f"Query: {query}")
        logger.info(f"Answer: {answer.data}")
        for i, doc in enumerate(answer.documents):
            logger.info(f"[{i}] {doc.meta}")
        docs = [doc.to_dict(flatten=False) for doc in answer.documents]
        for doc in docs:
            if "embedding" in doc:
                del doc["embedding"]
            if "score" in doc and doc['score'] is not None:
                if not isinstance(doc['score'], float):
                    doc["score"] = doc["score"].item()

        output.append({"query": query, "answer_data": answer.data, "answer_docs": docs})
    if args.output_json_path is not None:
        with open(Path(args.output_json_path), "w") as f:
            json.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and run a RAG pipeline")
    parser.add_argument(
        "-q",
        "--query_input",
        required=True,
        help="Path to a csv with queries. Or a single query string.",
    )
    parser.add_argument(
        "-c",
        "--query_csv_column",
        required=False,
        default="Question",
        help="Column in the csv with the queries",
    )
    parser.add_argument(
        "-o", "--output_json_path", required=False, help="Path for output json"
    )
    parser.add_argument(
        "-p",
        "--pipeline_name",
        choices=[
            "rag_pipeline_v1",
            "rag_pipeline_v2",
            "rag_pipeline_v2b",
            "rag_pipeline_v2c",
            "rag_pipeline_v2d",
        ],
        default="rag_pipeline_v1",
        help="Name of the pipeline to run (default: rag_pipeline_v1)",
    )

    args = parser.parse_args()
    main(args)
