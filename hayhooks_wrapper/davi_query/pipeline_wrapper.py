from typing import Dict, Any, Generator, Optional, Union, List

import logging
import os

from haystack.dataclasses import ChatMessage
from hayhooks import BasePipelineWrapper, get_last_user_message, streaming_generator

from davi.document_store import main_doc_store
from davi.llm import create_generator
from davi.components.retrieval_query_expander import expand_retrieval_query
from davi.rag import (
    compute_query_top_k,
    count_filtered_sources,
    create_rag_pipeline,
    log_rag_pipeline_startup,
)


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
logging.getLogger("davi.rag").setLevel(logging.INFO)
logger = logging.getLogger("davi_query")
logger.setLevel(logging.INFO)


class PipelineWrapper(BasePipelineWrapper):

    def setup(self) -> None:
        pipeline_name = os.environ.get("RAG_QUERY_PIPELINE_NAME", "rag_pipeline_v2d")
        self.pipeline = create_rag_pipeline(
            document_store=main_doc_store,
            generator_creation_fn=create_generator,
            pipeline_name=pipeline_name,
        )
        self.pipeline.warm_up()
        log_rag_pipeline_startup(pipeline_name, self.pipeline)

    def run_api(self, query: str, index_id: Optional[str]=None, filters: Optional[Dict[str, Any]] = None) -> tuple[Any, Any]:
        if filters is None:
            filters = {}

        if index_id:
            doc_store = self.pipeline.get_component('bm25_retriever')._document_store
            doc_store._index = index_id

        num_sources = count_filtered_sources(filters)
        top_k = compute_query_top_k(num_sources) if num_sources > 0 else None

        retriever_kwargs: Dict[str, Any] = {"filters": filters}
        if top_k is not None:
            retriever_kwargs["top_k"] = top_k

        pipeline_input = {
            "interaction_router": {"input_messages": [ChatMessage.from_user(text=query)]},
            "bm25_retriever": retriever_kwargs,
            "embedding_retriever": retriever_kwargs,
        }

        include_outputs = {"answer_builder", "token_limiter"}
        try:
            self.pipeline.get_component("retrieval_postprocessor")
            include_outputs.add("retrieval_postprocessor")
        except ValueError:
            pass
        try:
            self.pipeline.get_component("meta_field_grouping_ranker")
            include_outputs.add("meta_field_grouping_ranker")
        except ValueError:
            pass

        expanded_query = expand_retrieval_query(query)
        logger.info(
            "RAG query retrieval: sources=%s top_k=%s index_id=%s expanded_query=%s",
            num_sources,
            top_k,
            index_id,
            expanded_query if expanded_query != query else query,
        )

        result = self.pipeline.run(pipeline_input, include_outputs_from=include_outputs)

        answer = result["answer_builder"]["answers"][0]
        final_docs = answer.documents or []

        postprocessed = result.get("retrieval_postprocessor", {}).get("documents", [])
        if postprocessed:
            file_ids = sorted(
                {
                    str((doc.meta or {}).get("file_id", "unknown"))
                    for doc in postprocessed
                }
            )
            logger.info(
                "RAG query chunks: postprocessor=%s token_limiter=%s unique_files=%s file_ids=%s",
                len(postprocessed),
                len(result.get("token_limiter", {}).get("documents", [])),
                len(file_ids),
                file_ids,
            )

        return final_docs, answer

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
        chat_messages = [ChatMessage.from_openai_dict_format(message) for message in messages]

        # Consider Alternative to setting the index at each call
        doc_store = self.pipeline.get_component('bm25_retriever').document_store
        doc_store._index = body.get('index_id', 'default')

        pipeline_input = {
            "interaction_router": {"input_messages": chat_messages},
        }

        return streaming_generator(pipeline=self.pipeline, pipeline_run_args=pipeline_input)