from typing import Dict, Any, Generator, Optional, Union, List

import logging

from hayhooks import BasePipelineWrapper, get_last_user_message, streaming_generator

from davi.document_store import main_doc_store
from davi.llm import create_generator
from davi.web_search import create_web_search_pipeline


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


class PipelineWrapper(BasePipelineWrapper):

    def setup(self) -> None:
        self.pipeline = create_web_search_pipeline(
            generator_creation_fn=create_generator,
        )
        self.pipeline.warm_up()

    def run_api(self, query: str, sites: Optional[List[str]]=None) -> tuple[Any, Any]:

        pipeline_input = {
            "query_variation_prompt_builder": {"question": query},
            "final_prompt_builder": {"question": query},
            "answer_builder": {"query": query},
        }
        if sites is not None:
            pipeline_input["serper_searcher"] = {"allowed_domains": [sites, sites, sites]}

        result = self.pipeline.run(
            pipeline_input, include_outputs_from={"all_result_docs", "answer_builder"}
        )

        return result["all_result_docs"]["output"][0], result["answer_builder"]["answers"][0]

    def run_chat_completion(self, model: str, messages: List[dict], body: dict) -> Union[str, Generator]:
        query = get_last_user_message(messages)

        pipeline_input = {
            "query_variation_prompt_builder": {"question": query},
            "final_prompt_builder": {"question": query},
            "answer_builder": {"query": query},
        }

        return streaming_generator(pipeline=self.pipeline, pipeline_run_args=pipeline_input)
