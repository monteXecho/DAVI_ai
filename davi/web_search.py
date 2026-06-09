import typing

from haystack import Document
from haystack import Pipeline
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.components.builders.prompt_builder import PromptBuilder
# from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.utils import Secret

from davi.components.serper_dev import CustomSerperDevWebSearch
from davi.components.parallel_executor import DeepsetParallelExecutor
from davi.llm import create_generator
from davi.prompts_web_search import QUERY_GENERATION_PROMPT, WEB_QA_PROMPT


def create_web_search_pipeline(generator_creation_fn):
    multi_queries = OutputAdapter(
        template="{{ replies[0].split(';') }}",
        output_type=typing.List[str],
        unsafe=True,
    )
    query_variation_generator = generator_creation_fn()
    query_variation_prompt_builder = PromptBuilder(
        template=QUERY_GENERATION_PROMPT,
        required_variables=["question"],
    )
    all_result_docs = OutputAdapter(
        template="{% set flat = doc_lists | sum(start=[]) %}{{ flat }}",
        output_type=typing.Optional[typing.List[Document]],
        unsafe=True,
    )
    serper_searcher = DeepsetParallelExecutor(
        component=CustomSerperDevWebSearch(
            api_key=Secret.from_env_var("SERPERDEV_API_KEY"),
            top_k=10,
            allowed_domains=[
                "www.rijksoverheid.nl",
                "kinderopvang.nl",
                "ondernemersplein.overheid.nl",
                "wetten.overheid.nl",
                "www.kinderopvang-werkt.nl",
                "business.gov.nl",
                "ggdghor.nl",
                "lokaleregelgeving.overheid.nl",
                "www.belastingdienst.nl",
                "www.boink.info",
                "www.fnv.nl",
                "www.gezondekinderopvang.nl",
                "www.ggd.amsterdam.nl",
                "www.ggdrotterdamrijnmond.nl",
                "www.maatschappelijkekinderopvang.nl",
                "www.nji.nl",
            ],
            search_params=None,
        ),
        max_workers=4,
        max_retries=3,
        progress_bar=False,
        raise_on_failure=True,
        flatten_output=False,
    )
    final_prompt_builder = PromptBuilder(
        template=WEB_QA_PROMPT,
        required_variables=["docs", "question"],
    )
    final_llm = generator_creation_fn()
    answer_builder = AnswerBuilder(last_message_only=False)

    pipeline = Pipeline()
    pipeline.add_component("multi_queries", multi_queries)
    pipeline.add_component(
        "query_variation_generator", query_variation_generator
    )
    pipeline.add_component("query_variation_prompt_builder", query_variation_prompt_builder)
    pipeline.add_component("all_result_docs", all_result_docs)
    pipeline.add_component("serper_searcher", serper_searcher)
    pipeline.add_component("final_prompt_builder", final_prompt_builder)
    pipeline.add_component("final_llm", final_llm)
    pipeline.add_component("answer_builder", answer_builder)
    pipeline.connect(
        "query_variation_prompt_builder.prompt", "query_variation_generator.prompt"
    )
    pipeline.connect(
        "query_variation_generator.replies", "multi_queries.replies"
    )
    pipeline.connect("multi_queries.output", "serper_searcher.query")
    pipeline.connect("serper_searcher.documents", "all_result_docs.doc_lists")
    pipeline.connect("all_result_docs.output", "final_prompt_builder.docs")
    pipeline.connect("final_prompt_builder.prompt", "final_llm.prompt")
    pipeline.connect("final_llm.replies", "answer_builder.replies")
    pipeline.connect("all_result_docs.output", "answer_builder.documents")

    return pipeline


def main():
    import sys
    q = sys.argv[1]

    pipeline = create_web_search_pipeline(create_generator)
    pipeline.warm_up()
    result = pipeline.run(
        data={
            "query_variation_prompt_builder": {"question": q},
            "final_prompt_builder": {"question": q},
            "answer_builder": {"query": q},
        },
        include_outputs_from={"all_result_docs", "answer_builder", "final_llm"}
    )
    print('## Docs')
    for d in result["all_result_docs"]["output"]:
        print(d.content)
        print(d.meta["link"])
    print('---' * 4)
    print('## Answer')
    print(result["answer_builder"]["answers"][0].data)
    print('---' * 4)
    print('## LLM Call Notes')
    print(result["final_llm"]["meta"])


if __name__ == '__main__':
    main()