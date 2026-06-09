import os

from haystack.components.generators.openai import OpenAIGenerator 
from haystack.components.generators.chat.openai import OpenAIChatGenerator


LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost/v1")
LLM_BASE_MODEL = os.environ.get("LLM_BASE_MODEL", "granite3.2:8b-instruct-fp16")


def create_generator():
    return OpenAIGenerator(
        model=LLM_BASE_MODEL,
        api_base_url=LLM_BASE_URL,
        timeout=200,
    )


def create_chat_generator():
    return OpenAIChatGenerator(
        model=LLM_BASE_MODEL,
        api_base_url=LLM_BASE_URL,
        timeout=200,
    )