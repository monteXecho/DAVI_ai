from haystack import component, Document
from transformers import AutoTokenizer


@component
class TokenLimiter:
    def __init__(self, max_token_limit: int, model_name: str):
        self.max_token_limit = max_token_limit
        self.model_name = model_name
        self.tokenizer = None

    def warm_up(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if self.tokenizer is None:
            self.warm_up()
        total_tokens = 0
        chosen_documents: list[Document] = []
        for doc in documents:
            token_ids = self.tokenizer.encode(doc.content, add_special_tokens=False)
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            # If we go over the token limit when adding the doc, skip it
            if (total_tokens + len(tokens)) >= self.max_token_limit:
                continue
            chosen_documents.append(doc)
            total_tokens += len(tokens)
        return {'documents': chosen_documents}


if __name__ == '__main__':
    limiter = TokenLimiter(max_token_limit=100, model_name="Qwen/Qwen3-4B")
    docs = [Document(content=("hello " * 50)), Document(content=("polymerase " * 50)), Document(content=("world " * 2))]
    chosen_docs = limiter.run(docs)['documents']
    for doc in chosen_docs:
        print(doc.content.split()[0])