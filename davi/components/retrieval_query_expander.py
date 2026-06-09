"""
Expand user queries with Dutch retrieval terms before BM25 / dense search.

Users often ask in English while source documents and filenames are Dutch.
"""

import unicodedata

from haystack import component

# (English/multi phrases in query) -> extra Dutch terms for retrieval only
_RETRIEVAL_PHRASE_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("school time", "school hours", "school schedule", "weekly schedule", "hours per week"),
        ("schooltijden", "schooltijd", "lesuren", "rooster"),
    ),
    (
        ("vacation", "holiday", "school holiday"),
        ("vakantie", "vakantierooster"),
    ),
    (
        ("after school", "childcare", "daycare"),
        ("naschoolse opvang", "bso", "tussenschoolse opvang"),
    ),
    (
        ("absence", "sick", "illness"),
        ("ziek", "ziekmelding", "verzuim"),
    ),
    (
        ("enrollment", "admission"),
        ("aanname", "aanmelding", "inschrijving"),
    ),
)


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(ch) != "Mn"
    )


def expand_retrieval_query(query: str) -> str:
    if not query or not query.strip():
        return query

    normalized_query = _normalize_text(query)
    extra_terms: list[str] = []
    for phrases, dutch_terms in _RETRIEVAL_PHRASE_EXPANSIONS:
        if any(_normalize_text(phrase) in normalized_query for phrase in phrases):
            extra_terms.extend(dutch_terms)

    if not extra_terms:
        return query

    deduped = list(dict.fromkeys(extra_terms))
    return f"{query} {' '.join(deduped)}"


@component
class RetrievalQueryExpander:
    """Append Dutch retrieval terms; the original query still goes to the LLM prompt."""

    @component.output_types(expanded_query=str)
    def run(self, query: str) -> dict[str, str]:
        expanded = expand_retrieval_query(query)
        return {"expanded_query": expanded}
