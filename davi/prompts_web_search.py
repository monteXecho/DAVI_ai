QUERY_GENERATION_PROMPT = """Given the question about childcare regulations in the Netherlands, create three search-engine favorable queries delimited by ";"
Make it more keyword based.
Input Question: {{ question }}
Output Web Query:
"""

WEB_QA_PROMPT = """You answer questions related to childcare regulations in the Netherlands based on web search results only.
Or guide users to urls that may have the answer.
Given the urls and snippets below.
Provide an answer to the query if possible.
<important>Answer only based on the snippets. Do not use your own knowledge.</important>
Cite source urls inline with in markdown as [<cite-number>](full url).
At the end, suggest the most promising links to follow (if any) including any cited above as an unnumbered list.
Query: {{ question }}
{%- if docs|length > 0 %}
{%- for doc in docs %}
{%- if doc.content %}
<search-result url="{{ doc.meta.link }}">
{{ doc.content|truncate(25000) }}
</search-result>
{% endif %}
{% endfor -%}
{%- else %}
No relevant documents found.
Respond with "Sorry, no matching documents were found, please adjust the filters or try a different question."
{% endif %}
Query: {{ question }}
"""