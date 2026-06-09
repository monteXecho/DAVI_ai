# %%
import dotenv

dotenv.load_dotenv('../.env')

import pandas as pd
QUERIES_FILE = "../module-1-eval-set-20.csv"

df = pd.read_csv(QUERIES_FILE)
df.shape
# %%
df.head()
# %%
import json

PIPELINE_OUTPUT_FILE = '../index-v2_rag-v2_e20_r2.json'
with open(PIPELINE_OUTPUT_FILE) as f:
    pipeline_output = json.load(f)

answers: list[str] = [p['answer_data'] for p in pipeline_output]
df['pipeline_answer'] = answers
df['retrieved_docs'] = [(set(f"{doc['meta']['file_path']}:{doc['meta']['page_number']}" for doc in p['answer_docs'])) for p in pipeline_output]
df.head()
# %%
import unicodedata

recalled = []
for source, p in zip(df['Source'], pipeline_output):
    source_file = source.split(',')[0]
    page_num = source.split(',')[1].strip().replace('p', '').split('-')[0]
    recall = False
    for d in p['answer_docs']:
        if (
            d['meta']['file_path'] == unicodedata.normalize("NFC", source_file)
            and d['meta']['page_number'] == int(page_num)
        ):
            recall = True
            break
    recalled.append(recall)
df['recall'] = recalled
df[['Source', 'retrieved_docs', 'recall']]
# %%
from haystack.components.generators.openai import OpenAIGenerator
llm = OpenAIGenerator(
    model='gpt-4o'
)

grades = []

for _, row in df.iterrows():
    print(row['Question'])
    prompt = (
        "You are an expert judge fluent in Dutch.\n"
        "Check if the provided answer is correct or not against the actual answer.\n"
        "Output one of four grades:\n"
        "- Correct: The provided answer is pretty much exactly correct and complete. It may contain extra information but also fine if so as long as the core answer is present.\n"
        "- Correct-Incomplete: The provided answer is pretty much exactly correct but slightly incomplete. E.g misses a few details.\n"
        "- Incomplete: The provided answer isn't saying anything wrong but is quite incomplete. E.g misses main details.\n"
        "- Incorrect: The provided answer is providing incorrect information.\n"
        "Output the judgment in English as: <grade>::<very-concise-reasoning>"
        f"Question: {row['Question']}\n"
        f"Actual Answer: {row['Answer']}\n"
        f"Provided Answer: {row['pipeline_answer']}\n"
        f"Grade:"
    )
    
    grades.append(llm.run(prompt=prompt)['replies'][0])

df['grade'] = grades
print(df['grade'].to_list())
# %%
df['grade'] = df['grade'].map(
    lambda x: (
        x.replace('Correct:', 'Correct::')
         .replace('Incorrect:', 'Incorrect::')
         .replace('Correct-Incomplete:', 'Correct-Incomplete::')
         .replace('Incomplete:', 'Incomplete::')
    )
)
df['grade_label'] = df['grade'].map(lambda x: x.split('::')[0])
df['grade_label'].value_counts()
# %%
df[['recall', 'grade_label', 'Comment']]
# %%
df[~df['recall']][['Source', 'retrieved_docs']].to_dict()
# %%
df.to_csv(PIPELINE_OUTPUT_FILE.replace('.json', '_judged.csv'), index=False)
# %%
