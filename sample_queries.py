import json

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.query import AgenticRAG

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory="src/chroma_db",
    embedding_function=embeddings,
    collection_name="DOC_HR"
)

rag = AgenticRAG(db, model_name="gpt-4o-mini")
print("TOTAL DOCS:", db._collection.count())

with open("Data/questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    
results = []

for q in data["questions"]:
    result = rag.ask(q)
    results.append(json.loads(result))

with open("Output/sample_queries.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
