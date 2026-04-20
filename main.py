import argparse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.query import AgenticRAG


def main():
    # 1. Argumentos CLI
    parser = argparse.ArgumentParser(description="RAG HR Assistant")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Pregunta del usuario"
    )

    args = parser.parse_args()

    # 2. Pregunta (fallback si no viene parámetro)
    question = args.query if args.query else "¿Qué políticas de vacaciones existen?"

    # 3. Inicialización
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma(
        persist_directory="src/chroma_db",
        embedding_function=embeddings,
        collection_name="DOC_HR"
    )

    rag = AgenticRAG(db, model_name="gpt-4o-mini")

    print("TOTAL DOCS:", db._collection.count())
    print("\nPregunta:", question)

    # 4. Ejecutar RAG
    result = rag.ask(question)

    print("\nRespuesta:")
    print(result)


if __name__ == "__main__":
    main()