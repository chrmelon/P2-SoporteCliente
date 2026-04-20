import os
import time
import json
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.logger import get_logger

logger = get_logger(__name__)


class AgenticRAG:

    def __init__(self, vector_store: Chroma, model_name: str, top_k: int = 8):
        self.retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": 10}
        )

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
        )

        logger.info(f"RAG listo con modelo {model_name}")

    def ask(self, question: str) -> str:
        start = time.time()

        # 1. Buscar contexto
        docs = self.retriever.invoke(question)
        chunks_related = [
                            {
                                "content": d.page_content,
                                "metadata": d.metadata
                            }
                            for d in docs[:3]
                        ]

        if not docs:
            result = {
                "user_question": question,
                "system_answer": "No está en la base de conocimiento.",
                "chunks_related": [],
                "evaluation": {
                    "score": 0,
                    "justification": "No se encontraron documentos relevantes"
                }
            }
            return json.dumps(result, indent=2, ensure_ascii=False)

        # 2. Limpiar y reducir contexto
        unique_texts = []
        seen = set()

        for d in docs:
            text = d.page_content.strip()
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        context = "\n\n".join(unique_texts[:2])
        
        # 3. Prompt
        messages = [
                SystemMessage(content=f"""
                Sos un asistente de RRHH.

                Tu tarea es responder usando SOLO el contexto.

                Reglas:
                - SIEMPRE usar el contexto
                - NO inventar información
                - Si hay pasos → listarlos claramente
                - Si hay info parcial → usarla igual

                Contexto:
                {context}

                Pregunta:
                {question}

                Respuesta:
                """),
            HumanMessage(content=question)
        ]

        # 4. LLM
        response = self.llm.invoke(messages)

        evaluation = self.evaluate_answer(
            question,
            response.content,
            chunks_related
            )

        result = {
                "user_question": question,
                "system_answer": response.content,
                "chunks_related": chunks_related,
                "evaluation": evaluation}

        elapsed = round(time.time() - start, 2)
        logger.info(f"Tiempo respuesta: {elapsed}s")

        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def evaluate_answer(self, question: str, answer: str, chunks: list) -> dict:

        eval_prompt = f"""
        Sos un evaluador de calidad de un sistema RAG.

        Tenés que puntuar la respuesta del sistema del 0 al 10.

        Evaluá según:
        1. Relevancia respecto a la pregunta
        2. Uso correcto del contexto (chunks)
        3. Precisión
        4. Completitud

        Pregunta:
        {question}

        Respuesta:
        {answer}

        Contexto:
        {chunks}

        Devolvé SOLO un JSON con este formato:
        {{
            "score": number,
            "justification": "explicación breve"
        }}
        """

        response = self.llm.invoke([
            SystemMessage(content="Sos un evaluador objetivo y estricto."),
            HumanMessage(content=eval_prompt)
        ])

        try:
            return json.loads(response.content)
        except:
            return {
                "score": 0,
                "justification": "Error al evaluar la respuesta"
            }