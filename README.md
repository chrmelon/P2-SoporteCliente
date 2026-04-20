# 🧠 Sistema de Asistente Virtual basado en RAG

Este proyecto implementa un sistema de **FAQs inteligente basado en Retrieval-Augmented Generation (RAG)** para una plataforma HR SaaS.

---

# 📁 Estructura del Proyecto


|-- src/
|   |-- build_index.py
|   |-- query.py
|   |-- logger.py
|   |-- chroma_db/
|-- output/
|   |-- sample_queries.json                   
|-- data/
|   |-- faq_document.txt                     
|   |-- questions.json    # preguntas usadas para generar sample_queries.json
|-- pyproject.toml        # Dependencias y config del proyecto
|-- config.yaml                              
|-- main.py               # ejecuta una pregunta al asistente
|-- sample_queries.py     # genera sample_queries.json
|-- README.md
|-- .end
|-- .gitignore

---

# ⚙️ Setup

## 🐍 Python
Python 3.12

## 🔑 API Key

Crear archivo `.env`:

OPENAI_API_KEY=tu_api_key
LANGCHAIN_API_KEY=tu_api_key

---

## 📦 Instalación

py -3.12 -m venv .venv
.venv\Scripts\activate

pip install .

---

# 🚀 Uso

## 1. Indexación

python src/build_index.py

## 2. Consultas

Se puede usar de 2 maneras:

  1 - Sin parámetros
        python main.py   
        En este caso, por defecto, se realiza la siguiente pregunta: ¿Qué políticas de vacaciones existen?
  2 - Con parámetro de pregunta 
        python main.py --query "¿Cómo solicito vacaciones?"


---

# 🧩 Arquitectura RAG

1. Carga de documento
2. Chunking (RecursiveCharacterTextSplitter)
3. Embeddings (OpenAI)
4. Vector store (Chroma)
5. Recuperación (MMR)
6. Generación con LLM

---

# 🔎 Decisiones Técnicas

- Chunking: RecursiveCharacterTextSplitter
- Búsqueda: MMR (Max Marginal Relevance)
- Vector DB: Chroma
- Embeddings: OpenAI

Se utiliza MMR como estrategia híbrida para balancear relevancia y diversidad

---

# 📤 Output JSON

{
  "user_question": "...",
  "system_answer": "...",
  "chunks_related": [...],
  "evaluation": "..."
}

Incluye transparencia y auditabilidad.
Evaluación 

    Score 0-10 basado en:
    - relevancia
    - precisión
    - completitud

---

# 🚀 Conclusión

Sistema RAG completo listo para responder FAQs automáticamente.
