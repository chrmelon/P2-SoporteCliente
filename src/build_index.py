#######################
# ---- Libraries ---- #
#######################

import yaml
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger import get_logger


SCRIPT_DIR = Path(__file__).parent

PROJECT_ROOT = Path(__file__).parent.parent

def build_index():
    ###########################
    # ---- Logger Design ---- #
    ###########################

    logger = get_logger(__name__)

    ##########################
    # ---- Load Config ---- #
    ##########################

    try:
        with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(
            f"Configuración cargada — "
            f"colección: '{config['collection_name']}', "
            f"chunk_size: {config['chunk_size']}"
        )
    except FileNotFoundError:
        logger.error("No se encontró config.yaml")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error al parsear config.yaml: {e}")
        raise

    ######################
    # ---- Call API ---- #
    ######################

    if not load_dotenv():
        logger.warning("No se encontró .env — buscando OPENAI_API_KEY en el sistema")

    try:
        embeddings = OpenAIEmbeddings(model=config["embedding_model"])
        logger.info(f"Embeddings listos: {config['embedding_model']}")
    except Exception as e:
        logger.error(f"No se pudo inicializar embeddings: {e}")
        raise RuntimeError("Verifica tu OPENAI_API_KEY en el archivo .env") from e
        

    DATA_PATH = PROJECT_ROOT / "data" / "faq_document.txt"

    try:
        loader = TextLoader(str(DATA_PATH), encoding="utf-8")
        CORPUS = loader.load()
        logger.info(f"Documento cargado correctamente: {len(CORPUS)} documentos")
    except Exception as e:
        logger.error(f"Error al cargar el documento: {e}")
        raise
        
        
    # Chunking e indexación
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )

    chunks: list[Document] = []
    for doc in CORPUS:
        sub_chunks = splitter.split_text(doc.page_content)
        for c in sub_chunks:
            chunks.append(Document(page_content=c.strip(), metadata=doc.metadata))

    logger.info(f"Corpus dividido en {len(chunks)} chunks")

    # limpiar db previa (si querés rebuild limpio)
    if config.get("rebuild", True):
         shutil.rmtree(SCRIPT_DIR / "chroma_db", ignore_errors=True)


    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=config["collection_name"],
            persist_directory=str(SCRIPT_DIR / "chroma_db")
        )
        logger.info(f"Vectorstore indexado: {vectorstore._collection.count()} chunks")
    except Exception as e:
        logger.error(f"Error al crear vectorstore: {e}")
        raise

    retriever_base = vectorstore.as_retriever(
        search_kwargs={"k": config["retrieval_k"]},
    )



if __name__ == "__main__":
    build_index()