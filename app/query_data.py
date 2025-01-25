import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnablePassthrough
import logging
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constantes
CHROMA_PATH = "./app/chroma"
DEFAULT_MODEL = "phi4"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Additional web search results:

{web_results}

---

Answer the question in the user's language: {question}
"""


class RetrievalEvaluator(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Evaluación de relevancia del documento: 'yes' o 'no'"
    )


def initialize_components():
    """Inicializa y configura todos los componentes del sistema"""
    # Configuración del evaluador de documentos
    retrieval_evaluator_llm = OllamaLLM(model=DEFAULT_MODEL, temperature=0)
    evaluator_parser = JsonOutputParser(pydantic_object=RetrievalEvaluator)

    system_retrieval = """Evalúa la relevancia del documento para responder la pregunta. 
    Responde EXCLUSIVAMENTE con JSON válido usando este esquema:
    {{
        "binary_score": "yes"|"no"
    }}"""

    retrieval_evaluator_prompt = ChatPromptTemplate.from_messages([
        ("system", system_retrieval + "\n\n{format_instructions}"),
        ("human", "Documento:\n{document}\n\nPregunta: {question}")
    ]).partial(format_instructions=evaluator_parser.get_format_instructions())

    # Base de datos vectorial
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    return {
        "retrieval_grader": retrieval_evaluator_prompt | retrieval_evaluator_llm | evaluator_parser,
        "question_rewriter": create_question_rewriter(),
        "web_search_tool": TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), k=3),
        "vector_db": db,
        "retriever": db.as_retriever(search_kwargs={"k": 4}),
        "main_model": OllamaLLM(model=DEFAULT_MODEL, temperature=0.1),
        "answer_chain": create_answer_chain()
    }


def create_question_rewriter():
    """Crea la cadena para reescribir preguntas"""
    return ChatPromptTemplate.from_messages([
        ("system", "Mejora la claridad y precisión de la pregunta manteniendo su intención original."),
        ("human", "Pregunta original: {question}\nPregunta mejorada:")
    ]) | OllamaLLM(model=DEFAULT_MODEL, temperature=0) | StrOutputParser()


def create_answer_chain():
    """Crea la cadena principal de generación de respuestas"""
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return (
            RunnablePassthrough.assign(
                context=lambda x: "\n\n".join([doc.page_content for doc in x["docs"]]),
                web_results=lambda x: "\n".join([
                    f"{res.get('url', 'Sin URL')}: {res.get('content', 'Sin contenido')}"
                    for res in x["web_results"]
                ])
            )
            | prompt
            | OllamaLLM(model=DEFAULT_MODEL)
            | StrOutputParser()
    )


# Sistema de inicialización perezosa
_components = None


def get_components():
    global _components
    if _components is None:
        _components = initialize_components()
    return _components


def process_query(query_text: str) -> dict:
    """Procesa consultas y devuelve respuestas estructuradas"""
    try:
        components = get_components()
        logger.info(f"Procesando consulta: {query_text}")

        # 1. Mejorar la pregunta
        improved_question = components["question_rewriter"].invoke({"question": query_text})

        # 2. Recuperar documentos relevantes
        retriever = components["retriever"]
        docs = retriever.invoke(improved_question)

        # 3. Evaluar relevancia de documentos con manejo de errores
        relevance_result = components["retrieval_grader"].invoke({
            "document": docs[0].page_content if docs else "No hay documentos",
            "question": improved_question
        })

        try:
            relevance = RetrievalEvaluator(**relevance_result)
        except ValidationError as e:
            logger.error(f"Error validando evaluación: {e}")
            relevance = RetrievalEvaluator(binary_score="no")

        # 4. Buscar en web y generar respuesta
        web_results = components["web_search_tool"].invoke(improved_question)
        final_answer = components["answer_chain"].invoke({
            "question": improved_question,
            "docs": docs,
            "web_results": web_results
        })

        web_urls = list({res.get('url') for res in web_results if res.get('url')})

        return {
            "answer": final_answer,
            "sources": list({doc.metadata.get("source", "unknown") for doc in docs}),
            "web_urls": web_urls,
            "rewritten_question": improved_question,
            "relevance_check": relevance.binary_score,
            "confidence": "high" if relevance.binary_score == "yes" else "low"
        }

    except Exception as e:
        logger.error(f"Error crítico: {str(e)}", exc_info=True)
        return {
            "error": "Error procesando la consulta",
            "details": str(e),
            "status": "error"
        }