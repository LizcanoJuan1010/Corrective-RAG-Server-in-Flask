import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnablePassthrough
import logging
import os
from .db_utils import get_db_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase

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
        "answer_chain": create_answer_chain(),
        "sql_agent_executor": create_sql_agent_executor()
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

def create_sql_agent_executor():
    """Crea y devuelve un agente SQL de LangChain."""
    db_engine = get_db_engine()
    db = SQLDatabase(engine=db_engine)
    llm = OllamaLLM(model=DEFAULT_MODEL, temperature=0)

    # Define table names for the agent
    table_names = ["licitacion", "flags_licitaciones", "flags", "banco_flagueado", "flags_log"]

    # Create the SQL agent
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="tool-calling",
        verbose=True,
        table_names=table_names,
        return_intermediate_steps=True,
    )

    # Create a new chain to extract licitacion_id from the agent's observation
    def get_licitacion_ids(sql_result):
        """
        Parses the result from the SQL agent's execution to extract licitacion_ids.
        This avoids re-executing the query.
        """
        # The observation is the result of the SQL query, returned as a string.
        # It's the second element of the tuple in the first intermediate step.
        observation = sql_result["intermediate_steps"][0][1]
        import ast
        try:
            # The result is a string representation of a list of tuples, e.g., "[(1,), (2,)]"
            # We use ast.literal_eval for safe evaluation of this string.
            results_list = ast.literal_eval(observation)
            # Extract the first element (the ID) from each tuple.
            licitacion_ids = [item[0] for item in results_list if isinstance(item, tuple) and len(item) > 0]
        except (ValueError, SyntaxError):
            # If parsing fails (e.g., empty string or different format), return an empty list.
            licitacion_ids = []
        return licitacion_ids

    return agent_executor | get_licitacion_ids


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

        # 2. Consultar a la base de datos SQL con el agente
        sql_agent_executor = components["sql_agent_executor"]
        sql_result = sql_agent_executor.invoke({"input": improved_question})

        # 3. Recuperar documentos relevantes y filtrar por licitacion_id
        retriever = components["retriever"]
        if sql_result:
            retriever = components["vector_db"].as_retriever(
                search_kwargs={"k": 10, "filter": {"licitacion_id": {"$in": sql_result}}}
            )
        docs = retriever.invoke(improved_question)

        # 4. Evaluar relevancia de documentos con manejo de errores
        relevance_result = components["retrieval_grader"].invoke({
            "document": docs[0].page_content if docs else "No hay documentos",
            "question": improved_question
        })

        try:
            relevance = RetrievalEvaluator(**relevance_result)
        except ValidationError as e:
            logger.error(f"Error validando evaluación: {e}")
            relevance = RetrievalEvaluator(binary_score="no")

        # 5. Buscar en web y generar respuesta
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
            "confidence": "high" if relevance.binary_score == "yes" else "low",
            "sql_result": sql_result
        }

    except Exception as e:
        logger.error(f"Error crítico: {str(e)}", exc_info=True)
        return {
            "error": "Error procesando la consulta",
            "details": str(e),
            "status": "error"
        }
