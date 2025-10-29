# app/query_data.py
# -*- coding: utf-8 -*-
"""
Módulo que construye un agente Text-to-SQL sobre PostgreSQL + pgvector.
- Usa LangChain + Gemini (chat + embeddings 1024d).
- Conecta a la BD vía SQLAlchemy (engine provisto por get_db_engine()).
- Expone process_query() que recibe una pregunta en texto y devuelve una respuesta limpia.

Requisitos en .env:
  GEMINI_API_KEY=...
  DB_HOST=db
  DB_PORT=5432
  DB_NAME=rag_db
  DB_USER=user123
  DB_PASSWORD=password123
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, Tuple

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from .db_utils import get_db_engine


# ---------------------------------------------------------------------------
# Logging básico
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config de modelos
# ---------------------------------------------------------------------------
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada.")

# Modelo conversacional (rápido y con tool-calling)
DEFAULT_MODEL = "gemini-2.5-flash"

# Embeddings (salida 1024-d para que coincida con vector(1024))
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 1024  # ¡Debe coincidir con vector(1024) en la BD!


# ---------------------------------------------------------------------------
# Prompt del agente SQL
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Eres un asistente experto en PostgreSQL trabajando con LangChain.

TAREA:
1) Recibirás una pregunta del usuario (en {input}).
2) Genera UNA sola consulta SQL válida para PostgreSQL que responda la pregunta usando las tablas disponibles.
3) Ejecuta la consulta.
4) Responde en lenguaje natural basándote ESTRICTAMENTE en los resultados de esa consulta (no muestres el SQL).

REGLAS:
- Si la consulta devuelve 0 filas, dilo explícitamente.
- No inventes datos ni hagas suposiciones fuera de los resultados obtenidos.
- No reveles la consulta SQL generada.
- Usa nombres de tabla y columna reales (ver esquema abajo).
- Prioriza consultas eficientes (usa índices cuando existan).

ESQUEMA (nombres reales):
- public.licitacion (
    id, entidad, objeto, cuantia, modalidad, numero, estado,
    fecha_public, ubicacion, act_econ, enlace, portal_origen,
    texto_indexado, embedding, objeto_vec vector(1024)
  )

- public.licitacion_chunk (
    id, licitacion_id, chunk_idx, chunk_text,
    embedding, embedding_vec vector(1024)
  )

- public.flags, public.flags_licitaciones, public.flags_log
- Otras: public.chunks, public.documents, public.doc_section_hits,
         public.document_text_samples, public.licitacion_keymap

GUÍA PARA CONSULTAS:
- Búsqueda semántica: si la pregunta es abierta ("¿de qué trata...?",
  "busca info sobre..."), considera ordenar por similitud coseno
  usando el índice IVFFlat:
    ORDER BY public.licitacion_chunk.embedding_vec <=> '[VECTOR_1024D]'
    LIMIT 10
  (usa el operador de distancia coseno `<=>` con vector_cosine_ops).

- Si piden por flags/alertas específicos, une licitacion ↔ flags_licitaciones ↔ flags.

- Para preguntas simples (conteos, filtros por estado/fecha/entidad), consulta
  directamente public.licitacion (usa índices por estado/fecha si existen).

- Devuelve SIEMPRE una respuesta narrativa basada en los datos obtenidos, no el SQL.
"""


# ---------------------------------------------------------------------------
# Helper: normaliza la salida del modelo a texto limpio
# ---------------------------------------------------------------------------
def _to_text(x: Any) -> str:
    """
    Convierte la salida (que a veces es lista/dict con 'text' o 'extras')
    a un string limpio. Evita que aparezcan campos como "extras/signature".
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x

    # Listas de partes (LangChain/Gemini puede devolver múltiples "content parts")
    if isinstance(x, list):
        parts = []
        for p in x:
            if isinstance(p, dict):
                if "text" in p:
                    parts.append(p["text"])
                elif p.get("type") == "text" and "text" in p:
                    parts.append(p["text"])
                else:
                    # Evita volcar "extras" poco útiles
                    val = p.get("text") or ""
                    if val:
                        parts.append(val)
            else:
                parts.append(str(p))
        return " ".join([s for s in parts if s]).strip()

    # Diccionarios (algunos wrappers devuelven {"type":"text","text":"..."})
    if isinstance(x, dict):
        if x.get("type") == "text" and "text" in x:
            return x["text"]
        if "output" in x:
            return _to_text(x["output"])
        # Como último recurso, intenta acceder a un campo 'text'
        if "text" in x:
            return str(x["text"])

    return str(x)


# ---------------------------------------------------------------------------
# Construcción del agente SQL
# ---------------------------------------------------------------------------
def create_sql_agent_executor():
    """
    Crea y devuelve un agente SQL LangChain configurado para tu esquema real.
    - Conecta vía SQLAlchemy.
    - Limita el contexto a tablas relevantes para reducir alucinaciones.
    """
    db_engine = get_db_engine()

    # Tablas reales (sin prefijo "public_")
    include_tables = [
        "licitacion",
        "licitacion_chunk",
        "flags",
        "flags_licitaciones",
        "flags_log",
        "chunks",
        "documents",
        "doc_section_hits",
        "document_text_samples",
        "licitacion_keymap",
    ]

    # SQLDatabase le da al agente el esquema y ejecuta las queries
    db = SQLDatabase(engine=db_engine, include_tables=include_tables)

    # LLM conversacional
    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        temperature=0,
        convert_system_message_to_human=True,
    )

    # Prompt (system + user + scratchpad)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Agente orientado a herramientas SQL (tool-calling/openai-tools)
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        prompt=prompt,
    )

    return agent_executor


# ---------------------------------------------------------------------------
# Inicialización perezosa de componentes
# ---------------------------------------------------------------------------
_agent_executor = None
_embedding_function = None


def get_components() -> Tuple[Any, GoogleGenerativeAIEmbeddings]:
    """
    Devuelve (agente, embedding_function) inicializándolos una sola vez.
    """
    global _agent_executor, _embedding_function
    if _agent_executor is None:
        _agent_executor = create_sql_agent_executor()
    if _embedding_function is None:
        # ¡IMPORTANTE!: salida de dimensión 1024 para que coincida con vector(1024)
        _embedding_function = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            output_dimensionality=EMBED_DIM,
        )
    return _agent_executor, _embedding_function


# ---------------------------------------------------------------------------
# Punto de entrada para consultas del usuario
# ---------------------------------------------------------------------------
def process_query(query_text: str) -> Dict[str, Any]:
    """
    Procesa la consulta del usuario con el agente SQL.
    - Genera embedding 1024d del texto y lo adjunta al input para que el agente
      pueda usarlo como literal en consultas vectoriales (si lo necesita).
    - Ejecuta la acción del agente y devuelve una respuesta limpia.

    Returns:
        dict: {
          "answer": <str>,
          "sql_query": <list> (si verbose/intermediate_steps está disponible)
        }
    """
    try:
        q = (query_text or "").strip()
        if not q:
            return {"error": "La consulta está vacía.", "status": "error"}

        logger.info("Procesando consulta: %s", q)

        agent_executor, embedding_function = get_components()

        # Genera embedding 1024d
        query_embedding = embedding_function.embed_query(q)

        # Inyecta el embedding en el prompt para que el agente pueda usarlo
        # como literal SQL si decide hacer búsqueda semántica:
        #  ORDER BY licitacion_chunk.embedding_vec <=> '[v1, v2, ...]' LIMIT 10;
        formatted_input = (
            f"{q}\n\n"
            "Embedding_1024_para_busqueda_semantica:\n"
            f"{str(query_embedding)}"
        )

        # Ejecuta el agente
        result = agent_executor.invoke({"input": formatted_input})

        # Limpia la salida del modelo a 'str'
        answer_raw = result.get("output") if isinstance(result, dict) else result
        answer_text = _to_text(answer_raw)

        return {
            "answer": answer_text or "No se encontró una respuesta.",
            "sql_query": result.get("intermediate_steps", []) if isinstance(result, dict) else [],
        }

    except Exception as e:
        logger.error("Error crítico en process_query: %s", str(e), exc_info=True)
        return {
            "error": "Error procesando la consulta",
            "details": str(e),
            "status": "error",
        }
