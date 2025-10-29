# app/query_data.py
# -*- coding: utf-8 -*-
"""
Agente Text-to-SQL sobre PostgreSQL + pgvector usando LangChain + Gemini.

- Conecta a la BD vía SQLAlchemy (engine provisto por get_db_engine()).
- Genera y ejecuta UNA consulta SQL por pregunta.
- Evita respuestas genéricas y siempre responde con base en resultados.
- Devuelve, además del texto final, las consultas SQL detectadas.

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
import json
import logging
from typing import Any, Dict, List, Tuple

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
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada.")

DEFAULT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 1024  # Debe coincidir con vector(1024) en la BD


# ---------------------------------------------------------------------------
# Prompt del agente SQL
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Eres un asistente experto en PostgreSQL trabajando con LangChain.

OBJETIVO (OBLIGATORIO):
- Para cada pregunta del usuario, DEBES generar y ejecutar UNA única consulta SQL válida
  para PostgreSQL usando únicamente las tablas y columnas reales disponibles.
- La respuesta final debe estar basada ESTRICTAMENTE en los resultados de esa consulta.
- No muestres la consulta SQL en la respuesta final.
- Si la consulta devuelve 0 filas, dilo explícitamente (sin inventar datos).

REGLAS IMPORTANTES:
- Nunca pidas al usuario que reformule la pregunta: decide y ejecuta.
- Si el usuario pide “resumen por …” o “distribución por …”, usa GROUP BY sobre las columnas
  solicitadas y calcula COUNT, SUM, AVG y, si se pide mediana, usa:
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cuantia).
- Para filtros por estado/fecha/entidad, consulta directamente public.licitacion.
- Para búsqueda semántica en texto, puedes ordenar por similitud coseno con el índice ivfflat:
    ORDER BY public.licitacion_chunk.embedding_vec <=> [VECTOR_1024D]  LIMIT 10
  (usa el operador `<=>` con vector_cosine_ops). Solo hazlo cuando la pregunta lo indique.

ESQUEMA DISPONIBLE (nombres reales):
- public.licitacion: id, entidad, objeto, cuantia, modalidad, numero, estado,
  fecha_public, ubicacion, act_econ, enlace, portal_origen, texto_indexado,
  embedding, objeto_vec (vector(1024))
- public.licitacion_chunk: id, licitacion_id, chunk_idx, chunk_text,
  embedding, embedding_vec (vector(1024))
- public.flags, public.flags_licitaciones, public.flags_log
- Otras: public.chunks, public.documents, public.doc_section_hits,
         public.document_text_samples, public.licitacion_keymap

FORMATO DE TRABAJO:
- Piensa brevemente, genera la consulta SQL, ejecútala y resume los hallazgos.
- No inventes conclusiones fuera de los datos devueltos.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_text(x: Any) -> str:
    """Normaliza la salida del LLM a texto (sin 'extras/signature')."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        out: List[str] = []
        for p in x:
            if isinstance(p, dict):
                if "text" in p:
                    out.append(str(p["text"]))
                elif p.get("type") == "text" and "text" in p:
                    out.append(str(p["text"]))
            else:
                out.append(str(p))
        return " ".join(s for s in out if s).strip()
    if isinstance(x, dict):
        if x.get("type") == "text" and "text" in x:
            return str(x["text"])
        if "output" in x:
            return _to_text(x["output"])
        if "text" in x:
            return str(x["text"])
    return str(x)


def _extract_sql_steps(result: Any) -> List[str]:
    """
    Intenta extraer consultas SQL desde 'intermediate_steps' en distintos formatos.
    Es tolerante a versiones de LangChain.
    """
    sqls: List[str] = []
    if not isinstance(result, dict):
        return sqls

    steps = result.get("intermediate_steps")
    if not steps:
        return sqls

    for step in steps:
        try:
            # Formato típico: (action, observation)
            action, _obs = step
            # action puede ser una estructura con .tool_input o un dict
            ti = getattr(action, "tool_input", None)
            if ti is None and isinstance(action, dict):
                ti = action.get("tool_input") or action.get("input")
            if ti:
                sqls.append(str(ti))
                continue
        except Exception:
            pass
        # Último recurso: stringify del step
        try:
            s = json.dumps(step, ensure_ascii=False)
        except Exception:
            s = str(step)
        sqls.append(s)
    return sqls


def _looks_semantic_query(q: str) -> bool:
    """
    Heurística simple: activa embedding sólo si la intención parece semántica.
    Evita confundir al planner en consultas puramente tabulares.
    """
    ql = q.lower()
    keywords = [
        "similar", "semejante", "parecido", "semántic", "buscar texto",
        "snippet", "de qué trata", "contenido", "chunk", "embedding", "vector",
        "relacionado con", "parecidas a", "similar a"
    ]
    return any(k in ql for k in keywords)


# ---------------------------------------------------------------------------
# Construcción del agente
# ---------------------------------------------------------------------------
def create_sql_agent_executor():
    """
    Crea el agente SQL con tool-calling real y prompt estricto para evitar respuestas genéricas.
    Hace fallback automático a 'openai-tools' si la versión no soporta 'tool-calling'.
    """
    db_engine = get_db_engine()
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
    db = SQLDatabase(engine=db_engine, include_tables=include_tables)

    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        temperature=0,
        convert_system_message_to_human=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Intento 1: tool-calling con pasos intermedios
    try:
        agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="tool-calling",
            verbose=True,
            prompt=prompt,
            return_intermediate_steps=True,  # algunas versiones lo aceptan
        )
        return agent
    except TypeError:
        logger.warning("create_sql_agent no acepta return_intermediate_steps con 'tool-calling'. Reintentando sin ese argumento…")
        try:
            agent = create_sql_agent(
                llm=llm,
                db=db,
                agent_type="tool-calling",
                verbose=True,
                prompt=prompt,
            )
            return agent
        except Exception:
            logger.warning("Fallback a agent_type='openai-tools'.")
            agent = create_sql_agent(
                llm=llm,
                db=db,
                agent_type="openai-tools",
                verbose=True,
                prompt=prompt,
            )
            return agent


# ---------------------------------------------------------------------------
# Componentes (lazy)
# ---------------------------------------------------------------------------
_agent_executor = None
_embedding_function = None


def get_components() -> Tuple[Any, GoogleGenerativeAIEmbeddings]:
    global _agent_executor, _embedding_function
    if _agent_executor is None:
        _agent_executor = create_sql_agent_executor()
    if _embedding_function is None:
        _embedding_function = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            output_dimensionality=EMBED_DIM,
        )
    return _agent_executor, _embedding_function


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def process_query(query_text: str) -> Dict[str, Any]:
    """
    Ejecuta el agente con la pregunta del usuario.
    - Activa embedding SOLO si la intención sugiere búsqueda semántica.
    - Devuelve la respuesta limpia y, si es posible, las queries SQL detectadas.
    """
    try:
        q = (query_text or "").strip()
        if not q:
            return {"error": "La consulta está vacía.", "status": "error"}

        logger.info("Consulta recibida: %s", q)

        agent_executor, embedding_function = get_components()

        # Decide si adjuntar embedding (para consultas semánticas)
        formatted_input = q
        if _looks_semantic_query(q):
            try:
                vec = embedding_function.embed_query(q)
                # Usamos sintaxis de array JSON legible por extensión vector (como literal),
                # el agente lo puede inyectar en ORDER BY … <=> [v1, v2, …].
                formatted_input = f"{q}\n\nVECTOR_1024D:\n{json.dumps(vec)}"
            except Exception as e:
                logger.warning("Fallo generando embedding (se continúa sin vector): %s", e)

        # Invoca agente
        result = agent_executor.invoke({"input": formatted_input})

        # Respuesta de texto limpia
        answer_raw = result.get("output") if isinstance(result, dict) else result
        answer_text = _to_text(answer_raw).strip()

        # Si por alguna razón el modelo cae en una respuesta genérica, refuérzalo:
        if not answer_text or "proporciona la pregunta" in answer_text.lower():
            answer_text = "No pude derivar una respuesta válida desde la base de datos. Reformula de forma más concreta (incluye columnas / filtros)."

        # Extraer SQL (si disponible)
        sql_steps = _extract_sql_steps(result)

        return {
            "answer": answer_text or "Sin respuesta.",
            "sql_query": sql_steps,
        }

    except Exception as e:
        logger.error("Error crítico en process_query: %s", str(e), exc_info=True)
        return {
            "error": "Error procesando la consulta",
            "details": str(e),
            "status": "error",
        }
