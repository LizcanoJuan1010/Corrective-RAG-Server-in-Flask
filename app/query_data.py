from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from .db_utils import get_db_engine
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"

# --- Plantilla de Prompt para el Agente SQL con pgvector y esquema complejo ---
# Se reestructura el prompt para usar el formato de mensajes de LangChain
SYSTEM_PROMPT = """
Eres un asistente experto en PostgreSQL que trabaja con LangChain. Tu tarea es generar consultas SQL para una base de datos de licitaciones.

**Esquema de la Base de Datos:**

1.  `licitacion`: Contiene los detalles principales de cada licitación.
    *   `id`, `entidad`, `objeto`, `cuantia`, `modalidad`, `numero`, `estado`, `fecha_public`, `ubicacion`, `act_econ`, `enlace`, `portal_origen`.
    *   `nombre_licitacion_embedding` (VECTOR): Embedding del título/objeto de la licitación para búsqueda semántica.
    *   `texto_indexado` (TEXT): El texto completo usado para generar el embedding anterior.

2.  `chunks`: Contiene fragmentos de texto detallados de los documentos de la licitación.
    *   `id`, `licitacion_id` (se une a `licitacion.id`), `texto`, `embedding` (VECTOR).

3.  `flags`: Describe diferentes tipos de "banderas" o alertas (ej. riesgos, condiciones especiales).
    *   `id`, `codigo`, `nombre`, `descripcion`.

4.  `flags_licitaciones`: Tabla de unión que indica qué flags tiene cada licitación.
    *   `id`, `licitacion_id` (se une a `licitacion.id`), `flag_id` (se une a `flags.id`), `valor` (BOOLEAN).

5.  `flags_log`: Un registro de auditoría de los cambios en los flags.

**Instrucciones para Generar Consultas:**

1.  **Genera una única consulta SQL válida** para PostgreSQL.
2.  **Búsqueda Semántica:** Para preguntas abiertas sobre el contenido, prioriza la búsqueda en la tabla `chunks`. Para preguntas sobre el objeto o título de una licitación, usa `nombre_licitacion_embedding`.
    *   Utiliza el operador `<=>` (distancia coseno) y el embedding de la pregunta del usuario proporcionado en `{user_question_embedding}`.
    *   **Ejemplo:** `ORDER BY chunks.embedding <=> '{user_question_embedding}'`
3.  **Consultas con Filtros y Uniones (JOINs):**
    *   Si el usuario pregunta por una licitación con un "flag" específico, debes unir `licitacion` con `flags_licitaciones` y `flags`.
    *   **Ejemplo:** "Encuentra licitaciones con el flag 'red1'":
        ```sql
        SELECT l.objeto FROM licitacion l
        JOIN flags_licitaciones fl ON l.id = fl.licitacion_id
        JOIN flags f ON fl.flag_id = f.id
        WHERE f.codigo = 'red1' AND fl.valor = TRUE;
        ```
    *   Si se pide buscar texto dentro de licitaciones con un flag, une las tres tablas y además `chunks`, combinando `WHERE` con `ORDER BY` semántico.
4.  **Selecciona Columnas Relevantes:** Devuelve las columnas que respondan mejor a la pregunta. Si es una pregunta general, `licitacion.objeto` o `chunks.texto` suelen ser las más útiles.
5.  **Limita los resultados** a un número razonable (ej. `LIMIT 10`).
"""

def create_sql_agent_executor():
    """Crea y devuelve un agente SQL de LangChain configurado para el nuevo esquema."""
    db_engine = get_db_engine()
    # Incluimos todas las tablas relevantes para que el agente las conozca
    include_tables = ['licitacion', 'chunks', 'flags', 'flags_licitaciones', 'flags_log']
    db = SQLDatabase(engine=db_engine, include_tables=include_tables)

    llm = OllamaLLM(model=DEFAULT_MODEL, temperature=0)

    # Construcción del prompt con el placeholder requerido
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="{input}\n\nEmbedding de la pregunta para usar en la consulta:\n{user_question_embedding}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        prompt=prompt
    )
    return agent_executor

# Sistema de inicialización perezosa
_agent_executor = None
_embedding_function = None

def get_components():
    global _agent_executor, _embedding_function
    if _agent_executor is None:
        _agent_executor = create_sql_agent_executor()
    if _embedding_function is None:
        _embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return _agent_executor, _embedding_function

def process_query(query_text: str) -> dict:
    """Procesa la consulta del usuario utilizando un agente SQL avanzado."""
    try:
        logger.info(f"Procesando consulta: {query_text}")
        agent_executor, embedding_function = get_components()
        
        query_embedding = embedding_function.embed_query(query_text)

        result = agent_executor.invoke({
            "input": query_text,
            "user_question_embedding": str(query_embedding),
        })

        return {
            "answer": result.get("output", "No se encontró una respuesta."),
            "sql_query": result.get("intermediate_steps", [])
        }

    except Exception as e:
        logger.error(f"Error crítico: {str(e)}", exc_info=True)
        return {
            "error": "Error procesando la consulta",
            "details": str(e),
            "status": "error"
        }
