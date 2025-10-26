from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from .db_utils import get_db_engine
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"

# --- Plantilla de Prompt para el Agente SQL con pgvector ---
SQL_PROMPT_TEMPLATE = """
Eres un experto en PostgreSQL que trabaja con LangChain. Tu tarea es interactuar con una base de datos que contiene información sobre licitaciones y chunks de documentos.

**Capacidades de la Base de Datos:**
1.  **Búsqueda Semántica con pgvector:** La tabla `chunks` contiene una columna `texto_embedding` de tipo `vector`. Puedes realizar búsquedas de similitud semántica utilizando el operador `<=>` (distancia coseno).
2.  **Relaciones entre Tablas:**
    *   La tabla `licitaciones` contiene información general (`id`, `nombre_licitacion`).
    *   La tabla `chunks` contiene fragmentos de texto (`id`, `id_licitacion`, `texto`, `texto_embedding`).
    *   Puedes unir (`JOIN`) estas tablas usando `licitaciones.id = chunks.id_licitacion`.

**Instrucciones para Generar Consultas:**

1.  **Genera una consulta SQL válida** para PostgreSQL.
2.  **Utiliza el embedding proporcionado:** En lugar de generar un embedding, utiliza el vector que te proporciono en el parámetro `user_question_embedding`. Insértalo directamente en la consulta.
    *   **Ejemplo:** `ORDER BY texto_embedding <=> '{user_question_embedding}'`
3.  **Combina filtros y búsqueda semántica:** Si la pregunta del usuario menciona un nombre o ID de licitación, usa una cláusula `WHERE` y luego ordena por similitud semántica.
    *   **Ejemplo con JOIN:**
        ```sql
        SELECT chunks.texto
        FROM licitaciones
        JOIN chunks ON licitaciones.id = chunks.id_licitacion
        WHERE licitaciones.nombre_licitacion LIKE '%nombre específico%'
        ORDER BY chunks.texto_embedding <=> '{user_question_embedding}'
        LIMIT 5;
        ```
4.  **Devuelve el texto:** Asegúrate de que tu consulta final seleccione la columna `chunks.texto` y limita los resultados (`LIMIT 5`).

Aquí tienes la pregunta del usuario:
{input}
"""

def create_sql_agent_executor():
    """Crea y devuelve un agente SQL de LangChain configurado para pgvector."""
    db_engine = get_db_engine()
    db = SQLDatabase(engine=db_engine, include_tables=['licitaciones', 'chunks'])
    llm = OllamaLLM(model=DEFAULT_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_template(SQL_PROMPT_TEMPLATE)

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
    """
    Procesa la consulta del usuario utilizando un agente SQL.
    """
    try:
        logger.info(f"Procesando consulta: {query_text}")
        agent_executor, embedding_function = get_components()
        
        # 1. Generar el embedding de la pregunta del usuario
        query_embedding = embedding_function.embed_query(query_text)

        # 2. Invocar al agente con la pregunta y el embedding precalculado
        result = agent_executor.invoke({
            "input": query_text,
            "user_question_embedding": str(query_embedding) # Pasar como string
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
