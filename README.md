# Servidor RAG Correctivo en Flask

Este proyecto es un servidor basado en Flask que implementa un sistema RAG (Generación Aumentada por Recuperación) híbrido. Utiliza un Agente SQL de LangChain para realizar consultas que pueden combinar filtros SQL estándar con búsquedas semánticas vectoriales en una base de datos PostgreSQL con la extensión `pgvector`. El sistema está completamente contenedorizado con Docker para facilitar su configuración y despliegue.

## Prerrequisitos

-   Docker
-   Docker Compose
-   `sudo` o permisos de administrador para ejecutar comandos de Docker.

## Configuración

### 1. Coloca el Respaldo de la Base de Datos

-   Consigue el archivo de respaldo de la base de datos que deseas restaurar.
-   Renombra el archivo a `backup.dump`.
-   Coloca este archivo en el **directorio raíz del proyecto**, al mismo nivel que `docker-compose.yml`.

### 2. Configura las Variables de Entorno

1.  Crea un archivo llamado `.env` en la raíz del proyecto. Puedes hacerlo copiando el archivo de ejemplo:
    ```bash
    cp .env.example .env
    ```
2.  Abre el archivo `.env` con un editor de texto. Deberás configurar las siguientes variables:

    ```
    # Configuración de PostgreSQL
    DB_USER=user
    DB_PASSWORD=password
    DB_HOST=db
    DB_PORT=5432
    DB_NAME=rag_db

    # Clave de API de Tavily (Opcional, para búsqueda web)
    TAVILY_API_KEY=tu_clave_api_de_tavily
    ```
    **Nota Importante:**
    -   Los valores `DB_HOST=db` y `DB_PORT=5432` están preconfigurados para la comunicación entre contenedores y **no deben ser modificados**.
    -   Puedes personalizar `DB_USER`, `DB_PASSWORD`, y `DB_NAME`, pero los valores por defecto funcionarán sin problemas. El sistema usará estas credenciales para inicializar la base de datos.

### 3. Construye y Ejecuta los Contenedores

Desde el directorio raíz del proyecto, ejecuta el siguiente comando:

```bash
sudo docker compose up --build -d
```

Este comando realizará las siguientes acciones:
1.  Construirá la imagen Docker para la aplicación Flask.
2.  Descargará las imágenes oficiales de `pgvector/pgvector` para la base de datos y `ollama/ollama` para el modelo de lenguaje.
3.  Iniciará todos los servicios (API, base de datos y Ollama).
4.  **Restaurará automáticamente la base de datos** usando el archivo `backup.dump`.
5.  Descargará los modelos de IA requeridos (`gemma3:4b` y `nomic-embed-text`) a través del servicio de configuración de Ollama.
6.  Creará volúmenes de Docker (`postgres_data` y `ollama_data`) para persistir los datos de tu base de datos y los modelos de Ollama.

Una vez que el comando finalice, la API estará disponible en `http://localhost:8000`.

## Uso

### Endpoints de la API

#### `/query`

-   **Método:** POST
-   **URL:** `http://localhost:8000/query`
-   **Cuerpo (Body):**
    ```json
    {
        "prompt": "Tu consulta aquí"
    }
    ```

## Cómo Detener la Aplicación

Para detener todos los contenedores, ejecuta:

```bash
sudo docker compose down
```

Si además deseas eliminar los volúmenes (y con ello, todos los datos persistidos de la base de datos y los modelos de Ollama), utiliza:
```bash
sudo docker compose down -v
```
