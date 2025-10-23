# Corrective-RAG-Server-in-Flask

This project is a Flask-based server that implements a hybrid RAG (Retrieval-Augmented Generation) system. It can query both a PostgreSQL database for structured data and a ChromaDB vector store for unstructured data. This project is fully containerized with Docker for easy setup and deployment.

## Prerequisites

-   Docker
-   Docker Compose

## Setup

### 1. Configure Environment Variables

1.  Create a file named `.env` in the root of the project by copying the example file:
    ```bash
    cp .env.example .env
    ```
2.  Open the `.env` file and **edit the variables** to match your desired configuration. At a minimum, you must set `TAVILY_API_KEY`. The database credentials will be used by Docker Compose to initialize the PostgreSQL container.

    ```
    # PostgreSQL Settings
    DB_USER=user
    DB_PASSWORD=password
    DB_HOST=db
    DB_PORT=5432
    DB_NAME=rag_db

    # Tavily API Key for web search
    TAVILY_API_KEY=your_tavily_api_key
    ```
    **Note:** The `DB_HOST` must be `db`, which is the service name of the PostgreSQL container in `docker-compose.yml`.

### 2. Build and Run the Containers

From the root of the project, run the following command:

```bash
docker-compose up --build
```

This command will:
1.  Build the Docker image for the Flask application.
2.  Download the official PostgreSQL image.
3.  Start both the application and the database containers.
4.  Create two Docker volumes (`postgres_data` and `chroma_data`) to persist your data.

The API will be available at `http://localhost:8000`.

### 3. Set Up the Database Tables

1.  After starting the containers, you need to create the necessary tables in the PostgreSQL database. You can connect to the database using a client like `psql` or DBeaver with the credentials from your `.env` file.
2.  Execute the SQL statements required to create the schema shown in `image.png`.

### 4. Populate the Vector Store

1.  Place your PDF documents in the `app/data` directory.
2.  Run the `populate_database.py` script **inside the running container**:
    ```bash
    docker-compose exec api python -m app.populate_database
    ```

## Usage

### API Endpoints

#### `/query`

-   **Method:** POST
-   **URL:** `http://localhost:8000/query`
-   **Body:**
    ```json
    {
        "prompt": "Your query here"
    }
    ```

#### `/upload`

-   **Method:** POST
-   **URL:** `http://localhost:8000/upload`
-   **Body:** `multipart/form-data` with a `file` field.

## Stopping the Application

To stop the containers, press `Ctrl+C` in the terminal where `docker-compose` is running, or run:

```bash
docker-compose down
```
