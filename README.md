# Corrective-RAG-Server-in-Flask

This project is a Flask-based server that implements a hybrid RAG (Retrieval-Augmented Generation) system. It can query both a PostgreSQL database for structured data and a ChromaDB vector store for unstructured data.

## Setup

### 1. Install Dependencies

```bash
pip install -r app/requirements.txt
```

### 2. Set Up PostgreSQL

1.  Install PostgreSQL.
2.  Create a database.
3.  Create the tables defined in the schema (see `image.png`).

### 3. Configure Environment Variables

Create a `.env` file in the root of the project and add the following environment variables:

```
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
DB_PORT=your_db_port
DB_NAME=your_db_name
TAVILY_API_KEY=your_tavily_api_key
```

### 4. Populate the Vector Store

To populate the ChromaDB vector store, place your PDF documents in the `app/data` directory and run the following command:

```bash
python -m app.populate_database
```

### 5. Run the Server

```bash
python main.py
```

## API Endpoints

### `/query`

*   **Method:** POST
*   **Body:**
    ```json
    {
        "prompt": "Your query here"
    }
    ```
*   **Response:**
    ```json
    {
        "answer": "The answer to your query.",
        "sources": ["source1.pdf", "source2.pdf"],
        "web_urls": ["url1", "url2"],
        "rewritten_question": "The rewritten question.",
        "relevance_check": "yes" | "no",
        "confidence": "high" | "low",
        "sql_result": [...]
    }
    ```

### `/upload`

*   **Method:** POST
*   **Body:**
    ```
    multipart/form-data
    ```
    with a `file` field containing the PDF file to upload.
*   **Response:**
    ```json
    {
        "message": "File uploaded successfully.",
        "file_path": "app/data/your_file.pdf"
    }
    ```
