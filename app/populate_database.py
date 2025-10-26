import os
import argparse
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from sqlalchemy import create_engine, text
from pgvector.sqlalchemy import Vector
import pandas as pd
from app.db_utils import get_db_engine

DATA_PATH = "app/data"
EMBEDDING_MODEL = "nomic-embed-text"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create schema and tables
    engine = get_db_engine()
    with engine.connect() as connection:
        connection.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        connection.execute(text("""
        CREATE TABLE IF NOT EXISTS licitaciones (
            id SERIAL PRIMARY KEY,
            nombre_licitacion VARCHAR(255) UNIQUE
        )
        """))
        connection.execute(text("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            id_licitacion INTEGER REFERENCES licitaciones(id),
            texto TEXT,
            texto_embedding VECTOR(768)
        )
        """))
        connection.commit()

    # Load documents and add to database
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_postgres(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ".", "!", "?"],
    )
    return text_splitter.split_documents(documents)

def add_to_postgres(chunks):
    engine = get_db_engine()
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)

    with engine.connect() as connection:
        for chunk in chunks:
            source = chunk.metadata.get("source")
            if source:
                # Insert licitacion if it doesn't exist
                licitacion_name = os.path.basename(source)
                res = connection.execute(text("SELECT id FROM licitaciones WHERE nombre_licitacion = :name"), {"name": licitacion_name}).fetchone()
                if res:
                    licitacion_id = res[0]
                else:
                    result = connection.execute(text("INSERT INTO licitaciones (nombre_licitacion) VALUES (:name) RETURNING id"), {"name": licitacion_name})
                    licitacion_id = result.fetchone()[0]

                # Insert chunk
                embedding = embedding_function.embed_query(chunk.page_content)
                connection.execute(
                    text("""
                    INSERT INTO chunks (id_licitacion, texto, texto_embedding)
                    VALUES (:id_licitacion, :texto, :embedding)
                    """),
                    {"id_licitacion": licitacion_id, "texto": chunk.page_content, "embedding": embedding}
                )
        connection.commit()
    print("✅ Documents added to PostgreSQL")

def clear_database():
    engine = get_db_engine()
    with engine.connect() as connection:
        connection.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS licitaciones CASCADE"))
        connection.commit()

if __name__ == "__main__":
    main()
