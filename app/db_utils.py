import os
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

def get_db_engine():
    """
    Creates and returns a SQLAlchemy engine for connecting to the PostgreSQL database.
    Reads database credentials from environment variables.
    """
    url_object = URL.create(
        "postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
    )
    engine = create_engine(url_object)
    return engine
