# src/database/connection.py

import os
from sqlalchemy import create_engine


def get_engine():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME", "ecommerce")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")

    url = (
        f"postgresql+psycopg2://{user}:{password}"
        f"@{host}:{port}/{database}"
    )

    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600
    )

    return engine