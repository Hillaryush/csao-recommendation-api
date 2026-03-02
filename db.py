from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback for local development
if not DATABASE_URL:
    DATABASE_URL = "postgresql://csao_db_user:3QNahFHSfp4EQL4IanfAu2Gbh2zRPCcj@dpg-d6iin6cr85hc73a3kfv0-a.singapore-postgres.render.com/csao_db"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)