# db_connect.py
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")

# SQLAlchemy URI for MySQL with PyMySQL driver
DB_URI = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DB_URI, pool_pre_ping=True)

def test_conn():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT NOW()")).fetchone()
        print("Connected, server time:", result[0])
        # list tables
        tbls = conn.execute(text("SHOW TABLES")).fetchall()
        print("Tables:", tbls)

if __name__ == "__main__":
    test_conn()
