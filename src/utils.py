import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Paths:
    root: str = "."
    data_dir: str = "data"
    raw_dir: str = os.path.join("data", "raw")
    interim_dir: str = os.path.join("data", "interim")
    processed_dir: str = os.path.join("data", "processed")
    models_dir: str = "models"
    reports_dir: str = os.path.join("reports", "figures")

def ensure_dirs(p: Paths):
    os.makedirs(p.data_dir, exist_ok=True)
    os.makedirs(p.raw_dir, exist_ok=True)
    os.makedirs(p.interim_dir, exist_ok=True)
    os.makedirs(p.processed_dir, exist_ok=True)
    os.makedirs(p.models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(p.reports_dir), exist_ok=True)

def get_env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)

def get_env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def db_connect():

    url = get_env_str("DATABASE_URL", "")
    if not url:
        return None
    try:
        from sqlalchemy import create_engine
        engine = create_engine(url)
        conn = engine.connect()
        return conn
    except Exception as e:
        print(f"No se pudo conectar a la BD: {e}")
        return None
