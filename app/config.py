import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class Config:
    # Flask settings
    SECRET_KEY = os.getenv("SECRET_KEY", "clipsmith-secret-key-default-2026")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))

    # Hardware & Performance
    DEVICE = os.getenv("DEVICE", "cuda")  # 'cuda' or 'cpu'
    COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")  # 'float16' or 'int8'
    OMP_NUM_THREADS = int(os.getenv("OMP_NUM_THREADS", 4))

    # Speech-to-text (WhisperX)
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

    # LLM (Ollama)
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
    LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", 8192))

    # Video & FFmpeg
    VIDEO_ENCODER = os.getenv("VIDEO_ENCODER", "h264_nvenc")
    VIDEO_PRESET = os.getenv("VIDEO_PRESET", "p4")

    # Storage Paths
    STORAGE_DIR = BASE_DIR / os.getenv("STORAGE_DIR", "storage")
    STORAGE_RESULTS = STORAGE_DIR / "results"
    STORAGE_TEMP = STORAGE_DIR / "temp"

    # Database
    db_env = os.getenv("DATABASE_URL")
    if db_env:
        if db_env.startswith("sqlite:///") and not os.path.isabs(db_env.replace("sqlite:///", "")):
            # convert relative sqlite path to absolute
            rel_path = db_env.replace("sqlite:///", "")
            SQLALCHEMY_DATABASE_URI = f"sqlite:///{BASE_DIR / rel_path}"
        else:
            SQLALCHEMY_DATABASE_URI = db_env
    else:
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{STORAGE_DIR / 'clipsmith.db'}"

    SQLALCHEMY_TRACK_MODIFICATIONS = False
