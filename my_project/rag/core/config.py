from pathlib import Path


class Settings:
    BASE_DIR = Path(__file__).resolve().parents[2]
    VECTOR_DATABASE_PATH = BASE_DIR / "vectorstore"
    DATA_PATH = BASE_DIR / "files"
    CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "langgraph.db"

    HF_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
    GOOGLE_EMBEDDER = "gemini-embedding-001"

    GOOGLE_MODEL_LITE = "gemini-2.5-flash-lite"
    GOOGLE_MODEL_HEAVY = "gemini-2.5-flash"
    HF_LLM = "meta-llama/Llama-3.2-3B-Instruct"

    API_KEY = "Chatbot_API_Key_220405"

settings = Settings()