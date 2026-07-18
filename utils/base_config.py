import os
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


class AppConfig:
    # Ollama Service
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # LLM Model Settings
    DEBUG_LLM_MODEL: str = os.getenv("DEBUG_LLM_MODEL")
    LIGHT_DEV_LLM_MODEL: str = os.getenv("LIGHT_DEV_LLM_MODEL")
    SHOWCASE_LLM_MODEL: str = os.getenv("SHOWCASE_LLM_MODEL")
    LLM_TEMPERATURE: str = os.getenv("LLM_TEMPERATURE")

    # Retrieval Models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL")
    RERANK_TOP_N: str = os.getenv("RERANK_TOP_N")

    # Persistent Storage Paths
    CHROMA_STORAGE_PATH: str = os.getenv("CHROMA_STORAGE_PATH")
    LOG_SAVE_PATH: str = os.getenv("LOG_SAVE_PATH")
    TEST_DOCUMENT_FOLDER: str = os.getenv("TEST_DOCUMENT_FOLDER")
    CHAT_SESSION_STORAGE: str = os.getenv("CHAT_SESSION_STORAGE")

    @classmethod
    def get_all_configs(cls) -> Dict[str, str]:
        """Return all config values as dictionary for debugging"""
        return {
            "OLLAMA_BASE_URL": cls.OLLAMA_BASE_URL,
            "DEBUG_LLM_MODEL": cls.DEBUG_LLM_MODEL,
            "LIGHT_DEV_LLM_MODEL": cls.LIGHT_DEV_LLM_MODEL,
            "SHOWCASE_LLM_MODEL": cls.SHOWCASE_LLM_MODEL,
            "EMBEDDING_MODEL": cls.EMBEDDING_MODEL,
            "RERANK_MODEL": cls.RERANK_MODEL,
            "RERANK_TOP_N": cls.RERANK_TOP_N,
            "CHROMA_STORAGE_PATH": cls.CHROMA_STORAGE_PATH,
            "LOG_SAVE_PATH": cls.LOG_SAVE_PATH,
            "TEST_DOCUMENT_FOLDER": cls.TEST_DOCUMENT_FOLDER,
            "CHAT_SESSION_STORAGE": cls.CHAT_SESSION_STORAGE,
            "LLM_TEMPERATURE": cls.LLM_TEMPERATURE,
        }


app_config = AppConfig()
