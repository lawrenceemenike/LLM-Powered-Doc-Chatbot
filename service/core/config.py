import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "LLM-Powered Doc Chatbot"
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", 8000))
    CHROMA_COLLECTION: str = "documents"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt2" # Placeholder/Default
    
    class Config:
        env_file = ".env"

settings = Settings()
