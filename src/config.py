import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "AI-Powered University CBT Exam Preparation Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Groq LLM Configuration
    GROQ_API_KEY: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    GROQ_TEMPERATURE: float = 0.2

    # Pinecone Vector Store Configuration
    PINECONE_API_KEY: Optional[str] = Field(default=None, alias="PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = Field(default="us-east-1", alias="PINECONE_ENVIRONMENT")
    PINECONE_INDEX: str = Field(default="exam", alias="PINECONE_INDEX")

    # Embeddings Configuration
    EMBEDDING_PROVIDER: Literal["fastembed", "sentence-transformers", "openai", "local"] = Field(
        default="fastembed", alias="EMBEDDING_PROVIDER"
    )
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    OPENAI_API_KEY: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    # Persistence Database (SQLite or PostgreSQL)
    DATABASE_URL: str = Field(
        default=f"sqlite:///{BASE_DIR / 'cbt_exam.db'}",
        alias="DATABASE_URL",
    )

    # Document & Course Paths
    COURSES_DIR: Path = Field(default=BASE_DIR / "courses", alias="COURSES_DIR")
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 80

    # CBT Defaults
    DEFAULT_EXAM_DURATION_MINUTES: int = 45
    PASSING_PERCENTAGE: float = 50.0
    WEAK_TOPIC_THRESHOLD: float = 60.0
    DEFAULT_QUESTIONS_COUNT: int = 15

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_mode(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"development", "dev"}:
                return True
        return value

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def normalize_database_url(cls, value):
        if not isinstance(value, str):
            return value

        normalized = value.strip().strip("'\"")
        if normalized.startswith("postgres://"):
            return normalized.replace("postgres://", "postgresql://", 1)
        return normalized


# Instantiate global settings
settings = Settings()
