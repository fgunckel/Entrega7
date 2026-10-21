"""Application settings and configurable LLM provider."""

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Ignorar variables de entorno vacías (p. ej. GROQ_API_KEY="" en la
        # sesión): sin esto, el valor vacío pisa la clave real del archivo .env.
        env_ignore_empty=True,
    )

    groq_api_key: SecretStr | None = None
    llm_provider: Literal["groq"] = Field(default="groq", alias="ADA_LLM_PROVIDER")
    llm_model: str = Field(default="openai/gpt-oss-20b", alias="ADA_LLM_MODEL")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, alias="ADA_TEMPERATURE")
    max_steps: int = Field(default=8, ge=1, le=30, alias="ADA_MAX_STEPS")
    max_retries: int = Field(default=2, ge=0, le=5, alias="ADA_MAX_RETRIES")
    top_k: int = Field(default=5, ge=1, le=20, alias="ADA_TOP_K")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="ADA_EMBEDDING_MODEL",
    )

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def _empty_key_becomes_none(cls, value: Any) -> Any:
        """Tratar una clave vacía (p. ej. variable de entorno sin valor) como ausente.

        Las variables de entorno tienen prioridad sobre el archivo .env en
        pydantic-settings; una GROQ_API_KEY vacía en la sesión pisa la clave
        real del archivo. Normalizamos: cadena vacía o solo espacios -> None.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Return cached settings."""
    return Settings()


def get_chat_model():
    """Create the configured chat model.

    Provider selection is isolated here so graph nodes do not depend directly on Groq.
    """
    settings = get_settings()

    if settings.llm_provider == "groq":
        if not settings.groq_api_key or not settings.groq_api_key.get_secret_value().strip():
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to .env or export it in your environment."
            )

        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key.get_secret_value(),
            temperature=settings.temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
