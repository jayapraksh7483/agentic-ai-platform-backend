from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # =========================================================
    # APP
    # =========================================================

    APP_NAME: str = "Agentic AI Platform - Backend"
    ENV: str = "development"

    # =========================================================
    # DATABASE
    # =========================================================

    DATABASE_URL: str = (
        "postgresql://postgres:postgres@localhost:5432/project_agent"
    )

    # =========================================================
    # AUTH / JWT
    # =========================================================

    SECRET_KEY: str = "CHANGE_ME_TO_A_RANDOM_SECRET"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # =========================================================
    # GEMINI
    # =========================================================

    GEMINI_API_KEY: str = ""

    GEMINI_MODEL_1: str = "gemini-3.7-flash"
    GEMINI_MODEL_2: str = "gemini-3.6-flash"
    GEMINI_MODEL_3: str = "gemini-3.5-flash-lite"

    GEMINI_DEFAULT_MODEL: str = "gemini-3.7-flash"

    # =========================================================
    # GROQ
    # =========================================================

    GROQ_API_KEY: str = ""

    GROQ_MODEL_1: str = "llama-3.1-8b-instant"
    GROQ_MODEL_2: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_3: str = "openai/gpt-oss-120b"

    GROQ_DEFAULT_MODEL: str = "openai/gpt-oss-120b"

    # =========================================================
    # ANTHROPIC
    # =========================================================

    ANTHROPIC_API_KEY: str = ""

    ANTHROPIC_MODEL_1: str = "claude-opus-4-6"
    ANTHROPIC_MODEL_2: str = "claude-sonnet-4-6"
    ANTHROPIC_MODEL_3: str = "claude-haiku-4-5-20251001"

    ANTHROPIC_DEFAULT_MODEL: str = "claude-sonnet-4-6"

    # =========================================================
    # OPENAI
    # =========================================================

    OPENAI_API_KEY: str = ""

    OPENAI_MODEL_1: str = "gpt-5"
    OPENAI_MODEL_2: str = "gpt-5-mini"
    OPENAI_MODEL_3: str = "gpt-5-nano"

    OPENAI_DEFAULT_MODEL: str = "gpt-5-mini"

    # =========================================================
    # EXECUTION ENGINE
    # =========================================================

    EXECUTION_TIMEOUT_SECONDS: int = 30
    MAX_CONCURRENT_TASKS: int = 10

    # =========================================================
    # MANAGER AGENT / ORCHESTRATION
    # =========================================================

    MAX_ORCHESTRATION_STEPS: int = 10
    MAX_ORCHESTRATION_RETRIES: int = 2
    MAX_ORCHESTRATION_EXECUTION_TIME: int = 120

    MAX_ORCHESTRATION_TASKS: int = 20
    MAX_DEPENDENCY_DEPTH: int = 8

    MAX_CONTEXT_CHARS: int = 16000

    MAX_RETRIEVED_CHUNKS: int = 20

    ALLOW_MOCK_EMBEDDINGS: bool = False

    # =========================================================
    # GOOGLE OAUTH
    # =========================================================

    # Loaded securely from .env
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    GOOGLE_REDIRECT_URI: str = (
        "http://localhost:9000/api/oauth/google/callback"
    )

    GOOGLE_OAUTH_SCOPES: str = (
        "openid "
        "https://www.googleapis.com/auth/userinfo.email "
        "https://www.googleapis.com/auth/gmail.readonly "
        "https://www.googleapis.com/auth/gmail.send "
        "https://www.googleapis.com/auth/drive.readonly "
        "https://www.googleapis.com/auth/calendar "
        "https://www.googleapis.com/auth/spreadsheets"
    )

    # Fernet encryption key loaded securely from .env
    TOKEN_ENCRYPTION_KEY: str = ""

    # =========================================================
    # PYDANTIC SETTINGS
    # =========================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()