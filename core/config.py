from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "Agentic AI Platform - Backend"
    ENV: str = "development"

    # --- Database ---
    DATABASE_URL: str = (
        "postgresql://postgres:postgres@localhost:5432/project_agent"
    )

    # --- Auth / JWT ---
    SECRET_KEY: str = "CHANGE_ME_TO_A_RANDOM_SECRET"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24
    # Phase 5A addition -- refresh tokens are DB-backed (models/user.py
    # RefreshToken), not JWTs, so this only controls how long a session
    # row stays valid, not anything encoded in a token.
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
    # EXECUTION ENGINE
    # =========================================================

    EXECUTION_TIMEOUT_SECONDS: int = 30

    # =========================================================
    # MANAGER AGENT / ORCHESTRATION (Phase 4)
    # =========================================================

    # Hard ceiling on the number of plan "waves" the Manager Agent will
    # execute for a single orchestration request. Prevents an uncontrolled
    # replanning/execution loop.
    MAX_ORCHESTRATION_STEPS: int = 10

    # Per-step retry budget for a step whose underlying agent execution
    # fails (e.g. transient LLM/tool error). 0 = no retries.
    MAX_ORCHESTRATION_RETRIES: int = 2

    # Wall-clock budget for a single orchestration request, in seconds.
    MAX_ORCHESTRATION_EXECUTION_TIME: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()