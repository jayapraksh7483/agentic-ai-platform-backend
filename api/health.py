from typing import Dict, Any

from fastapi import APIRouter, Depends
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session

from core.database import get_db, engine
from core.config import settings

router = APIRouter(tags=["Health"])

# Tables the app expects to exist. If any are missing, something went
# wrong with startup's create_all() or a manual migration is incomplete.
EXPECTED_TABLES = [
    "agents", "agent_versions", "agent_capabilities",
    "agent_executions",
    "knowledge_bases", "knowledge_documents", "knowledge_chunks",
]


def _check_database(db: Session) -> Dict[str, Any]:
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _check_tables(db: Session) -> Dict[str, Any]:
    try:
        existing = set(inspect(engine).get_table_names())
        missing = [t for t in EXPECTED_TABLES if t not in existing]
        if missing:
            return {"status": "error", "detail": f"Missing tables: {', '.join(missing)}"}
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _check_pgvector(db: Session) -> Dict[str, Any]:
    if engine.dialect.name != "postgresql":
        return {"status": "skipped", "detail": f"Not applicable on {engine.dialect.name} (dev/test only)"}
    try:
        result = db.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).first()
        if result:
            return {"status": "ok"}
        return {
            "status": "error",
            "detail": "pgvector extension not enabled. Run: CREATE EXTENSION IF NOT EXISTS vector;",
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _check_llm_providers() -> Dict[str, Any]:
    providers = {}
    providers["gemini"] = "configured" if settings.GEMINI_API_KEY else "missing (mock responses only)"
    providers["groq"] = "configured" if settings.GROQ_API_KEY else "missing (mock responses only)"

    # At least one real provider should be configured for anything beyond
    # local testing -- flag it clearly if neither is set, without failing
    # the whole health check (mock mode is a valid dev setup).
    status = "ok" if (settings.GEMINI_API_KEY or settings.GROQ_API_KEY) else "warning"
    return {"status": status, "providers": providers}


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """
    Liveness + dependency check. Keeps the original top-level
    {"status", "database"} shape for backward compatibility with anything
    already parsing those two fields, and adds a "checks" breakdown for
    everything else (pgvector, tables, LLM provider config).
    """
    database_check = _check_database(db)
    tables_check = _check_tables(db) if database_check["status"] == "ok" else {"status": "skipped", "detail": "database check failed"}
    pgvector_check = _check_pgvector(db) if database_check["status"] == "ok" else {"status": "skipped", "detail": "database check failed"}
    llm_check = _check_llm_providers()

    checks = {
        "database": database_check,
        "tables": tables_check,
        "pgvector": pgvector_check,
        "llm_providers": llm_check,
    }

    # Overall status: "ok" only if every non-skipped check is "ok".
    # A "warning" (e.g. no LLM key configured) doesn't fail the whole
    # check -- it's a valid (if limited) dev configuration.
    hard_failures = [name for name, c in checks.items() if c["status"] == "error"]
    if hard_failures:
        overall = "degraded"
    elif any(c["status"] == "warning" for c in checks.values()):
        overall = "ok_with_warnings"
    else:
        overall = "ok"

    return {
        "status": overall,
        # Kept for backward compatibility with the original spec's exact
        # {"status": "ok", "database": "connected"} shape.
        "database": "connected" if database_check["status"] == "ok" else "error",
        "checks": checks,
    }


@router.get("/health/live")
def liveness():
    """Bare liveness probe -- responds instantly, no DB/dependency checks.
    Use this for container orchestration liveness probes (e.g. Docker
    healthcheck, Kubernetes livenessProbe) where you only want to know
    the process is up and responding, not whether its dependencies are
    healthy (that's what /health is for)."""
    return {"status": "alive"}