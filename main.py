from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.database import Base, engine, SessionLocal

# Import models so SQLAlchemy knows about them before create_all() runs
from models import agent, execution, knowledge, orchestration, user  # noqa: F401

from api import (
    health, agents, executions, discovery, knowledge as knowledge_api,
    orchestration as orchestration_api, auth,
)
 

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend / control layer for the Agentic AI Platform - "
                 "Agent Registry, Execution Engine, Discovery & Auth.",
    version="1.0.0",
)

# Allow the React frontend (Sumith's layer / Phase 8) to call this API.
# >>> WHAT YOU NEED TO CHANGE <<<
# Replace "*" with the actual frontend origin(s) in production, e.g.
# ["http://localhost:3000", "https://your-frontend-domain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Create all tables on startup.
# This is fine for development. For production, switch to Alembic migrations
# (see README "Going to production" section).
#
# >>> RE-ENABLED FOR PHASE 5A <<<
# This block (table creation, including pgvector setup) was commented
# out in the codebase this phase started from -- meaning NO tables at
# all were being created on startup, not just the new auth ones. That
# wasn't something Phase 5A broke; it was already off. Re-enabling it
# here since without it, `users` and `refresh_tokens` (and every other
# existing table) never get created and every DB-touching endpoint
# fails with "relation does not exist."
#
# >>> WHAT YOU NEED TO DO FIRST (one-time, RAG / Knowledge Base feature) <<<
# The pgvector Postgres EXTENSION must be installed at the OS/Postgres level
# before this will work -- `pip install pgvector` only installs the PYTHON
# client library, not the actual Postgres extension binary. On most
# systems: `sudo apt install postgresql-16-pgvector` (or the matching
# version for your Postgres), then the CREATE EXTENSION below enables it
# for this specific database. If the extension isn't installed at the OS
# level, the line below will fail with "could not open extension control
# file" -- that error means you still need the OS-level install step.
with engine.connect() as _conn:
    from sqlalchemy import text as _sql_text
    if engine.dialect.name == "postgresql":
        _conn.execute(_sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
        _conn.commit()

Base.metadata.create_all(bind=engine)

 

# --- Routers ---
# NOTE: discovery.router must be included BEFORE agents.router.
# Both define paths under /api/agents, and agents.router has a catch-all
# /api/agents/{agent_id} route that would otherwise swallow /api/agents/discover.
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(discovery.router)
app.include_router(agents.router)
app.include_router(executions.router)
app.include_router(knowledge_api.router)
app.include_router(orchestration_api.router)


@app.get("/")
def root():
    return {
        "message": "Agentic AI Platform backend is running",
        "docs": "/docs",
    }