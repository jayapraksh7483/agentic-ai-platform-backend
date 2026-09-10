from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.database import Base, engine

# Import models so SQLAlchemy knows about them before create_all() runs
from models import (
    agent,
    execution,
    knowledge,
    orchestration,
    user,
    conversation,
    message,
)

from api import (
    health,
    agents,
    executions,
    discovery,
    knowledge as knowledge_api,
    orchestration as orchestration_api,
    auth,
    conversations,
)


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Backend / control layer for the Agentic AI Platform - "
        "Agent Registry, Execution Engine, Discovery, "
        "Orchestration, Authentication & Persistent Conversations."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================
#
# Allow the React frontend to call this API.
#
# For production, replace "*" with the actual frontend origin(s),
# for example:
#
# allow_origins=[
#     "http://localhost:3000",
#     "https://your-frontend-domain.com",
# ]
#
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Database initialization
# ============================================================
#
# Create the pgvector extension when PostgreSQL is being used.
#
# NOTE:
# `pip install pgvector` installs the Python package/client.
# It does NOT install the PostgreSQL extension itself.
#
# The PostgreSQL server must already have the vector extension
# installed at the OS/PostgreSQL level.
#

with engine.connect() as _conn:
    from sqlalchemy import text as _sql_text

    if engine.dialect.name == "postgresql":
        _conn.execute(
            _sql_text(
                "CREATE EXTENSION IF NOT EXISTS vector"
            )
        )
        _conn.commit()


# Create all SQLAlchemy tables.
#
# Development approach:
#   Base.metadata.create_all()
#
# Production approach:
#   Use Alembic migrations.
#
# Phase 5C models included here:
#   conversations
#   messages
#
Base.metadata.create_all(bind=engine)


# ============================================================
# API Routers
# ============================================================
#
# IMPORTANT:
# discovery.router must be registered BEFORE agents.router.
#
# Both use /api/agents paths and agents.router contains:
#
#   /api/agents/{agent_id}
#
# which could otherwise catch:
#
#   /api/agents/discover
#
# ============================================================

app.include_router(health.router)

# Authentication
app.include_router(auth.router)

# Agent discovery must come before agent catch-all routes
app.include_router(discovery.router)

# Agent registry / management / execution
app.include_router(agents.router)
app.include_router(executions.router)

# Knowledge Base / RAG
app.include_router(knowledge_api.router)

# Manager Agent / Dynamic Orchestration
app.include_router(orchestration_api.router)

# Persistent Conversations + Messages
app.include_router(conversations.router)


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Agentic AI Platform backend is running",
        "docs": "/docs",
    }