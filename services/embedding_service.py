 
"""
Embedding generation for the Agentic AI Platform.

Architecture:
    LangChain RAG
        ↓
    Embedding service
        ↓
    Gemini gemini-embedding-001
        ↓
    768-dimensional vector
        ↓
    PostgreSQL + pgvector

The existing embedding API is intentionally preserved so the current
ingestion/search pipeline continues to work while LangChain is introduced.

IMPORTANT:
- GEMINI_API_KEY must be configured in .env.
- gemini-embedding-001 is used instead of the retired text-embedding-004.
- The vector dimension must remain 768 because the database currently
  uses Vector(768).
- Do not change EMBEDDING_DIM without regenerating all existing vectors.
"""

import random
import hashlib
from typing import List

from core.config import settings


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "models/gemini-embedding-001"

# Must match models.knowledge.EMBEDDING_DIM and the pgvector column.
EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Gemini SDK
# ---------------------------------------------------------------------------

try:
    import google.generativeai as genai

    _SDK_AVAILABLE = True
except ImportError:
    genai = None
    _SDK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mock embedding
# ---------------------------------------------------------------------------

def _mock_embedding(text: str) -> List[float]:
    """
    Deterministic pseudo-embedding used when Gemini is unavailable.

    This allows local development and pipeline testing without requiring
    a real Gemini API key.

    The vectors are NOT semantically meaningful and should not be used
    for production RAG quality evaluation.
    """
    rng = random.Random(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big"))

    return [
        rng.uniform(-1, 1)
        for _ in range(EMBEDDING_DIM)
    ]


# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

def _configure_gemini() -> None:
    """
    Configure the Gemini SDK using the platform API key.
    """
    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    if not _SDK_AVAILABLE:
        raise RuntimeError(
            "google-generativeai package is not installed."
        )

    genai.configure(
        api_key=settings.GEMINI_API_KEY
    )


# ---------------------------------------------------------------------------
# Single embedding
# ---------------------------------------------------------------------------

def embed_text(
    text: str,
    task_type: str = "retrieval_document",
) -> List[float]:
    """
    Generate one embedding.

    task_type:
        retrieval_document
            Used when storing document chunks.

        retrieval_query
            Used when embedding a user's search/query text.

    Returns:
        A vector containing exactly EMBEDDING_DIM values.
    """

    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM

    # Keep the existing mock path for local/testing environments.
    if not settings.GEMINI_API_KEY or not _SDK_AVAILABLE:
        if not settings.ALLOW_MOCK_EMBEDDINGS or settings.ENV == "production":
            raise RuntimeError("Embedding provider is not configured; document indexing is unavailable.")
        return _mock_embedding(text)

    _configure_gemini()

    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
        output_dimensionality=EMBEDDING_DIM,
        request_options={"timeout": settings.EXECUTION_TIMEOUT_SECONDS},
    )

    embedding = result["embedding"]

    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"Gemini returned an embedding with dimension "
            f"{len(embedding)}, expected {EMBEDDING_DIM}."
        )

    return embedding


# ---------------------------------------------------------------------------
# Batch embedding
# ---------------------------------------------------------------------------

def embed_batch(
    texts: List[str],
    task_type: str = "retrieval_document",
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.

    The current Gemini SDK path intentionally processes the texts one at
    a time. This keeps behavior predictable and preserves compatibility
    with the existing ingestion pipeline.

    Returns:
        A list of vectors, each containing EMBEDDING_DIM values.
    """

    if not texts:
        return []

    if not settings.GEMINI_API_KEY or not _SDK_AVAILABLE:
        if not settings.ALLOW_MOCK_EMBEDDINGS or settings.ENV == "production":
            raise RuntimeError("Embedding provider is not configured; document indexing is unavailable.")
        return [
            _mock_embedding(text)
            for text in texts
        ]

    _configure_gemini()

    embeddings: List[List[float]] = []

    for text in texts:
        embedding = embed_text(
            text=text,
            task_type=task_type,
        )

        embeddings.append(embedding)

    return embeddings


# ---------------------------------------------------------------------------
# LangChain adapter
# ---------------------------------------------------------------------------

class GeminiEmbeddingFunction:
    """
    Small LangChain-compatible embedding adapter.

    This lets LangChain's RAG components use the existing Gemini embedding
    service without forcing us to rewrite the current embedding pipeline.

    The adapter intentionally delegates to embed_text/embed_batch so there
    remains only one source of truth for Gemini embedding configuration.
    """

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Embed document chunks for vector storage.
        """
        return embed_batch(
            texts=texts,
            task_type="retrieval_document",
        )

    def embed_query(
        self,
        text: str,
    ) -> List[float]:
        """
        Embed a user query for vector retrieval.
        """
        return embed_text(
            text=text,
            task_type="retrieval_query",
        )
 
