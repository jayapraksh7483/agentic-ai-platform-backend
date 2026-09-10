"""
Embedding generation via the Gemini Embedding API.

>>> WHAT YOU NEED TO CHANGE <<<
GEMINI_API_KEY must be set in .env (same key used for chat completions --
this is a separate endpoint on the same API, not a separate product).

>>> IMPORTANT MODEL UPDATE (2026) <<<
text-embedding-004 was permanently shut down by Google on January 14, 2026.
This now uses gemini-embedding-001, the current stable replacement.
gemini-embedding-001 defaults to 3072-dimensional output, so we explicitly
request 768 dimensions via output_dimensionality to match
models.knowledge.EMBEDDING_DIM. If you ever change EMBEDDING_DIM, this
output_dimensionality value AND models.knowledge.EMBEDDING_DIM must change
together, and every existing chunk's embedding needs to be regenerated
(you cannot mix vector dimensions in the same pgvector column).
"""
import random
from typing import List, Optional

from core.config import settings

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768

try:
    import google.generativeai as genai
    _SDK_AVAILABLE = True
except ImportError:
    genai = None
    _SDK_AVAILABLE = False


def _mock_embedding(text: str) -> List[float]:
    """Deterministic pseudo-embedding (seeded by text hash) so the same
    input always produces the same mock vector -- lets you test the
    chunking/storage/search pipeline end-to-end without a real API key,
    though similarity scores won't be meaningful."""
    rng = random.Random(hash(text) % (2**32))
    return [rng.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]


def embed_text(text: str, task_type: str = "retrieval_document") -> List[float]:
    """
    task_type should be "retrieval_document" when embedding chunks to
    store, and "retrieval_query" when embedding a search query -- Gemini's
    embedding model is trained to produce better retrieval results when
    told which side of the search it's embedding.
    """
    if not settings.GEMINI_API_KEY or not _SDK_AVAILABLE:
        return _mock_embedding(text)

    genai.configure(api_key=settings.GEMINI_API_KEY)
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
        output_dimensionality=EMBEDDING_DIM,
    )
    return result["embedding"]


def embed_batch(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """Embeds multiple texts. Falls back to one-at-a-time if the SDK/key
    isn't available (mock path), otherwise uses the batch-friendly form
    of embed_content where supported."""
    if not settings.GEMINI_API_KEY or not _SDK_AVAILABLE:
        return [_mock_embedding(t) for t in texts]

    genai.configure(api_key=settings.GEMINI_API_KEY)
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        )
        embeddings.append(result["embedding"])
    return embeddings