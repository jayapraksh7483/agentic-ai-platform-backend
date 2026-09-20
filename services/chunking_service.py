
"""
Text chunking for the LangChain RAG pipeline.

Target configuration:
    Chunk size: 500 tokens
    Chunk overlap: 50 tokens

LangChain is now responsible for text splitting.

We use RecursiveCharacterTextSplitter because it preserves natural
document boundaries where possible (paragraphs, lines, sentences, etc.)
instead of blindly cutting every N words.

The existing chunk_text() function is preserved so the rest of the
application can migrate without changing its API immediately.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Default RAG chunk configuration
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


# ---------------------------------------------------------------------------
# LangChain splitter
# ---------------------------------------------------------------------------

def _create_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Create the LangChain text splitter.

    The separator hierarchy attempts to preserve:
        1. paragraphs
        2. lines
        3. sentences
        4. words
        5. individual characters

    This gives better RAG chunks than a simple word-count split.
    """

    if chunk_size <= chunk_overlap:
        raise ValueError(
            "chunk_size must be greater than overlap"
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            ", ",
            " ",
            "",
        ],
        length_function=_approximate_token_count,
        is_separator_regex=False,
    )


# ---------------------------------------------------------------------------
# Token-size approximation
# ---------------------------------------------------------------------------

def _approximate_token_count(text: str) -> int:
    """
    Approximate token count.

    This keeps the existing platform behavior where approximately one
    word-like unit is treated as one token.

    Later, if exact model token counting becomes necessary, this function
    can be replaced with a model-specific tokenizer without changing the
    rest of the RAG architecture.
    """

    return len(text.split())


# ---------------------------------------------------------------------------
# Public chunking API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping LangChain RAG chunks.

    Args:
        text:
            Source document text.

        chunk_size:
            Target chunk size, approximately measured in tokens.

        overlap:
            Approximate overlap between adjacent chunks.

    Returns:
        List of chunk strings.
    """

    if not text or not text.strip():
        return []

    splitter = _create_splitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    documents = splitter.create_documents([text])

    return [
        document.page_content
        for document in documents
        if document.page_content.strip()
    ]
