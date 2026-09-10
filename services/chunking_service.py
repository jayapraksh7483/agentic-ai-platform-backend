"""
Text chunking for RAG (Sumith spec: 500 tokens, 50 token overlap).

NOTE ON TOKEN COUNTING: this uses a simple word-count approximation
(1 "token" ~= 1 word-ish unit) rather than a real tokenizer, since the
project doesn't otherwise depend on tiktoken or a model-specific
tokenizer. This is close enough for chunk-sizing purposes -- the exact
boundary doesn't need to be precise, it just needs to keep chunks a
reasonable, consistent size with overlap for context continuity. If you
want exact token counts matching a specific model's tokenizer, install
`tiktoken` and swap _approximate_tokenize() for a real tokenizer's
encode/decode.
"""
from typing import List


def _approximate_tokenize(text: str) -> List[str]:
    return text.split()


def _detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks. Each chunk has `chunk_size`
    tokens (approx.), and each subsequent chunk starts `chunk_size -
    overlap` tokens after the previous one's start, so `overlap` tokens
    of context carry over between adjacent chunks.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    tokens = _approximate_tokenize(text)
    if not tokens:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(_detokenize(chunk_tokens))
        if end >= len(tokens):
            break
        start += step

    return chunks