from typing import List, Optional

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from models.knowledge import KnowledgeBase, KnowledgeDocument, KnowledgeChunk, DocumentStatus
from schemas.knowledge import KnowledgeBaseCreate
from services import document_parser, chunking_service, embedding_service

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def create_knowledge_base(db: Session, kb_in: KnowledgeBaseCreate, created_by: Optional[int] = None) -> KnowledgeBase:
    existing = db.query(KnowledgeBase).filter(KnowledgeBase.name == kb_in.name).first()
    if existing:
        raise ValueError(f"A knowledge base named '{kb_in.name}' already exists")

    kb = KnowledgeBase(name=kb_in.name, description=kb_in.description, created_by=created_by)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


def list_knowledge_bases(db: Session) -> List[KnowledgeBase]:
    return db.query(KnowledgeBase).order_by(KnowledgeBase.name).all()


def get_knowledge_base(db: Session, kb_id: str) -> Optional[KnowledgeBase]:
    return db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()


def process_upload(db: Session, kb_id: str, filename: str, file_bytes: bytes) -> KnowledgeDocument:
    """
    Full pipeline (Sumith spec): validate -> extract text -> chunk ->
    generate embeddings -> store vectors -> store document metadata.

    Runs synchronously in the request for simplicity. For large documents
    in production, consider moving this to a background task/queue so the
    upload endpoint doesn't block -- the document row's status field
    (processing/ready/failed) is already designed to support that later
    without an API contract change.
    """
    kb = get_knowledge_base(db, kb_id)
    if not kb:
        raise ValueError(f"Knowledge base '{kb_id}' not found")

    document = KnowledgeDocument(kb_id=kb_id, filename=filename, status=DocumentStatus.PROCESSING)
    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        text = document_parser.extract_text(filename, file_bytes)
        chunks = chunking_service.chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        if not chunks:
            raise ValueError("Document produced no text content to index")

        embeddings = embedding_service.embed_batch(chunks, task_type="retrieval_document")

        for idx, (chunk_text_val, embedding) in enumerate(zip(chunks, embeddings)):
            db.add(KnowledgeChunk(
                doc_id=document.id,
                kb_id=kb_id,
                text=chunk_text_val,
                chunk_index=idx,
                embedding=embedding,
            ))

        document.status = DocumentStatus.READY
        document.chunk_count = len(chunks)
        db.commit()

        kb.document_count = db.query(KnowledgeDocument).filter(
            KnowledgeDocument.kb_id == kb_id, KnowledgeDocument.status == DocumentStatus.READY
        ).count()
        db.commit()

    except (document_parser.UnsupportedFileTypeError, document_parser.DocumentParseError, ValueError) as e:
        document.status = DocumentStatus.FAILED
        document.error_message = str(e)
        db.commit()
        raise

    db.refresh(document)
    return document


def search_knowledge_base(db: Session, kb_id: str, query: str, top_k: int = 5) -> List[dict]:
    """
    Embeds the query and finds the top_k most similar chunks in this
    knowledge base using pgvector's cosine distance operator (<=>).

    Returns [{"text": ..., "score": ...}] per Sumith's exact spec, where
    score is a similarity in [0, 1] (1 = identical), NOT a raw distance.
    """
    kb = get_knowledge_base(db, kb_id)
    if not kb:
        raise ValueError(f"Knowledge base '{kb_id}' not found")

    query_embedding = embedding_service.embed_text(query, task_type="retrieval_query")

    # pgvector's <=> operator returns cosine DISTANCE (0 = identical,
    # 2 = opposite). We convert to a similarity score by 1 - distance/2
    # so smaller distance => score closer to 1, matching Sumith's
    # {"score": 0.9}-style example where higher = more relevant.
    #
    # NOTE: this raw SQL requires the pgvector extension to be installed
    # (CREATE EXTENSION IF NOT EXISTS vector;) and only works on
    # PostgreSQL -- it will not run against SQLite.
    # pgvector's text input format is "[0.1,0.2,...]" -- build it
    # explicitly rather than relying on Python's str(list) spacing.
    query_embedding_literal = "[" + ",".join(str(v) for v in query_embedding) + "]"

    result = db.execute(
        sql_text(
            """
            SELECT text, 1 - (embedding <=> CAST(:query_embedding AS vector)) / 2 AS score
            FROM knowledge_chunks
            WHERE kb_id = :kb_id AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
            """
        ),
        {"query_embedding": query_embedding_literal, "kb_id": kb_id, "top_k": top_k},
    )

    return [{"text": row.text, "score": float(row.score)} for row in result]


STRICT_RAG_INSTRUCTION = """
You must answer ONLY using the information in the "Retrieved context" section
below. Do not use any outside knowledge, even if you know the answer from
general training. If the retrieved context does not contain enough
information to answer the question, respond exactly with:
"I couldn't find this information in the provided documents."
When you do answer from the context, do not fabricate details that are not
present in it.
""".strip()


def build_strict_rag_prompt(base_system_prompt: str, retrieved_chunks: List[dict]) -> str:
    """
    Builds the augmented system prompt for a document/RAG-based agent
    (Sumith spec item 7 - Strict Document-Only RAG). Prepends the original
    agent system prompt with a hard instruction to answer only from the
    retrieved chunks, and includes the chunks themselves as context.

    Called from services/executor_service.py whenever agent.knowledge_base_id
    is set -- see _call_llm().
    """
    if not retrieved_chunks:
        context_block = "(No relevant documents were found for this query.)"
    else:
        context_block = "\n\n".join(
            f"[Source {i+1}, relevance {c['score']:.2f}]\n{c['text']}"
            for i, c in enumerate(retrieved_chunks)
        )

    return (
        f"{base_system_prompt}\n\n"
        f"{STRICT_RAG_INSTRUCTION}\n\n"
        f"--- Retrieved context ---\n{context_block}\n--- End of retrieved context ---"
    )