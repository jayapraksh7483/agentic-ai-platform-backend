from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user

from schemas.knowledge import (
    KnowledgeBaseCreate, KnowledgeBaseOut, KnowledgeDocumentOut, KnowledgeSearchResultItem,
    KnowledgeBaseReadinessOut,
)
from services import knowledge_service, document_parser
from services.upload_service import read_upload
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/api/knowledge-bases", tags=["Knowledge Bases"])

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("", response_model=KnowledgeBaseOut, status_code=status.HTTP_201_CREATED)
def create_knowledge_base(
    kb_in: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        return knowledge_service.create_knowledge_base(
            db, kb_in, created_by=current_user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("", response_model=List[KnowledgeBaseOut])
def list_knowledge_bases(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    return knowledge_service.list_knowledge_bases(db, user_id=current_user.id)


@router.get("/{kb_id}", response_model=KnowledgeBaseOut)
def get_knowledge_base(
    kb_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    kb = knowledge_service.get_knowledge_base(db, kb_id, user_id=current_user.id)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")
    return kb


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_knowledge_base(
    kb_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        knowledge_service.delete_knowledge_base(
            db, kb_id, user_id=current_user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{kb_id}/readiness", response_model=KnowledgeBaseReadinessOut)
def get_knowledge_base_readiness(
    kb_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    readiness = knowledge_service.knowledge_base_readiness(
        db, kb_id, user_id=current_user.id
    )
    if not readiness.get("exists"):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return readiness


@router.post("/{kb_id}/upload", response_model=KnowledgeDocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    file_bytes = await read_upload(file, MAX_UPLOAD_SIZE_BYTES)
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB limit")

    try:
        return await run_in_threadpool(knowledge_service.process_upload,
            db, kb_id, file.filename, file_bytes, user_id=current_user.id
        )
    except (ValueError, document_parser.UnsupportedFileTypeError, document_parser.DocumentParseError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/{kb_id}/search", response_model=List[KnowledgeSearchResultItem])
def search_knowledge_base(
    kb_id: str,
    q: str,
    top_k: int = 5,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        results = knowledge_service.search_knowledge_base(
            db, kb_id, q, top_k=top_k, user_id=current_user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return results
