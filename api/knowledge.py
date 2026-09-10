from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from core.database import get_db
 
from schemas.knowledge import (
    KnowledgeBaseCreate, KnowledgeBaseOut, KnowledgeDocumentOut, KnowledgeSearchResultItem
)
from services import knowledge_service, document_parser

router = APIRouter(prefix="/api/knowledge-bases", tags=["Knowledge Bases"])

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("", response_model=KnowledgeBaseOut, status_code=status.HTTP_201_CREATED)
def create_knowledge_base(
    kb_in: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    
):
    try:
        return knowledge_service.create_knowledge_base(db, kb_in, created_by=None)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("", response_model=List[KnowledgeBaseOut])
def list_knowledge_bases(
    db: Session = Depends(get_db),
    
):
    return knowledge_service.list_knowledge_bases(db)


@router.get("/{kb_id}", response_model=KnowledgeBaseOut)
def get_knowledge_base(
    kb_id: str,
    db: Session = Depends(get_db),
):
    kb = knowledge_service.get_knowledge_base(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{kb_id}' not found")
    return kb


@router.post("/{kb_id}/upload", response_model=KnowledgeDocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    
):
    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB limit")

    try:
        return knowledge_service.process_upload(db, kb_id, file.filename, file_bytes)
    except (ValueError, document_parser.UnsupportedFileTypeError, document_parser.DocumentParseError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/{kb_id}/search", response_model=List[KnowledgeSearchResultItem])
def search_knowledge_base(
    kb_id: str,
    q: str,
    top_k: int = 5,
    db: Session = Depends(get_db),
  
):
    try:
        results = knowledge_service.search_knowledge_base(db, kb_id, q, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return results