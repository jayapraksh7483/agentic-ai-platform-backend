from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
 
from models.execution import AgentExecution
 
from schemas.execution import ExecutionOut

router = APIRouter(prefix="/api/executions", tags=["Executions"])


@router.get("", response_model=List[ExecutionOut])
def list_executions(
    agent_id: Optional[str] = None,
    db: Session = Depends(get_db),
 
):
    query = db.query(AgentExecution)
    if agent_id:
        query = query.filter(AgentExecution.agent_id == agent_id)
    return query.order_by(AgentExecution.start_time.desc()).all()


@router.get("/{execution_id}", response_model=ExecutionOut)
def get_execution(
    execution_id: str,
    db: Session = Depends(get_db),
   
):
    execution = db.query(AgentExecution).filter(AgentExecution.id == execution_id).first()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    return execution
