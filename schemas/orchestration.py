from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from models.orchestration import OrchestrationStatus, StepStatus


class OrchestrateRequest(BaseModel):
    user_input: str

    @field_validator("user_input")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("user_input cannot be empty")
        return v


class OrchestrateResponse(BaseModel):
    execution_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None


class StepOut(BaseModel):
    step_key: str
    agent_id: Optional[str]
    capability: Optional[str]
    task: str
    depends_on: Optional[List[str]] = None
    status: StepStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ExecutionStatusResponse(BaseModel):
    execution_id: str
    status: OrchestrationStatus
    user_input: str
    required_capabilities: Optional[List[str]] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = []
    steps: List[StepOut] = []
    final_response: Optional[str] = None
    error: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    latency_ms: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class ApprovalRequest(BaseModel):
    approved: bool
    reason: Optional[str] = None