from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.orchestration import OrchestrationStatus, StepStatus


# ============================================================
# ORCHESTRATION REQUEST
# ============================================================

class OrchestrateRequest(BaseModel):
    user_input: str

    # Optional conversation association.
    #
    # Direct /api/orchestrate calls can omit this.
    # Conversation chat calls will provide it.
    conversation_id: Optional[str] = None

    @field_validator("user_input")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("user_input cannot be empty")
        return v


# ============================================================
# PROPOSED AGENT
# ============================================================

class ProposedAgentOut(BaseModel):
    """Editable agent/lifecycle proposal returned to the frontend."""

    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    tools: Optional[List[str]] = None
    is_rag: bool = False
    knowledge_base_id: Optional[str] = None
    visibility: str = "private"
    timeout_seconds: int = 30
    max_retries: int = 2
    requires_approval: bool = True
    reason: Optional[str] = None

    # Trusted lifecycle metadata, returned read-only to the client.
    operation: str = "create"
    target_agent_id: Optional[str] = None
    proposal_state: str = "proposed"
    allow_duplicate: bool = False


class KnowledgeBaseOptionOut(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    document_count: int = 0
    status: str
    readiness: str
    ready: bool = False


class ProposalEditRequest(BaseModel):
    """Fields the user is allowed to edit before approval.

    Backend-owned fields such as created_by/status/current_version/IDs are
    intentionally absent and extra fields are rejected.
    """

    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    capabilities: Optional[List[str]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    tools: Optional[List[str]] = None
    is_rag: Optional[bool] = None
    knowledge_base_id: Optional[str] = None
    visibility: Optional[str] = None
    timeout_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    requires_approval: Optional[bool] = None
    allow_duplicate: Optional[bool] = None

    model_config = ConfigDict(extra="forbid")


# ============================================================
# ORCHESTRATION RESPONSE
# ============================================================

class OrchestrateResponse(BaseModel):
    decision: Optional[str] = None
    failure_code: Optional[str] = None
    knowledge_source_choices: List[str] = Field(default_factory=list)
    created_agent_id: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    execution_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None

    # Populated when status is pending_agent_approval.
    proposed_agent: Optional[ProposedAgentOut] = None
    knowledge_source_required: bool = False
    available_knowledge_bases: List[KnowledgeBaseOptionOut] = Field(default_factory=list)


# ============================================================
# ORCHESTRATION STEP
# ============================================================

class StepOut(BaseModel):
    tool_name: Optional[str] = None
    step_key: str
    agent_id: Optional[str]
    capability: Optional[str]
    task: str
    depends_on: Optional[List[str]] = None
    status: StepStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True
    )


# ============================================================
# ORCHESTRATION EXECUTION STATUS
# ============================================================

class ExecutionStatusResponse(BaseModel):
    decision: Optional[str] = None
    failure_code: Optional[str] = None
    knowledge_source_required: bool = False
    knowledge_source_choices: List[str] = Field(default_factory=list)
    available_knowledge_bases: List[KnowledgeBaseOptionOut] = Field(default_factory=list)
    # SQLAlchemy model uses `id`.
    # Public API exposes it as `execution_id`.
    execution_id: str = Field(
        validation_alias="id",
        serialization_alias="execution_id",
    )

    status: OrchestrationStatus
    user_input: str

    required_capabilities: Optional[List[str]] = None
    current_step: Optional[str] = None

    completed_steps: List[str] = Field(
        default_factory=list
    )

    steps: List[StepOut] = Field(
        default_factory=list
    )

    final_response: Optional[str] = None
    error: Optional[str] = None

    # Proposed agent is returned when the execution is waiting
    # for user approval.
    proposed_agent: Optional[ProposedAgentOut] = None

    start_time: datetime
    end_time: Optional[datetime] = None
    latency_ms: Optional[float] = None

    model_config = ConfigDict(
        from_attributes=True,
        # `execution_id` has validation_alias="id" so that building this
        # model directly from the ORM object (which uses `id`) works.
        # Without populate_by_name=True, pydantic v2 ONLY accepts the
        # alias ("id") during validation and rejects "execution_id" --
        # which breaks manager/service.py's get_execution(), which
        # constructs this model manually with execution_id=execution.id.
        # This caused every GET /api/orchestrate/{execution_id} call to
        # 500 with "Field required: id".
        populate_by_name=True,
    )


# ============================================================
# APPROVAL REQUEST
# ============================================================

class ApprovalRequest(BaseModel):
    approved: bool
    reason: Optional[str] = None

    # Optional edited proposal submitted together with approval.
    # The API saves these edits before creating the agent.
    proposed_agent: Optional[ProposalEditRequest] = None

    model_config = ConfigDict(extra="forbid")
