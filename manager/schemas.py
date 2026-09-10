"""
Internal planning schemas for the Manager Agent.

These are distinct from schemas/orchestration.py (the API-facing request/
response contract) -- these model the Manager's own reasoning artifacts
(capability extraction, the execution plan, per-step results).
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CapabilityExtraction(BaseModel):
    """Structured output of Stage 1 (Request Understanding)."""

    required_capabilities: List[str] = Field(default_factory=list)

    # "sequential"   -> each capability's step depends on the previous one
    # "parallel"     -> independent steps, no depends_on between them
    # "conditional"  -> later steps only run if an earlier one's result
    #                   matches `condition_keyword` (see ExecutionStep)
    execution_mode: str = "sequential"

    # Only used when execution_mode == "conditional". Free-text keyword the
    # Manager expects to see in the *previous* step's result for the next
    # step to run (e.g. "suspicious", "flagged"). Optional.
    condition_keyword: Optional[str] = None

    # Set by the LLM ONLY when required_capabilities came back empty --
    # a short plain-language description of the capability that WOULD be
    # needed to satisfy the request (e.g. "calculating percentages").
    # Used downstream to draft a new-agent proposal. Null when at least
    # one registered capability was selected.
    unmatched_description: Optional[str] = None

    @field_validator("execution_mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        v = (v or "sequential").lower().strip()
        if v not in {"sequential", "parallel", "conditional"}:
            return "sequential"
        return v


class AgentSelection(BaseModel):
    capability: str
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    found: bool = False


class ExecutionStep(BaseModel):
    step_id: str
    capability: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    task: str
    depends_on: List[str] = Field(default_factory=list)

    # The selected agent's OWN configuration (spec section 18: the Manager
    # must execute the selected agent as-is, not substitute its own
    # provider/model/prompt). Populated from the Agent Registry row at
    # plan-build time.
    system_prompt: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None

    # Set only for execution_mode == "conditional" steps that are gated on
    # a previous step's result.
    condition_on: Optional[str] = None       # step_id whose result gates this step
    condition_keyword: Optional[str] = None  # substring to look for (case-insensitive)


class ExecutionPlan(BaseModel):
    request: str
    steps: List[ExecutionStep] = Field(default_factory=list)
    unmatched_capabilities: List[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    step_id: str
    agent_id: Optional[str] = None
    status: str  # "success" | "failed" | "skipped"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ProposedAgentSpec(BaseModel):
    """
    A new-agent proposal, generated ONLY when no registered
    capability/agent can satisfy the request. Never registered
    automatically -- the platform must get explicit user approval
    before an Agent row is created from this spec.

    `capabilities` should be a BROAD, reusable capability domain
    (e.g. "arithmetic and percentage calculations") rather than a
    single-purpose capability for the exact query that triggered the
    proposal (e.g. NOT "calculate 15 percent of 200").
    """

    name: str
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    system_prompt: str
    provider: str = "gemini"
    model: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    # Short, user-facing explanation of why this agent is being proposed
    # and why it's scoped the way it is. Shown in the approval prompt.
    reason: Optional[str] = None