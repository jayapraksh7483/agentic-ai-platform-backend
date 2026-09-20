"""
Internal planning schemas for the Manager Agent.

These are distinct from schemas/orchestration.py (the API-facing request/
response contract) -- these model the Manager's own reasoning artifacts
(capability extraction, the execution plan, per-step results).
"""
import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# =====================================================================
# PHASE 1 -- MANAGER CONTROL-PLANE CONTRACTS
#
# These model the Manager's *decision* layer (what should happen next
# and why), as opposed to the planning artifacts further below (how the
# selected agents will be executed).
#
# Design rule: intent and requirements are deliberately SEPARATE
# concepts. Intent answers "what kind of request is this"; requirements
# answer "what does satisfying it actually need". Collapsing both into
# one enum is what forces every future behavior to become a new enum
# member, and is why the previous binary general/agent_task split could
# not express things like "answerable directly, but needs document
# context" without mis-routing.
# =====================================================================


class ManagerIntent(str, enum.Enum):
    """
    What kind of request this is.

    Only GENERAL_ANSWER, AGENT_EXECUTION, MULTI_AGENT_ORCHESTRATION and
    RAG_QUERY have executing paths in Phase 1. The lifecycle/knowledge
    members exist so classification can already recognize and route
    them to a controlled "not implemented yet" outcome instead of
    silently mis-classifying them into agent execution -- which, with
    the old binary split, is exactly how "create an agent that..."
    ended up in capability extraction.
    """

    GENERAL_ANSWER = "GENERAL_ANSWER"
    AGENT_EXECUTION = "AGENT_EXECUTION"
    MULTI_AGENT_ORCHESTRATION = "MULTI_AGENT_ORCHESTRATION"

    AGENT_CREATE = "AGENT_CREATE"
    AGENT_UPDATE = "AGENT_UPDATE"
    AGENT_DELETE = "AGENT_DELETE"
    AGENT_QUERY = "AGENT_QUERY"

    RAG_QUERY = "RAG_QUERY"
    DOCUMENT_OPERATION = "DOCUMENT_OPERATION"
    KNOWLEDGE_BASE_OPERATION = "KNOWLEDGE_BASE_OPERATION"

    UNKNOWN = "UNKNOWN"

    @classmethod
    def coerce(cls, value: Any) -> Optional["ManagerIntent"]:
        """
        Validate an LLM-supplied intent. Returns None (NOT a default)
        when the value is absent or not a real member, so the caller
        can record CLASSIFICATION_FAILED rather than silently
        substituting a plausible-looking intent the classifier never
        actually chose.
        """

        if value is None:
            return None

        try:
            return cls(str(value).strip().upper())
        except ValueError:
            return None


LIFECYCLE_INTENTS = frozenset(
    {
        ManagerIntent.AGENT_CREATE,
        ManagerIntent.AGENT_UPDATE,
        ManagerIntent.AGENT_DELETE,
        ManagerIntent.AGENT_QUERY,
    }
)

KNOWLEDGE_INTENTS = frozenset(
    {
        ManagerIntent.RAG_QUERY,
        ManagerIntent.DOCUMENT_OPERATION,
        ManagerIntent.KNOWLEDGE_BASE_OPERATION,
    }
)


class LifecycleOperation(str, enum.Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"


class ProposalState(str, enum.Enum):
    PROPOSED = "proposed"
    KNOWLEDGE_SOURCE_REQUIRED = "knowledge_source_required"
    EDITING = "editing"
    READY_FOR_APPROVAL = "ready_for_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    FAILED = "failed"
    CREATED = "created"


class LifecycleRequest(BaseModel):
    """Validated, LLM-derived lifecycle intent details.

    IDs are intentionally absent: the model may name an agent/KB, but
    database identifiers are always resolved by application code inside
    the authenticated user's scope.
    """

    operation: LifecycleOperation
    target_agent_name: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    tools: Optional[List[str]] = None
    is_rag: Optional[bool] = None
    knowledge_base_name: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    visibility: Optional[str] = None
    timeout_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    requires_approval: bool = True
    force_separate: bool = False

    @field_validator("capabilities", mode="before")
    @classmethod
    def _normalize_lifecycle_capabilities(cls, value: Any) -> List[str]:
        if not isinstance(value, (list, tuple, set)):
            return []
        seen = set()
        result = []
        for item in value:
            text = str(item or "").strip()
            if text and text.casefold() not in seen:
                seen.add(text.casefold())
                result.append(text)
        return result

    @field_validator("tools", mode="before")
    @classmethod
    def _normalize_tools(cls, value: Any):
        if value is None:
            return None
        if not isinstance(value, (list, tuple, set)):
            return []
        return [str(item).strip() for item in value if str(item or "").strip()]

    @field_validator("visibility")
    @classmethod
    def _valid_visibility(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip().lower()
        if value not in {"private", "platform"}:
            raise ValueError("Unsupported agent visibility")
        return value


class ManagerFailureCode(str, enum.Enum):
    """
    Why execution did not proceed normally.

    These are deliberately distinct values rather than one catch-all
    error string, because they have completely different future
    behavior: NO_MATCHING_CAPABILITY may eventually justify proposing
    an agent (with approval); CLASSIFICATION_FAILED and PLANNING_FAILED
    never may, because nothing was ever successfully understood.
    """

    CLASSIFICATION_FAILED = "CLASSIFICATION_FAILED"
    UNKNOWN_INTENT = "UNKNOWN_INTENT"
    PLANNING_FAILED = "PLANNING_FAILED"
    NO_MATCHING_CAPABILITY = "NO_MATCHING_CAPABILITY"
    AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    POLICY_DENIED = "POLICY_DENIED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    TIMEOUT = "TIMEOUT"

    # ---- Phase 2 (routing) ----
    NO_REQUIREMENT = "NO_REQUIREMENT"
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    AGENT_NOT_AUTHORIZED = "AGENT_NOT_AUTHORIZED"
    ROUTING_FAILED = "ROUTING_FAILED"

    # ---- Lifecycle / Knowledge ----
    AGENT_ALREADY_EXISTS = "AGENT_ALREADY_EXISTS"
    AGENT_UPDATE_NOT_ALLOWED = "AGENT_UPDATE_NOT_ALLOWED"
    KNOWLEDGE_SOURCE_REQUIRED = "KNOWLEDGE_SOURCE_REQUIRED"
    KNOWLEDGE_BASE_NOT_READY = "KNOWLEDGE_BASE_NOT_READY"
    KNOWLEDGE_BASE_NOT_FOUND = "KNOWLEDGE_BASE_NOT_FOUND"
    APPROVAL_NOT_FOUND = "APPROVAL_NOT_FOUND"
    AMBIGUOUS_APPROVAL = "AMBIGUOUS_APPROVAL"


class ManagerStage(str, enum.Enum):
    """Which stage of the control plane produced a failure."""

    CLASSIFICATION = "classification"
    PLANNING = "planning"
    ROUTING = "routing"
    RETRIEVAL = "retrieval"
    EXECUTION = "execution"
    SYNTHESIS = "synthesis"
    SYSTEM = "system"


class ManagerFailure(BaseModel):
    """
    Structured failure information.

    `user_message` is the ONLY field intended to be shown to an end
    user -- `message` may contain internal detail (exception text) and
    must not be surfaced directly.
    """

    code: ManagerFailureCode
    stage: ManagerStage
    message: Optional[str] = None
    user_message: Optional[str] = None
    retryable: bool = False


class RequestRequirements(BaseModel):
    """
    What satisfying this request actually needs -- independent of
    intent. A GENERAL_ANSWER request has all of these false/empty; an
    AGENT_EXECUTION request has requires_agent true and at least one
    required capability.
    """

    requested_agent: Optional[str] = None
    requires_agent: bool = False
    requires_multiple_agents: bool = False
    requires_rag: bool = False
    requires_document: bool = False
    requires_knowledge_base: bool = False
    requires_external_tool: bool = False
    requires_approval: bool = False

    required_capabilities: List[str] = Field(default_factory=list)

    @field_validator("required_capabilities", mode="before")
    @classmethod
    def _clean_capabilities(cls, v: Any) -> List[str]:
        if not isinstance(v, (list, tuple, set)):
            return []
        return [
            str(item).strip()
            for item in v
            if item is not None and str(item).strip()
        ]


class RequestUnderstanding(BaseModel):
    """
    The validated result of request understanding -- the single
    structured contract the routing layer consumes.

    Note `intent` is Optional and `failure` carries the reason when it
    is None. A classification failure MUST remain distinguishable from
    a confident GENERAL_ANSWER; defaulting a failed classification to
    GENERAL_ANSWER would erase that distinction (and is what previously
    made an LLM outage look like a successful small-talk reply).
    """

    intent: Optional[ManagerIntent] = None
    confidence: float = 0.0
    reason: Optional[str] = None

    requirements: RequestRequirements = Field(
        default_factory=RequestRequirements
    )

    failure: Optional[ManagerFailure] = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _bound_confidence(cls, v: Any) -> float:
        """Clamp to [0,1]; treat non-numeric as 0.0 rather than raising."""
        try:
            value = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, value))

    @property
    def ok(self) -> bool:
        return self.intent is not None and self.failure is None


# =====================================================================
# PHASE 2 -- ROUTING CONTRACTS
#
# Routing answers "which registered agent should serve this
# requirement, and why" -- deterministically, from the registry.
#
# The LLM's only role upstream is naming the capability. Every object
# below is constructed by application code from real database rows, so
# an agent_id the registry never returned cannot appear here, and
# therefore cannot reach execution.
# =====================================================================


class MatchQuality(str, enum.Enum):
    """
    How a candidate's capability matched the requirement. Ordered
    best-to-worst; used as the primary deterministic ranking key so
    selection never depends on row order or LLM opinion.
    """

    EXPLICIT = "EXPLICIT"            # user named this agent directly
    EXACT = "EXACT"                  # byte-identical capability name
    NORMALIZED = "NORMALIZED"        # case/punctuation/spacing-insensitive
    TOKEN_OVERLAP = "TOKEN_OVERLAP"  # shares significant word(s)

    @property
    def rank(self) -> int:
        return {
            MatchQuality.EXPLICIT: 0,
            MatchQuality.EXACT: 1,
            MatchQuality.NORMALIZED: 2,
            MatchQuality.TOKEN_OVERLAP: 3,
        }[self]


class RejectionReason(str, enum.Enum):
    """
    Why a discovered candidate was filtered out. Recorded per-candidate
    so a routing decision stays explainable after the fact, and so
    "nothing ran" can be attributed to the right cause.
    """

    INACTIVE = "INACTIVE"
    NOT_OWNED = "NOT_OWNED"
    NO_EXECUTABLE_VERSION = "NO_EXECUTABLE_VERSION"
    MISSING_RAG_SUPPORT = "MISSING_RAG_SUPPORT"


class AgentCandidate(BaseModel):
    """
    A registry row projected into routing. Always built from a real
    Agent object -- never from LLM output.
    """

    agent_id: str
    agent_name: str
    agent_version: int

    capability: str
    match_quality: MatchQuality

    is_default: bool = False
    is_rag: bool = False
    knowledge_base_id: Optional[str] = None

    provider: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None

    eligible: bool = True
    rejection_reason: Optional[RejectionReason] = None

    def sort_key(self) -> tuple:
        """
        Deterministic ordering. Every component is registry-derived and
        stable, so the same registry always yields the same selection.

        Preference order:
          1. better capability match
          2. the user's own specialized agent over a platform default
          3. higher agent version
          4. name (stable tiebreak)
        """
        return (
            self.match_quality.rank,
            1 if self.is_default else 0,
            -self.agent_version,
            self.agent_name.lower(),
        )


class CapabilityRouting(BaseModel):
    """Routing outcome for ONE required capability."""

    capability: str

    candidates: List["AgentCandidate"] = Field(default_factory=list)
    selected: Optional["AgentCandidate"] = None

    # Set when candidates existed but none survived eligibility. This
    # is what keeps "the Calculator exists but is disabled"
    # (AGENT_UNAVAILABLE) from being misreported as "no calculation
    # capability exists at all" (CAPABILITY_GAP).
    blocked_reason: Optional[RejectionReason] = None

    @property
    def matched(self) -> bool:
        return self.selected is not None

    @property
    def had_candidates(self) -> bool:
        return bool(self.candidates)


class RoutingStatus(str, enum.Enum):
    RESOLVED = "RESOLVED"        # every requirement matched
    PARTIAL = "PARTIAL"          # some matched, some did not
    UNRESOLVED = "UNRESOLVED"    # nothing matched
    NO_REQUIREMENT = "NO_REQUIREMENT"


class RoutingResult(BaseModel):
    """
    The complete, traceable routing decision.

    Phase 3 will consume `routings` (per-capability candidates +
    selection) to build a real task graph; Phase 4 consumes
    `missing_capabilities` to decide whether to propose an agent.
    """

    status: RoutingStatus = RoutingStatus.NO_REQUIREMENT

    requirements: List[str] = Field(default_factory=list)
    routings: List[CapabilityRouting] = Field(default_factory=list)

    reason: Optional[str] = None

    @property
    def matched_capabilities(self) -> List[str]:
        return [r.capability for r in self.routings if r.matched]

    @property
    def missing_capabilities(self) -> List[str]:
        """
        Capabilities with NO candidate at all -- a genuine registry
        gap. Capabilities whose providers all existed but were
        ineligible are deliberately EXCLUDED here and reported via
        unavailable_capabilities instead: that is an availability
        problem, not a capability gap, and must not feed agent
        proposal.
        """
        return [
            r.capability
            for r in self.routings
            if not r.matched and not r.had_candidates
        ]

    @property
    def unavailable_capabilities(self) -> List[str]:
        """Capability exists in the registry, but no provider was usable."""
        return [
            r.capability
            for r in self.routings
            if not r.matched and r.had_candidates
        ]

    @property
    def selected_agents(self) -> List["AgentCandidate"]:
        return [r.selected for r in self.routings if r.selected]


class CapabilityExtraction(BaseModel):
    """Structured output of Stage 1 (Request Understanding)."""

    required_capabilities: List[str] = Field(default_factory=list)
    task_objectives: Dict[str, str] = Field(default_factory=dict)
    task_dependencies: Dict[str, List[str]] = Field(default_factory=dict)

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
            raise ValueError("Invalid execution mode")
        return v


class AgentSelection(BaseModel):
    capability: str
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    found: bool = False


class ExecutionStep(BaseModel):
    step_id: str
    tool_name: Optional[str] = None
    capability: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    task: str
    depends_on: List[str] = Field(default_factory=list)
    agent_version: Optional[int] = None

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

    # Capabilities with NO provider in the registry -- a genuine gap.
    unmatched_capabilities: List[str] = Field(default_factory=list)

    # Capabilities whose providers exist but were all ineligible
    # (inactive, wrong owner, no RAG support...). Tracked separately
    # from unmatched_capabilities because this is an availability
    # problem, and must never feed the agent-proposal path.
    unavailable_capabilities: List[str] = Field(default_factory=list)

    # Full Phase 2 routing trace: candidates considered, candidates
    # rejected and why, and the selection per capability. Phase 3 will
    # consume this to build a real task graph.
    routing: Optional["RoutingResult"] = None

    @model_validator(mode="after")
    def validate_graph(self):
        from core.config import settings
        if len(self.steps) > settings.MAX_ORCHESTRATION_TASKS:
            raise ValueError("Task limit exceeded")
        by_id = {step.step_id: step for step in self.steps}
        if len(by_id) != len(self.steps):
            raise ValueError("Duplicate task IDs")
        visiting, depths = set(), {}

        def visit(task_id):
            if task_id in visiting:
                raise ValueError("Task dependency cycle")
            if task_id not in by_id:
                raise ValueError("Unknown dependency")
            if task_id in depths:
                return depths[task_id]
            visiting.add(task_id)
            step = by_id[task_id]
            if bool(step.agent_id) == bool(step.tool_name):
                raise ValueError("Task requires exactly one registered agent or tool")
            if step.condition_on and step.condition_on not in step.depends_on:
                raise ValueError("Condition must reference a dependency")
            depth = 1 + max((visit(dep) for dep in step.depends_on), default=0)
            if depth > settings.MAX_DEPENDENCY_DEPTH:
                raise ValueError("Dependency depth limit exceeded")
            visiting.remove(task_id)
            depths[task_id] = depth
            return depth

        for task_id in by_id:
            visit(task_id)
        return self


class AgentResult(BaseModel):
    step_id: str
    agent_id: Optional[str] = None
    status: str  # "success" | "failed" | "skipped"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ProposedAgentSpec(BaseModel):
    """Editable proposal persisted with an orchestration execution.

    Backend-generated metadata (operation/target/state) is produced by
    trusted application code.  Client edit schemas deliberately do not
    expose ownership/status/generated IDs.
    """

    name: str
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    system_prompt: str
    provider: str = "gemini"
    model: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    tools: Optional[List[str]] = None
    is_rag: bool = False
    knowledge_base_id: Optional[str] = None
    visibility: str = "private"
    timeout_seconds: int = 30
    max_retries: int = 2
    requires_approval: bool = True
    reason: Optional[str] = None

    # Trusted lifecycle metadata. Stored in the JSON proposal so approval
    # remains durable even across process restarts.
    operation: LifecycleOperation = LifecycleOperation.CREATE
    target_agent_id: Optional[str] = None
    proposal_state: ProposalState = ProposalState.PROPOSED
    allow_duplicate: bool = False
    resume_task: bool = True

    @field_validator("name", "system_prompt")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("Field cannot be blank")
        return value

    @field_validator("capabilities", mode="before")
    @classmethod
    def _clean_proposal_capabilities(cls, value: Any) -> List[str]:
        if not isinstance(value, (list, tuple, set)):
            return []
        seen = set()
        out = []
        for item in value:
            text = str(item or "").strip()
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                out.append(text)
        return out

    @field_validator("timeout_seconds")
    @classmethod
    def _valid_timeout(cls, value: int) -> int:
        value = int(value)
        if not 1 <= value <= 600:
            raise ValueError("timeout_seconds must be between 1 and 600")
        return value

    @field_validator("max_retries")
    @classmethod
    def _valid_retries(cls, value: int) -> int:
        value = int(value)
        if not 0 <= value <= 10:
            raise ValueError("max_retries must be between 0 and 10")
        return value

    @field_validator("visibility")
    @classmethod
    def _valid_proposal_visibility(cls, value: str) -> str:
        value = (value or "private").strip().lower()
        if value not in {"private", "platform"}:
            raise ValueError("Unsupported agent visibility")
        return value

    @model_validator(mode="after")
    def _rag_consistency(self):
        if not self.is_rag:
            self.knowledge_base_id = None
        if self.operation in {LifecycleOperation.CREATE, LifecycleOperation.UPDATE}:
            if not self.capabilities:
                raise ValueError("Agent proposal requires at least one capability")
        return self
