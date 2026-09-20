from typing import Optional, List, Dict, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.agent import AgentStatus


def _normalize_capabilities(v):
    """
    Accept capabilities as either:
        ["sentiment_analysis"]

    or:
        [{"name": "sentiment_analysis", "description": "..."}]
    """

    if not isinstance(v, list):
        return v

    normalized = []

    for item in v:
        if isinstance(item, str):
            normalized.append(item)

        elif isinstance(item, dict) and "name" in item:
            normalized.append(str(item["name"]))

        else:
            normalized.append(str(item))

    return normalized


class ModelConfig(BaseModel):
    """
    WRITE shape -- accepted on create/update only.

    api_key is accepted here (and only here) so the caller can set or
    change it. It is never echoed back: responses use ModelConfigOut
    instead, which has no field capable of holding the raw key.
    """

    provider: str = "gemini"
    model: str

    # Optional per-agent override. When omitted/None on create, the
    # agent has no agent-specific key and execution falls back to the
    # platform-wide key for `provider`. When omitted/None on update,
    # the agent's existing stored key (if any) is left untouched --
    # there is no way to distinguish "not provided" from "clear it" in
    # a single optional field, so clearing a key is done with the
    # explicit sentinel "" (empty string) instead of null.
    api_key: Optional[str] = None

    # Optional per-agent sampling temperature. None -> provider default.
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
    )


class ModelConfigOut(BaseModel):
    """
    READ shape -- returned by GET/POST/PUT responses.

    Deliberately has no field that can hold a raw API key. Callers can
    only tell whether a key is configured (api_key_configured) and see
    a masked preview (api_key_preview, e.g. "••••••••abcd") of the last
    4 characters, never the key itself.
    """

    provider: str
    model: str
    temperature: Optional[float] = None

    api_key_configured: bool = False
    api_key_preview: Optional[str] = None


class AgentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    system_prompt: str

    capabilities: List[str] = Field(default_factory=list)

    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    model_cfg: ModelConfig = Field(validation_alias="model")

    # Registered tool names this agent may use (see
    # tools/registry.py::tool_registry). None -> unrestricted (every
    # registered tool is available), matching the behavior of every
    # agent created before this field existed.
    tools: Optional[List[str]] = None

    is_rag: bool = False
    knowledge_base_id: Optional[str] = None

    visibility: str = Field(default="private", pattern="^(private|platform)$")
    timeout_seconds: int = Field(default=30, ge=1, le=600)
    max_retries: int = Field(default=2, ge=0, le=10)
    requires_approval: bool = False

    model_config = ConfigDict(
        populate_by_name=True,
        protected_namespaces=(),
    )

    @field_validator("capabilities", mode="before")
    @classmethod
    def _validate_capabilities(cls, v):
        return _normalize_capabilities(v)


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None

    capabilities: Optional[List[str]] = None

    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    model_cfg: Optional[ModelConfig] = Field(
        default=None,
        validation_alias="model",
    )

    tools: Optional[List[str]] = None

    status: Optional[AgentStatus] = None

    is_rag: Optional[bool] = None
    # Empty string explicitly clears an existing KB association; None
    # means "leave unchanged" for backwards compatibility.
    knowledge_base_id: Optional[str] = None

    visibility: Optional[str] = Field(default=None, pattern="^(private|platform)$")
    timeout_seconds: Optional[int] = Field(default=None, ge=1, le=600)
    max_retries: Optional[int] = Field(default=None, ge=0, le=10)
    requires_approval: Optional[bool] = None

    model_config = ConfigDict(
        populate_by_name=True,
        protected_namespaces=(),
    )

    @field_validator("capabilities", mode="before")
    @classmethod
    def _validate_capabilities(cls, v):
        return (
            _normalize_capabilities(v)
            if v is not None
            else v
        )


class AgentStatusUpdate(BaseModel):
    status: AgentStatus


class AgentOut(BaseModel):
    id: str
    name: str
    description: Optional[str]

    status: AgentStatus
    current_version: int

    system_prompt: str

    capabilities: List[str] = Field(
        default_factory=list
    )

    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    tools: Optional[List[str]] = None

    is_rag: bool = False
    knowledge_base_id: Optional[str] = None

    visibility: str = "private"
    timeout_seconds: int = 30
    max_retries: int = 2
    requires_approval: bool = False

    # Frontend uses this to separate:
    # Platform Agents vs My Agents
    is_default: bool = False

    model_cfg: ModelConfigOut = Field(
        serialization_alias="model"
    )

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=(),
    )


class AgentExecuteRequest(BaseModel):
    input: Dict[str, Any] | str
    parameters: Optional[Dict[str, Any]] = None


class AgentExecuteResponse(BaseModel):
    execution_id: str
    agent_id: str
    status: str
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
