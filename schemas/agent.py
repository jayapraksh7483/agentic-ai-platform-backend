from typing import Optional, List, Dict, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.agent import AgentStatus


def _normalize_capabilities(v):
    """
    Accepts capabilities as either plain strings (["sentiment_analysis"])
    or as objects with a name field ([{"name": "sentiment_analysis",
    "description": "..."}]) -- the AI Layer's exact output shape has
    varied between these, and rejecting one of them with a 422 is worse
    than just normalizing it here.
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
    provider: str = "gemini"
    model: str


class AgentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    system_prompt: str
    capabilities: List[str] = []
     
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    model_cfg: ModelConfig = Field(alias="model")
    # RAG agents: is_rag marks intent at creation time. knowledge_base_id
    # is intentionally left unset here -- documents get attached later,
    # on first Execute, not during creation.
    is_rag: bool = False
    knowledge_base_id: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    @field_validator("capabilities", mode="before")
    @classmethod
    def _validate_capabilities(cls, v):
        return _normalize_capabilities(v)


class AgentUpdate(BaseModel):
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    capabilities: Optional[List[str]] = None
     
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    model_cfg: Optional[ModelConfig] = Field(default=None, alias="model")
    status: Optional[AgentStatus] = None
    is_rag: Optional[bool] = None
    knowledge_base_id: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    @field_validator("capabilities", mode="before")
    @classmethod
    def _validate_capabilities(cls, v):
        return _normalize_capabilities(v) if v is not None else v


class AgentStatusUpdate(BaseModel):
    status: AgentStatus


class AgentOut(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: AgentStatus
    current_version: int
    system_prompt: str
    capabilities: List[str] = []
     
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    is_rag: bool = False
    knowledge_base_id: Optional[str] = None
    model_cfg: ModelConfig = Field(serialization_alias="model")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True, protected_namespaces=())


class AgentExecuteRequest(BaseModel):
    input: Dict[str, Any] | str
    parameters: Optional[Dict[str, Any]] = None


class AgentExecuteResponse(BaseModel):
    execution_id: str
    agent_id: str
    status: str
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None