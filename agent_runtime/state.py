from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    retryable: bool
    agent_id: str
    user_id: Optional[int]

    input: Any

    thread_id: Optional[str]
    execution_id: Optional[str]

    system_prompt: Optional[str]

    provider: Optional[str]
    model: Optional[str]

    # Per-agent overrides (models/agent.py: Agent.api_key_encrypted /
    # temperature / tools). All three are None for every agent that
    # doesn't set them -- the Manager Agent and the platform default
    # agents never set any of these, so they see identical behavior
    # to before this feature existed.
    api_key: Optional[str]
    temperature: Optional[float]
    allowed_tools: Optional[List[str]]

    # Persistent conversation history supplied by
    # Conversation -> Manager -> Agent Runtime.
    conversation_history: List[Dict[str, Any]]

    # Messages used internally by the current LangGraph execution.
    messages: List[Dict[str, Any]]

    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    # Number of completed tool execution rounds.
    tool_rounds: int

    output: Any

    status: str

    error: Optional[str]