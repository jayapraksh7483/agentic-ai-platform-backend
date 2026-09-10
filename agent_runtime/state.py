 
from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    agent_id: str
    user_id: Optional[str]

    input: Any

    thread_id: Optional[str]
    execution_id: Optional[str]

    system_prompt: Optional[str]

    provider: Optional[str]
    model: Optional[str]

    messages: List[Dict[str, Any]]

    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    # Number of completed tool execution rounds.
    tool_rounds: int

    output: Any

    status: str

    error: Optional[str]
 
