from typing import Any, Dict, Optional

from .graph import build_agent_graph


class AgentRuntime:
    def __init__(self):
        self.graph = build_agent_graph()

    def execute(
        self,
        agent_id: str,
        user_input: Any,
        provider: str,
        model: str,
        system_prompt: Optional[str] = None,
        user_id: Optional[int] = None,
        thread_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        conversation_history: Optional[list] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        allowed_tools: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Execute an agent through the LangGraph runtime.

        conversation_history contains the messages that occurred
        before the current user request.

        api_key / temperature / allowed_tools are the selected agent's
        OWN configuration (models/agent.py), resolved and decrypted by
        the caller (services/executor_service.py for direct execution,
        manager/planner.py + manager/manager_agent.py for orchestrated
        steps). None for all three means "no agent-specific override" --
        the LLM service layer falls back to the platform-wide key for
        `provider` and every registered tool stays available, which is
        the existing behavior for every agent that predates this
        feature (the Manager Agent and the platform default agents
        never pass any of these).
        """

        initial_state = {
            "agent_id": agent_id,
            "user_id": user_id,
            "input": user_input,

            "thread_id": thread_id,
            "execution_id": execution_id,

            "system_prompt": system_prompt or "",

            "provider": provider,
            "model": model,

            "api_key": api_key,
            "temperature": temperature,
            "allowed_tools": allowed_tools,

            "conversation_history": conversation_history or [],

            "messages": [],

            "tool_calls": [],
            "tool_results": [],

            "tool_rounds": 0,

            "output": None,

            "status": "pending",

            "error": None,
        }

        try:
            result = self.graph.invoke(initial_state)

            return {
                "agent_id": result.get("agent_id"),
                "execution_id": result.get("execution_id"),
                "output": result.get("output"),
                "status": result.get("status"),
                "error": result.get("error"),
                "retryable": result.get("retryable", False),
                "messages": result.get(
                    "messages",
                    [],
                ),
                "tool_calls": result.get(
                    "tool_calls",
                    [],
                ),
                "tool_results": result.get(
                    "tool_results",
                    [],
                ),
                "tool_rounds": result.get(
                    "tool_rounds",
                    0,
                ),
            }

        except Exception as exc:
            return {
                "agent_id": agent_id,
                "execution_id": execution_id,
                "output": None,
                "status": "failed",
                "error": str(exc),
                "messages": [],
                "tool_calls": [],
                "tool_results": [],
                "tool_rounds": 0,
            }


agent_runtime = AgentRuntime()