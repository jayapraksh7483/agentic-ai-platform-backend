
from typing import Any, Dict, Optional

from .graph import build_agent_graph


class AgentRuntime:
    """
    Main runtime responsible for executing agents through LangGraph.

    The runtime prepares the initial AgentState and invokes the compiled
    graph. LangGraph then controls:

        prepare_input
            ->
        call_llm
            ->
        execute_tool
            ->
        call_llm
            ->
        ...
            ->
        finalize_execution
    """

    def __init__(self):
        self.graph = build_agent_graph()

    def execute(
        self,
        agent_id: str,
        user_input: Any,
        provider: str,
        model: str,
        system_prompt: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        execution_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an agent through the LangGraph runtime.
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

            "messages": [],

            "tool_calls": [],
            "tool_results": [],

            # Number of completed tool execution rounds.
            "tool_rounds": 0,

            "output": None,

            "status": "pending",

            "error": None,
        }

        try:
            result = self.graph.invoke(
                initial_state
            )

            return {
                "agent_id": result.get(
                    "agent_id"
                ),

                "execution_id": result.get(
                    "execution_id"
                ),

                "output": result.get(
                    "output"
                ),

                "status": result.get(
                    "status"
                ),

                "error": result.get(
                    "error"
                ),

                "messages": result.get(
                    "messages",
                    []
                ),

                "tool_calls": result.get(
                    "tool_calls",
                    []
                ),

                "tool_results": result.get(
                    "tool_results",
                    []
                ),

                "tool_rounds": result.get(
                    "tool_rounds",
                    0
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


# Shared runtime instance used by the execution service.
agent_runtime = AgentRuntime()
 
