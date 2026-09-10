import json
from typing import Any, Dict

from tools import tool_registry

from .state import AgentState


def execute_tool(state: AgentState) -> AgentState:
    """
    Execute the tool requested by the LLM.

    The selected tool name and arguments are read from
    state['tool_calls'] and executed through the central
    Tool Registry.
    """

    tool_calls = state.get("tool_calls", [])

    if not tool_calls:
        state["tool_results"] = []
        return state

    tool_results = state.get("tool_results", [])

    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})

        if not tool_name:
            tool_results.append(
                {
                    "name": None,
                    "success": False,
                    "result": None,
                    "error": "Tool name is missing.",
                }
            )
            continue

        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            result = tool_registry.execute(
                tool_name=tool_name,
                arguments=arguments,
            )

            tool_results.append(
                {
                    "name": tool_name,
                    "success": True,
                    "result": result,
                    "error": None,
                }
            )

        except Exception as exc:
            tool_results.append(
                {
                    "name": tool_name,
                    "success": False,
                    "result": None,
                    "error": str(exc),
                }
            )

    state["tool_results"] = tool_results
    return state