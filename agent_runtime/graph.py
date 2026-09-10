from langgraph.graph import END, START, StateGraph

from .nodes import (
    call_llm,
    execute_tool,
    finalize_execution,
    prepare_input,
)
from .state import AgentState


# Maximum number of LLM -> tool -> LLM cycles for one execution.
# This prevents an agent from getting stuck requesting tools forever.
MAX_TOOL_ROUNDS = 3


def route_after_llm(state: AgentState) -> str:
    """
    Decide where execution goes after the LLM responds.

    Routes:

        error
            -> finalize_execution

        tool calls available and round limit not reached
            -> increment_tool_round

        otherwise
            -> finalize_execution
    """

    if state.get("error"):
        return "finalize_execution"

    tool_calls = state.get(
        "tool_calls",
        [],
    )

    tool_rounds = state.get(
        "tool_rounds",
        0,
    )

    # LLM requested one or more tools and the limit
    # has not been reached.
    if tool_calls and tool_rounds < MAX_TOOL_ROUNDS:
        return "increment_tool_round"

    # LLM is still requesting tools after the maximum
    # number of allowed rounds.
    if tool_calls and tool_rounds >= MAX_TOOL_ROUNDS:
        state["error"] = (
            f"Maximum tool execution rounds "
            f"({MAX_TOOL_ROUNDS}) exceeded."
        )

        state["status"] = "failed"

        return "finalize_execution"

    # No tool calls means the LLM has produced the
    # final answer.
    return "finalize_execution"


def route_after_tool(state: AgentState) -> str:
    """
    After executing tools, decide whether to call the LLM again.

    Normally:

        execute_tool -> call_llm

    If an error occurred:

        execute_tool -> finalize_execution
    """

    if state.get("error"):
        return "finalize_execution"

    return "call_llm"


def increment_tool_round(state: AgentState) -> AgentState:
    """
    Increment the number of completed tool rounds.

    This is kept as a separate graph node so the state
    transition is explicit.
    """

    current_rounds = state.get(
        "tool_rounds",
        0,
    )

    state["tool_rounds"] = current_rounds + 1

    return state


def build_agent_graph():
    """
    Build and compile the LangGraph agent execution graph.

    Flow:

        START
          |
          v
        prepare_input
          |
          v
        call_llm
          |
          +-----------------------------+
          |                             |
       no tools                       tools
          |                             |
          v                             v
    finalize_execution       increment_tool_round
          |                             |
          |                             v
          |                       execute_tool
          |                             |
          |                             v
          |                         call_llm
          |                             |
          |                             +----> ...
          |
          v
         END
    """

    workflow = StateGraph(
        AgentState
    )

    # ---------------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------------

    workflow.add_node(
        "prepare_input",
        prepare_input,
    )

    workflow.add_node(
        "call_llm",
        call_llm,
    )

    workflow.add_node(
        "increment_tool_round",
        increment_tool_round,
    )

    workflow.add_node(
        "execute_tool",
        execute_tool,
    )

    workflow.add_node(
        "finalize_execution",
        finalize_execution,
    )

    # ---------------------------------------------------------------
    # Initial execution
    # ---------------------------------------------------------------

    workflow.add_edge(
        START,
        "prepare_input",
    )

    workflow.add_edge(
        "prepare_input",
        "call_llm",
    )

    # ---------------------------------------------------------------
    # LLM -> tool or final answer
    # ---------------------------------------------------------------

    workflow.add_conditional_edges(
        "call_llm",
        route_after_llm,
        {
            "increment_tool_round": "increment_tool_round",
            "finalize_execution": "finalize_execution",
        },
    )

    # ---------------------------------------------------------------
    # Increment tool round
    # ---------------------------------------------------------------

    workflow.add_edge(
        "increment_tool_round",
        "execute_tool",
    )

    # ---------------------------------------------------------------
    # Tool -> LLM or failure
    # ---------------------------------------------------------------

    workflow.add_conditional_edges(
        "execute_tool",
        route_after_tool,
        {
            "call_llm": "call_llm",
            "finalize_execution": "finalize_execution",
        },
    )

    # ---------------------------------------------------------------
    # Final state
    # ---------------------------------------------------------------

    workflow.add_edge(
        "finalize_execution",
        END,
    )

    return workflow.compile()