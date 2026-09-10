 
from typing import Any, Dict, List
import json

from services.llm_service import build_tool_spec, get_llm_client
from .state import AgentState
from .tool_loop import normalize_tool_calls


def prepare_input(state: AgentState) -> AgentState:
    user_input = state.get("input")

    if user_input is None:
        raise ValueError("Input is required")

    if isinstance(user_input, str) and not user_input.strip():
        raise ValueError("Input cannot be empty")

    state["status"] = "running"

    if "messages" not in state:
        state["messages"] = []

    state["messages"].append(
        {
            "role": "user",
            "content": str(user_input),
        }
    )

    return state


def _tool_specs() -> List[Dict[str, Any]]:
    from tools import tool_registry

    definitions = tool_registry.get_tool_definitions()

    return [
        build_tool_spec(
            name=tool["name"],
            description=tool["description"],
            input_schema=tool["input_schema"],
        )
        for tool in definitions
    ]


def call_llm(state: AgentState) -> AgentState:
    try:
        provider = state.get("provider") or "gemini"
        model = state.get("model")

        if not model:
            raise ValueError("Model is required")

        system_prompt = state.get("system_prompt") or ""

        client = get_llm_client(provider)

        tool_specs = _tool_specs()

        messages = state.get("messages", [])

        # ---------------------------------------------------------
        # Tool-enabled execution
        # ---------------------------------------------------------
        if tool_specs:

            response = client.generate_with_tool_calls(
                system_prompt=system_prompt,
                messages=messages,
                model_name=model,
                tool_specs=tool_specs,
            )

            content = response.get("content")

            raw_tool_calls = response.get(
                "tool_calls",
                [],
            )

            tool_calls = normalize_tool_calls(
                raw_tool_calls
            )

            state["tool_calls"] = tool_calls

            # -----------------------------------------------------
            # If the model requested tools, preserve the assistant
            # tool-call message in conversation history.
            # -----------------------------------------------------
            if tool_calls:

                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                }

                state["messages"].append(
                    assistant_message
                )

                # The tool call is not the final answer.
                state["output"] = None

                return state

            # -----------------------------------------------------
            # No tool call means this is the final LLM response.
            # -----------------------------------------------------
            if content:

                state["output"] = content

                state["messages"].append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

            else:

                state["output"] = None

            return state

        # ---------------------------------------------------------
        # Normal LLM execution without tools
        # ---------------------------------------------------------
        user_input = str(
            state.get("input", "")
        )

        response_text = client.generate(
            system_prompt=system_prompt,
            user_input=user_input,
            model_name=model,
        )

        if response_text is None:
            raise RuntimeError(
                "LLM returned no response"
            )

        state["output"] = str(
            response_text
        )

        state["tool_calls"] = []

        state["messages"].append(
            {
                "role": "assistant",
                "content": str(response_text),
            }
        )

        return state

    except Exception as exc:

        state["error"] = str(exc)
        state["status"] = "failed"

        return state


def execute_tool(state: AgentState) -> AgentState:
    from tools import tool_registry

    tool_calls = state.get(
        "tool_calls",
        [],
    )

    if not tool_calls:

        state["tool_results"] = []

        return state

    tool_results = state.get(
        "tool_results",
        [],
    )

    current_results = []

    for tool_call in tool_calls:

        tool_name = tool_call.get(
            "name"
        )

        arguments = tool_call.get(
            "arguments",
            {},
        )

        call_id = tool_call.get(
            "call_id"
        )

        if not tool_name:

            result = {
                "name": None,
                "success": False,
                "result": None,
                "error": "Tool name is missing.",
                "call_id": call_id,
            }

            tool_results.append(result)
            current_results.append(result)

            continue

        try:

            if isinstance(arguments, str):

                arguments = json.loads(
                    arguments
                )

            result_value = tool_registry.execute(
                tool_name=tool_name,
                arguments=arguments,
            )

            result = {
                "name": tool_name,
                "success": True,
                "result": result_value,
                "error": None,
                "call_id": call_id,
            }

        except Exception as exc:

            result = {
                "name": tool_name,
                "success": False,
                "result": None,
                "error": str(exc),
                "call_id": call_id,
            }

        tool_results.append(result)
        current_results.append(result)

    state["tool_results"] = tool_results

    # -------------------------------------------------------------
    # Add each tool result to the conversation.
    #
    # call_id is preserved so providers that support native
    # tool-call protocols can associate the result with the
    # corresponding tool request.
    # -------------------------------------------------------------
    for tool_result in current_results:

        state["messages"].append(
            {
                "role": "tool",
                "content": json.dumps(
                    tool_result,
                    default=str,
                ),
                "name": tool_result.get(
                    "name"
                ),
                "call_id": tool_result.get(
                    "call_id"
                ),
            }
        )

    # The current tool calls have now been executed.
    state["tool_calls"] = []

    return state


def finalize_execution(state: AgentState) -> AgentState:
    if state.get("error"):

        state["status"] = "failed"

        return state

    if state.get("output") is None:

        state["status"] = "failed"

        state["error"] = (
            "LLM execution completed without an output"
        )

        return state

    state["status"] = "success"

    return state
 
