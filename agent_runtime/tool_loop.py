from typing import Any, Dict, List, Optional


class ToolCall:
    """
    Normalized representation of an LLM-requested tool call.

    Provider-specific responses are converted into this structure
    before LangGraph processes them.

    thought_signature is preserved for providers such as Gemini that
    require the model's tool-call reasoning signature to be returned
    unchanged on the next model request.
    """

    def __init__(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
        thought_signature: Optional[str] = None,
    ) -> None:
        if not name:
            raise ValueError("Tool call name is required.")

        self.name = name
        self.arguments = arguments or {}
        self.call_id = call_id
        self.thought_signature = thought_signature

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }

        if self.thought_signature:
            result["thought_signature"] = self.thought_signature

        return result


class ToolResult:
    """
    Normalized result produced after executing a tool.
    """

    def __init__(
        self,
        name: str,
        result: Any = None,
        error: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> None:
        self.name = name
        self.result = result
        self.error = error
        self.call_id = call_id

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "result": self.result,
            "error": self.error,
            "success": self.success,
            "call_id": self.call_id,
        }


def normalize_tool_calls(
    tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Normalize raw provider tool-call dictionaries into the platform
    format while preserving provider-specific metadata required for
    subsequent tool-call continuation.
    """

    normalized: List[Dict[str, Any]] = []

    for call in tool_calls:
        if not isinstance(call, dict):
            raise TypeError(
                "Each tool call must be a dictionary."
            )

        name = call.get("name")

        if not name:
            raise ValueError(
                "Tool call name is required."
            )

        arguments = call.get(
            "arguments",
            {},
        )

        if arguments is None:
            arguments = {}

        if not isinstance(arguments, dict):
            raise TypeError(
                f"Arguments for tool '{name}' "
                "must be a dictionary."
            )

        normalized.append(
            ToolCall(
                name=name,
                arguments=arguments,
                call_id=call.get("call_id"),
                thought_signature=call.get(
                    "thought_signature"
                ),
            ).to_dict()
        )

    return normalized
 
