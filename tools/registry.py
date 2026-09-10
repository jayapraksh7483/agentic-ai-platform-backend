from typing import Any, Dict, List

from .base import BaseTool


class ToolRegistry:
    """
    Central registry for all tools available to the Agentic AI Platform.

    Responsibilities:
    - Register tools
    - Discover tools
    - Check whether a tool exists
    - Enable/disable tools
    - Return LLM-compatible tool definitions
    - Execute registered tools
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._enabled: Dict[str, bool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry.
        """

        if not isinstance(tool, BaseTool):
            raise TypeError("Tool must inherit from BaseTool.")

        if not tool.name:
            raise ValueError("Tool name is required.")

        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered."
            )

        self._tools[tool.name] = tool
        self._enabled[tool.name] = True

    def unregister(self, tool_name: str) -> None:
        """
        Remove a tool from the registry.
        """

        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' is not registered.")

        del self._tools[tool_name]
        del self._enabled[tool_name]

    def get(self, tool_name: str) -> BaseTool:
        """
        Get a registered tool by name.
        """

        tool = self._tools.get(tool_name)

        if tool is None:
            raise KeyError(
                f"Tool '{tool_name}' is not registered."
            )

        if not self._enabled.get(tool_name, False):
            raise RuntimeError(
                f"Tool '{tool_name}' is disabled."
            )

        return tool

    def exists(self, tool_name: str) -> bool:
        """
        Check whether a tool is registered.
        """

        return tool_name in self._tools

    def enable(self, tool_name: str) -> None:
        """
        Enable a registered tool.
        """

        if tool_name not in self._tools:
            raise KeyError(
                f"Tool '{tool_name}' is not registered."
            )

        self._enabled[tool_name] = True

    def disable(self, tool_name: str) -> None:
        """
        Disable a registered tool.
        """

        if tool_name not in self._tools:
            raise KeyError(
                f"Tool '{tool_name}' is not registered."
            )

        self._enabled[tool_name] = False

    def is_enabled(self, tool_name: str) -> bool:
        """
        Check whether a registered tool is enabled.
        """

        if tool_name not in self._tools:
            return False

        return self._enabled.get(tool_name, False)

    def list_tools(self) -> List[BaseTool]:
        """
        Return all registered tools.
        """

        return list(self._tools.values())

    def list_enabled_tools(self) -> List[BaseTool]:
        """
        Return only enabled tools.
        """

        return [
            tool
            for name, tool in self._tools.items()
            if self._enabled.get(name, False)
        ]

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Return LLM-compatible definitions for enabled tools.
        """

        return [
            tool.get_definition()
            for tool in self.list_enabled_tools()
        ]

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """
        Execute a registered tool with the supplied arguments.
        """

        tool = self.get(tool_name)

        if arguments is None:
            arguments = {}

        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a dictionary.")

        return tool.execute(**arguments)


# Shared application-level registry.

tool_registry = ToolRegistry()