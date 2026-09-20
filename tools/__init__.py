from .base import BaseTool
from .calculator import CalculatorTool
from .web_search import WebSearchTool
from .registry import ToolRegistry, tool_registry


def initialize_tools() -> ToolRegistry:
    """
    Initialize all built-in platform tools.
    """

    if not tool_registry.exists("calculator"):
        tool_registry.register(CalculatorTool())

    # The default "Web Search" agent's system prompt (see
    # services/default_agent_service.py) explicitly promises a
    # backend-controlled DuckDuckGo search tool. Register it here so
    # that promise is actually backed by a real tool.
    if not tool_registry.exists("web_search"):
        tool_registry.register(WebSearchTool())

    return tool_registry


# Automatically initialize built-in tools when the package is imported.
initialize_tools()


__all__ = [
    "BaseTool",
    "CalculatorTool",
    "WebSearchTool",
    "ToolRegistry",
    "tool_registry",
    "initialize_tools",
]