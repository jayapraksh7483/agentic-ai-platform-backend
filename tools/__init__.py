from .base import BaseTool
from .calculator import CalculatorTool
from .registry import ToolRegistry, tool_registry


def initialize_tools() -> ToolRegistry:
    """
    Initialize all built-in platform tools.
    """

    if not tool_registry.exists("calculator"):
        tool_registry.register(CalculatorTool())

    return tool_registry


# Automatically initialize built-in tools when the package is imported.
initialize_tools()


__all__ = [
    "BaseTool",
    "CalculatorTool",
    "ToolRegistry",
    "tool_registry",
    "initialize_tools",
]