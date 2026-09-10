from typing import Any

from .base import BaseTool


class CalculatorTool(BaseTool):
    """
    Basic calculator tool for the Agentic AI Platform.

    Supported operations:
    - add
    - subtract
    - multiply
    - divide
    """

    name = "calculator"
    description = (
        "Performs basic arithmetic operations on two numbers. "
        "Supports add, subtract, multiply, and divide."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "Arithmetic operation to perform.",
                "enum": ["add", "subtract", "multiply", "divide"],
            },
            "a": {
                "type": "number",
                "description": "First number.",
            },
            "b": {
                "type": "number",
                "description": "Second number.",
            },
        },
        "required": ["operation", "a", "b"],
    }

    def execute(self, **kwargs) -> Any:
        operation = kwargs.get("operation")
        a = kwargs.get("a")
        b = kwargs.get("b")

        if operation not in {"add", "subtract", "multiply", "divide"}:
            raise ValueError(
                "Unsupported operation. "
                "Use add, subtract, multiply, or divide."
            )

        if a is None or b is None:
            raise ValueError("Both 'a' and 'b' are required.")

        if operation == "add":
            return a + b

        if operation == "subtract":
            return a - b

        if operation == "multiply":
            return a * b

        if operation == "divide":
            if b == 0:
                raise ValueError("Division by zero is not allowed.")
            return a / b

        raise ValueError("Invalid calculator operation.")