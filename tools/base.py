from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """
    Base interface for every tool in the Agentic AI Platform.

    Every tool must provide:
    - name
    - description
    - input schema
    - execution logic
    """

    name: str
    description: str
    input_schema: Dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute the tool with validated input.
        """
        raise NotImplementedError

    def get_definition(self) -> Dict[str, Any]:
        """
        Return the tool definition that can be exposed to an LLM.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }