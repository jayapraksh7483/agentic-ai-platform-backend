import json
import re

from services.llm_service import get_llm_client


BUILDER_MODEL = "gemini-3.5-flash-lite"

BUILDER_SYSTEM_PROMPT = """
You are an expert AI Agent Designer.

Your job is to take a user's requirement and generate a complete Agent Specification in JSON format.

You must return ONLY a valid JSON object with this exact structure:

{
    "name": "Agent Name",
    "description": "What this agent does",
    "capabilities": ["capability1", "capability2"],
    "tools": [],
    "input_schema": {
        "field_name": "field_type"
    },
    "output_schema": {
        "field_name": "field_type"
    },
    "system_prompt": "Detailed system prompt for this agent"
}

Rules:
1. name must be short and descriptive ending with Agent
2. capabilities must be a list of strings describing what the agent can do
3. system_prompt must be detailed and specific to the agent's purpose
4. input_schema and output_schema must reflect the agent's expected inputs and outputs
5. Return ONLY the JSON object, no explanation, no markdown, no code blocks
"""


class BuilderService:
    def build_agent(
        self,
        user_prompt: str,
        provider: str = "gemini",
        model: str = BUILDER_MODEL,
    ) -> dict:
        try:
            client = get_llm_client(provider)

            raw_response = client.generate(
                system_prompt=BUILDER_SYSTEM_PROMPT,
                user_input=user_prompt,
                model_name=model,
            )

            cleaned = self._clean_response(raw_response)
            spec = json.loads(cleaned)

            return {
                "name": spec.get("name", "Unnamed Agent"),
                "description": spec.get("description", ""),
                "capabilities": spec.get("capabilities", []),
                "tools": spec.get("tools", []),
                "model": {
                    "provider": provider,
                    "model": model,
                },
                "input_schema": spec.get("input_schema", {}),
                "output_schema": spec.get("output_schema", {}),
                "system_prompt": spec.get("system_prompt", ""),
            }

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse agent specification: {exc}"
            ) from exc

        except Exception as exc:
            raise RuntimeError(
                f"Agent Builder failed: {exc}"
            ) from exc

    @staticmethod
    def _clean_response(response: str) -> str:
        cleaned = response.strip()

        cleaned = re.sub(
            r"^```json\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"^```\s*",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        return cleaned.strip()
