"""
LLM Provider Abstraction Layer.

LangGraph owns orchestration and tool execution.

Provider responsibilities:
    - Send messages to the selected LLM.
    - Return normal text responses.
    - Return requested tool calls.
    - Never execute tools inside generate_with_tool_calls().

Supported providers:
    - Gemini
    - Groq
"""

from __future__ import annotations

import abc
import base64
import json
import os
from typing import Optional, List, Dict, Any, Callable

from dotenv import load_dotenv

from core.config import settings
from google import genai
from google.genai import types


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

load_dotenv()

MAX_TOOL_ITERATIONS = 3


def _env_value(
    name: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    value = os.getenv(name)

    if value is not None:
        value = value.strip()

    if value:
        return value

    return fallback


def _get_gemini_api_key() -> Optional[str]:
    return _env_value(
        "GEMINI_API_KEY",
        getattr(settings, "GEMINI_API_KEY", None),
    )


def _get_groq_api_key() -> Optional[str]:
    return _env_value(
        "GROQ_API_KEY",
        getattr(settings, "GROQ_API_KEY", None),
    )


# ---------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------

def build_tool_spec(
    name: str,
    description: str,
    input_schema: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Convert a Tool Registry schema into a provider-independent
    function-calling schema.
    """

    input_schema = input_schema or {}

    if (
        input_schema.get("type") == "object"
        and isinstance(input_schema.get("properties"), dict)
    ):
        parameters = input_schema

    else:
        properties: Dict[str, Dict[str, Any]] = {}

        for field_name, field_type in input_schema.items():

            if isinstance(field_type, dict):
                properties[field_name] = field_type

            else:
                json_type = (
                    field_type
                    if field_type in (
                        "string",
                        "number",
                        "integer",
                        "boolean",
                        "array",
                        "object",
                    )
                    else "string"
                )

                properties[field_name] = {
                    "type": json_type
                }

        parameters = {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
        }

    return {
        "name": name,
        "description": description or "",
        "parameters": parameters,
    }


# =====================================================================
# BASE LLM CLIENT
# =====================================================================

class LLMClient(abc.ABC):

    @abc.abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_input: str,
        model_name: str,
        temperature: Optional[float] = None,
    ) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_with_tools(
        self,
        system_prompt: str,
        user_input: str,
        model_name: str,
        tool_specs: List[Dict[str, Any]],
        tool_call_handler: Callable[
            [str, Dict[str, Any]],
            Dict[str, Any],
        ],
        temperature: Optional[float] = None,
    ) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_with_tool_calls(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        model_name: str,
        tool_specs: List[Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


# =====================================================================
# GEMINI
# =====================================================================

class GeminiClient(LLMClient):

    def __init__(self):

        self.api_key = _get_gemini_api_key()

        self.default_model = _env_value(
            "GEMINI_DEFAULT_MODEL",
            getattr(
                settings,
                "GEMINI_DEFAULT_MODEL",
                None,
            ),
        )

        self._client: Optional[genai.Client] = None

    # -----------------------------------------------------------------
    # Client
    # -----------------------------------------------------------------

    def _get_client(self) -> genai.Client:

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        if self._client is None:
            self._client = genai.Client(
                api_key=self.api_key
            )

        return self._client

    # -----------------------------------------------------------------
    # Normal generation
    # -----------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        model_name: str,
        temperature: Optional[float] = None,
    ) -> str:

        if not self.api_key:
            return (
                "[MOCK GEMINI RESPONSE - set GEMINI_API_KEY "
                "in .env for real calls]\n"
                f"model={model_name}\n"
                f"system_prompt={system_prompt[:80]}...\n"
                f"user_input={user_input}"
            )

        client = self._get_client()

        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_prompt,
        }

        if temperature is not None:
            config_kwargs["temperature"] = temperature

        response = client.models.generate_content(
            model=model_name,
            contents=user_input,
            config=types.GenerateContentConfig(
                **config_kwargs
            ),
        )

        return response.text or ""

    # -----------------------------------------------------------------
    # Gemini tool declarations
    # -----------------------------------------------------------------

    def _build_gemini_tools(
        self,
        tool_specs: List[Dict[str, Any]],
    ) -> List[types.Tool]:

        declarations: List[types.FunctionDeclaration] = []

        for spec in tool_specs:

            declaration = types.FunctionDeclaration(
                name=spec["name"],
                description=spec.get(
                    "description",
                    "",
                ),
                parameters_json_schema=spec.get(
                    "parameters",
                    {
                        "type": "object",
                        "properties": {},
                    },
                ),
            )

            declarations.append(
                declaration
            )

        if not declarations:
            return []

        return [
            types.Tool(
                function_declarations=declarations
            )
        ]

    # -----------------------------------------------------------------
    # Generate with tool calls
    # -----------------------------------------------------------------

    def generate_with_tool_calls(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        model_name: str,
        tool_specs: List[Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:

        if not self.api_key:

            return {
                "content": self.generate(
                    system_prompt=system_prompt,
                    user_input=_extract_last_user_message(
                        messages
                    ),
                    model_name=model_name,
                    temperature=temperature,
                ),
                "tool_calls": [],
            }

        client = self._get_client()

        tools = self._build_gemini_tools(
            tool_specs
        )

        contents = _convert_messages_to_gemini_contents(
            messages
        )

        if not contents:

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=""
                        )
                    ],
                )
            ]

        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_prompt,
            "tools": tools,
            "automatic_function_calling": (
                types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            ),
        }

        if temperature is not None:
            config_kwargs["temperature"] = temperature

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                **config_kwargs
            ),
        )

        return _parse_gemini_response(
            response
        )

    # -----------------------------------------------------------------
    # Legacy-compatible tool loop API
    # -----------------------------------------------------------------

    def generate_with_tools(
        self,
        system_prompt: str,
        user_input: str,
        model_name: str,
        tool_specs: List[Dict[str, Any]],
        tool_call_handler: Callable[
            [str, Dict[str, Any]],
            Dict[str, Any],
        ],
        temperature: Optional[float] = None,
    ) -> str:

        if not self.api_key or not tool_specs:

            return self.generate(
                system_prompt,
                user_input,
                model_name,
                temperature,
            )

        client = self._get_client()

        tools = self._build_gemini_tools(
            tool_specs
        )

        contents: List[Any] = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=user_input
                    )
                ],
            )
        ]

        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_prompt,
            "tools": tools,
            "automatic_function_calling": (
                types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            ),
        }

        if temperature is not None:
            config_kwargs["temperature"] = temperature

        config = types.GenerateContentConfig(
            **config_kwargs
        )

        for _ in range(MAX_TOOL_ITERATIONS):

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )

            function_calls = (
                response.function_calls
                or []
            )

            if not function_calls:
                return response.text or ""

            model_content = (
                response.candidates[0].content
                if response.candidates
                else None
            )

            if model_content is not None:
                contents.append(
                    model_content
                )

            response_parts = []

            for function_call in function_calls:

                tool_name = function_call.name

                tool_args = dict(
                    function_call.args or {}
                )

                tool_result = tool_call_handler(
                    tool_name,
                    tool_args,
                )

                response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={
                            "result": tool_result
                        },
                    )
                )

            contents.append(
                types.Content(
                    # Gemini does not accept role="tool"; function-response
                    # content must be sent back with role="user".
                    role="user",
                    parts=response_parts,
                )
            )

        return "[No final response after tool calls]"


# ---------------------------------------------------------------------
# Gemini response parsing
# ---------------------------------------------------------------------

def _extract_thought_signature(
    part: Any,
) -> Optional[str]:
    """
    Extract and safely encode Gemini's opaque thought signature.

    Gemini returns this as bytes. We encode it as base64 so it can
    safely travel through our provider-independent LangGraph state.
    """

    signature = getattr(
        part,
        "thought_signature",
        None,
    )

    if signature is None:
        return None

    if isinstance(signature, bytes):

        return base64.b64encode(
            signature
        ).decode("ascii")

    if isinstance(signature, str):

        return signature

    try:
        return base64.b64encode(
            bytes(signature)
        ).decode("ascii")
    except Exception:
        return str(signature)


def _decode_thought_signature(
    signature: Any,
) -> Optional[bytes]:

    if signature is None:
        return None

    if isinstance(signature, bytes):
        return signature

    if not isinstance(signature, str):
        return None

    try:
        return base64.b64decode(
            signature.encode("ascii")
        )
    except Exception:
        return signature.encode("utf-8")


def _parse_gemini_response(
    response: Any,
) -> Dict[str, Any]:

    tool_calls: List[Dict[str, Any]] = []
    content_parts: List[str] = []

    if not response.candidates:

        return {
            "content": None,
            "tool_calls": [],
        }

    candidate = response.candidates[0]

    if not candidate.content:

        return {
            "content": None,
            "tool_calls": [],
        }

    for part in candidate.content.parts:

        function_call = getattr(
            part,
            "function_call",
            None,
        )

        if (
            function_call is not None
            and getattr(
                function_call,
                "name",
                None,
            )
        ):

            arguments = dict(
                getattr(
                    function_call,
                    "args",
                    {},
                )
                or {}
            )

            tool_call: Dict[str, Any] = {
                "name": function_call.name,
                "arguments": arguments,
                "call_id": getattr(
                    function_call,
                    "id",
                    None,
                ),
            }

            thought_signature = (
                _extract_thought_signature(
                    part
                )
            )

            if thought_signature:

                tool_call[
                    "thought_signature"
                ] = thought_signature

            tool_calls.append(
                tool_call
            )

            continue

        text = getattr(
            part,
            "text",
            None,
        )

        if text:
            content_parts.append(
                text
            )

    return {
        "content": (
            "\n".join(content_parts)
            if content_parts
            else None
        ),
        "tool_calls": tool_calls,
    }


# =====================================================================
# GROQ
# =====================================================================

class GroqClient(LLMClient):

    def __init__(self):

        self.api_key = _get_groq_api_key()

        self.default_model = _env_value(
            "GROQ_DEFAULT_MODEL",
            getattr(
                settings,
                "GROQ_DEFAULT_MODEL",
                None,
            ),
        )

        try:

            import groq

            self._groq = groq
            self._sdk_available = True

        except ImportError:

            self._groq = None
            self._sdk_available = False

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        model_name: str,
        temperature: Optional[float] = None,
    ) -> str:

        if (
            not self.api_key
            or not self._sdk_available
        ):

            return (
                "[MOCK GROQ RESPONSE - set GROQ_API_KEY "
                "in .env for real calls]\n"
                f"model={model_name}\n"
                f"system_prompt={system_prompt[:80]}...\n"
                f"user_input={user_input}"
            )

        client = self._groq.Groq(
            api_key=self.api_key
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
            temperature=(
                temperature
                if temperature is not None
                else 0.7
            ),
        )

        return (
            response.choices[0]
            .message.content
        )

    def generate_with_tool_calls(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        model_name: str,
        tool_specs: List[Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:

        if (
            not self.api_key
            or not self._sdk_available
            or not tool_specs
        ):

            return {
                "content": self.generate(
                    system_prompt=system_prompt,
                    user_input=_extract_last_user_message(
                        messages
                    ),
                    model_name=model_name,
                    temperature=temperature,
                ),
                "tool_calls": [],
            }

        client = self._groq.Groq(
            api_key=self.api_key
        )

        tools_payload = [
            {
                "type": "function",
                "function": spec,
            }
            for spec in tool_specs
        ]

        groq_messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        for message in messages:

            role = message.get("role")

            if role == "user":

                groq_messages.append(
                    {
                        "role": "user",
                        "content": str(
                            message.get(
                                "content",
                                "",
                            )
                        ),
                    }
                )

            elif role == "assistant":

                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get(
                        "content"
                    ),
                }

                internal_tool_calls = message.get(
                    "tool_calls",
                    [],
                )

                if internal_tool_calls:

                    provider_tool_calls = []

                    for tool_call in internal_tool_calls:

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
                            continue

                        if not call_id:

                            call_id = (
                                "tool_call_"
                                f"{len(provider_tool_calls) + 1}"
                            )

                        provider_tool_calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(
                                        arguments
                                    ),
                                },
                            }
                        )

                    if provider_tool_calls:

                        assistant_message[
                            "tool_calls"
                        ] = provider_tool_calls

                groq_messages.append(
                    assistant_message
                )

            elif role == "tool":

                call_id = message.get(
                    "call_id"
                )

                content = message.get(
                    "content",
                    "",
                )

                if not call_id:
                    call_id = "unknown_tool_call"

                groq_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": str(content),
                    }
                )

        response = client.chat.completions.create(
            model=model_name,
            messages=groq_messages,
            tools=tools_payload,
            temperature=(
                temperature
                if temperature is not None
                else 0.7
            ),
        )

        message = response.choices[0].message

        tool_calls: List[Dict[str, Any]] = []

        if message.tool_calls:

            for tool_call in message.tool_calls:

                try:

                    arguments = (
                        json.loads(
                            tool_call.function.arguments
                        )
                        if tool_call.function.arguments
                        else {}
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                ):

                    arguments = {}

                tool_calls.append(
                    {
                        "name": (
                            tool_call.function.name
                        ),
                        "arguments": arguments,
                        "call_id": tool_call.id,
                    }
                )

        return {
            "content": message.content,
            "tool_calls": tool_calls,
        }

    def generate_with_tools(
        self,
        system_prompt: str,
        user_input: str,
        model_name: str,
        tool_specs: List[Dict[str, Any]],
        tool_call_handler: Callable[
            [str, Dict[str, Any]],
            Dict[str, Any],
        ],
        temperature: Optional[float] = None,
    ) -> str:

        if (
            not self.api_key
            or not self._sdk_available
            or not tool_specs
        ):

            return self.generate(
                system_prompt,
                user_input,
                model_name,
                temperature,
            )

        client = self._groq.Groq(
            api_key=self.api_key
        )

        tools_payload = [
            {
                "type": "function",
                "function": spec,
            }
            for spec in tool_specs
        ]

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]

        for _ in range(MAX_TOOL_ITERATIONS):

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools_payload,
                temperature=(
                    temperature
                    if temperature is not None
                    else 0.7
                ),
            )

            message = response.choices[0].message

            if not message.tool_calls:
                return message.content

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": (
                                    tc.function.name
                                ),
                                "arguments": (
                                    tc.function.arguments
                                ),
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tc in message.tool_calls:

                try:

                    tool_args = (
                        json.loads(
                            tc.function.arguments
                        )
                        if tc.function.arguments
                        else {}
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                ):

                    tool_args = {}

                tool_result = tool_call_handler(
                    tc.function.name,
                    tool_args,
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            tool_result,
                            default=str,
                        ),
                    }
                )

        return "[No final response after tool calls]"


# =====================================================================
# GEMINI MESSAGE CONVERSION
# =====================================================================

def _convert_messages_to_gemini_contents(
    messages: List[Dict[str, Any]],
) -> List[types.Content]:

    converted: List[types.Content] = []

    for message in messages:

        role = message.get("role")

        # -------------------------------------------------------------
        # User
        # -------------------------------------------------------------

        if role == "user":

            converted.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=str(
                                message.get(
                                    "content",
                                    "",
                                )
                            )
                        )
                    ],
                )
            )

            continue

        # -------------------------------------------------------------
        # Assistant / Gemini model response
        # -------------------------------------------------------------

        if role == "assistant":

            parts: List[types.Part] = []

            content = message.get(
                "content"
            )

            if content:

                parts.append(
                    types.Part.from_text(
                        text=str(content)
                    )
                )

            tool_calls = message.get(
                "tool_calls",
                [],
            )

            for tool_call in tool_calls:

                tool_name = tool_call.get(
                    "name"
                )

                arguments = tool_call.get(
                    "arguments",
                    {},
                )

                if not tool_name:
                    continue

                part = types.Part.from_function_call(
                    name=tool_name,
                    args=arguments,
                )

                # -----------------------------------------------------
                # CRITICAL:
                #
                # Gemini 3 requires the original thought signature
                # to be attached to the exact function-call Part.
                # -----------------------------------------------------

                thought_signature = (
                    tool_call.get(
                        "thought_signature"
                    )
                )

                if thought_signature:

                    decoded_signature = (
                        _decode_thought_signature(
                            thought_signature
                        )
                    )

                    if decoded_signature:

                        part.thought_signature = (
                            decoded_signature
                        )

                parts.append(part)

            if parts:

                converted.append(
                    types.Content(
                        role="model",
                        parts=parts,
                    )
                )

            continue

        # -------------------------------------------------------------
        # Tool result
        # -------------------------------------------------------------

        if role == "tool":

            tool_name = message.get(
                "name"
            )

            if not tool_name:
                continue

            raw_content = message.get(
                "content",
                "",
            )

            try:

                response_data = json.loads(
                    raw_content
                )

            except (
                TypeError,
                json.JSONDecodeError,
            ):

                response_data = {
                    "result": raw_content
                }

            converted.append(
                types.Content(
                    # Gemini does not accept role="tool"; function-response
                    # content must be sent back with role="user".
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=tool_name,
                            response=response_data,
                        )
                    ],
                )
            )

            continue

    return converted


# =====================================================================
# HELPERS
# =====================================================================

def _extract_last_user_message(
    messages: List[Dict[str, Any]],
) -> str:

    for message in reversed(messages):

        if message.get("role") != "user":
            continue

        content = message.get(
            "content"
        )

        if content is None:
            return ""

        return str(content)

    return ""


# =====================================================================
# PROVIDER REGISTRY
# =====================================================================

PROVIDER_REGISTRY = {
    "gemini": GeminiClient,
    "groq": GroqClient,
}


_client_cache: Dict[
    str,
    LLMClient,
] = {}


def get_llm_client(
    provider: str,
) -> LLMClient:

    provider_key = (
        provider or "gemini"
    ).lower()

    if provider_key not in PROVIDER_REGISTRY:

        supported = ", ".join(
            PROVIDER_REGISTRY.keys()
        )

        raise ValueError(
            f"Unsupported LLM provider "
            f"'{provider}'. "
            f"Supported providers: {supported}"
        )

    if provider_key not in _client_cache:

        _client_cache[provider_key] = (
            PROVIDER_REGISTRY[
                provider_key
            ]()
        )

    return _client_cache[
        provider_key
    ]