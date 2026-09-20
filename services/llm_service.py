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

# Provider SDKs are optional at import time.  Keeping Gemini imports
# lazy/failable means the application and tests can still start when a
# provider-specific package is not installed, as long as that provider
# is not actually used.
try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - exercised only in minimal installs
    genai = None
    types = None


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

load_dotenv()

MAX_TOOL_ITERATIONS = 3

def is_transient_overload_error(exc: Exception) -> bool:
    """
    PLACEHOLDER -- reconstructed from the import site only, not from
    your original implementation (which I have not seen). Treats an
    error as transient/retryable if it looks like a rate-limit or
    overload response from any of the four providers. Replace this
    with your real logic once you can paste it to me, or tell me your
    original retry conditions and I'll match them exactly.
    """
    text = str(exc).lower()

    transient_markers = (
        "429",
        "503",
        "rate limit",
        "rate_limit",
        "resource_exhausted",
        "overloaded",
        "quota",
        "too many requests",
        "retry",
    )

    if any(marker in text for marker in transient_markers):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code in (429, 503):
        return True

    return False



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
        and isinstance(
            input_schema.get("properties"),
            dict,
        )
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
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate a normal LLM response.

        `messages` is optional so existing callers remain compatible.

        When supplied, messages contains the persistent conversation
        history plus the current user message.
        """
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

def _safe_gemini_text(response) -> str:
    """
    Safely extract text from a Gemini response.

    response.text (the SDK property) internally iterates
    `candidate.content.parts` with no None-guard. When Gemini returns a
    response with no candidates, or a candidate with no content/parts
    (e.g. blocked by safety settings, hit MAX_TOKENS with only a partial/
    empty completion, or a function-call-only turn), that property
    raises "'NoneType' object is not iterable" instead of just meaning
    "no text". This unwraps the same structure defensively and returns
    "" for any of those cases instead of crashing the whole request.
    """
    try:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""

        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            return ""

        return "".join(
            getattr(part, "text", "") or "" for part in parts
        ).strip()
    except Exception:
        # Belt-and-suspenders: never let response parsing itself be the
        # thing that raises out of generate().
        return ""


class GeminiClient(LLMClient):

    def __init__(self, api_key: Optional[str] = None):

        # An explicit api_key (a per-agent override, e.g. a custom
        # agent's own stored key) takes priority over the platform-wide
        # key from settings/.env. Every existing call site passes no
        # api_key at all, so this is fully backward compatible.
        self.api_key = api_key or _get_gemini_api_key()

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

        if genai is None or types is None:
            raise RuntimeError(
                "Gemini provider requires the 'google-genai' package."
            )

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
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:

        # -------------------------------------------------------------
        # Mock mode
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Persistent conversation history
        # -------------------------------------------------------------

        if messages:

            contents = (
                _convert_messages_to_gemini_contents(
                    messages
                )
            )

            if not contents:

                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text=user_input
                            )
                        ],
                    )
                ]

        else:

            contents = user_input

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                **config_kwargs
            ),
        )

        return _safe_gemini_text(response)

    # -----------------------------------------------------------------
    # Gemini tool declarations
    # -----------------------------------------------------------------

    def _build_gemini_tools(
        self,
        tool_specs: List[Dict[str, Any]],
    ) -> List[types.Tool]:

        declarations: List[
            types.FunctionDeclaration
        ] = []

        for spec in tool_specs:

            declaration = (
                types.FunctionDeclaration(
                    name=spec["name"],
                    description=spec.get(
                        "description",
                        "",
                    ),
                    parameters_json_schema=(
                        spec.get(
                            "parameters",
                            {
                                "type": "object",
                                "properties": {},
                            },
                        )
                    ),
                )
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
                    user_input=(
                        _extract_last_user_message(
                            messages
                        )
                    ),
                    model_name=model_name,
                    temperature=temperature,
                    messages=messages,
                ),
                "tool_calls": [],
            }

        client = self._get_client()

        tools = self._build_gemini_tools(
            tool_specs
        )

        contents = (
            _convert_messages_to_gemini_contents(
                messages
            )
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

        for _ in range(
            MAX_TOOL_ITERATIONS
        ):

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
                return _safe_gemini_text(response)

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
                    role="user",
                    parts=response_parts,
                )
            )

        return (
            "[No final response after tool calls]"
        )


# ---------------------------------------------------------------------
# Gemini response parsing
# ---------------------------------------------------------------------

def _extract_thought_signature(
    part: Any,
) -> Optional[str]:

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

    tool_calls: List[
        Dict[str, Any]
    ] = []

    content_parts: List[str] = []

    if not response.candidates:

        return {
            "content": None,
            "tool_calls": [],
        }

    candidate = response.candidates[0]

    if not candidate.content or not candidate.content.parts:

        finish_reason = getattr(
            candidate, "finish_reason", None
        )

        return {
            "content": None,
            "tool_calls": [],
            "finish_reason": (
                str(finish_reason)
                if finish_reason is not None
                else None
            ),
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

    def __init__(self, api_key: Optional[str] = None):

        # See GeminiClient.__init__ -- same override precedence.
        self.api_key = api_key or _get_groq_api_key()

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

    # -----------------------------------------------------------------
    # Normal generation
    # -----------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        model_name: str,
        temperature: Optional[float] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
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

        groq_messages: List[
            Dict[str, Any]
        ] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        # -------------------------------------------------------------
        # Persistent conversation history
        # -------------------------------------------------------------

        if messages:

            for message in messages:

                role = message.get(
                    "role"
                )

                if role not in (
                    "user",
                    "assistant",
                ):
                    continue

                content = message.get(
                    "content",
                    "",
                )

                groq_messages.append(
                    {
                        "role": role,
                        "content": str(
                            content
                        ),
                    }
                )

        else:

            groq_messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

        response = client.chat.completions.create(
            model=model_name,
            messages=groq_messages,
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

    # -----------------------------------------------------------------
    # Tool calls
    # -----------------------------------------------------------------

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
                    user_input=(
                        _extract_last_user_message(
                            messages
                        )
                    ),
                    model_name=model_name,
                    temperature=temperature,
                    messages=messages,
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

        groq_messages: List[
            Dict[str, Any]
        ] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        for message in (messages or []):

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

                assistant_message: Dict[
                    str,
                    Any,
                ] = {
                    "role": "assistant",
                    "content": message.get(
                        "content"
                    ),
                }

                internal_tool_calls = (
                    message.get(
                        "tool_calls",
                        [],
                    )
                )

                if internal_tool_calls:

                    provider_tool_calls = []

                    for tool_call in (
                        internal_tool_calls
                    ):

                        tool_name = (
                            tool_call.get(
                                "name"
                            )
                        )

                        arguments = (
                            tool_call.get(
                                "arguments",
                                {},
                            )
                        )

                        call_id = (
                            tool_call.get(
                                "call_id"
                            )
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
                    call_id = (
                        "unknown_tool_call"
                    )

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

        tool_calls: List[
            Dict[str, Any]
        ] = []

        if message.tool_calls:

            for tool_call in (
                message.tool_calls
            ):

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

        for _ in range(
            MAX_TOOL_ITERATIONS
        ):

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

        return (
            "[No final response after tool calls]"
        )


# =====================================================================
# OPENAI / ANTHROPIC
# =====================================================================

def _get_anthropic_api_key() -> Optional[str]:
    return _env_value(
        "ANTHROPIC_API_KEY",
        getattr(settings, "ANTHROPIC_API_KEY", None),
    )


def _get_openai_api_key() -> Optional[str]:
    return _env_value(
        "OPENAI_API_KEY",
        getattr(settings, "OPENAI_API_KEY", None),
    )


# =====================================================================
# OPENAI
#
# The OpenAI Python SDK's chat.completions.create() has the same
# request/response shape Groq's SDK already mirrors here, so this
# class is a near-exact copy of GroqClient with the SDK import and
# mock-response text swapped.
# =====================================================================

class OpenAIClient(LLMClient):

    def __init__(self, api_key: Optional[str] = None):

        self.api_key = api_key or _get_openai_api_key()

        self.default_model = _env_value(
            "OPENAI_DEFAULT_MODEL",
            getattr(settings, "OPENAI_DEFAULT_MODEL", None),
        )

        try:
            import openai

            self._openai = openai
            self._sdk_available = True

        except ImportError:
            self._openai = None
            self._sdk_available = False

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        model_name: str,
        temperature: Optional[float] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:

        if not self.api_key or not self._sdk_available:
            return (
                "[MOCK OPENAI RESPONSE - set OPENAI_API_KEY "
                "in .env for real calls]\n"
                f"model={model_name}\n"
                f"system_prompt={system_prompt[:80]}...\n"
                f"user_input={user_input}"
            )

        client = self._openai.OpenAI(api_key=self.api_key)

        openai_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

        if messages:
            for message in messages:
                role = message.get("role")
                if role not in ("user", "assistant"):
                    continue
                openai_messages.append(
                    {
                        "role": role,
                        "content": str(message.get("content", "")),
                    }
                )
        else:
            openai_messages.append(
                {"role": "user", "content": user_input}
            )

        response = client.chat.completions.create(
            model=model_name,
            messages=openai_messages,
            temperature=(
                temperature if temperature is not None else 0.7
            ),
        )

        return response.choices[0].message.content

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
                    user_input=_extract_last_user_message(messages),
                    model_name=model_name,
                    temperature=temperature,
                    messages=messages,
                ),
                "tool_calls": [],
            }

        client = self._openai.OpenAI(api_key=self.api_key)

        tools_payload = [
            {"type": "function", "function": spec}
            for spec in tool_specs
        ]

        openai_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

        for message in (messages or []):
            role = message.get("role")

            if role == "user":
                openai_messages.append(
                    {
                        "role": "user",
                        "content": str(message.get("content", "")),
                    }
                )

            elif role == "assistant":
                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content"),
                }

                internal_tool_calls = message.get("tool_calls", [])

                if internal_tool_calls:
                    provider_tool_calls = []

                    for tool_call in internal_tool_calls:
                        tool_name = tool_call.get("name")
                        arguments = tool_call.get("arguments", {})
                        call_id = tool_call.get("call_id")

                        if not tool_name:
                            continue

                        if not call_id:
                            call_id = (
                                f"tool_call_{len(provider_tool_calls) + 1}"
                            )

                        provider_tool_calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        )

                    if provider_tool_calls:
                        assistant_message["tool_calls"] = (
                            provider_tool_calls
                        )

                openai_messages.append(assistant_message)

            elif role == "tool":
                call_id = message.get("call_id") or "unknown_tool_call"
                content = message.get("content", "")

                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": str(content),
                    }
                )

        response = client.chat.completions.create(
            model=model_name,
            messages=openai_messages,
            tools=tools_payload,
            temperature=(
                temperature if temperature is not None else 0.7
            ),
        )

        message = response.choices[0].message

        tool_calls: List[Dict[str, Any]] = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    arguments = (
                        json.loads(tool_call.function.arguments)
                        if tool_call.function.arguments
                        else {}
                    )
                except (json.JSONDecodeError, TypeError):
                    arguments = {}

                tool_calls.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": arguments,
                        "call_id": tool_call.id,
                    }
                )

        return {"content": message.content, "tool_calls": tool_calls}

    def generate_with_tools(
        self,
        system_prompt: str,
        user_input: str,
        model_name: str,
        tool_specs: List[Dict[str, Any]],
        tool_call_handler: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> str:

        if (
            not self.api_key
            or not self._sdk_available
            or not tool_specs
        ):
            return self.generate(
                system_prompt, user_input, model_name, temperature
            )

        client = self._openai.OpenAI(api_key=self.api_key)

        tools_payload = [
            {"type": "function", "function": spec}
            for spec in tool_specs
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        for _ in range(MAX_TOOL_ITERATIONS):

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools_payload,
                temperature=(
                    temperature if temperature is not None else 0.7
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
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tc in message.tool_calls:
                try:
                    tool_args = (
                        json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {}
                    )
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}

                tool_result = tool_call_handler(
                    tc.function.name, tool_args
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, default=str),
                    }
                )

        return "[No final response after tool calls]"


# =====================================================================
# ANTHROPIC
#
# Different shape from the two above: `system` is a top-level request
# param (never a message), there is no "tool" role -- a tool result is
# sent back as a "user" message containing a tool_result content
# block -- and every response is a list of content blocks
# ("text" and/or "tool_use"), not a single message.content string.
# max_tokens is required by Anthropic's API (no server-side default).
# =====================================================================

ANTHROPIC_MAX_TOKENS = 4096


class AnthropicClient(LLMClient):

    def __init__(self, api_key: Optional[str] = None):

        self.api_key = api_key or _get_anthropic_api_key()

        self.default_model = _env_value(
            "ANTHROPIC_DEFAULT_MODEL",
            getattr(settings, "ANTHROPIC_DEFAULT_MODEL", None),
        )

        try:
            import anthropic

            self._anthropic = anthropic
            self._sdk_available = True

        except ImportError:
            self._anthropic = None
            self._sdk_available = False

    def _client(self):
        return self._anthropic.Anthropic(api_key=self.api_key)

    @staticmethod
    def _extract_text(content_blocks) -> str:
        return "".join(
            block.text
            for block in content_blocks
            if getattr(block, "type", None) == "text"
        ).strip()

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        model_name: str,
        temperature: Optional[float] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:

        if not self.api_key or not self._sdk_available:
            return (
                "[MOCK ANTHROPIC RESPONSE - set ANTHROPIC_API_KEY "
                "in .env for real calls]\n"
                f"model={model_name}\n"
                f"system_prompt={system_prompt[:80]}...\n"
                f"user_input={user_input}"
            )

        client = self._client()

        anthropic_messages: List[Dict[str, Any]] = []

        if messages:
            for message in messages:
                role = message.get("role")
                if role not in ("user", "assistant"):
                    continue
                anthropic_messages.append(
                    {
                        "role": role,
                        "content": str(message.get("content", "")),
                    }
                )
        else:
            anthropic_messages.append(
                {"role": "user", "content": user_input}
            )

        response = client.messages.create(
            model=model_name,
            system=system_prompt,
            messages=anthropic_messages,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            temperature=(
                temperature if temperature is not None else 0.7
            ),
        )

        return self._extract_text(response.content)

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
                    user_input=_extract_last_user_message(messages),
                    model_name=model_name,
                    temperature=temperature,
                    messages=messages,
                ),
                "tool_calls": [],
            }

        client = self._client()

        tools_payload = [
            {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "input_schema": spec.get(
                    "parameters",
                    {"type": "object", "properties": {}},
                ),
            }
            for spec in tool_specs
        ]

        anthropic_messages: List[Dict[str, Any]] = []

        for message in (messages or []):
            role = message.get("role")

            if role == "user":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": str(message.get("content", "")),
                    }
                )

            elif role == "assistant":
                blocks: List[Dict[str, Any]] = []

                text_content = message.get("content")
                if text_content:
                    blocks.append(
                        {"type": "text", "text": str(text_content)}
                    )

                for tool_call in message.get("tool_calls", []):
                    tool_name = tool_call.get("name")
                    if not tool_name:
                        continue

                    call_id = tool_call.get("call_id") or (
                        f"toolu_{len(blocks) + 1}"
                    )

                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": tool_name,
                            "input": tool_call.get("arguments", {}),
                        }
                    )

                if blocks:
                    anthropic_messages.append(
                        {"role": "assistant", "content": blocks}
                    )

            elif role == "tool":
                # Anthropic has no "tool" role -- a tool result goes
                # back as a user message containing a tool_result
                # block referencing the matching tool_use id.
                call_id = message.get("call_id") or "unknown_tool_call"
                content = message.get("content", "")

                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call_id,
                                "content": str(content),
                            }
                        ],
                    }
                )

        response = client.messages.create(
            model=model_name,
            system=system_prompt,
            messages=anthropic_messages,
            tools=tools_payload,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            temperature=(
                temperature if temperature is not None else 0.7
            ),
        )

        tool_calls: List[Dict[str, Any]] = []

        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    {
                        "name": block.name,
                        "arguments": block.input or {},
                        "call_id": block.id,
                    }
                )

        return {
            "content": self._extract_text(response.content),
            "tool_calls": tool_calls,
        }

    def generate_with_tools(
        self,
        system_prompt: str,
        user_input: str,
        model_name: str,
        tool_specs: List[Dict[str, Any]],
        tool_call_handler: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> str:

        if (
            not self.api_key
            or not self._sdk_available
            or not tool_specs
        ):
            return self.generate(
                system_prompt, user_input, model_name, temperature
            )

        client = self._client()

        tools_payload = [
            {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "input_schema": spec.get(
                    "parameters",
                    {"type": "object", "properties": {}},
                ),
            }
            for spec in tool_specs
        ]

        messages = [{"role": "user", "content": user_input}]

        for _ in range(MAX_TOOL_ITERATIONS):

            response = client.messages.create(
                model=model_name,
                system=system_prompt,
                messages=messages,
                tools=tools_payload,
                max_tokens=ANTHROPIC_MAX_TOKENS,
                temperature=(
                    temperature if temperature is not None else 0.7
                ),
            )

            tool_use_blocks = [
                block
                for block in response.content
                if getattr(block, "type", None) == "tool_use"
            ]

            if not tool_use_blocks:
                return self._extract_text(response.content)

            messages.append(
                {"role": "assistant", "content": response.content}
            )

            result_blocks = []

            for block in tool_use_blocks:
                tool_result = tool_call_handler(
                    block.name, block.input or {}
                )

                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(tool_result, default=str),
                    }
                )

            messages.append({"role": "user", "content": result_blocks})

        return "[No final response after tool calls]"


# =====================================================================
# GEMINI MESSAGE CONVERSION
# =====================================================================

def _convert_messages_to_gemini_contents(
    messages: List[Dict[str, Any]],
) -> List[types.Content]:

    converted: List[
        types.Content
    ] = []

    for message in (messages or []):

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

            parts: List[
                types.Part
            ] = []

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

                part = (
                    types.Part.from_function_call(
                        name=tool_name,
                        args=arguments,
                    )
                )

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
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}


_client_cache: Dict[
    str,
    LLMClient,
] = {}


def get_llm_client(
    provider: str,
    api_key: Optional[str] = None,
) -> LLMClient:
    """
    Resolve the LLMClient for `provider`.

    Two paths:

    api_key is None (the vast majority of calls -- the Manager Agent,
    default agents, and any custom agent with no key of its own):
        Return the existing process-wide cached singleton for this
        provider, exactly as before this parameter existed. Behavior
        for every current call site is unchanged.

    api_key is provided (a custom agent with its own stored,
    decrypted key -- see services/agent_service.py):
        Build a fresh, UNCACHED client instance bound to that key and
        return it directly. This deliberately bypasses _client_cache:
        the cache is a single shared, process-wide dict keyed only by
        provider name, so caching a keyed client there would leak one
        user's API key into every other request for that provider,
        including other users' agents and the platform default agents.
        A short-lived instance per call is the correct tradeoff here.
    """

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

    if api_key:

        return PROVIDER_REGISTRY[
            provider_key
        ](api_key=api_key)

    if provider_key not in _client_cache:

        _client_cache[provider_key] = (
            PROVIDER_REGISTRY[
                provider_key
            ]()
        )

    return _client_cache[
        provider_key
    ]

 