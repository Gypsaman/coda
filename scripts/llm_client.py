"""
Unified LLM client abstraction for OpenAI, Anthropic, and Gemini APIs.
Handles provider-specific formatting so pipeline scripts stay provider-agnostic.

Provider-specific field names/SDK shapes below (Gemini especially) should be
re-verified against current provider docs before a live run — these APIs move
fast and the exact kwargs can shift between SDK versions.
"""

import json
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)  # .env must win over stray shell-exported keys of the same name

# Anthropic has no first-class "response_format=json_schema" mode; we emulate
# one by forcing a single synthetic tool call and re-serializing its input
# back into the `text` field, so callers can treat all three providers
# identically and existing json.loads-based parsers keep working unchanged.
_SCHEMA_TOOL_NAME = "emit_result"


class LLMClient:
    """Unified interface for calling OpenAI, Anthropic, and Gemini models."""

    def __init__(self):
        self._openai_client = None
        self._anthropic_client = None
        self._gemini_client = None

    @property
    def openai(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._openai_client

    @property
    def anthropic(self):
        if self._anthropic_client is None:
            from anthropic import Anthropic
            self._anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._anthropic_client

    @property
    def gemini(self):
        if self._gemini_client is None:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    def complete(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        response_schema: dict | None = None,
        reasoning_tier: bool = False,
        enable_prompt_cache: bool = False,
    ) -> dict:
        """
        Send a completion request and return a standardized response.

        Args:
            tools:          list of {name, description, input_schema} tool defs.
            tool_choice:    None/"auto" (model decides), "required" (must call
                            some tool), or a specific tool name to force.
            response_schema: JSON Schema dict. When set, the response is forced
                            into that shape and returned (as JSON text) in
                            `text`, regardless of provider.
            reasoning_tier: True for reasoning-class OpenAI models, which take
                            `max_completion_tokens` instead of `max_tokens` and
                            only support default temperature.
            enable_prompt_cache: Anthropic-only. Marks the system prompt as an
                            ephemeral cache breakpoint. Only pays off once the
                            system prompt exceeds the provider's minimum
                            cacheable-prefix length.

        Returns:
            {
                "text": str,              # text content of the response
                "tool_calls": list[dict], # list of {name, arguments} if any
                "usage": {"input_tokens": int, "output_tokens": int},
                "latency_ms": float,
                "raw": Any,               # raw API response for debugging
            }
        """
        start = time.time()

        if provider == "openai":
            result = self._complete_openai(
                model, system_prompt, user_message, temperature, max_tokens,
                tools, tool_choice, response_schema, reasoning_tier,
            )
        elif provider == "anthropic":
            result = self._complete_anthropic(
                model, system_prompt, user_message, temperature, max_tokens,
                tools, tool_choice, response_schema, enable_prompt_cache,
            )
        elif provider == "gemini":
            result = self._complete_gemini(
                model, system_prompt, user_message, temperature, max_tokens,
                tools, tool_choice, response_schema,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        result["latency_ms"] = (time.time() - start) * 1000
        return result

    def _complete_openai(
        self, model, system_prompt, user_message, temperature, max_tokens,
        tools, tool_choice, response_schema, reasoning_tier,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        if reasoning_tier:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens

        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", t.get("parameters", {})),
                    },
                }
                for t in tools
            ]
            if tool_choice == "required":
                kwargs["tool_choice"] = "required"
            elif tool_choice and tool_choice != "auto":
                kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        if response_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": response_schema, "strict": True},
            }

        response = self.openai.chat.completions.create(**kwargs)
        choice = response.choices[0]

        text = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
                })

        return {
            "text": text,
            "tool_calls": tool_calls,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            "raw": response,
        }

    def _complete_anthropic(
        self, model, system_prompt, user_message, temperature, max_tokens,
        tools, tool_choice, response_schema, enable_prompt_cache,
    ) -> dict:
        system_field: Any = system_prompt
        if enable_prompt_cache:
            system_field = [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }]

        kwargs: dict[str, Any] = {
            "model": model,
            "system": system_field,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        anthropic_tools = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema", t.get("parameters", {})),
            }
            for t in (tools or [])
        ]

        forced_schema_tool = False
        if response_schema:
            anthropic_tools.append({
                "name": _SCHEMA_TOOL_NAME,
                "description": "Emit the final structured result.",
                "input_schema": response_schema,
            })
            forced_schema_tool = True

        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        if forced_schema_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": _SCHEMA_TOOL_NAME}
        elif tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        elif tool_choice and tool_choice != "auto":
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        response = self.anthropic.messages.create(**kwargs)

        text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                if forced_schema_tool and block.name == _SCHEMA_TOOL_NAME:
                    text = json.dumps(block.input)
                else:
                    tool_calls.append({
                        "name": block.name,
                        "arguments": block.input,
                    })

        return {
            "text": text,
            "tool_calls": tool_calls,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "raw": response,
        }

    def _complete_gemini(
        self, model, system_prompt, user_message, temperature, max_tokens,
        tools, tool_choice, response_schema,
    ) -> dict:
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        if tools:
            config_kwargs["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("input_schema", t.get("parameters", {})),
                )
                for t in tools
            ])]
            if tool_choice == "required":
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                )
            elif tool_choice and tool_choice != "auto":
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY", allowed_function_names=[tool_choice],
                    )
                )

        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        response = self.gemini.models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        text = ""
        tool_calls = []
        candidate = response.candidates[0]
        for part in candidate.content.parts:
            if getattr(part, "text", None):
                text += part.text
            elif getattr(part, "function_call", None):
                fc = part.function_call
                tool_calls.append({"name": fc.name, "arguments": dict(fc.args)})

        usage = response.usage_metadata
        return {
            "text": text,
            "tool_calls": tool_calls,
            "usage": {
                "input_tokens": usage.prompt_token_count,
                "output_tokens": usage.candidates_token_count,
            },
            "raw": response,
        }
