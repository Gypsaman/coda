"""
Unified LLM client abstraction for OpenAI and Anthropic APIs.
Handles provider-specific formatting so pipeline scripts stay provider-agnostic.
"""

import json
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Unified interface for calling OpenAI and Anthropic models."""

    def __init__(self):
        self._openai_client = None
        self._anthropic_client = None

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

    def complete(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
    ) -> dict:
        """
        Send a completion request and return a standardized response.

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
                model, system_prompt, user_message, temperature, max_tokens, tools
            )
        elif provider == "anthropic":
            result = self._complete_anthropic(
                model, system_prompt, user_message, temperature, max_tokens, tools
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        result["latency_ms"] = (time.time() - start) * 1000
        return result

    def _complete_openai(
        self, model, system_prompt, user_message, temperature, max_tokens, tools
    ) -> dict:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
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
        self, model, system_prompt, user_message, temperature, max_tokens, tools
    ) -> dict:
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("input_schema", t.get("parameters", {})),
                }
                for t in tools
            ]

        response = self.anthropic.messages.create(**kwargs)

        text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
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
