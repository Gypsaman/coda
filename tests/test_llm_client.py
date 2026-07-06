"""
Mocked unit checks for scripts/llm_client.py -- no live API calls.

Verifies provider-specific kwarg translation (tools, tool_choice,
response_schema, reasoning_tier, prompt caching) and response normalization,
by substituting a fake client object for the real OpenAI/Anthropic/Gemini
SDK client so the real network call is never made.
"""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from scripts.llm_client import LLMClient  # noqa: E402

SAMPLE_TOOLS = [
    {"name": "search_web", "description": "Search.", "input_schema": {"type": "object", "properties": {}}},
]


def make_openai_response(text="hello", tool_calls=None):
    message = SimpleNamespace(content=text, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def make_anthropic_response(blocks):
    return SimpleNamespace(content=blocks, usage=SimpleNamespace(input_tokens=10, output_tokens=5))


class TestOpenAIBackend(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient()
        self.fake_openai = MagicMock()
        self.client._openai_client = self.fake_openai

    def test_basic_completion(self):
        self.fake_openai.chat.completions.create.return_value = make_openai_response("hi there")
        result = self.client.complete(
            provider="openai", model="gpt-5.5", system_prompt="sys", user_message="hello",
        )
        self.assertEqual(result["text"], "hi there")
        self.assertEqual(result["usage"], {"input_tokens": 10, "output_tokens": 5})
        kwargs = self.fake_openai.chat.completions.create.call_args.kwargs
        self.assertIn("temperature", kwargs)
        self.assertIn("max_tokens", kwargs)

    def test_reasoning_tier_uses_max_completion_tokens(self):
        self.fake_openai.chat.completions.create.return_value = make_openai_response("hi")
        self.client.complete(
            provider="openai", model="gpt-5.5", system_prompt="sys", user_message="hello",
            reasoning_tier=True, max_tokens=500,
        )
        kwargs = self.fake_openai.chat.completions.create.call_args.kwargs
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("max_tokens", kwargs)
        self.assertEqual(kwargs["max_completion_tokens"], 500)

    def test_tool_choice_required(self):
        self.fake_openai.chat.completions.create.return_value = make_openai_response("", tool_calls=None)
        self.client.complete(
            provider="openai", model="gpt-5.5", system_prompt="sys", user_message="hello",
            tools=SAMPLE_TOOLS, tool_choice="required",
        )
        kwargs = self.fake_openai.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], "required")
        self.assertEqual(kwargs["tools"][0]["function"]["name"], "search_web")

    def test_tool_choice_named(self):
        self.fake_openai.chat.completions.create.return_value = make_openai_response("")
        self.client.complete(
            provider="openai", model="gpt-5.5", system_prompt="sys", user_message="hello",
            tools=SAMPLE_TOOLS, tool_choice="search_web",
        )
        kwargs = self.fake_openai.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], {"type": "function", "function": {"name": "search_web"}})

    def test_response_schema_translates_to_json_schema(self):
        self.fake_openai.chat.completions.create.return_value = make_openai_response('{"a": 1}')
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        self.client.complete(
            provider="openai", model="gpt-5.5", system_prompt="sys", user_message="hello",
            response_schema=schema,
        )
        kwargs = self.fake_openai.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")
        self.assertEqual(kwargs["response_format"]["json_schema"]["schema"], schema)
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])

    def test_tool_call_parsing(self):
        tc = SimpleNamespace(function=SimpleNamespace(name="search_web", arguments=json.dumps({"query": "x"})))
        self.fake_openai.chat.completions.create.return_value = make_openai_response("", tool_calls=[tc])
        result = self.client.complete(
            provider="openai", model="gpt-5.5", system_prompt="sys", user_message="hello", tools=SAMPLE_TOOLS,
        )
        self.assertEqual(result["tool_calls"], [{"name": "search_web", "arguments": {"query": "x"}}])


class TestAnthropicBackend(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient()
        self.fake_anthropic = MagicMock()
        self.client._anthropic_client = self.fake_anthropic

    def test_basic_completion_and_tool_use(self):
        blocks = [
            SimpleNamespace(type="text", text="hi "),
            SimpleNamespace(type="tool_use", name="search_web", input={"query": "x"}),
        ]
        self.fake_anthropic.messages.create.return_value = make_anthropic_response(blocks)
        result = self.client.complete(
            provider="anthropic", model="claude-sonnet-5", system_prompt="sys", user_message="hello",
            tools=SAMPLE_TOOLS,
        )
        self.assertEqual(result["text"], "hi ")
        self.assertEqual(result["tool_calls"], [{"name": "search_web", "arguments": {"query": "x"}}])

    def test_prompt_cache_wraps_system(self):
        self.fake_anthropic.messages.create.return_value = make_anthropic_response([SimpleNamespace(type="text", text="hi")])
        self.client.complete(
            provider="anthropic", model="claude-sonnet-5", system_prompt="sys prompt", user_message="hello",
            enable_prompt_cache=True,
        )
        kwargs = self.fake_anthropic.messages.create.call_args.kwargs
        self.assertEqual(kwargs["system"], [{"type": "text", "text": "sys prompt", "cache_control": {"type": "ephemeral"}}])

    def test_response_schema_forces_tool_and_reserializes_to_text(self):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        blocks = [SimpleNamespace(type="tool_use", name="emit_result", input={"a": 1})]
        self.fake_anthropic.messages.create.return_value = make_anthropic_response(blocks)
        result = self.client.complete(
            provider="anthropic", model="claude-haiku-4-5-20251001", system_prompt="sys", user_message="hello",
            response_schema=schema,
        )
        kwargs = self.fake_anthropic.messages.create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], {"type": "tool", "name": "emit_result"})
        self.assertEqual(json.loads(result["text"]), {"a": 1})
        self.assertEqual(result["tool_calls"], [])  # schema tool call isn't surfaced as a regular tool call

    def test_tool_choice_any(self):
        self.fake_anthropic.messages.create.return_value = make_anthropic_response([SimpleNamespace(type="text", text="hi")])
        self.client.complete(
            provider="anthropic", model="claude-sonnet-5", system_prompt="sys", user_message="hello",
            tools=SAMPLE_TOOLS, tool_choice="required",
        )
        kwargs = self.fake_anthropic.messages.create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], {"type": "any"})


class TestGeminiBackend(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient()
        self.fake_gemini = MagicMock()
        self.client._gemini_client = self.fake_gemini

    def test_basic_completion_and_function_call(self):
        from google.genai import types

        part_text = SimpleNamespace(text="hi ", function_call=None)
        part_fc = SimpleNamespace(text=None, function_call=types.FunctionCall(name="search_web", args={"query": "x"}))
        candidate = SimpleNamespace(content=SimpleNamespace(parts=[part_text, part_fc]))
        self.fake_gemini.models.generate_content.return_value = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
        )
        result = self.client.complete(
            provider="gemini", model="gemini-3.5-flash", system_prompt="sys", user_message="hello",
            tools=SAMPLE_TOOLS,
        )
        self.assertEqual(result["text"], "hi ")
        self.assertEqual(result["tool_calls"], [{"name": "search_web", "arguments": {"query": "x"}}])
        self.assertEqual(result["usage"], {"input_tokens": 10, "output_tokens": 5})

    def test_response_schema_sets_json_mime_type(self):
        candidate = SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text='{"a": 1}', function_call=None)]))
        self.fake_gemini.models.generate_content.return_value = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
        )
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        self.client.complete(
            provider="gemini", model="gemini-3.5-flash", system_prompt="sys", user_message="hello",
            response_schema=schema,
        )
        call_kwargs = self.fake_gemini.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_schema, schema)


if __name__ == "__main__":
    unittest.main()
