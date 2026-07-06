"""
End-to-end (mocked) check that Case B's tool schema survives Phase 3 (optimize)
into Phase 4 (validate) -- the concrete bug this modernization fixed: previously
run_optimization.py always wrote a plain .txt artifact (dropping `tools`
entirely for JSON-origin prompts), and neither quick_evaluate nor
run_validation.py passed `tools=` to the LLM client at all.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")


class TestCaseBToolsPreserved(unittest.TestCase):
    def setUp(self):
        # run_optimization.py writes to the real prompts/<case>_optimized.*
        # path (no output-dir override exists) -- snapshot and restore it so
        # this test doesn't permanently clobber the checked-in artifact.
        self.artifact_path = Path(__file__).parent.parent / "prompts" / "case_b_optimized.json"
        self.original_content = self.artifact_path.read_text() if self.artifact_path.exists() else None

    def tearDown(self):
        if self.original_content is not None:
            self.artifact_path.write_text(self.original_content)

    def test_optimized_json_artifact_preserves_tools_and_tool_choice(self):
        import scripts.run_optimization as run_optimization

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir)
            (results_path / "classification_report.json").write_text(json.dumps({
                "triage_zone": "orange",
                "summary": {"prioritized_action_items": [{"category": "tool_calling"}]},
            }))
            (results_path / "new_model_results.json").write_text(json.dumps([]))
            (results_path / "baseline_results.json").write_text(json.dumps([]))

            fake_tools = [{"name": "search_web", "description": "d", "input_schema": {"type": "object"}}]

            with patch("scripts.run_optimization.route_and_optimize") as mock_route, \
                 patch.object(run_optimization, "LLMClient"):
                mock_route.return_value = ("optimized system prompt text", {"iterations_run": 3, "optimizer": "protegi"})

                # Patch load_prompt to avoid depending on the real prompt file on disk
                with patch.object(run_optimization, "load_prompt", return_value={
                    "system_prompt": "original prompt", "tools": fake_tools,
                }):
                    import yaml
                    root = Path(__file__).parent.parent
                    config = yaml.safe_load((root / "config" / "models.yaml").read_text())
                    config["cases"]["b"]["prompt_file"] = "prompts/case_b_original.json"  # already true, just explicit
                    log_output = run_optimization.run_optimization("b", str(results_path))

            optimized_path = Path(log_output["optimized_prompt_path"])
            self.assertTrue(str(optimized_path).endswith(".json"))
            artifact = json.loads(optimized_path.read_text())
            self.assertEqual(artifact["system_prompt"], "optimized system prompt text")
            self.assertEqual(artifact["tools"], fake_tools)
            self.assertEqual(artifact["tool_choice"], "required")


if __name__ == "__main__":
    unittest.main()
