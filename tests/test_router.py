"""
Mocked unit checks for optimizers/router.py -- verifies the safety
short-circuit and that meta_model_config is correctly forwarded into the
selected optimizer backend, without making any real LLM calls.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import optimizers.router as router_module  # noqa: E402
from optimizers.router import route_and_optimize  # noqa: E402


def make_classification(zone, primary_category):
    return {
        "triage_zone": zone,
        "summary": {"prioritized_action_items": [{"category": primary_category}]},
    }


class TestSafetyShortCircuit(unittest.TestCase):
    def test_safety_refusal_short_circuits_regardless_of_zone(self):
        for zone in ["yellow", "orange", "red", "critical"]:
            classification = make_classification(zone, "safety_refusal")
            best_prompt, log = route_and_optimize(
                classification_report=classification,
                prompt="original prompt",
                test_cases=[],
                eval_fn=lambda p: 1.0,
                client=None,
                model_config={"provider": "openai", "model": "gpt-5.5"},
                optimizer_configs={},
            )
            self.assertEqual(best_prompt, "original prompt")
            self.assertEqual(log["optimizer"], "human_review")
            self.assertEqual(log["iterations_run"], 0)


class TestMetaModelThreading(unittest.TestCase):
    def setUp(self):
        # _REGISTRY caches optimizer function references at first use; reset
        # it so each test's @patch is picked up fresh instead of an earlier
        # test's cached (possibly already-restored) reference.
        router_module._REGISTRY = None

    @patch("optimizers.protegi.optimize")
    def test_meta_model_config_reaches_protegi(self, mock_optimize):
        mock_optimize.return_value = ("optimized prompt", {"iterations_run": 1})
        classification = make_classification("orange", "instruction_drift")
        meta_model_config = {"provider": "anthropic", "model": "claude-opus-4-8"}
        optimizer_configs = {"protegi": {"max_iterations": 10, "meta_model_config": meta_model_config}}

        route_and_optimize(
            classification_report=classification,
            prompt="original prompt",
            test_cases=[],
            eval_fn=lambda p: 1.0,
            client=None,
            model_config={"provider": "openai", "model": "gpt-5.5-mini"},
            optimizer_configs=optimizer_configs,
        )

        # optimize(prompt, test_cases, eval_fn, client, model_config, config)
        self.assertEqual(mock_optimize.call_args[0][5]["meta_model_config"], meta_model_config)

    @patch("optimizers.opro.optimize")
    def test_zone_default_routes_to_opro_for_red_zone(self, mock_optimize):
        mock_optimize.return_value = ("optimized prompt", {"iterations_run": 1})
        classification = make_classification("red", "instruction_drift")
        route_and_optimize(
            classification_report=classification,
            prompt="original prompt",
            test_cases=[],
            eval_fn=lambda p: 1.0,
            client=None,
            model_config={"provider": "openai", "model": "gpt-5.5-mini"},
            optimizer_configs={"opro": {}},
        )
        mock_optimize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
