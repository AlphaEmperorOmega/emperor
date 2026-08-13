from __future__ import annotations

import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from model_runtime.runs import PlanTooLarge
from models.project_cli.main import run_experiment


class ProjectCliDispatchContracts(unittest.TestCase):
    def test_repeated_selectors_use_the_last_value_and_forward_other_flags(
        self,
    ) -> None:
        with patch(
            "models.project_cli.main.run_model_command",
            return_value=17,
        ) as run_model_command:
            result = run_experiment(
                [
                    "--model-type",
                    "bert",
                    "--model",
                    "linear",
                    "--model-type",
                    "gpt",
                    "--model",
                    "expert_linear",
                    "--preset",
                    "baseline",
                ]
            )

        self.assertEqual(result, 17)
        run_model_command.assert_called_once_with(
            "gpt",
            "expert_linear",
            ["--preset", "baseline"],
        )

    def test_listing_flags_keep_their_existing_precedence(self) -> None:
        with (
            patch("models.project_cli.main.show_model_type_list_usage") as types,
            patch("models.project_cli.main.show_model_list_usage") as models,
        ):
            result = run_experiment(["--list-models", "--list-model-types"])

        self.assertEqual(result, 0)
        types.assert_called_once_with()
        models.assert_not_called()

        with (
            patch("models.project_cli.main.model_id_from_parts", return_value="id"),
            patch("models.project_cli.main.show_dataset_list_usage") as datasets,
            patch("models.project_cli.main.show_monitor_list_usage") as monitors,
        ):
            result = run_experiment(
                [
                    "--model-type",
                    "linears",
                    "--model",
                    "linear",
                    "--list-monitors",
                    "--list-datasets",
                ]
            )

        self.assertEqual(result, 0)
        datasets.assert_called_once_with("linears", "linear")
        monitors.assert_not_called()

    def test_missing_selector_value_keeps_exact_output_and_return_code(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = run_experiment(["--model-type"])

        self.assertEqual(result, 1)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            stdout.getvalue(),
            "Error: --model-type requires a value.\n"
            "\n"
            "Run 'mise run experiment --' to see available flags.\n",
        )

    def test_expected_planning_failures_are_rendered_without_a_traceback(self) -> None:
        stderr = StringIO()
        with (
            patch(
                "models.project_cli.main.run_model_command",
                side_effect=PlanTooLarge(
                    "Training search requested 20 axes; limit 16. "
                    "Select fewer axes with --search-keys."
                ),
            ),
            redirect_stderr(stderr),
        ):
            result = run_experiment(["--model-type", "bert", "--model", "linear"])

        self.assertEqual(result, 2)
        self.assertEqual(
            stderr.getvalue(),
            "Error: Training search requested 20 axes; limit 16. "
            "Select fewer axes with --search-keys.\n",
        )


if __name__ == "__main__":
    unittest.main()
