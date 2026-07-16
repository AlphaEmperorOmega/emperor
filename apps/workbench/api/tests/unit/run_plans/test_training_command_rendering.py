from __future__ import annotations

import unittest

from emperor_workbench.run_plans._command_projection import (
    render_posix_command,
    render_powershell_command,
)


class TrainingCommandRenderingTests(unittest.TestCase):
    def test_shell_renderers_quote_from_the_same_argument_array(self) -> None:
        argv = [
            "mise",
            "run",
            "experiment",
            "--",
            "--logdir",
            "O'Brien runs",
            "--config",
            "",
        ]

        self.assertEqual(
            render_posix_command(argv),
            "mise run experiment -- --logdir 'O'\"'\"'Brien runs' --config ''",
        )
        self.assertEqual(
            render_powershell_command(argv),
            "mise run experiment -- --logdir 'O''Brien runs' --config ''",
        )


if __name__ == "__main__":
    unittest.main()
