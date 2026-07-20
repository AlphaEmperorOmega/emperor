import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPORTS = (
    "HaltingConfig",
    "SoftHaltingConfig",
    "StickBreakingConfig",
    "HaltingHiddenStateModeOptions",
    "HaltingInterface",
    "HaltingBase",
    "HaltingStateBase",
    "SoftHalting",
    "SoftHaltingState",
    "StickBreaking",
    "StickBreakingState",
    "HaltingMonitorCallback",
    "HaltingUsageTracker",
    "HaltingUsageTrackerManager",
)

EXPECTED_OWNERS = {
    "HaltingConfig": "emperor.halting._config",
    "SoftHaltingConfig": "emperor.halting._config",
    "StickBreakingConfig": "emperor.halting._config",
    "HaltingHiddenStateModeOptions": "emperor.halting._config",
    "HaltingInterface": "emperor.halting._interface",
    "HaltingBase": "emperor.halting._base",
    "HaltingStateBase": "emperor.halting._base",
    "SoftHalting": "emperor.halting._strategies.soft",
    "SoftHaltingState": "emperor.halting._strategies.soft",
    "StickBreaking": "emperor.halting._strategies.stick_breaking",
    "StickBreakingState": "emperor.halting._strategies.stick_breaking",
    "HaltingMonitorCallback": "emperor.halting._monitoring.callback",
    "HaltingUsageTracker": "emperor.halting._monitoring.tracking",
    "HaltingUsageTrackerManager": "emperor.halting._monitoring.tracking",
}


class TestHaltingPublicInterface(unittest.TestCase):
    def test_unknown_attribute_raises_exact_module_error(self):
        import emperor.halting as halting

        with self.assertRaisesRegex(
            AttributeError,
            r"^module 'emperor\.halting' has no attribute 'unknown_halting'$",
        ):
            _ = halting.unknown_halting  # type: ignore[attr-defined]

    def test_exact_exports_resolve_lazily_from_their_owning_modules(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json
import sys

import emperor.halting as halting

private_modules = (
    "emperor.halting._config",
    "emperor.halting._interface",
    "emperor.halting._base",
    "emperor.halting._strategies.soft",
    "emperor.halting._strategies.stick_breaking",
    "emperor.halting._monitoring",
    "emperor.halting._monitoring.callback",
    "emperor.halting._monitoring.diagnostics",
    "emperor.halting._monitoring.tracking",
)
before = {name: name in sys.modules for name in private_modules}
runtime_before = {
    "lightning": "lightning" in sys.modules,
    "torch": "torch" in sys.modules,
}
owners = {name: getattr(halting, name).__module__ for name in halting.__all__}

print(json.dumps({
    "after": {name: name in sys.modules for name in private_modules},
    "all": halting.__all__,
    "before": before,
    "owners": owners,
    "removed_exports": {
        name: hasattr(halting, name)
        for name in ("HaltingOptions", "StickBreakingValidator")
    },
    "runtime_before": runtime_before,
}))
""",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            env={
                **os.environ,
                "MPLCONFIGDIR": str(
                    Path(tempfile.gettempdir()) / "matplotlib-halting-interface"
                ),
            },
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)

        self.assertEqual(tuple(result["all"]), EXPECTED_EXPORTS)
        self.assertEqual(result["owners"], EXPECTED_OWNERS)
        self.assertEqual(result["before"], dict.fromkeys(result["before"], False))
        self.assertEqual(result["after"], dict.fromkeys(result["after"], True))
        self.assertEqual(
            result["runtime_before"],
            {"lightning": False, "torch": False},
        )
        self.assertEqual(
            result["removed_exports"],
            {"HaltingOptions": False, "StickBreakingValidator": False},
        )


if __name__ == "__main__":
    unittest.main()
