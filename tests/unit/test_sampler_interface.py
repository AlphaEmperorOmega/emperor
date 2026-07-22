import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from emperor.sampler import RouterConfig, SamplerConfig, SamplerModel

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPORTS = (
    "RouterConfig",
    "SamplerConfig",
    "RouterModel",
    "SamplerModel",
    "SamplerMonitorCallback",
)

EXPECTED_OWNERS = {
    "RouterConfig": "emperor.sampler._config",
    "SamplerConfig": "emperor.sampler._config",
    "RouterModel": "emperor.sampler._router",
    "SamplerModel": "emperor.sampler._sampler",
    "SamplerMonitorCallback": "emperor.sampler._monitoring",
}

ROUTER_FIELDS = (
    "input_dim",
    "num_experts",
    "noisy_topk_flag",
    "model_config",
)

SAMPLER_FIELDS = (
    "top_k",
    "threshold",
    "filter_above_threshold",
    "num_topk_samples",
    "normalize_probabilities_flag",
    "noisy_topk_flag",
    "num_experts",
    "coefficient_of_variation_loss_weight",
    "switch_loss_weight",
    "zero_centred_loss_weight",
    "mutual_information_loss_weight",
    "router_config",
)


class SamplerPublicInterfaceTests(unittest.TestCase):
    def test_routing_interface_returns_only_sampler_probabilities(self) -> None:
        self.assertTrue(hasattr(SamplerModel, "sample_probabilities_and_indices"))
        self.assertFalse(
            hasattr(SamplerModel, "sample_probabilities_log_scores_and_indices")
        )
        self.assertFalse(
            hasattr(
                SamplerModel,
                "sample_probabilities_log_scores_router_scores_and_indices",
            )
        )

    def test_unknown_attribute_raises_exact_module_error(self) -> None:
        import emperor.sampler as sampler

        with self.assertRaisesRegex(
            AttributeError,
            "^module 'emperor.sampler' has no attribute 'missing_sampler_export'$",
        ):
            _ = sampler.missing_sampler_export  # type: ignore[attr-defined]

    def test_exact_exports_resolve_eagerly_from_their_owning_modules(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json
import sys

import emperor.sampler as sampler

private_modules = (
    "emperor.sampler._config",
    "emperor.sampler._router",
    "emperor.sampler._sampler",
    "emperor.sampler._monitoring",
)
before = {name: name in sys.modules for name in private_modules}
runtime_before = {
    "lightning": "lightning" in sys.modules,
    "torch": "torch" in sys.modules,
}
owners = {name: getattr(sampler, name).__module__ for name in sampler.__all__}

print(json.dumps({
    "after": {name: name in sys.modules for name in private_modules},
    "all": sampler.__all__,
    "before": before,
    "owners": owners,
    "private_exports": {
        name: hasattr(sampler, name)
        for name in ("SamplerBase", "SamplerUsageTracker", "SamplerModelValidator")
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
                    Path(tempfile.gettempdir()) / "matplotlib-sampler-interface"
                ),
            },
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)

        self.assertEqual(tuple(result["all"]), EXPECTED_EXPORTS)
        self.assertEqual(result["owners"], EXPECTED_OWNERS)
        expected_loaded = dict.fromkeys(result["before"], True)
        self.assertEqual(result["before"], expected_loaded)
        self.assertEqual(result["after"], expected_loaded)
        self.assertEqual(
            result["runtime_before"],
            {"lightning": True, "torch": True},
        )
        self.assertEqual(
            result["private_exports"],
            {
                "SamplerBase": False,
                "SamplerModelValidator": False,
                "SamplerUsageTracker": False,
            },
        )

    def test_config_schema_registry_owners_and_defaults_are_preserved(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(RouterConfig)), ROUTER_FIELDS
        )
        self.assertEqual(
            tuple(field.name for field in fields(SamplerConfig)),
            SAMPLER_FIELDS,
        )
        self.assertTrue(
            all(
                getattr(RouterConfig(), field_name) is None
                for field_name in ROUTER_FIELDS
            )
        )
        self.assertTrue(
            all(
                getattr(SamplerConfig(), field_name) is None
                for field_name in SAMPLER_FIELDS
            )
        )
        self.assertEqual(RouterConfig().registry_owner().__name__, "RouterModel")
        self.assertEqual(SamplerConfig().registry_owner().__name__, "SamplerModel")


if __name__ == "__main__":
    unittest.main()
