import unittest

from model_runtime.inspection.runtime_defaults import runtime_defaults_spec
from models.catalog import discover_model_packages

EXISTING_RECURRENT_CONTROL_KEYS = frozenset(
    {
        "RECURRENT_MAX_STEPS",
        "RECURRENT_HALTING_THRESHOLD",
    }
)
MINIMUM_AND_PONDER_CONTROL_KEYS = frozenset(
    {
        "RECURRENT_MIN_STEPS",
        "RECURRENT_PONDER_COST_WEIGHT",
    }
)
RECURRENT_MODEL_PACKAGES = frozenset(
    {
        "bert/expert_linear",
        "bert/expert_linear_adaptive",
        "bert/linear",
        "bert/linear_adaptive",
        "experts/linear",
        "experts/linear_adaptive",
        "gpt/expert_linear",
        "gpt/expert_linear_adaptive",
        "gpt/linear",
        "gpt/linear_adaptive",
        "linears/linear",
        "linears/linear_adaptive",
        "mlp_mixer/expert_linear",
        "mlp_mixer/expert_linear_adaptive",
        "mlp_mixer/linear",
        "mlp_mixer/linear_adaptive",
        "neuron/expert_linear",
        "neuron/expert_linear_adaptive",
        "neuron/linear",
        "neuron/linear_adaptive",
        "transformer/expert_linear",
        "transformer/expert_linear_adaptive",
        "transformer/linear",
        "transformer/linear_adaptive",
        "vit/expert_linear",
        "vit/expert_linear_adaptive",
        "vit/linear",
        "vit/linear_adaptive",
    }
)


class TestRecurrentRuntimeControlScope(unittest.TestCase):
    def test_minimum_and_ponder_controls_have_an_explicit_package_scope(self) -> None:
        supported_keys = {
            package.catalog_key: frozenset(
                runtime_defaults_spec(package).supported_keys
            )
            for package in discover_model_packages()
        }
        recurrent_packages = {
            catalog_key
            for catalog_key, keys in supported_keys.items()
            if EXISTING_RECURRENT_CONTROL_KEYS <= keys
        }

        self.assertEqual(recurrent_packages, RECURRENT_MODEL_PACKAGES)
        self.assertEqual(
            {
                catalog_key
                for catalog_key in recurrent_packages
                if MINIMUM_AND_PONDER_CONTROL_KEYS <= supported_keys[catalog_key]
            },
            {"gpt/expert_linear_adaptive"},
        )
        for catalog_key in recurrent_packages - {"gpt/expert_linear_adaptive"}:
            with self.subTest(catalog_key=catalog_key):
                self.assertTrue(
                    MINIMUM_AND_PONDER_CONTROL_KEYS.isdisjoint(
                        supported_keys[catalog_key]
                    )
                )


if __name__ == "__main__":
    unittest.main()
