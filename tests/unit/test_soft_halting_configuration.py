import unittest

from torch import nn

from emperor.halting import (
    HaltingHiddenStateModeOptions,
    HaltingStateBase,
    SoftHalting,
    SoftHaltingConfig,
    SoftHaltingState,
)


class SoftHaltingConfigurationTests(unittest.TestCase):
    def test_default_gate_uses_the_canonical_two_projection_shape(self) -> None:
        model = SoftHalting(
            SoftHaltingConfig(
                input_dim=3,
                threshold=0.999,
                dropout_probability=0.25,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=None,
            )
        )

        self.assertEqual(model.threshold, 0.999)
        self.assertIsInstance(model._gate, nn.Sequential)
        self.assertEqual(
            tuple(type(module) for module in model._gate),
            (nn.Linear, nn.GELU, nn.Dropout, nn.Linear),
        )
        self.assertEqual(model._gate[2].p, 0.25)
        self.assertEqual(model._gate[-1].out_features, 2)
        self.assertEqual(model._gate[-1].weight.count_nonzero().item(), 0)

    def test_raw_hidden_belongs_to_the_soft_state_only(self) -> None:
        self.assertNotIn("raw_hidden", HaltingStateBase.__dataclass_fields__)
        self.assertIn("raw_hidden", SoftHaltingState.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
