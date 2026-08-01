import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    SoftHalting,
    SoftHaltingConfig,
)
from emperor.neuron._validation import NeuronClusterValidator


class _DuckTypedHalting:
    @staticmethod
    def update_halting_state(previous_state, model_hidden_state):
        return previous_state, model_hidden_state

    @staticmethod
    def finalize_weighted_accumulation(state, current_hidden):
        return current_hidden, current_hidden.new_zeros(())


@dataclass
class _DuckTypedHaltingConfig(HaltingConfig):
    def _registry_owner(self):
        return _DuckTypedHalting


def _cluster_config_with(halting_config: HaltingConfig):
    return SimpleNamespace(
        halting_config=halting_config,
        neuron_config=SimpleNamespace(
            terminal_config=SimpleNamespace(input_dim=4),
        ),
    )


class TestNeuronHaltingInterfaceValidation(unittest.TestCase):
    def test_duck_typed_halting_owner_is_rejected(self) -> None:
        config = _cluster_config_with(_DuckTypedHaltingConfig(input_dim=4))

        with self.assertRaisesRegex(
            ValueError,
            "does not implement the HaltingBase lifecycle required by NeuronCluster",
        ):
            NeuronClusterValidator.validate_halting_config(config)

    def test_supported_halting_does_not_require_a_preflight_hook(self) -> None:
        halting_config = SoftHaltingConfig(
            input_dim=None,
            threshold=0.999,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=None,
        )

        NeuronClusterValidator.validate_halting_config(
            _cluster_config_with(halting_config)
        )

        self.assertFalse(hasattr(SoftHalting, "validate_resolved_config"))
        self.assertIsNone(halting_config.input_dim)
        self.assertEqual(halting_config.threshold, 0.999)
