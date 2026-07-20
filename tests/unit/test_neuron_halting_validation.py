import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from emperor.halting import HaltingConfig
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
