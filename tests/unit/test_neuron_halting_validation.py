import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from emperor.halting import HaltingBase, HaltingConfig
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


class _CompleteHalting(HaltingBase):
    validated_config = None

    @classmethod
    def validate_resolved_config(cls, cfg) -> None:
        cls.validated_config = cfg
        cfg.halting_gate_config.marker = "validated"
        if not 0.0 < cfg.threshold <= 1.0:
            raise ValueError("resolved threshold is invalid")

    def update_halting_state(self, previous_state, model_hidden_state):
        return previous_state, model_hidden_state

    def finalize_weighted_accumulation(self, state, current_hidden):
        return current_hidden, current_hidden.new_zeros(())


@dataclass
class _CompleteHaltingConfig(HaltingConfig):
    DEFAULT_THRESHOLD = 0.875

    def _registry_owner(self):
        return _CompleteHalting


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


class TestNeuronResolvedHaltingValidation(unittest.TestCase):
    def test_preflight_resolves_defaults_on_a_deep_copy(self) -> None:
        halting_config = _CompleteHaltingConfig(
            input_dim=None,
            threshold=None,
            halting_gate_config=SimpleNamespace(marker="original"),
        )

        NeuronClusterValidator.validate_halting_config(
            _cluster_config_with(halting_config)
        )

        resolved_config = _CompleteHalting.validated_config
        self.assertIsNot(resolved_config, halting_config)
        self.assertEqual(resolved_config.input_dim, 4)
        self.assertEqual(resolved_config.threshold, 0.875)
        self.assertEqual(resolved_config.halting_gate_config.marker, "validated")
        self.assertIsNone(halting_config.input_dim)
        self.assertIsNone(halting_config.threshold)
        self.assertEqual(halting_config.halting_gate_config.marker, "original")

    def test_invalid_resolved_config_is_rejected(self) -> None:
        halting_config = _CompleteHaltingConfig(
            input_dim=4,
            threshold=2.0,
            halting_gate_config=SimpleNamespace(marker="original"),
        )

        with self.assertRaisesRegex(ValueError, "resolved threshold is invalid"):
            NeuronClusterValidator.validate_halting_config(
                _cluster_config_with(halting_config)
            )
