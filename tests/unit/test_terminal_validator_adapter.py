import unittest

import torch

from emperor.neuron import Terminal, TerminalConfig
from emperor.neuron._terminal.routing import TerminalRoutingTreeDelegate
from emperor.neuron._terminal.validation import (
    RoutingTreeDelegateValidator,
    Validator,
)


class TestTerminalValidatorAdapter(unittest.TestCase):
    def test_module_declares_its_validator_adapter(self):
        self.assertIs(Terminal.VALIDATOR, Validator)

    def test_pre_initialization_config_validation_uses_adapter(self):
        class TrackingValidator(Validator):
            @classmethod
            def validate_required_fields(cls, cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingTerminal(Terminal):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingTerminal(TerminalConfig())

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(Validator):
            @classmethod
            def validate_forward_input(cls, model, input):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingTerminal(Terminal):
            VALIDATOR = RejectingValidator

        model = RejectingTerminal.__new__(RejectingTerminal)
        torch.nn.Module.__init__(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model(torch.ones(1, 3))


class TestRoutingTreeDelegateValidatorAdapter(unittest.TestCase):
    def test_module_declares_its_validator_adapter(self):
        self.assertIs(
            TerminalRoutingTreeDelegate.VALIDATOR,
            RoutingTreeDelegateValidator,
        )

    def test_construction_config_validation_uses_adapter(self):
        class RejectingValidator(RoutingTreeDelegateValidator):
            @classmethod
            def validate_routing_tree_config(cls, routing_tree_config):
                raise RuntimeError("substituted construction validator was called")

        class RejectingDelegate(TerminalRoutingTreeDelegate):
            VALIDATOR = RejectingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            RejectingDelegate(
                TerminalConfig(),
                torch.zeros((1, 3), dtype=torch.long),
            )

    def test_runtime_validation_uses_adapter(self):
        class RejectingValidator(RoutingTreeDelegateValidator):
            @classmethod
            def validate_forward_inputs(cls, model, input_matrix, skip_mask):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingDelegate(TerminalRoutingTreeDelegate):
            VALIDATOR = RejectingValidator

        model = RejectingDelegate.__new__(RejectingDelegate)
        torch.nn.Module.__init__(model)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime validator was called",
        ):
            model.sample_probabilities_and_indices(torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()
