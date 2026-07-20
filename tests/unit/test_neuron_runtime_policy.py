import unittest

import torch
import torch.nn as nn

from emperor.neuron._cluster.runtime_policy import inherit_runtime_policy


class _TrainingPolicyObserver(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.register_buffer("running_value", torch.ones((), dtype=torch.float32))
        self.training_observations: list[
            tuple[bool, torch.dtype, bool, torch.dtype]
        ] = []

    def train(self, mode: bool = True) -> "_TrainingPolicyObserver":
        self.training_observations.append(
            (
                mode,
                self.weight.dtype,
                self.weight.requires_grad,
                self.running_value.dtype,
            )
        )
        return super().train(mode)


class TestNeuronRuntimePolicy(unittest.TestCase):
    def test_unmatched_tensor_roles_inherit_cluster_fallback_context(self) -> None:
        module = nn.Module()
        module.extra_parameter = nn.Parameter(torch.ones((), dtype=torch.float32))
        module.register_buffer(
            "extra_float_buffer",
            torch.ones((), dtype=torch.float32),
        )
        module.register_buffer(
            "extra_integer_buffer",
            torch.ones((), dtype=torch.int32),
        )
        module.extra_child = nn.ReLU()
        template = nn.Module().eval()

        inherit_runtime_policy(
            module,
            template,
            fallback_device=torch.device("cpu"),
            fallback_dtype=torch.float64,
        )

        self.assertFalse(module.training)
        self.assertEqual(module.extra_parameter.device, torch.device("cpu"))
        self.assertEqual(module.extra_float_buffer.device, torch.device("cpu"))
        self.assertEqual(module.extra_integer_buffer.device, torch.device("cpu"))
        self.assertEqual(module.extra_parameter.dtype, torch.float64)
        self.assertEqual(module.extra_float_buffer.dtype, torch.float64)
        self.assertEqual(module.extra_integer_buffer.dtype, torch.int32)

    def test_unmatched_float_role_keeps_dtype_without_fallback_dtype(self) -> None:
        module = nn.Module()
        module.extra_parameter = nn.Parameter(torch.ones((), dtype=torch.float32))

        inherit_runtime_policy(
            module,
            nn.Module(),
            fallback_device=torch.device("cpu"),
            fallback_dtype=None,
        )

        self.assertEqual(module.extra_parameter.dtype, torch.float32)

    def test_later_alias_roles_in_template_are_inherited_completely(self) -> None:
        module = nn.Module()
        module.later_parameter = nn.Parameter(
            torch.ones((), dtype=torch.float32),
            requires_grad=True,
        )
        module.register_buffer(
            "later_buffer",
            torch.ones((), dtype=torch.float32),
        )
        module.later_child = nn.ReLU()

        template = nn.Module()
        shared_parameter = nn.Parameter(
            torch.ones((), dtype=torch.float64),
            requires_grad=False,
        )
        template.earlier_parameter = shared_parameter
        template.later_parameter = shared_parameter
        shared_buffer = torch.ones((), dtype=torch.float64)
        template.register_buffer("earlier_buffer", shared_buffer)
        template.register_buffer("later_buffer", shared_buffer)
        shared_child = nn.ReLU().eval()
        template.earlier_child = shared_child
        template.later_child = shared_child

        inherit_runtime_policy(
            module,
            template,
            fallback_device=torch.device("cpu"),
            fallback_dtype=torch.float32,
        )

        self.assertFalse(module.later_child.training)
        self.assertEqual(module.later_parameter.dtype, torch.float64)
        self.assertFalse(module.later_parameter.requires_grad)
        self.assertEqual(module.later_buffer.dtype, torch.float64)

    def test_unmatched_child_before_matched_child_does_not_stop_inheritance(
        self,
    ) -> None:
        module = nn.Module()
        module.unmatched_child = nn.ReLU()
        module.matched_child = nn.ReLU()
        template = nn.Module()
        template.matched_child = nn.ReLU().eval()

        inherit_runtime_policy(
            module,
            template,
            fallback_device=torch.device("cpu"),
            fallback_dtype=torch.float64,
        )

        self.assertTrue(module.unmatched_child.training)
        self.assertFalse(module.matched_child.training)

    def test_tensor_policy_is_applied_before_custom_training_mode(self) -> None:
        module = _TrainingPolicyObserver()
        original_parameter = module.weight
        original_buffer = module.running_value
        template = nn.Module()
        template.weight = nn.Parameter(
            torch.ones((), dtype=torch.float64),
            requires_grad=False,
        )
        template.register_buffer(
            "running_value",
            torch.ones((), dtype=torch.float64),
        )
        template.eval()

        inherit_runtime_policy(
            module,
            template,
            fallback_device=torch.device("cpu"),
            fallback_dtype=torch.float32,
        )

        self.assertIs(module.weight, original_parameter)
        self.assertIs(module.running_value, original_buffer)
        self.assertEqual(
            module.training_observations,
            [(False, torch.float64, False, torch.float64)],
        )
        self.assertFalse(module.training)

    def test_tied_module_mode_conflict_is_actionable_and_atomic(self) -> None:
        module = nn.Module()
        module.anchor = nn.Parameter(torch.ones((), dtype=torch.float32))
        shared_child = nn.ReLU()
        module.tied_role_a = shared_child
        module.tied_role_b = shared_child
        module.eval()

        template = nn.Module()
        template.anchor = nn.Parameter(
            torch.ones((), dtype=torch.float64),
            requires_grad=False,
        )
        template.tied_role_a = nn.ReLU()
        template.tied_role_b = nn.ReLU().eval()

        expected_message = (
            "Cannot inherit Neuron runtime policy because grown module roles "
            "'tied_role_a' and 'tied_role_b' are tied but their template "
            "training modes differ."
        )
        with self.assertRaises(RuntimeError) as raised:
            inherit_runtime_policy(
                module,
                template,
                fallback_device=torch.device("cpu"),
                fallback_dtype=torch.float64,
            )

        self.assertEqual(str(raised.exception), expected_message)
        self.assertEqual(module.anchor.dtype, torch.float32)
        self.assertTrue(module.anchor.requires_grad)
        self.assertFalse(module.training)

    def test_tied_parameter_trainability_conflict_is_actionable_and_atomic(
        self,
    ) -> None:
        module = nn.Module()
        module.anchor = nn.Parameter(torch.ones((), dtype=torch.float32))
        shared_parameter = nn.Parameter(torch.ones((), dtype=torch.float32))
        module.tied_role_a = shared_parameter
        module.tied_role_b = shared_parameter
        module.eval()

        template = nn.Module()
        template.anchor = nn.Parameter(
            torch.ones((), dtype=torch.float64),
            requires_grad=False,
        )
        template.tied_role_a = nn.Parameter(
            torch.ones((), dtype=torch.float32),
            requires_grad=True,
        )
        template.tied_role_b = nn.Parameter(
            torch.ones((), dtype=torch.float32),
            requires_grad=False,
        )

        expected_message = (
            "Cannot inherit Neuron runtime policy because grown parameter roles "
            "'tied_role_a' and 'tied_role_b' are tied but their template "
            "requires_grad policies differ."
        )
        with self.assertRaises(RuntimeError) as raised:
            inherit_runtime_policy(
                module,
                template,
                fallback_device=torch.device("cpu"),
                fallback_dtype=torch.float64,
            )

        self.assertEqual(str(raised.exception), expected_message)
        self.assertEqual(module.anchor.dtype, torch.float32)
        self.assertTrue(module.anchor.requires_grad)
        self.assertFalse(module.training)

    def test_conflict_preflight_does_not_partially_mutate_module(self) -> None:
        module = nn.Module()
        module.first_parameter = nn.Parameter(torch.ones((), dtype=torch.float32))
        shared_parameter = nn.Parameter(torch.ones((), dtype=torch.float32))
        module.tied_role_a = shared_parameter
        module.tied_role_b = shared_parameter

        template = nn.Module()
        template.first_parameter = nn.Parameter(torch.ones((), dtype=torch.float64))
        template.tied_role_a = nn.Parameter(torch.ones((), dtype=torch.float32))
        template.tied_role_b = nn.Parameter(torch.ones((), dtype=torch.float64))

        expected_message = (
            "Cannot inherit Neuron runtime policy because grown parameter roles "
            "'tied_role_a' and 'tied_role_b' are tied but their template device "
            "or dtype contexts differ."
        )
        with self.assertRaises(RuntimeError) as raised:
            inherit_runtime_policy(
                module,
                template,
                fallback_device=torch.device("cpu"),
                fallback_dtype=torch.float64,
            )

        self.assertEqual(str(raised.exception), expected_message)
        self.assertEqual(module.first_parameter.dtype, torch.float32)

    def test_tied_parameter_device_conflict_is_detected_before_mutation(self) -> None:
        module = nn.Module()
        module.anchor = nn.Parameter(torch.ones((), dtype=torch.float32))
        shared_parameter = nn.Parameter(torch.ones((), dtype=torch.float32))
        module.tied_role_a = shared_parameter
        module.tied_role_b = shared_parameter
        module.eval()

        template = nn.Module()
        template.anchor = nn.Parameter(
            torch.ones((), dtype=torch.float64),
            requires_grad=False,
        )
        template.tied_role_a = nn.Parameter(
            torch.ones((), dtype=torch.float32, device="cpu")
        )
        template.tied_role_b = nn.Parameter(
            torch.ones((), dtype=torch.float32, device="meta")
        )

        expected_message = (
            "Cannot inherit Neuron runtime policy because grown parameter roles "
            "'tied_role_a' and 'tied_role_b' are tied but their template device "
            "or dtype contexts differ."
        )
        with self.assertRaises(RuntimeError) as raised:
            inherit_runtime_policy(
                module,
                template,
                fallback_device=torch.device("cpu"),
                fallback_dtype=torch.float64,
            )

        self.assertEqual(str(raised.exception), expected_message)
        self.assertEqual(module.anchor.dtype, torch.float32)
        self.assertTrue(module.anchor.requires_grad)
        self.assertFalse(module.training)

    def test_tied_buffer_context_conflict_is_actionable_and_atomic(self) -> None:
        module = nn.Module()
        module.anchor = nn.Parameter(torch.ones((), dtype=torch.float32))
        module.register_buffer(
            "first_buffer",
            torch.ones((), dtype=torch.float32),
        )
        shared_buffer = torch.ones((), dtype=torch.float32)
        module.register_buffer("tied_role_a", shared_buffer)
        module.register_buffer("tied_role_b", shared_buffer)
        module.eval()

        template = nn.Module()
        template.anchor = nn.Parameter(
            torch.ones((), dtype=torch.float64),
            requires_grad=False,
        )
        template.register_buffer(
            "first_buffer",
            torch.ones((), dtype=torch.float64),
        )
        template.register_buffer(
            "tied_role_a",
            torch.ones((), dtype=torch.float32),
        )
        template.register_buffer(
            "tied_role_b",
            torch.ones((), dtype=torch.float64),
        )

        expected_message = (
            "Cannot inherit Neuron runtime policy because grown buffer roles "
            "'tied_role_a' and 'tied_role_b' are tied but their template device "
            "or dtype contexts differ."
        )
        with self.assertRaises(RuntimeError) as raised:
            inherit_runtime_policy(
                module,
                template,
                fallback_device=torch.device("cpu"),
                fallback_dtype=torch.float64,
            )

        self.assertEqual(str(raised.exception), expected_message)
        self.assertEqual(module.anchor.dtype, torch.float32)
        self.assertTrue(module.anchor.requires_grad)
        self.assertEqual(module.first_buffer.dtype, torch.float32)
        self.assertFalse(module.training)

    def test_tied_buffer_device_conflict_is_detected_before_mutation(self) -> None:
        module = nn.Module()
        module.anchor = nn.Parameter(torch.ones((), dtype=torch.float32))
        module.register_buffer(
            "first_buffer",
            torch.ones((), dtype=torch.float32),
        )
        shared_buffer = torch.ones((), dtype=torch.float32)
        module.register_buffer("tied_role_a", shared_buffer)
        module.register_buffer("tied_role_b", shared_buffer)
        module.eval()

        template = nn.Module()
        template.anchor = nn.Parameter(
            torch.ones((), dtype=torch.float64),
            requires_grad=False,
        )
        template.register_buffer(
            "first_buffer",
            torch.ones((), dtype=torch.float64),
        )
        template.register_buffer(
            "tied_role_a",
            torch.ones((), dtype=torch.float32, device="cpu"),
        )
        template.register_buffer(
            "tied_role_b",
            torch.ones((), dtype=torch.float32, device="meta"),
        )

        expected_message = (
            "Cannot inherit Neuron runtime policy because grown buffer roles "
            "'tied_role_a' and 'tied_role_b' are tied but their template device "
            "or dtype contexts differ."
        )
        with self.assertRaises(RuntimeError) as raised:
            inherit_runtime_policy(
                module,
                template,
                fallback_device=torch.device("cpu"),
                fallback_dtype=torch.float64,
            )

        self.assertEqual(str(raised.exception), expected_message)
        self.assertEqual(module.anchor.dtype, torch.float32)
        self.assertTrue(module.anchor.requires_grad)
        self.assertEqual(module.first_buffer.dtype, torch.float32)
        self.assertFalse(module.training)


if __name__ == "__main__":
    unittest.main()
