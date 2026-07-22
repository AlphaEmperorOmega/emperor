import math
import unittest

import torch
from lightning import LightningModule, Trainer
from torch import nn

from emperor.layers import LayerState
from emperor.memory import (
    GatedResidualDynamicMemoryConfig,
    MemoryMonitorCallback,
    WeightedDynamicMemoryConfig,
)
from emperor.memory._base import DynamicMemoryAbstract
from emperor.memory._monitoring import (
    _MemoryDiagnostics,
    _MemoryObservation,
)
from support.monitor import orchestration_calls
from unit.test_memory import make_memory_config


def build_memory(
    config_cls=GatedResidualDynamicMemoryConfig,
):
    return make_memory_config(
        config_cls=config_cls,
        input_dim=4,
        output_dim=4,
    ).build()


class _MemoryOwner(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.memory = build_memory()
        self.logged_scalars: list[tuple[str, torch.Tensor]] = []

    def log(
        self,
        name: str,
        value: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if torch.is_tensor(value):
            self.logged_scalars.append((name, value))


class _GateFreeMemory(DynamicMemoryAbstract):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        self.VALIDATOR.validate_forward_inputs(logits, self.memory_dim)
        return logits + 1.0


class _GateFreeOwner(_MemoryOwner):
    def __init__(self) -> None:
        LightningModule.__init__(self)
        self.memory = _GateFreeMemory(make_memory_config(input_dim=4, output_dim=4))
        self.logged_scalars = []


class TestMemoryMonitorCallback(unittest.TestCase):
    def test_tracking_orchestration_lists_each_tracked_fact(self) -> None:
        orchestration = (
            MemoryMonitorCallback._MemoryMonitorCallback__track_memory_diagnostics
        )

        self.assertEqual(
            orchestration_calls(orchestration),
            (
                "__track_output_mean",
                "__track_output_variance",
                "__track_output_l2_norm",
                "__track_contribution_delta_mean",
                "__track_contribution_delta_variance",
                "__track_contribution_delta_norm",
                "__track_contribution_relative_delta_norm",
                "__track_gate_open_mean",
                "__track_gate_open_fraction",
                "__track_gate_saturation_fraction",
            ),
        )

    def test_constructor_preserves_defaults_and_rejects_invalid_intervals(
        self,
    ) -> None:
        self.assertEqual(MemoryMonitorCallback().log_every_n_steps, 100)
        self.assertEqual(
            MemoryMonitorCallback(log_every_n_steps=7).log_every_n_steps,
            7,
        )
        for value in (True, 1.5, "1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    TypeError,
                    "^log_every_n_steps must be a positive integer, "
                    f"received {type(value).__name__}\\.$",
                ):
                    MemoryMonitorCallback(log_every_n_steps=value)
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "^log_every_n_steps must be greater than 0\\.$",
                ):
                    MemoryMonitorCallback(log_every_n_steps=value)

    def test_diagnostics_match_exact_output_and_contribution_equations(self) -> None:
        inputs = torch.tensor(
            [[1.0, -2.0], [3.0, 0.5]],
            dtype=torch.float64,
        )
        outputs = torch.tensor(
            [[2.0, -1.0], [1.0, 4.5]],
            dtype=torch.float64,
        )
        metrics = _MemoryDiagnostics.calculate(inputs, outputs)
        delta = outputs - inputs

        torch.testing.assert_close(metrics.output_mean, outputs.mean())
        torch.testing.assert_close(
            metrics.output_variance,
            outputs.var(unbiased=False),
        )
        torch.testing.assert_close(metrics.output_norm, outputs.norm())
        torch.testing.assert_close(metrics.delta_mean, delta.mean())
        torch.testing.assert_close(
            metrics.delta_variance,
            delta.var(unbiased=False),
        )
        torch.testing.assert_close(metrics.delta_norm, delta.norm())
        torch.testing.assert_close(
            metrics.relative_delta_norm,
            delta.norm() / inputs.norm().clamp_min(1e-6),
        )

        zero_inputs = torch.zeros(2, 2)
        nonzero_outputs = torch.ones(2, 2)
        zero_metrics = _MemoryDiagnostics.calculate(
            zero_inputs,
            nonzero_outputs,
        )
        torch.testing.assert_close(
            zero_metrics.relative_delta_norm,
            (nonzero_outputs - zero_inputs).norm() / 1e-6,
        )

    def test_gate_diagnostics_distinguish_weighted_and_sigmoid_variants(
        self,
    ) -> None:
        low_saturation_logit = math.log(0.01 / 0.99)
        high_saturation_logit = math.log(0.99 / 0.01)
        sigmoid_logits = torch.tensor(
            [
                -10.0,
                low_saturation_logit,
                0.0,
                high_saturation_logit,
                10.0,
                1.0,
            ]
        )
        sigmoid_metrics = _MemoryDiagnostics.calculate_gate(
            build_memory(GatedResidualDynamicMemoryConfig),
            sigmoid_logits,
        )
        sigmoid_values = torch.sigmoid(sigmoid_logits)
        torch.testing.assert_close(
            sigmoid_metrics.open_mean,
            sigmoid_values.mean(),
        )
        torch.testing.assert_close(
            sigmoid_metrics.open_fraction,
            (sigmoid_values > 0.5).float().mean(),
        )
        torch.testing.assert_close(
            sigmoid_metrics.saturation_fraction,
            (
                (sigmoid_logits < low_saturation_logit)
                | (sigmoid_logits > high_saturation_logit)
            )
            .float()
            .mean(),
        )

        weighted_logits = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 2.0], [3.0, -1.0]],
                [[1.0, 2.0], [-2.0, 1.0], [4.0, 4.0]],
            ]
        )
        weighted_metrics = _MemoryDiagnostics.calculate_gate(
            build_memory(WeightedDynamicMemoryConfig),
            weighted_logits,
        )
        memory_share = torch.softmax(weighted_logits, dim=-1)[..., -1]
        torch.testing.assert_close(
            weighted_metrics.open_mean,
            memory_share.mean(),
        )
        torch.testing.assert_close(
            weighted_metrics.open_fraction,
            (memory_share > 0.5).float().mean(),
        )
        self.assertIsNone(weighted_metrics.saturation_fraction)

    def test_hidden_extraction_accepts_tensor_and_layer_state_only(self) -> None:
        extract = MemoryMonitorCallback._MemoryMonitorCallback__extract_hidden_tensor
        tensor = torch.tensor([[1.0, 2.0]])
        self.assertIs(extract(tensor), tensor)
        state = LayerState(hidden=tensor)
        self.assertIs(extract(state), tensor)
        self.assertIsNone(extract(object()))

    def test_gate_discovery_uses_real_variant_submodules(self) -> None:
        find_gate = MemoryMonitorCallback._MemoryMonitorCallback__find_gate_submodule
        gated = build_memory(GatedResidualDynamicMemoryConfig)
        weighted = build_memory(WeightedDynamicMemoryConfig)

        self.assertIs(find_gate(gated), gated.memory_gate_model)
        self.assertIs(find_gate(weighted), weighted.memory_weight_model)
        self.assertIsNone(find_gate(nn.Linear(2, 2)))

    def test_capture_and_pre_hooks_replace_only_fresh_tensor_logits(self) -> None:
        callback = MemoryMonitorCallback()
        memory = build_memory()
        capture = callback._MemoryMonitorCallback__make_gate_capture_hook("memory")
        clear = callback._MemoryMonitorCallback__make_memory_pre_hook("memory")
        first = torch.tensor([[1.0, 2.0]])
        second = torch.tensor([[3.0, 4.0]])

        capture(memory.memory_gate_model, (), object())
        self.assertEqual(callback._latest_gate_logits, {})
        capture(memory.memory_gate_model, (), LayerState(hidden=first))
        torch.testing.assert_close(
            callback._latest_gate_logits["memory"],
            first,
        )
        capture(memory.memory_gate_model, (), second)
        torch.testing.assert_close(
            callback._latest_gate_logits["memory"],
            second,
        )
        clear(memory, ())
        self.assertEqual(callback._latest_gate_logits, {})

    def test_malformed_forward_hook_payloads_are_ignored_after_gate_cleanup(
        self,
    ) -> None:
        callback = MemoryMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        owner = _MemoryOwner()
        memory = owner.memory
        hook = callback._MemoryMonitorCallback__make_memory_forward_hook(
            "memory",
            trainer,
            owner,
        )

        for inputs, output in (
            ((), torch.ones(1, 4)),
            ((object(),), torch.ones(1, 4)),
            ((torch.ones(1, 4),), object()),
        ):
            callback._latest_gate_logits["memory"] = torch.ones(1, 4)
            hook(memory, inputs, output)
            self.assertEqual(callback._latest_gate_logits, {})
        self.assertEqual(owner.logged_scalars, [])

    def test_build_context_detaches_inputs_outputs_and_optional_gate(self) -> None:
        owner = _MemoryOwner()
        inputs = torch.tensor([[1.0, -2.0]], requires_grad=True)
        outputs = torch.tensor([[3.0, 4.0]], requires_grad=True)
        observation = _MemoryObservation(
            memory_module=owner.memory,
            input_values=inputs,
            output_values=outputs,
            gate_logits=None,
        )

        context = MemoryMonitorCallback._MemoryMonitorCallback__build_tracking_context(
            owner,
            "memory",
            observation,
        )

        self.assertIsNone(context.gate_metrics)
        for metric in (
            context.memory_metrics.output_mean,
            context.memory_metrics.output_variance,
            context.memory_metrics.output_norm,
            context.memory_metrics.delta_mean,
            context.memory_metrics.delta_variance,
            context.memory_metrics.delta_norm,
            context.memory_metrics.relative_delta_norm,
        ):
            self.assertEqual(metric.dtype, torch.float32)
            self.assertFalse(metric.requires_grad)

        callback = MemoryMonitorCallback()
        callback._MemoryMonitorCallback__track_memory_diagnostics(context)
        self.assertEqual(len(owner.logged_scalars), 7)
        self.assertFalse(any("/gate/" in name for name, _ in owner.logged_scalars))

    def test_owner_without_memory_registers_no_hooks(self) -> None:
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        callback = MemoryMonitorCallback()
        owner = LightningModule()

        callback.on_fit_start(trainer, owner)

        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._latest_gate_logits, {})

    def test_gate_free_memory_registers_only_memory_hooks_and_logs_no_gate(
        self,
    ) -> None:
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        callback = MemoryMonitorCallback(log_every_n_steps=1)
        owner = _GateFreeOwner()

        callback.on_fit_start(trainer, owner)
        callback._latest_gate_logits["memory"] = torch.full((2, 4), 100.0)
        output = owner.memory(torch.zeros(2, 4))

        self.assertEqual(len(callback._hooks), 2)
        torch.testing.assert_close(output, torch.ones(2, 4))
        self.assertEqual(len(owner.logged_scalars), 7)
        self.assertFalse(any("/gate/" in name for name, _ in owner.logged_scalars))
        callback.on_fit_end(trainer, owner)

    def test_real_objects_allow_repeated_start_and_cleanup_without_duplicates(
        self,
    ) -> None:
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        owner = _MemoryOwner()
        callback = MemoryMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(trainer, owner)
        first_hook_count = len(callback._hooks)
        callback.on_fit_start(trainer, owner)

        self.assertEqual(len(callback._hooks), first_hook_count)
        self.assertEqual(first_hook_count, 3)
        owner.memory(torch.ones(2, 4))
        self.assertTrue(owner.logged_scalars)
        callback.on_fit_end(trainer, owner)
        callback.on_fit_end(trainer, owner)
        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._latest_gate_logits, {})


if __name__ == "__main__":
    unittest.main()
