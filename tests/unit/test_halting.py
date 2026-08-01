import math
import unittest

import torch
import torch.nn as nn

from emperor.halting import (
    HaltingHiddenStateModeOptions,
    SoftHalting,
    SoftHaltingConfig,
    StickBreaking,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig


def gate_config(input_dim: int = 4, *, num_layers: int = 1) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=input_dim,
        output_dim=2,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DISABLED,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def stick_config(
    input_dim: int = 4,
    *,
    threshold: float | None = 0.99,
    mode: HaltingHiddenStateModeOptions = HaltingHiddenStateModeOptions.RAW,
) -> StickBreakingConfig:
    return StickBreakingConfig(
        input_dim=input_dim,
        threshold=threshold,
        dropout_probability=None,
        hidden_state_mode=mode,
        halting_gate_config=gate_config(input_dim),
    )


def soft_config(
    input_dim: int = 4,
    *,
    threshold: float | None = 0.999,
    dropout: float | None = 0.0,
    mode: HaltingHiddenStateModeOptions = HaltingHiddenStateModeOptions.RAW,
    custom_gate: bool = False,
) -> SoftHaltingConfig:
    return SoftHaltingConfig(
        input_dim=input_dim,
        threshold=threshold,
        dropout_probability=dropout,
        hidden_state_mode=mode,
        halting_gate_config=gate_config(input_dim) if custom_gate else None,
    )


def halting_cases(input_dim: int = 3):
    return (
        ("stick", StickBreaking(stick_config(input_dim)).eval()),
        ("soft", SoftHalting(soft_config(input_dim)).eval()),
    )


class HaltingConstructionTests(unittest.TestCase):
    def test_registry_builds_each_strategy_with_an_explicit_threshold(self) -> None:
        for cfg, strategy_type in (
            (stick_config(threshold=0.999), StickBreaking),
            (soft_config(threshold=0.999), SoftHalting),
        ):
            with self.subTest(strategy=strategy_type.__name__):
                self.assertIs(cfg._registry_owner(), strategy_type)
                model = cfg.build()
                self.assertIsInstance(model, strategy_type)
                self.assertEqual(model.threshold, 0.999)

    def test_explicit_threshold_and_overrides_are_authoritative(self) -> None:
        base = soft_config(3, threshold=0.8, custom_gate=True)
        overrides = SoftHaltingConfig(
            input_dim=5,
            threshold=0.6,
            hidden_state_mode=HaltingHiddenStateModeOptions.ACCUMULATED,
        )

        model = SoftHalting(base, overrides)

        self.assertEqual(model.input_dim, 5)
        self.assertEqual(model.threshold, 0.6)
        self.assertEqual(
            model.hidden_state_mode,
            HaltingHiddenStateModeOptions.ACCUMULATED,
        )
        self.assertEqual(model.halting_gate_config, base.halting_gate_config)

    def test_model_config_owner_is_accepted(self) -> None:
        halting_config = stick_config(2, threshold=0.75)
        model = StickBreaking(LayerConfig(halting_config=halting_config))

        self.assertIs(model.cfg, halting_config)
        self.assertEqual(model.threshold, 0.75)

        soft_halting_config = soft_config(2, threshold=0.75)
        soft_model = SoftHalting(LayerConfig(halting_config=soft_halting_config))

        self.assertIs(soft_model.cfg, soft_halting_config)
        self.assertEqual(soft_model.threshold, 0.75)

    def test_canonical_soft_gate_matches_sut_topology_and_initialization(self) -> None:
        with torch.random.fork_rng():
            torch.manual_seed(71)
            expected_first_layer = nn.Linear(4, 4, bias=True)

            torch.manual_seed(71)
            model = SoftHalting(soft_config(4, dropout=0.2))

        self.assertIsInstance(model._gate, nn.Sequential)
        self.assertEqual(len(model._gate), 4)
        self.assertIsInstance(model._gate[0], nn.Linear)
        self.assertIsInstance(model._gate[1], nn.GELU)
        self.assertIsInstance(model._gate[2], nn.Dropout)
        self.assertIsInstance(model._gate[3], nn.Linear)
        self.assertTrue(model._gate[0].bias is not None)
        self.assertIsNone(model._gate[3].bias)
        self.assertEqual(model._gate[2].p, 0.2)
        torch.testing.assert_close(
            model._gate[0].weight,
            expected_first_layer.weight,
        )
        torch.testing.assert_close(
            model._gate[0].bias,
            expected_first_layer.bias,
        )
        torch.testing.assert_close(
            model._gate[3].weight,
            torch.zeros_like(model._gate[3].weight),
        )

        no_dropout_model = SoftHalting(soft_config(4, dropout=None))
        self.assertEqual(no_dropout_model._gate[2].p, 0.0)

    def test_custom_soft_gate_uses_input_override_and_every_hidden_layer(self) -> None:
        custom_gate = gate_config(input_dim=5, num_layers=3)
        model = SoftHalting(
            SoftHaltingConfig(
                input_dim=2,
                threshold=0.9,
                dropout_probability=1.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=custom_gate,
            )
        )
        first, second, output = model._gate.layers
        self.assertEqual(first.model.weight_params.shape, (2, 5))

        with torch.no_grad():
            first.model.weight_params.copy_(
                torch.tensor(
                    (
                        (0.5, -0.2),
                        (0.1, 0.4),
                        (-0.3, 0.7),
                        (0.8, 0.2),
                        (-0.6, -0.1),
                    )
                ).T
            )
            first.model.bias_params.copy_(torch.tensor((0.2, -0.1, 0.3, -0.4, 0.5)))
            second.model.weight_params.copy_(torch.eye(5) * 0.7 + 0.1)
            second.model.bias_params.copy_(torch.tensor((-0.2, 0.1, 0.4, -0.3, 0.2)))
            output.model.weight_params.copy_(
                torch.tensor(
                    (
                        (0.6, -0.4, 0.2, 0.1, -0.3),
                        (-0.2, 0.7, -0.5, 0.4, 0.3),
                    )
                ).T
            )

        hidden = torch.tensor(
            (
                ((0.2, -0.5), (1.0, 0.3), (-0.7, 0.8)),
                ((0.4, 0.9), (-0.2, -0.6), (0.5, -0.1)),
            )
        )
        flat_hidden = hidden.reshape(-1, 2)
        first_hidden = torch.nn.functional.linear(
            flat_hidden,
            first.model.weight_params.T,
            first.model.bias_params,
        )
        second_hidden = torch.nn.functional.linear(
            first_hidden,
            second.model.weight_params.T,
            second.model.bias_params,
        )
        expected_eval_logits = torch.nn.functional.log_softmax(
            torch.nn.functional.linear(
                second_hidden,
                output.model.weight_params.T,
            ).reshape(2, 3, 2),
            dim=-1,
        )

        model.eval()
        torch.testing.assert_close(
            model._SoftHalting__compute_gate_logits(hidden),
            expected_eval_logits,
        )
        model.train()
        torch.testing.assert_close(
            model._SoftHalting__compute_gate_logits(hidden),
            torch.full((2, 3, 2), -math.log(2.0)),
        )

    def test_strategy_checkpoint_namespaces_round_trip_strictly(self) -> None:
        soft_cfg = soft_config(3)
        source_soft = SoftHalting(soft_cfg).eval()
        with torch.no_grad():
            source_soft._gate[0].weight.copy_(
                torch.tensor(((0.2, -0.1, 0.3), (0.4, 0.5, -0.2), (-0.3, 0.1, 0.6)))
            )
            source_soft._gate[0].bias.copy_(torch.tensor((0.1, -0.2, 0.3)))
            source_soft._gate[3].weight.copy_(
                torch.tensor(((0.7, -0.4, 0.2), (-0.1, 0.5, 0.3)))
            )
        hidden = torch.tensor(((1.0, -2.0, 0.5),))
        expected_soft = source_soft._SoftHalting__compute_gate_logits(hidden)
        restored_soft = SoftHalting(soft_cfg).eval()
        restored_soft.load_state_dict(source_soft.state_dict(), strict=True)

        self.assertEqual(
            tuple(source_soft.state_dict()),
            ("_gate.0.weight", "_gate.0.bias", "_gate.3.weight"),
        )
        torch.testing.assert_close(
            restored_soft._SoftHalting__compute_gate_logits(hidden),
            expected_soft,
        )

        stick_cfg = stick_config(3)
        source_stick = StickBreaking(stick_cfg).eval()
        expected_stick = source_stick._StickBreaking__compute_gate_logits(hidden)
        restored_stick = StickBreaking(stick_cfg).eval()
        restored_stick.load_state_dict(source_stick.state_dict(), strict=True)
        self.assertEqual(
            tuple(restored_stick.state_dict()),
            tuple(source_stick.state_dict()),
        )
        torch.testing.assert_close(
            restored_stick._StickBreaking__compute_gate_logits(hidden),
            expected_stick,
        )

    def test_custom_soft_gate_preserves_layer_stack_namespace(self) -> None:
        model = SoftHalting(soft_config(3, custom_gate=True))

        self.assertIsInstance(model._gate, LayerStack)
        self.assertEqual(
            tuple(model.state_dict()),
            ("_gate.layers.0.model.weight_params",),
        )


class CommonOfficialLifecycleTests(unittest.TestCase):
    def test_rank_two_rank_three_and_non_contiguous_inputs(self) -> None:
        inputs = (
            torch.randn(4, 3),
            torch.randn(2, 5, 3),
            torch.randn(2, 3, 5).transpose(1, 2),
        )
        self.assertFalse(inputs[-1].is_contiguous())

        for name, model in halting_cases():
            for hidden in inputs:
                with self.subTest(strategy=name, shape=tuple(hidden.shape)):
                    state, owner_output = model.update_halting_state(
                        None,
                        hidden + 1.0,
                    )
                    output, loss = model.finalize_weighted_accumulation(
                        state,
                        owner_output,
                    )
                    self.assertEqual(owner_output.shape, hidden.shape)
                    self.assertEqual(output.shape, hidden.shape)
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue(torch.isfinite(loss).all())

    def test_hidden_and_finalization_geometry_validation(self) -> None:
        for name, model in halting_cases():
            with self.subTest(strategy=name, case="non-tensor"):
                with self.assertRaisesRegex(TypeError, "must be a Tensor"):
                    model.update_halting_state(None, [[1.0, 2.0, 3.0]])
            with self.subTest(strategy=name, case="rank"):
                with self.assertRaisesRegex(ValueError, "rank >= 2"):
                    model.update_halting_state(None, torch.ones(3))
            with self.subTest(strategy=name, case="feature-dimension"):
                with self.assertRaisesRegex(ValueError, "final dimension"):
                    model.update_halting_state(None, torch.ones(2, 4))

            state, _ = model.update_halting_state(None, torch.ones(2, 3))
            with self.subTest(strategy=name, case="final-shape"):
                with self.assertRaisesRegex(ValueError, "current_hidden"):
                    model.finalize_weighted_accumulation(
                        state,
                        torch.ones(1, 3),
                    )

    def test_device_and_dtype_follow_hidden(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                for name, model in halting_cases():
                    with self.subTest(strategy=name, device=device, dtype=dtype):
                        model = model.to(device=device, dtype=dtype)
                        hidden = torch.ones(2, 3, device=device, dtype=dtype)
                        state, owner_output = model.update_halting_state(
                            None,
                            hidden + 1.0,
                        )
                        output, loss = model.finalize_weighted_accumulation(
                            state,
                            owner_output,
                        )
                        self.assertEqual(output.device, device)
                        self.assertEqual(output.dtype, dtype)
                        self.assertEqual(loss.device, device)
                        self.assertEqual(loss.dtype, dtype)
                        for value in vars(state).values():
                            if isinstance(value, torch.Tensor):
                                self.assertEqual(value.device, device)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_low_precision_smoke(self) -> None:
        device = torch.device("cuda")
        for dtype in (torch.float16, torch.bfloat16):
            for name, model in halting_cases():
                with self.subTest(strategy=name, dtype=dtype):
                    model = model.to(device=device, dtype=dtype)
                    hidden = torch.randn(2, 3, device=device, dtype=dtype)
                    state, owner_output = model.update_halting_state(None, hidden)
                    output, loss = model.finalize_weighted_accumulation(
                        state,
                        owner_output,
                    )
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue(torch.isfinite(loss).all())


class StickBreakingLifecycleTests(unittest.TestCase):
    def test_equal_logits_preserve_the_stick_breaking_recurrence(self) -> None:
        model = StickBreaking(stick_config(2)).eval()
        first_candidate = torch.tensor(((3.0, 4.0),))
        second_candidate = torch.tensor(((5.0, 6.0),))

        state, _ = model.update_halting_state(None, first_candidate)
        self.assertEqual(state.step_count, 0)
        torch.testing.assert_close(state.accumulated_hidden, first_candidate * 0.5)
        torch.testing.assert_close(
            state.accumulated_halt_probabilities,
            torch.tensor((0.5,)),
        )

        state, _ = model.update_halting_state(state, second_candidate)
        self.assertEqual(state.step_count, 1)
        torch.testing.assert_close(
            state.accumulated_hidden,
            first_candidate * 0.5 + second_candidate * 0.25,
        )
        torch.testing.assert_close(
            state.accumulated_halt_probabilities,
            torch.tensor((0.75,)),
        )
        torch.testing.assert_close(
            state.accumulated_ponder_cost,
            torch.tensor((0.25,)),
        )

        output, ponder = model.finalize_weighted_accumulation(
            state,
            second_candidate,
        )
        torch.testing.assert_close(
            output,
            first_candidate * 0.5 + second_candidate * 0.5,
        )
        torch.testing.assert_close(ponder, torch.tensor((0.75,)))

    def test_raw_and_accumulated_modes_select_owner_output(self) -> None:
        hidden = torch.tensor(((2.0, 4.0),))
        for mode in HaltingHiddenStateModeOptions:
            model = StickBreaking(stick_config(2, mode=mode)).eval()
            state, owner_output = model.update_halting_state(None, hidden)
            expected = (
                hidden * 0.5
                if mode == HaltingHiddenStateModeOptions.ACCUMULATED
                else hidden
            )
            with self.subTest(mode=mode):
                torch.testing.assert_close(state.output_hidden, expected)
                torch.testing.assert_close(owner_output, expected)

    def test_threshold_equality_halts(self) -> None:
        model = StickBreaking(stick_config(2, threshold=0.5)).eval()

        state, _ = model.update_halting_state(None, torch.ones(2, 2))

        self.assertTrue(state.halt_mask.all())

    def test_halted_rows_preserve_their_owner_output(self) -> None:
        model = StickBreaking(stick_config(2, threshold=0.5)).eval()
        first = torch.tensor(((1.0, 2.0), (3.0, 4.0)))
        state, _ = model.update_halting_state(None, first)
        previous_output = state.output_hidden.clone()

        state, _ = model.update_halting_state(state, torch.full_like(first, 99.0))

        torch.testing.assert_close(state.output_hidden, previous_output)

    def test_training_noise_is_seeded_and_eval_is_deterministic(self) -> None:
        model = StickBreaking(stick_config(3))
        hidden = torch.tensor(((1.0, -2.0, 3.0), (0.5, 1.5, -1.0)))
        model.train()
        torch.manual_seed(123)
        first = model._StickBreaking__compute_gate_logits(hidden)
        torch.manual_seed(123)
        second = model._StickBreaking__compute_gate_logits(hidden)
        torch.manual_seed(124)
        third = model._StickBreaking__compute_gate_logits(hidden)
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.allclose(first, third))

        model.eval()
        torch.testing.assert_close(
            model._StickBreaking__compute_gate_logits(hidden),
            model._StickBreaking__compute_gate_logits(hidden),
        )

    def test_backward_reaches_gate_and_input(self) -> None:
        model = StickBreaking(stick_config(3)).eval()
        hidden = torch.tensor(
            ((1.0, -2.0, 3.0), (0.5, 1.5, -1.0)),
            requires_grad=True,
        )
        candidate = hidden.square()
        state, _ = model.update_halting_state(None, candidate)
        output, ponder = model.finalize_weighted_accumulation(state, candidate)

        (output.sum() + ponder.sum()).backward()

        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.any(hidden.grad != 0))
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.halting_gate_model.parameters()
            )
        )


class SoftHaltingLifecycleTests(unittest.TestCase):
    def _nonzero_gate(self, *, dropout: float = 0.0) -> SoftHalting:
        model = SoftHalting(soft_config(2, threshold=0.8, dropout=dropout))
        with torch.no_grad():
            model._gate[0].weight.copy_(torch.tensor(((0.5, -0.2), (0.1, 0.4))))
            model._gate[0].bias.copy_(torch.tensor((0.2, -0.1)))
            model._gate[3].weight.copy_(torch.tensor(((0.6, -0.3), (-0.4, 0.7))))
        return model

    def test_first_step_skips_gate_and_uses_candidate_as_output(self) -> None:
        model = self._nonzero_gate().eval()
        gate_calls: list[torch.Tensor] = []
        handle = model._gate.register_forward_pre_hook(
            lambda _module, inputs: gate_calls.append(inputs[0].detach().clone())
        )
        candidate = torch.tensor(((3.0, 4.0),))

        state, owner_output = model.update_halting_state(None, candidate)
        handle.remove()

        self.assertEqual(gate_calls, [])
        self.assertIsNone(state.gate_input)
        self.assertIsNone(state.gate_logits)
        torch.testing.assert_close(state.raw_hidden, candidate)
        torch.testing.assert_close(state.output_hidden, candidate)
        torch.testing.assert_close(owner_output, candidate)
        torch.testing.assert_close(
            state.accumulated_hidden,
            torch.zeros_like(candidate),
        )
        torch.testing.assert_close(state.step_count, torch.zeros(1))

    def test_later_step_gates_the_previous_raw_hidden(self) -> None:
        model = self._nonzero_gate().eval()
        first_candidate = torch.tensor(((3.0, 5.0),))
        second_candidate = torch.tensor(((7.0, 11.0),))
        state, _ = model.update_halting_state(None, first_candidate)

        next_state, _ = model.update_halting_state(state, second_candidate)

        torch.testing.assert_close(next_state.gate_input, first_candidate)
        expected_halt_mass = next_state.gate_logits[..., 1].exp()
        torch.testing.assert_close(
            next_state.accumulated_hidden,
            expected_halt_mass.unsqueeze(-1) * first_candidate,
        )

    def test_strict_continuation_boundary_continues_and_lower_values_freeze(
        self,
    ) -> None:
        model = SoftHalting(soft_config(2, threshold=0.5)).eval()
        first = torch.tensor(((1.0, 2.0), (3.0, 4.0)))
        state, _ = model.update_halting_state(None, first)
        state.log_continuation = torch.tensor((0.0, math.log(0.5)))
        previous_output = state.output_hidden.clone()

        next_state, _ = model.update_halting_state(
            state,
            torch.full_like(first, 10.0),
        )

        self.assertEqual(next_state.continuation_probability[0].item(), 0.5)
        self.assertEqual(next_state.continuation_probability[1].item(), 0.0)
        self.assertFalse(next_state.halt_mask[0])
        self.assertTrue(next_state.halt_mask[1])
        torch.testing.assert_close(
            next_state.output_hidden[0],
            0.5 * first[0] + 0.5 * torch.full_like(first[0], 10.0),
        )
        torch.testing.assert_close(next_state.output_hidden[1], previous_output[1])
        torch.testing.assert_close(next_state.raw_hidden[1], first[1])

    def test_halted_rows_remain_frozen(self) -> None:
        model = SoftHalting(soft_config(2, threshold=0.5)).eval()
        candidate = torch.ones(2, 2)
        state, _ = model.update_halting_state(None, candidate)
        state, _ = model.update_halting_state(state, candidate * 2.0)
        state, _ = model.update_halting_state(state, candidate * 3.0)
        self.assertTrue(state.halt_mask.all())
        previous_output = state.output_hidden.clone()
        previous_raw = state.raw_hidden.clone()

        state, _ = model.update_halting_state(state, candidate * 100.0)

        torch.testing.assert_close(state.output_hidden, previous_output)
        torch.testing.assert_close(state.raw_hidden, previous_raw)

    def test_finalization_returns_the_soft_output_and_expected_depth(self) -> None:
        model = SoftHalting(soft_config(2)).eval()
        first = torch.tensor(((2.0, 4.0), (6.0, 8.0)))
        state, _ = model.update_halting_state(None, first)
        state, _ = model.update_halting_state(state, first * 2.0)

        output, loss = model.finalize_weighted_accumulation(
            state,
            state.raw_hidden,
        )

        self.assertIs(output, state.output_hidden)
        torch.testing.assert_close(loss, loss.new_tensor(0.5))

    def test_finalization_validation_names_current_hidden_exactly(self) -> None:
        model = SoftHalting(soft_config(2)).eval()
        state, _ = model.update_halting_state(None, torch.ones(2, 2))

        with self.assertRaisesRegex(TypeError, r"^current_hidden must be a Tensor"):
            model.finalize_weighted_accumulation(
                state,
                [[1.0, 2.0], [3.0, 4.0]],
            )
        with self.assertRaisesRegex(
            ValueError,
            r"^current_hidden must have shape \(2, 2\)",
        ):
            model.finalize_weighted_accumulation(state, torch.ones(3, 2))

    def test_train_mode_uses_only_seeded_dropout_not_gaussian_noise(self) -> None:
        hidden = torch.tensor(((1.0, -2.0), (0.5, 1.5)))
        no_dropout = self._nonzero_gate(dropout=0.0).train()
        torch.testing.assert_close(
            no_dropout._SoftHalting__compute_gate_logits(hidden),
            no_dropout._SoftHalting__compute_gate_logits(hidden),
        )

        with_dropout = self._nonzero_gate(dropout=0.5).train()
        torch.manual_seed(31)
        first_dropout = with_dropout._SoftHalting__compute_gate_logits(hidden)
        torch.manual_seed(31)
        second_dropout = with_dropout._SoftHalting__compute_gate_logits(hidden)
        torch.manual_seed(32)
        third_dropout = with_dropout._SoftHalting__compute_gate_logits(hidden)
        torch.testing.assert_close(first_dropout, second_dropout)
        self.assertFalse(torch.allclose(first_dropout, third_dropout))

        with_dropout.eval()
        torch.testing.assert_close(
            with_dropout._SoftHalting__compute_gate_logits(hidden),
            with_dropout._SoftHalting__compute_gate_logits(hidden),
        )

    def test_backward_reaches_every_canonical_gate_parameter_and_inputs(self) -> None:
        model = self._nonzero_gate().double().eval()
        second = torch.tensor(
            ((1.0, -2.0), (0.5, 1.5)),
            dtype=torch.float64,
            requires_grad=True,
        )
        third = (second.detach() * 2.0).requires_grad_()
        state, _ = model.update_halting_state(None, second)
        state, _ = model.update_halting_state(state, third)
        output, loss = model.finalize_weighted_accumulation(
            state,
            state.raw_hidden,
        )

        (output.square().sum() + loss).backward()

        self.assertIsNotNone(second.grad)
        self.assertIsNotNone(third.grad)
        for name, parameter in model._gate.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
