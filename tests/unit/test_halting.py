import math
import unittest

import torch
import torch.nn as nn

from emperor.halting import (
    HaltingHiddenStateModeOptions,
    SoftHalting,
    SoftHaltingConfig,
    SoftHaltingState,
    StickBreaking,
    StickBreakingConfig,
    StickBreakingState,
)
from emperor.halting._base import HaltingComputation
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
    threshold: float | None = None,
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


def run_step(
    model: StickBreaking | SoftHalting,
    previous_state: StickBreakingState | SoftHaltingState | None,
    raw_hidden: torch.Tensor,
    candidate: torch.Tensor | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
    update_mask: torch.Tensor | None = None,
) -> tuple[StickBreakingState | SoftHaltingState, HaltingComputation]:
    captured: list[HaltingComputation] = []

    def compute(computation: HaltingComputation) -> torch.Tensor:
        captured.append(computation)
        return computation.raw_hidden if candidate is None else candidate

    state = model.run_step(
        previous_state,
        raw_hidden,
        compute,
        valid_mask=valid_mask,
        update_mask=update_mask,
    )
    return state, captured[0]


def halting_cases(input_dim: int = 3):
    return (
        ("stick", StickBreaking(stick_config(input_dim)).eval()),
        ("soft", SoftHalting(soft_config(input_dim)).eval()),
    )


class HaltingConstructionTests(unittest.TestCase):
    def test_registry_builds_each_strategy_and_resolves_strategy_defaults(self) -> None:
        for cfg, strategy_type, default_threshold in (
            (stick_config(threshold=None), StickBreaking, 0.999),
            (soft_config(), SoftHalting, 0.999),
        ):
            with self.subTest(strategy=strategy_type.__name__):
                self.assertIs(cfg._registry_owner(), strategy_type)
                model = cfg.build()
                self.assertIsInstance(model, strategy_type)
                self.assertEqual(model.threshold, default_threshold)

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

    def test_custom_soft_gate_preserves_layer_stack_namespace(self) -> None:
        model = SoftHalting(soft_config(3, custom_gate=True))

        self.assertIsInstance(model._gate, LayerStack)
        self.assertEqual(
            tuple(model.state_dict()),
            ("_gate.layers.0.model.weight_params",),
        )

    def test_canonical_soft_gate_strict_round_trip_preserves_logits(self) -> None:
        cfg = soft_config(3)
        source = SoftHalting(cfg).eval()
        with torch.no_grad():
            source._gate[0].weight.copy_(
                torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.5, -0.2], [-0.3, 0.1, 0.6]])
            )
            source._gate[0].bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
            source._gate[3].weight.copy_(
                torch.tensor([[0.7, -0.4, 0.2], [-0.1, 0.5, 0.3]])
            )
        hidden = torch.tensor([[1.0, -2.0, 0.5]])
        expected = source._SoftHalting__compute_gate_logits(hidden)

        restored = SoftHalting(cfg).eval()
        restored.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(
            tuple(source.state_dict()),
            ("_gate.0.weight", "_gate.0.bias", "_gate.3.weight"),
        )
        torch.testing.assert_close(
            restored._SoftHalting__compute_gate_logits(hidden),
            expected,
        )

    def test_stick_checkpoint_keys_and_behavior_round_trip_strictly(self) -> None:
        cfg = stick_config(3)
        source = StickBreaking(cfg).eval()
        hidden = torch.tensor([[1.0, -2.0, 0.5]])
        expected = source._StickBreaking__compute_gate_logits(hidden)
        state_dict = source.state_dict()

        restored = StickBreaking(cfg).eval()
        restored.load_state_dict(state_dict, strict=True)

        self.assertEqual(tuple(restored.state_dict()), tuple(state_dict))
        torch.testing.assert_close(
            restored._StickBreaking__compute_gate_logits(hidden),
            expected,
        )


class CommonHaltingLifecycleTests(unittest.TestCase):
    def test_normalized_state_and_computation_record_contract(self) -> None:
        hidden = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        candidate = hidden + 10.0

        for name, model in halting_cases():
            with self.subTest(strategy=name):
                state, computation = run_step(model, None, hidden, candidate)
                self.assertIsInstance(computation, HaltingComputation)
                self.assertIs(computation.raw_hidden, hidden)
                self.assertIsNone(computation.context_hidden)
                torch.testing.assert_close(
                    computation.continuation_probability,
                    torch.ones(2),
                )
                self.assertTrue(computation.computation_mask.all())
                self.assertEqual(state.raw_hidden.shape, hidden.shape)
                self.assertEqual(state.output_hidden.shape, hidden.shape)
                self.assertEqual(state.accumulated_hidden.shape, hidden.shape)
                self.assertEqual(state.continuation_probability.shape, (2,))
                self.assertEqual(state.halt_mask.shape, (2,))
                self.assertTrue(state.valid_mask.all())
                self.assertFalse(state.finalized)

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
                    state, _ = run_step(model, None, hidden, hidden + 1)
                    output, loss = model.finalize(state, state.raw_hidden)
                    self.assertEqual(output.shape, hidden.shape)
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue(torch.isfinite(loss).all())

    def test_bool_integer_and_float_binary_masks_are_equivalent(self) -> None:
        hidden = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        mask_values = (
            torch.tensor([True, False, True]),
            torch.tensor([1, 0, 1], dtype=torch.int64),
            torch.tensor([1.0, 0.0, 1.0]),
        )

        for name, _ in halting_cases():
            states = []
            for mask in mask_values:
                model = (
                    StickBreaking(stick_config(3)).eval()
                    if name == "stick"
                    else SoftHalting(soft_config(3)).eval()
                )
                state, _ = run_step(
                    model,
                    None,
                    hidden,
                    hidden + 1,
                    valid_mask=mask,
                )
                states.append(state)
            for state in states[1:]:
                torch.testing.assert_close(state.output_hidden, states[0].output_hidden)
                self.assertTrue(torch.equal(state.valid_mask, states[0].valid_mask))

    def test_fractional_nonfinite_complex_wrong_shape_and_non_tensor_masks_fail(
        self,
    ) -> None:
        hidden = torch.ones(2, 3)
        invalid_cases = (
            (torch.tensor([1.0, 0.5]), ValueError, "binary"),
            (torch.tensor([1.0, float("nan")]), ValueError, "finite"),
            (torch.tensor([1 + 0j, 0 + 0j]), TypeError, "dtype"),
            (torch.ones(2, 1), ValueError, "shape"),
            ([True, False], TypeError, "Tensor"),
        )

        for name, model in halting_cases():
            for mask, error_type, message in invalid_cases:
                with self.subTest(strategy=name, mask=mask):
                    with self.assertRaisesRegex(error_type, message):
                        run_step(model, None, hidden, valid_mask=mask)

    def test_valid_mask_is_permanent(self) -> None:
        hidden = torch.ones(2, 3)
        for name, model in halting_cases():
            with self.subTest(strategy=name):
                state, _ = run_step(
                    model,
                    None,
                    hidden,
                    valid_mask=torch.tensor([True, False]),
                )
                with self.assertRaisesRegex(ValueError, "cannot change"):
                    run_step(
                        model,
                        state,
                        state.raw_hidden,
                        valid_mask=torch.tensor([True, True]),
                    )

    def test_inactive_and_invalid_rows_do_not_advance_or_contribute_to_loss(
        self,
    ) -> None:
        hidden = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        valid_mask = torch.tensor([True, True, False])
        update_mask = torch.tensor([True, False, True])

        for name, model in halting_cases():
            with self.subTest(strategy=name):
                state, computation = run_step(
                    model,
                    None,
                    hidden,
                    hidden + 100,
                    valid_mask=valid_mask,
                    update_mask=update_mask,
                )
                self.assertTrue(
                    torch.equal(
                        computation.computation_mask,
                        torch.tensor([True, False, False]),
                    )
                )
                torch.testing.assert_close(
                    computation.continuation_probability,
                    torch.tensor([1.0, 0.0, 0.0]),
                )
                torch.testing.assert_close(state.raw_hidden[1:], hidden[1:])
                torch.testing.assert_close(state.output_hidden[1:], hidden[1:])
                output, loss = model.finalize(state, state.raw_hidden)
                torch.testing.assert_close(output[1:], hidden[1:])
                if loss.dim() == 0:
                    torch.testing.assert_close(loss, hidden.new_zeros(()))
                else:
                    torch.testing.assert_close(loss[1:], hidden.new_zeros(2))

    def test_all_padding_is_finite_preserves_input_and_has_exact_zero_loss(
        self,
    ) -> None:
        hidden = torch.randn(2, 4, dtype=torch.float64)
        invalid = torch.zeros(2, dtype=torch.bool)

        for name, model in halting_cases(4):
            model = model.to(dtype=torch.float64)
            with self.subTest(strategy=name):
                state, _ = run_step(
                    model,
                    None,
                    hidden,
                    hidden * float("nan"),
                    valid_mask=invalid,
                )
                output, loss = model.finalize(state, state.raw_hidden)
                torch.testing.assert_close(output, hidden)
                self.assertTrue(torch.isfinite(output).all())
                self.assertIsInstance(loss, torch.Tensor)
                self.assertEqual(loss.dtype, hidden.dtype)
                torch.testing.assert_close(loss, torch.zeros_like(loss))

    def test_candidate_must_be_a_matching_feature_last_tensor(self) -> None:
        hidden = torch.ones(2, 3)
        invalid_candidates = (
            torch.ones(2),
            torch.ones(2, 4),
            torch.ones(1, 3),
        )

        for name, model in halting_cases():
            for candidate in invalid_candidates:
                with self.subTest(strategy=name, shape=tuple(candidate.shape)):
                    with self.assertRaises((ValueError, TypeError)):
                        run_step(model, None, hidden, candidate)

    def test_hidden_and_finalize_geometry_validation(self) -> None:
        model = SoftHalting(soft_config(3)).eval()
        with self.assertRaisesRegex(TypeError, "raw_hidden must be a Tensor"):
            model.run_step(None, [[1.0, 2.0, 3.0]], lambda computation: None)
        with self.assertRaisesRegex(ValueError, "rank >= 2"):
            run_step(model, None, torch.ones(3))
        with self.assertRaisesRegex(ValueError, "final dimension"):
            run_step(model, None, torch.ones(2, 4))

        state, _ = run_step(model, None, torch.ones(2, 3))
        with self.assertRaisesRegex(ValueError, "current_hidden"):
            model.finalize(state, torch.ones(1, 3))

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
                        state, _ = run_step(model, None, hidden, hidden + 1)
                        output, loss = model.finalize(state, state.raw_hidden)
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
                    state, _ = run_step(model, None, hidden, hidden + 1)
                    output, loss = model.finalize(state, state.raw_hidden)
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue(torch.isfinite(loss).all())


class StickBreakingLifecycleTests(unittest.TestCase):
    def test_shared_lifecycle_matches_the_unchanged_direct_api(self) -> None:
        for mode in HaltingHiddenStateModeOptions:
            cfg = stick_config(2, threshold=0.999, mode=mode)
            direct_model = StickBreaking(cfg).eval()
            shared_model = StickBreaking(cfg).eval()
            with torch.no_grad():
                direct_model.halting_gate_model[-1].model.weight_params.copy_(
                    torch.tensor(((0.35, -0.45), (-0.2, 0.5)))
                )
            shared_model.load_state_dict(direct_model.state_dict(), strict=True)
            candidates = (
                torch.tensor(((1.0, -2.0), (0.5, 1.5))),
                torch.tensor(((-0.5, 0.25), (2.0, -1.0))),
                torch.tensor(((1.5, 0.75), (-0.25, 0.5))),
            )
            direct_state = None
            shared_state = None

            for candidate in candidates:
                direct_state, direct_output = direct_model.update_halting_state(
                    direct_state,
                    candidate,
                )
                shared_state = shared_model.run_step(
                    shared_state,
                    candidate,
                    lambda _computation, value=candidate: value,
                )

                with self.subTest(mode=mode, step=direct_state.step_count):
                    for field_name in (
                        "halt_mask",
                        "log_continuation",
                        "accumulated_hidden",
                        "output_hidden",
                        "accumulated_halt_probabilities",
                        "accumulated_ponder_cost",
                    ):
                        torch.testing.assert_close(
                            getattr(shared_state, field_name),
                            getattr(direct_state, field_name),
                        )
                    self.assertEqual(shared_state.step_count, direct_state.step_count)
                    torch.testing.assert_close(
                        shared_state.output_hidden, direct_output
                    )

            current_hidden = candidates[-1] * 1.7
            direct_output, direct_loss = direct_model.finalize_weighted_accumulation(
                direct_state,
                current_hidden,
            )
            shared_output, shared_loss = shared_model.finalize(
                shared_state,
                current_hidden,
            )
            torch.testing.assert_close(shared_output, direct_output)
            torch.testing.assert_close(shared_loss, direct_loss)

    def test_computation_precedes_gate_and_equal_logits_preserve_recurrence(
        self,
    ) -> None:
        model = StickBreaking(stick_config(2)).eval()
        events: list[str] = []
        handle = model.halting_gate_model.register_forward_pre_hook(
            lambda _module, _inputs: events.append("gate")
        )
        hidden_0 = torch.tensor([[1.0, 2.0]])
        hidden_1 = torch.tensor([[3.0, 4.0]])

        state = model.run_step(
            None,
            hidden_0,
            lambda _computation: events.append("compute") or hidden_1,
        )
        handle.remove()

        self.assertEqual(events, ["compute", "gate"])
        self.assertEqual(state.step_count, 0)
        torch.testing.assert_close(state.accumulated_hidden, hidden_1 * 0.5)
        torch.testing.assert_close(state.continuation_probability, torch.tensor([0.5]))
        torch.testing.assert_close(
            state.accumulated_halt_probabilities,
            torch.tensor([0.5]),
        )
        torch.testing.assert_close(state.accumulated_ponder_cost, torch.tensor(0.0))

        hidden_2 = torch.tensor([[5.0, 6.0]])
        state, _ = run_step(model, state, state.raw_hidden, hidden_2)
        self.assertEqual(state.step_count, 1)
        torch.testing.assert_close(
            state.accumulated_hidden,
            hidden_1 * 0.5 + hidden_2 * 0.25,
        )
        torch.testing.assert_close(state.continuation_probability, torch.tensor([0.25]))
        torch.testing.assert_close(
            state.accumulated_halt_probabilities,
            torch.tensor([0.75]),
        )
        torch.testing.assert_close(state.accumulated_ponder_cost, torch.tensor([0.25]))

        output, ponder = model.finalize(state, state.raw_hidden)
        torch.testing.assert_close(output, hidden_1 * 0.5 + hidden_2 * 0.5)
        torch.testing.assert_close(ponder, torch.tensor([0.75]))
        self.assertTrue(state.finalized)

    def test_raw_and_accumulated_modes_select_owner_output(self) -> None:
        hidden = torch.tensor([[2.0, 4.0]])
        for mode in HaltingHiddenStateModeOptions:
            model = StickBreaking(stick_config(2, mode=mode)).eval()
            state, _ = run_step(model, None, hidden, hidden)
            expected = (
                hidden * 0.5
                if mode == HaltingHiddenStateModeOptions.ACCUMULATED
                else hidden
            )
            with self.subTest(mode=mode):
                torch.testing.assert_close(state.output_hidden, expected)

    def test_threshold_equality_halts_and_requests_early_stop(self) -> None:
        model = StickBreaking(stick_config(2, threshold=0.5)).eval()
        state, _ = run_step(model, None, torch.ones(2, 2))

        self.assertTrue(state.halt_mask.all())
        self.assertTrue(state.stop_requested)

    def test_halted_rows_preserve_output_while_legacy_computation_continues(
        self,
    ) -> None:
        model = StickBreaking(stick_config(2, threshold=0.5)).eval()
        first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        state, _ = run_step(model, None, first, first)
        previous_output = state.output_hidden.clone()
        candidate = torch.full_like(first, 99.0)

        state, computation = run_step(model, state, state.raw_hidden, candidate)

        self.assertTrue(computation.computation_mask.all())
        torch.testing.assert_close(state.raw_hidden, candidate)
        torch.testing.assert_close(state.output_hidden, previous_output)

    def test_training_noise_is_seeded_and_eval_is_deterministic(self) -> None:
        model = StickBreaking(stick_config(3))
        hidden = torch.tensor([[1.0, -2.0, 3.0], [0.5, 1.5, -1.0]])
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
        first_eval = model._StickBreaking__compute_gate_logits(hidden)
        second_eval = model._StickBreaking__compute_gate_logits(hidden)
        torch.testing.assert_close(first_eval, second_eval)

    def test_backward_reaches_gate_and_input(self) -> None:
        model = StickBreaking(stick_config(3)).eval()
        hidden = torch.tensor(
            [[1.0, -2.0, 3.0], [0.5, 1.5, -1.0]],
            requires_grad=True,
        )
        state, _ = run_step(model, None, hidden, hidden.square())
        output, ponder = model.finalize(state, state.raw_hidden)

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
            model._gate[0].weight.copy_(torch.tensor([[0.5, -0.2], [0.1, 0.4]]))
            model._gate[0].bias.copy_(torch.tensor([0.2, -0.1]))
            model._gate[3].weight.copy_(torch.tensor([[0.6, -0.3], [-0.4, 0.7]]))
        return model

    def test_first_step_skips_gate_and_uses_first_candidate_as_soft_output(
        self,
    ) -> None:
        model = self._nonzero_gate().eval()
        gate_calls: list[torch.Tensor] = []
        handle = model._gate.register_forward_pre_hook(
            lambda _module, inputs: gate_calls.append(inputs[0].detach().clone())
        )
        raw = torch.tensor([[1.0, 2.0]])
        candidate = torch.tensor([[3.0, 4.0]])

        state, computation = run_step(model, None, raw, candidate)
        handle.remove()

        self.assertEqual(gate_calls, [])
        self.assertIsNone(state.gate_input)
        self.assertIsNone(state.gate_logits)
        self.assertIsNone(computation.context_hidden)
        torch.testing.assert_close(state.raw_hidden, candidate)
        torch.testing.assert_close(state.output_hidden, candidate)
        torch.testing.assert_close(
            state.accumulated_hidden, torch.zeros_like(candidate)
        )
        torch.testing.assert_close(state.step_count, torch.zeros(1))

    def test_later_step_gates_previous_raw_and_passes_prior_soft_context(self) -> None:
        model = self._nonzero_gate().eval()
        initial_raw = torch.tensor([[1.0, 2.0]])
        first_candidate = torch.tensor([[3.0, 5.0]])
        second_candidate = torch.tensor([[7.0, 11.0]])
        state, _ = run_step(model, None, initial_raw, first_candidate)

        next_state, computation = run_step(
            model,
            state,
            state.raw_hidden,
            second_candidate,
        )

        torch.testing.assert_close(next_state.gate_input, first_candidate)
        torch.testing.assert_close(computation.raw_hidden, first_candidate)
        torch.testing.assert_close(computation.context_hidden, state.output_hidden)
        expected_halt_mass = next_state.gate_logits[..., 1].exp()
        torch.testing.assert_close(
            next_state.accumulated_hidden,
            expected_halt_mass.unsqueeze(-1) * first_candidate,
        )

    def test_accumulated_extension_passes_partial_accumulation_as_context(self) -> None:
        model = SoftHalting(
            soft_config(
                2,
                mode=HaltingHiddenStateModeOptions.ACCUMULATED,
            )
        ).eval()
        state, _ = run_step(model, None, torch.ones(1, 2), torch.full((1, 2), 2.0))
        state, _ = run_step(model, state, state.raw_hidden, torch.full((1, 2), 4.0))

        _next_state, computation = run_step(
            model,
            state,
            state.raw_hidden,
            torch.full((1, 2), 8.0),
        )

        torch.testing.assert_close(computation.context_hidden, state.accumulated_hidden)

    def test_strict_continuation_boundary_continues_and_lower_values_freeze(
        self,
    ) -> None:
        model = SoftHalting(soft_config(2, threshold=0.5)).eval()
        first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        state, _ = run_step(model, None, first, first)
        model.threshold = 0.5
        state.log_continuation = torch.tensor([0.0, math.log(0.5)])
        previous_output = state.output_hidden.clone()

        next_state, computation = run_step(
            model,
            state,
            state.raw_hidden,
            torch.full_like(first, 10.0),
        )

        self.assertTrue(computation.computation_mask[0])
        self.assertFalse(computation.computation_mask[1])
        self.assertEqual(next_state.continuation_probability[0].item(), 0.5)
        self.assertEqual(next_state.continuation_probability[1].item(), 0.0)
        self.assertFalse(next_state.halt_mask[0])
        self.assertTrue(next_state.halt_mask[1])
        torch.testing.assert_close(
            next_state.output_hidden[0],
            0.5 * first[0] + 0.5 * torch.full_like(first[0], 10.0),
        )
        torch.testing.assert_close(next_state.output_hidden[1], previous_output[1])

    def test_asynchronous_updates_accumulate_the_advanced_domain(self) -> None:
        model = SoftHalting(soft_config(2, threshold=0.999)).eval()
        hidden = torch.tensor(((1.0, 2.0), (3.0, 4.0)))
        state, _ = run_step(
            model,
            None,
            hidden,
            hidden + 1.0,
            update_mask=torch.tensor((True, False)),
        )
        state, _ = run_step(
            model,
            state,
            state.raw_hidden,
            state.raw_hidden + 1.0,
            update_mask=torch.tensor((False, True)),
        )

        self.assertTrue(torch.equal(state.advanced_mask, torch.tensor((True, True))))

    def test_never_updated_valid_rows_do_not_dilute_terminal_loss(self) -> None:
        model = SoftHalting(soft_config(2, threshold=0.999)).eval()
        hidden = torch.tensor(((1.0, 2.0), (3.0, 4.0)))
        update_mask = torch.tensor((True, False))
        state, _ = run_step(
            model,
            None,
            hidden,
            hidden + 1.0,
            update_mask=update_mask,
        )
        state, _ = run_step(
            model,
            state,
            state.raw_hidden,
            state.raw_hidden + 1.0,
            update_mask=update_mask,
        )

        _output, loss = model.finalize(state, state.raw_hidden)

        self.assertTrue(torch.equal(state.advanced_mask, update_mask))
        torch.testing.assert_close(loss, loss.new_tensor(0.5))

    def test_soft_never_requests_early_stop(self) -> None:
        model = SoftHalting(soft_config(2, threshold=0.5)).eval()
        state, _ = run_step(model, None, torch.ones(2, 2))
        for _ in range(4):
            state, _ = run_step(model, state, state.raw_hidden)

        self.assertTrue(state.halt_mask.all())
        self.assertFalse(state.stop_requested)

    def test_terminal_loss_is_masked_and_finalized_once_by_state(self) -> None:
        model = SoftHalting(soft_config(2)).eval()
        valid = torch.tensor([True, False])
        hidden = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        state, _ = run_step(model, None, hidden, hidden, valid_mask=valid)
        state, _ = run_step(
            model,
            state,
            state.raw_hidden,
            hidden * 2,
            valid_mask=valid,
        )

        output, loss = model.finalize(state, state.raw_hidden)

        self.assertIs(output, state.output_hidden)
        self.assertTrue(state.finalized)
        torch.testing.assert_close(loss, loss.new_tensor(0.5))

    def test_finalize_validation_names_current_hidden_exactly(self) -> None:
        model = SoftHalting(soft_config(2)).eval()
        state, _ = run_step(model, None, torch.ones(2, 2))

        with self.assertRaisesRegex(
            TypeError,
            r"^current_hidden must be a Tensor",
        ):
            model.finalize(state, [[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaisesRegex(
            ValueError,
            r"^current_hidden must have shape \(2, 2\)",
        ):
            model.finalize(state, torch.ones(3, 2))

    def test_train_mode_uses_only_seeded_dropout_not_gaussian_noise(self) -> None:
        hidden = torch.tensor([[1.0, -2.0], [0.5, 1.5]])
        no_dropout = self._nonzero_gate(dropout=0.0).train()
        first = no_dropout._SoftHalting__compute_gate_logits(hidden)
        second = no_dropout._SoftHalting__compute_gate_logits(hidden)
        torch.testing.assert_close(first, second)

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
        first = torch.tensor([[1.0, -2.0], [0.5, 1.5]], dtype=torch.float64)
        second = first.clone().requires_grad_()
        third = (first * 2).clone().requires_grad_()
        state, _ = run_step(model, None, first, second)
        state, _ = run_step(model, state, state.raw_hidden, third)
        output, loss = model.finalize(state, state.raw_hidden)

        (output.square().sum() + loss).backward()

        self.assertIsNotNone(second.grad)
        self.assertIsNotNone(third.grad)
        for name, parameter in model._gate.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
