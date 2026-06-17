from emperor.base.layer.residual import ResidualConnectionOptions
import torch
import unittest

from emperor.base.layer import LayerStack

from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.base.layer import (
    LayerConfig,
    LayerStackConfig,
)
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.halting.core.variants import (
    StickBreaking,
    StickBreakingState,
)
from emperor.linears.core.config import LinearLayerConfig
from emperor.base.options import LastLayerBiasOptions


class WrappedHaltingConfig:
    def __init__(self, halting_config: StickBreakingConfig):
        self.halting_config = halting_config


class TestHalting(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        gate_num_layers: int = 3,
        gate_activation: ActivationOptions = ActivationOptions.DISABLED,
        gate_residual_connection_option: ResidualConnectionOptions = (
            ResidualConnectionOptions.DISABLED
        ),
        gate_dropout_probability: float = 0.0,
        gate_bias_flag: bool = True,
        threshold: float = 0.99,
        halting_dropout: float = 0.0,
        hidden_state_mode: HaltingHiddenStateModeOptions = HaltingHiddenStateModeOptions.RAW,
    ) -> StickBreakingConfig:

        return StickBreakingConfig(
            input_dim=input_dim,
            threshold=threshold,
            halting_dropout=halting_dropout,
            hidden_state_mode=hidden_state_mode,
            halting_gate_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=input_dim,
                output_dim=2,
                num_layers=gate_num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=gate_activation,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_connection_option=gate_residual_connection_option,
                    dropout_probability=gate_dropout_probability,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=input_dim,
                        bias_flag=gate_bias_flag,
                    ),
                ),
            ),
        )

    def test_init_stores_all_config_attributes(self):
        cfg = self.preset()
        model = StickBreaking(cfg)

        self.assertIsInstance(model, StickBreaking)
        self.assertEqual(model.input_dim, cfg.input_dim)
        self.assertEqual(model.threshold, cfg.threshold)
        self.assertEqual(model.halting_gate_config, cfg.halting_gate_config)
        self.assertEqual(model.hidden_state_mode, cfg.hidden_state_mode)

    def test_config_registry_owner_and_build_dispatch(self):
        cfg = self.preset(input_dim=4)

        self.assertIs(cfg._registry_owner(), StickBreaking)
        model = cfg.build()

        self.assertIsInstance(model, StickBreaking)
        self.assertEqual(model.input_dim, cfg.input_dim)

    def test_overrides_take_precedence_and_keep_unset_base_fields(self):
        base = self.preset(
            input_dim=4,
            threshold=0.99,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
        )
        overrides = StickBreakingConfig(
            input_dim=6,
            threshold=0.25,
            hidden_state_mode=HaltingHiddenStateModeOptions.ACCUMULATED,
        )

        model = StickBreaking(base, overrides)

        self.assertEqual(model.input_dim, 6)
        self.assertEqual(model.threshold, 0.25)
        self.assertEqual(
            model.hidden_state_mode, HaltingHiddenStateModeOptions.ACCUMULATED
        )
        self.assertEqual(model.halting_gate_config, base.halting_gate_config)
        self.assertEqual(model.halting_gate_model[0].input_dim, 6)
        self.assertEqual(model.halting_gate_model[-1].output_dim, 2)

    def test_accepts_model_config_containing_halting_config(self):
        cfg = self.preset(input_dim=4, threshold=0.8)
        wrapped_cfg = WrappedHaltingConfig(cfg)

        model = StickBreaking(wrapped_cfg)

        self.assertEqual(model.cfg, cfg)
        self.assertEqual(model.input_dim, cfg.input_dim)
        self.assertEqual(model.threshold, cfg.threshold)

    def test_build_halting_gate_model(self):
        gate_layers = [1, 3]
        for num_layers in gate_layers:
            message = f"gate_num_layers={num_layers}"
            with self.subTest(msg=message):
                cfg = self.preset(gate_num_layers=num_layers)
                model = StickBreaking(cfg)
                gate = model._StickBreaking__build_halting_gate_model()

                self.assertIsNotNone(gate)
                self.assertIsInstance(gate, LayerStack)
                self.assertEqual(len(gate), num_layers)
                self.assertEqual(gate[-1].output_dim, 2)

    def test_init_gate_weights_zeroes_last_linear_weight(self):
        cfg = self.preset(input_dim=4, gate_num_layers=1)
        model = StickBreaking(cfg)
        first_linear = torch.nn.Linear(4, 4)
        last_linear = torch.nn.Linear(4, 2)
        torch.nn.init.ones_(last_linear.weight)
        model.halting_gate_model = torch.nn.Sequential(first_linear, last_linear)

        model._StickBreaking__init_gate_weights()

        torch.testing.assert_close(
            last_linear.weight,
            torch.zeros_like(last_linear.weight),
        )

    def test_validator_rejects_missing_required_fields(self):
        required_fields = [
            "input_dim",
            "threshold",
            "hidden_state_mode",
            "halting_gate_config",
        ]

        for field_name in required_fields:
            with self.subTest(field_name=field_name):
                cfg = self.preset(input_dim=4)
                setattr(cfg, field_name, None)

                with self.assertRaisesRegex(ValueError, field_name):
                    StickBreaking(cfg)

    def test_validator_allows_optional_halting_dropout_to_be_none(self):
        cfg = self.preset(input_dim=4)
        cfg.halting_dropout = None

        model = StickBreaking(cfg)

        self.assertIsInstance(model, StickBreaking)

    def test_validator_rejects_invalid_input_dim_and_threshold(self):
        invalid_cases = [
            ("input_dim_zero", {"input_dim": 0}, ValueError, "input_dim"),
            ("input_dim_negative", {"input_dim": -1}, ValueError, "input_dim"),
            ("threshold_zero", {"threshold": 0.0}, ValueError, "threshold"),
            ("threshold_negative", {"threshold": -0.1}, ValueError, "threshold"),
            ("threshold_above_one", {"threshold": 1.1}, ValueError, "threshold"),
        ]

        for name, kwargs, exception_type, pattern in invalid_cases:
            with self.subTest(name=name):
                cfg = self.preset(input_dim=4)
                for field_name, value in kwargs.items():
                    setattr(cfg, field_name, value)

                with self.assertRaisesRegex(exception_type, pattern):
                    StickBreaking(cfg)

    def test_validator_rejects_invalid_halting_gate_config(self):
        invalid_cases = [
            (
                "wrong_gate_config_type",
                lambda cfg: setattr(cfg, "halting_gate_config", object()),
                TypeError,
                "halting_gate_config",
            ),
            (
                "wrong_gate_output_dim",
                lambda cfg: setattr(cfg.halting_gate_config, "output_dim", 3),
                ValueError,
                "output_dim",
            ),
            (
                "enabled_last_layer_bias",
                lambda cfg: setattr(
                    cfg.halting_gate_config,
                    "last_layer_bias_option",
                    LastLayerBiasOptions.DEFAULT,
                ),
                ValueError,
                "last_layer_bias_option",
            ),
            (
                "nested_shared_halting",
                lambda cfg: setattr(
                    cfg.halting_gate_config,
                    "shared_halting_config",
                    object(),
                ),
                ValueError,
                "shared_halting_config",
            ),
            (
                "nested_shared_gate",
                lambda cfg: setattr(
                    cfg.halting_gate_config,
                    "shared_gate_config",
                    GateConfig(
                        model_config=self.preset(
                            input_dim=4, gate_num_layers=1
                        ).halting_gate_config,
                        option=LayerGateOptions.MULTIPLIER,
                    ),
                ),
                ValueError,
                "shared_gate_config",
            ),
            (
                "nested_layer_gate",
                lambda cfg: setattr(
                    cfg.halting_gate_config.layer_config,
                    "gate_config",
                    GateConfig(
                        model_config=self.preset(
                            input_dim=4, gate_num_layers=1
                        ).halting_gate_config,
                        option=LayerGateOptions.MULTIPLIER,
                    ),
                ),
                ValueError,
                "layer_config.gate_config",
            ),
            (
                "nested_layer_halting",
                lambda cfg: setattr(
                    cfg.halting_gate_config.layer_config,
                    "halting_config",
                    object(),
                ),
                ValueError,
                "layer_config.halting_config",
            ),
        ]

        for name, mutate, exception_type, pattern in invalid_cases:
            with self.subTest(name=name):
                cfg = self.preset(input_dim=4)
                mutate(cfg)

                with self.assertRaisesRegex(exception_type, pattern):
                    StickBreaking(cfg)

    def test_halting_gate_rejects_active_shared_gate_config(self):
        cfg = self.preset(input_dim=4)
        cfg.halting_gate_config.shared_gate_config = GateConfig(
            model_config=self.preset(input_dim=4, gate_num_layers=1).halting_gate_config,
            option=LayerGateOptions.MULTIPLIER,
        )

        with self.assertRaisesRegex(ValueError, "shared_gate_config"):
            StickBreaking(cfg)

    def test_halting_gate_allows_absent_shared_gate_config(self):
        cfg = self.preset(input_dim=4)
        cfg.halting_gate_config.shared_gate_config = None

        model = StickBreaking(cfg)

        self.assertIsInstance(model.halting_gate_model, LayerStack)
        self.assertTrue(
            all(layer.gate_model is None for layer in model.halting_gate_model)
        )

    def test_compute_gate_logits(self):
        batch_size = 4
        seq_len = 6
        input_dim = 12
        training_modes = [True, False]

        for training in training_modes:
            message = f"training={training}"
            with self.subTest(msg=message):
                cfg = self.preset(input_dim=input_dim)
                model = StickBreaking(cfg)
                model.train() if training else model.eval()

                hidden = torch.randn(batch_size, seq_len, input_dim)
                logits_1 = model._StickBreaking__compute_gate_logits(hidden)
                logits_2 = model._StickBreaking__compute_gate_logits(hidden)

                self.assertEqual(logits_1.shape, (batch_size, seq_len, 2))
                self.assertTrue(torch.all(logits_1 <= 0))
                probs = torch.exp(logits_1)
                sums = probs.sum(dim=-1)
                self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

                if training:
                    self.assertFalse(torch.equal(logits_1, logits_2))
                else:
                    self.assertTrue(torch.equal(logits_1, logits_2))

    def test_compute_gate_logits_handles_rank_two_and_rank_three_inputs(self):
        input_dim = 4
        cfg = self.preset(input_dim=input_dim)
        model = StickBreaking(cfg)
        model.eval()

        rank_two_hidden = torch.randn(5, input_dim)
        rank_three_hidden = torch.randn(2, 3, input_dim)

        rank_two_logits = model._StickBreaking__compute_gate_logits(rank_two_hidden)
        rank_three_logits = model._StickBreaking__compute_gate_logits(rank_three_hidden)

        self.assertEqual(rank_two_logits.shape, (5, 2))
        self.assertEqual(rank_three_logits.shape, (2, 3, 2))
        torch.testing.assert_close(
            rank_two_logits.exp().sum(dim=-1),
            torch.ones(5),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            rank_three_logits.exp().sum(dim=-1),
            torch.ones(2, 3),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_init_state(self):
        batch_size = 4
        seq_len = 6
        input_dims = [8, 12]
        thresholds = [0.01, 0.5, 0.99]

        for input_dim in input_dims:
            for threshold in thresholds:
                message = f"input_dim={input_dim}, threshold={threshold}"
                with self.subTest(msg=message):
                    cfg = self.preset(input_dim=input_dim, threshold=threshold)
                    model = StickBreaking(cfg)
                    model.eval()

                    hidden = torch.randn(batch_size, seq_len, input_dim)
                    log_gates = model._StickBreaking__compute_gate_logits(hidden)
                    state = model._StickBreaking__init_state(log_gates, hidden)

                    self.assertIsInstance(state, StickBreakingState)
                    self.assertEqual(state.step_count, 0)
                    self.assertEqual(state.accumulated_ponder_cost, torch.tensor(0.0))

                    log_cont, log_halt = torch.unbind(log_gates, dim=-1)
                    halt_prob = torch.exp(log_halt)

                    expected_log_continuation_shape = (batch_size, seq_len)
                    self.assertEqual(
                        state.log_continuation.shape, expected_log_continuation_shape
                    )
                    self.assertTrue(torch.equal(state.log_continuation, log_cont))

                    expected_accumulated_halt_probabilities_shape = (
                        batch_size,
                        seq_len,
                    )
                    self.assertEqual(
                        state.accumulated_halt_probabilities.shape,
                        expected_accumulated_halt_probabilities_shape,
                    )
                    self.assertTrue(
                        torch.equal(state.accumulated_halt_probabilities, halt_prob)
                    )

                    expected_halt_mask_shape = (batch_size, seq_len)
                    self.assertEqual(state.halt_mask.shape, expected_halt_mask_shape)
                    expected_mask = halt_prob >= threshold
                    self.assertTrue(torch.equal(state.halt_mask, expected_mask))

                    expected_accumulated_hidden_shape = (batch_size, seq_len, input_dim)
                    self.assertEqual(
                        state.accumulated_hidden.shape,
                        expected_accumulated_hidden_shape,
                    )
                    expected_hidden = halt_prob.unsqueeze(-1) * hidden
                    self.assertTrue(
                        torch.equal(state.accumulated_hidden, expected_hidden)
                    )
                    self.assertTrue(torch.equal(state.output_hidden, hidden))

    def test_update_state(self):
        batch_size = 4
        seq_len = 6
        input_dims = [8, 12]
        thresholds = [0.01, 0.5, 0.99]

        for input_dim in input_dims:
            for threshold in thresholds:
                message = f"input_dim={input_dim}, threshold={threshold}"
                with self.subTest(msg=message):
                    cfg = self.preset(input_dim=input_dim, threshold=threshold)
                    model = StickBreaking(cfg)
                    model.eval()

                    hidden_1 = torch.randn(batch_size, seq_len, input_dim)
                    hidden_2 = torch.randn(batch_size, seq_len, input_dim)

                    log_gates_1 = model._StickBreaking__compute_gate_logits(hidden_1)
                    previous_state = model._StickBreaking__init_state(
                        log_gates_1, hidden_1
                    )

                    log_gates_2 = model._StickBreaking__compute_gate_logits(hidden_2)
                    state = model._StickBreaking__update_state(
                        previous_state, log_gates_2, hidden_2
                    )

                    self.assertIsInstance(state, StickBreakingState)
                    self.assertEqual(state.step_count, previous_state.step_count + 1)

                    log_cont, halt_prob = (
                        model._StickBreaking__compute_step_halting_probability(
                            previous_state, log_gates_2
                        )
                    )

                    expected_log_continuation_shape = (batch_size, seq_len)
                    self.assertEqual(
                        state.log_continuation.shape, expected_log_continuation_shape
                    )
                    self.assertTrue(torch.equal(state.log_continuation, log_cont))

                    expected_accumulated_halt_probabilities_shape = (
                        batch_size,
                        seq_len,
                    )
                    expected_accumulated_halt_probabilities = (
                        previous_state.accumulated_halt_probabilities + halt_prob
                    )
                    self.assertEqual(
                        state.accumulated_halt_probabilities.shape,
                        expected_accumulated_halt_probabilities_shape,
                    )
                    self.assertTrue(
                        torch.equal(
                            state.accumulated_halt_probabilities,
                            expected_accumulated_halt_probabilities,
                        )
                    )

                    expected_halt_mask_shape = (batch_size, seq_len)
                    expected_halt_mask = (
                        expected_accumulated_halt_probabilities >= threshold
                    )
                    self.assertEqual(state.halt_mask.shape, expected_halt_mask_shape)
                    self.assertTrue(torch.equal(state.halt_mask, expected_halt_mask))

                    expected_accumulated_hidden_shape = (batch_size, seq_len, input_dim)
                    expected_accumulated_hidden = (
                        previous_state.accumulated_hidden
                        + halt_prob.unsqueeze(-1) * hidden_2
                    )
                    self.assertEqual(
                        state.accumulated_hidden.shape,
                        expected_accumulated_hidden_shape,
                    )
                    self.assertTrue(
                        torch.equal(
                            state.accumulated_hidden, expected_accumulated_hidden
                        )
                    )
                    expected_output_hidden = torch.where(
                        previous_state.halt_mask.unsqueeze(-1),
                        previous_state.output_hidden,
                        hidden_2,
                    )
                    self.assertTrue(
                        torch.equal(state.output_hidden, expected_output_hidden)
                    )

                    expected_accumulated_ponder_cost_shape = (batch_size, seq_len)
                    expected_accumulated_ponder_cost = (
                        previous_state.accumulated_ponder_cost
                        + halt_prob * state.step_count
                    )
                    self.assertEqual(
                        state.accumulated_ponder_cost.shape,
                        expected_accumulated_ponder_cost_shape,
                    )
                    self.assertTrue(
                        torch.equal(
                            state.accumulated_ponder_cost,
                            expected_accumulated_ponder_cost,
                        )
                    )

    def test_handcrafted_logits_match_stick_breaking_equations(self):
        cfg = self.preset(
            input_dim=2,
            threshold=0.7,
            hidden_state_mode=HaltingHiddenStateModeOptions.ACCUMULATED,
        )
        model = StickBreaking(cfg)
        hidden_1 = torch.tensor([[[10.0, 1.0], [20.0, 2.0]]])
        hidden_2 = torch.tensor([[[30.0, 3.0], [40.0, 4.0]]])
        log_gates_1 = torch.log(torch.tensor([[[0.6, 0.4], [0.2, 0.8]]]))
        log_gates_2 = torch.log(torch.tensor([[[0.5, 0.5], [0.1, 0.9]]]))

        initial_state = model._StickBreaking__init_state(log_gates_1, hidden_1)
        state = model._StickBreaking__update_state(
            initial_state, log_gates_2, hidden_2
        )
        output, ponder_loss = model.finalize_weighted_accumulation(state, hidden_2)

        expected_initial_accumulated = torch.tensor([[[4.0, 0.4], [16.0, 1.6]]])
        expected_initial_output = expected_initial_accumulated
        expected_initial_halt_mask = torch.tensor([[False, True]])
        torch.testing.assert_close(
            initial_state.accumulated_hidden, expected_initial_accumulated
        )
        torch.testing.assert_close(initial_state.output_hidden, expected_initial_output)
        self.assertTrue(torch.equal(initial_state.halt_mask, expected_initial_halt_mask))

        expected_halting_prob = torch.tensor([[0.3, 0.0]])
        expected_accumulated_prob = torch.tensor([[0.7, 0.8]])
        expected_accumulated_hidden = torch.tensor([[[13.0, 1.3], [16.0, 1.6]]])
        expected_output_hidden = torch.tensor([[[13.0, 1.3], [16.0, 1.6]]])
        expected_halt_mask = torch.tensor([[True, True]])
        _, halting_prob = model._StickBreaking__compute_step_halting_probability(
            initial_state, log_gates_2
        )

        torch.testing.assert_close(halting_prob, expected_halting_prob)
        torch.testing.assert_close(
            state.accumulated_halt_probabilities, expected_accumulated_prob
        )
        torch.testing.assert_close(state.accumulated_hidden, expected_accumulated_hidden)
        torch.testing.assert_close(state.output_hidden, expected_output_hidden)
        self.assertTrue(torch.equal(state.halt_mask, expected_halt_mask))
        self.assertEqual(state.step_count, 1)

        torch.testing.assert_close(output, expected_accumulated_hidden)
        torch.testing.assert_close(ponder_loss, torch.tensor([[0.3, 0.0]]))

    def test_compute_step_halting_probability(self):
        batch_size = 4
        seq_len = 6
        input_dims = [8, 12]
        thresholds = [0.01, 0.99]

        for input_dim in input_dims:
            for threshold in thresholds:
                message = f"input_dim={input_dim}, threshold={threshold}"
                with self.subTest(msg=message):
                    cfg = self.preset(input_dim=input_dim, threshold=threshold)
                    model = StickBreaking(cfg)
                    model.eval()

                    hidden_1 = torch.randn(batch_size, seq_len, input_dim)
                    hidden_2 = torch.randn(batch_size, seq_len, input_dim)

                    log_gates_1 = model._StickBreaking__compute_gate_logits(hidden_1)
                    previous_state = model._StickBreaking__init_state(
                        log_gates_1, hidden_1
                    )

                    log_gates_2 = model._StickBreaking__compute_gate_logits(hidden_2)
                    log_cont, halt_prob = (
                        model._StickBreaking__compute_step_halting_probability(
                            previous_state, log_gates_2
                        )
                    )

                    expected_log_continuation_shape = (batch_size, seq_len)
                    self.assertEqual(log_cont.shape, expected_log_continuation_shape)

                    expected_halting_probability_shape = (batch_size, seq_len)
                    self.assertEqual(
                        halt_prob.shape, expected_halting_probability_shape
                    )

                    self.assertTrue(torch.all(halt_prob >= 0.0))
                    self.assertTrue(torch.all(halt_prob <= 1.0))

                    if previous_state.halt_mask.any():
                        self.assertTrue(
                            torch.all(halt_prob[previous_state.halt_mask] == 0.0)
                        )

                    prev_log_cont = previous_state.log_continuation.unsqueeze(-1)
                    current_log_halting = prev_log_cont + log_gates_2
                    expected_log_cont, expected_log_halt = torch.unbind(
                        current_log_halting, dim=-1
                    )
                    expected_halt_prob = torch.exp(expected_log_halt)
                    expected_halt_prob = expected_halt_prob.masked_fill(
                        previous_state.halt_mask, 0.0
                    )

                    self.assertTrue(torch.equal(log_cont, expected_log_cont))
                    self.assertTrue(torch.equal(halt_prob, expected_halt_prob))

    def test_update_halting_state(self):
        batch_size = 4
        seq_len = 6
        input_dims = [8, 12]
        modes = [
            HaltingHiddenStateModeOptions.RAW,
            HaltingHiddenStateModeOptions.ACCUMULATED,
        ]
        previous_states = [None, "existing"]

        for input_dim in input_dims:
            for mode in modes:
                for prev in previous_states:
                    message = (
                        f"input_dim={input_dim}, "
                        f"hidden_state_mode={mode}, "
                        f"previous_state={prev}"
                    )
                    with self.subTest(msg=message):
                        cfg = self.preset(input_dim=input_dim, hidden_state_mode=mode)
                        model = StickBreaking(cfg)
                        model.eval()

                        hidden = torch.randn(batch_size, seq_len, input_dim)

                        if prev == "existing":
                            init_hidden = torch.randn(batch_size, seq_len, input_dim)
                            previous_state, _ = model.update_halting_state(
                                None, init_hidden
                            )
                        else:
                            previous_state = None

                        state, output = model.update_halting_state(
                            previous_state, hidden
                        )

                        self.assertIsInstance(state, StickBreakingState)
                        expected_output_shape = (batch_size, seq_len, input_dim)
                        self.assertEqual(output.shape, expected_output_shape)

                        if previous_state is None:
                            self.assertEqual(state.step_count, 0)
                        else:
                            self.assertEqual(
                                state.step_count, previous_state.step_count + 1
                            )

                        if mode == HaltingHiddenStateModeOptions.RAW:
                            expected_output = hidden
                        else:
                            expected_output = state.accumulated_hidden
                        if previous_state is not None:
                            expected_output = torch.where(
                                previous_state.halt_mask.unsqueeze(-1),
                                previous_state.output_hidden,
                                expected_output,
                            )
                        self.assertTrue(torch.equal(output, expected_output))
                        self.assertTrue(torch.equal(output, state.output_hidden))

    def test_update_halting_state_preserves_previous_output_for_halted_items(self):
        input_dim = 2
        hidden = torch.tensor([[[10.0, 10.0], [20.0, 20.0]]])
        previous_output = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
        previous_halt_mask = torch.tensor([[True, False]])

        for mode in (
            HaltingHiddenStateModeOptions.RAW,
            HaltingHiddenStateModeOptions.ACCUMULATED,
        ):
            with self.subTest(mode=mode):
                cfg = self.preset(input_dim=input_dim, hidden_state_mode=mode)
                model = StickBreaking(cfg)
                model.eval()
                previous_state = StickBreakingState(
                    halt_mask=previous_halt_mask,
                    log_continuation=torch.zeros(1, 2),
                    accumulated_hidden=torch.zeros(1, 2, input_dim),
                    output_hidden=previous_output,
                    accumulated_halt_probabilities=torch.tensor([[1.0, 0.0]]),
                    step_count=0,
                    accumulated_ponder_cost=torch.zeros(1, 2),
                )

                state, output = model.update_halting_state(previous_state, hidden)

                expected = state.accumulated_hidden.clone()
                if mode == HaltingHiddenStateModeOptions.RAW:
                    expected = hidden.clone()
                expected[:, 0, :] = previous_output[:, 0, :]
                torch.testing.assert_close(output, expected)
                torch.testing.assert_close(state.output_hidden, expected)

    def test_finalize_weighted_accumulation(self):
        batch_size = 4
        seq_len = 6
        input_dims = [8, 12]
        thresholds = [0.01, 0.5, 0.99]
        num_steps = [1, 3, 5]

        for input_dim in input_dims:
            for threshold in thresholds:
                for steps in num_steps:
                    message = (
                        f"input_dim={input_dim}, "
                        f"threshold={threshold}, "
                        f"num_steps={steps}"
                    )
                    with self.subTest(msg=message):
                        cfg = self.preset(input_dim=input_dim, threshold=threshold)
                        model = StickBreaking(cfg)
                        model.eval()

                        state = None
                        hidden = None
                        for _ in range(steps):
                            hidden = torch.randn(batch_size, seq_len, input_dim)
                            state, _ = model.update_halting_state(state, hidden)

                        output, ponder_loss = model.finalize_weighted_accumulation(
                            state, hidden
                        )

                        expected_output_shape = (batch_size, seq_len, input_dim)
                        self.assertEqual(output.shape, expected_output_shape)

                        expected_ponder_loss_shape = (batch_size, seq_len)
                        self.assertEqual(ponder_loss.shape, expected_ponder_loss_shape)
                        self.assertTrue(torch.all(ponder_loss >= 0))

                        remaining_prob = 1 - state.accumulated_halt_probabilities
                        remaining_prob = remaining_prob.masked_fill(
                            state.halt_mask, 0.0
                        )

                        expected_output = (
                            state.accumulated_hidden
                            + remaining_prob.unsqueeze(-1) * hidden
                        )
                        self.assertTrue(torch.equal(output, expected_output))

                        expected_ponder_loss = (
                            state.accumulated_ponder_cost
                            + remaining_prob * (state.step_count + 1)
                        )
                        self.assertTrue(torch.equal(ponder_loss, expected_ponder_loss))

                        if state.halt_mask.any():
                            self.assertTrue(
                                torch.all(remaining_prob[state.halt_mask] == 0.0)
                            )

    def test_backward_reaches_halting_gate_parameters(self):
        cfg = self.preset(
            input_dim=3,
            gate_num_layers=1,
            threshold=1.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.ACCUMULATED,
        )
        model = StickBreaking(cfg)
        model.eval()
        hidden = torch.tensor(
            [[[1.0, -2.0, 3.0], [0.5, 1.5, -1.0]]],
            requires_grad=True,
        )
        current_hidden = torch.tensor([[[2.0, 0.0, -1.0], [-2.0, 3.0, 1.0]]])

        state, output = model.update_halting_state(None, hidden)
        final_output, ponder_loss = model.finalize_weighted_accumulation(
            state, current_hidden
        )
        loss = output.sum() + final_output.sum() + ponder_loss.sum()
        loss.backward()

        parameter_gradients = [
            parameter.grad
            for parameter in model.halting_gate_model.parameters()
            if parameter.requires_grad
        ]
        nonzero_gradients = [
            gradient
            for gradient in parameter_gradients
            if gradient is not None and torch.any(gradient.abs() > 0)
        ]

        self.assertTrue(len(nonzero_gradients) > 0)
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.any(hidden.grad.abs() > 0))
