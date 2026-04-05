import torch
import unittest
import torch.nn as nn

from torch.nn import Sequential
from emperor.base.layer import Layer

from emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from emperor.base.layer import (
    LayerConfig,
    LayerStackConfig,
)
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.halting.utils.options.stick_breaking import (
    StickBreaking,
    StickBreakingState,
)
from emperor.linears.utils.config import LinearLayerConfig
from emperor.base.enums import LastLayerBiasOptions


class TestHalting(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        gate_num_layers: int = 3,
        gate_activation: ActivationOptions = ActivationOptions.DISABLED,
        gate_residual_flag: bool = False,
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
                    residual_flag=gate_residual_flag,
                    dropout_probability=gate_dropout_probability,
                    halting_config=None,
                    shared_halting_flag=False,
                    gate_config=None,
                    model_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=input_dim,
                        bias_flag=gate_bias_flag,
                        data_monitor=None,
                        parameter_monitor=None,
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

    def test_build_halting_gate_model(self):
        gate_layers = [1, 3]
        for num_layers in gate_layers:
            message = f"gate_num_layers={num_layers}"
            with self.subTest(msg=message):
                cfg = self.preset(gate_num_layers=num_layers)
                model = StickBreaking(cfg)
                gate = model._StickBreaking__build_halting_gate_model()

                self.assertIsNotNone(gate)
                if num_layers == 1:
                    self.assertIsInstance(gate, Layer)
                    self.assertEqual(gate.output_dim, 2)
                else:
                    self.assertIsInstance(gate, Sequential)
                    self.assertEqual(len(gate), num_layers)
                    self.assertEqual(gate[-1].output_dim, 2)

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
                            self.assertTrue(torch.equal(output, hidden))
                        else:
                            self.assertTrue(
                                torch.equal(output, state.accumulated_hidden)
                            )

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
