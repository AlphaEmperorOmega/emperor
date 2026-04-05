import torch
import unittest
import torch.nn as nn

from emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from emperor.base.layer import (
    LayerConfig,
    LayerStackConfig,
)
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.halting.utils.options.stick_breaking import StickBreaking, StickBreakingState
from emperor.linears.utils.config import LinearLayerConfig
from emperor.base.enums import LastLayerBiasOptions


class TestHalting(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        output_dim: int = 8,
        bias_flag: bool = True,
        activation: ActivationOptions = ActivationOptions.RELU,
        residual_flag: bool = True,
        dropout_probability: float = 0.2,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
        halting_gate_config: "LayerStackConfig | None" = None,
        gate_num_layers: int = 1,
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
        self.assertIsNotNone(model.halting_gate_model)

    def test_build_halting_gate_model(self):
        cfg = self.preset()
        model = StickBreaking(cfg)

        gate = model.halting_gate_model
        self.assertIsNotNone(gate)
        self.assertEqual(gate.output_dim, 2)

    def test_update_halting_state_init(self):
        batch_size = 4
        seq_len = 6
        input_dim = 12
        cfg = self.preset(input_dim=input_dim, output_dim=input_dim)
        model = StickBreaking(cfg)
        model.eval()

        hidden = torch.randn(batch_size, seq_len, input_dim)
        state, output = model.update_halting_state(None, hidden)

        self.assertIsInstance(state, StickBreakingState)
        self.assertEqual(state.step_count, 0)
        self.assertEqual(state.halt_mask.shape, (batch_size, seq_len))
        self.assertEqual(state.log_continuation.shape, (batch_size, seq_len))
        self.assertEqual(state.accumulated_hidden.shape, (batch_size, seq_len, input_dim))
        self.assertEqual(state.accumulated_halt_probabilities.shape, (batch_size, seq_len))
        self.assertEqual(output.shape, (batch_size, seq_len, input_dim))
        self.assertTrue(torch.equal(output, hidden))
