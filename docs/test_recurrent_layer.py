from dataclasses import dataclass
import unittest

import torch

from emperor.base.layer import (
    Layer,
    LayerConfig,
    LayerStackConfig,
    LayerState,
    RecurrentLayer,
    RecurrentLayerConfig,
    RecurrentLayerValidator,
)
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.utils import ConfigBase, Module, optional_field
from emperor.halting.config import SoftHaltingConfig, StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.halting.utils.options.base import HaltingStateBase


@dataclass
class AdditiveFeatureLastConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    increment: float | None = optional_field("Value added to every hidden element.")

    def _registry_owner(self) -> type:
        return AdditiveFeatureLastLayer


class AdditiveFeatureLastLayer(Module):
    def __init__(
        self,
        cfg: AdditiveFeatureLastConfig,
        overrides: AdditiveFeatureLastConfig | None = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.increment = self.cfg.increment
        self.call_count = 0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        if self.input_dim != self.output_dim:
            raise ValueError("AdditiveFeatureLastLayer requires stable dimensions")
        return X + self.increment


@dataclass
class ConstantFeatureLastConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    value: float | None = optional_field("Constant output value.")

    def _registry_owner(self) -> type:
        return ConstantFeatureLastLayer


class ConstantFeatureLastLayer(Module):
    def __init__(
        self,
        cfg: ConstantFeatureLastConfig,
        overrides: ConstantFeatureLastConfig | None = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.value = self.cfg.value

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        shape = (*X.shape[:-1], self.output_dim)
        return torch.full(shape, self.value, dtype=X.dtype, device=X.device)


@dataclass
class ThresholdHaltingGateConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    threshold: float | None = optional_field("Hidden value threshold.")
    high_logit: float | None = optional_field("High logit value.")
    low_logit: float | None = optional_field("Low logit value.")

    def _registry_owner(self) -> type:
        return ThresholdHaltingGateLayer


class ThresholdHaltingGateLayer(Module):
    def __init__(
        self,
        cfg: ThresholdHaltingGateConfig,
        overrides: ThresholdHaltingGateConfig | None = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.threshold = self.cfg.threshold
        self.high_logit = self.cfg.high_logit
        self.low_logit = self.cfg.low_logit

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        halt_now = X[..., 0] >= self.threshold
        continue_logit = torch.where(
            halt_now,
            torch.full_like(X[..., 0], self.low_logit),
            torch.full_like(X[..., 0], self.high_logit),
        )
        halt_logit = torch.where(
            halt_now,
            torch.full_like(X[..., 0], self.high_logit),
            torch.full_like(X[..., 0], self.low_logit),
        )
        return torch.stack((continue_logit, halt_logit), dim=-1)


@dataclass
class DummyHaltingState(HaltingStateBase):
    marker: str


class TestRecurrentLayer(unittest.TestCase):
    def layer_block_config(
        self,
        increment: float = 1.0,
        input_dim: int | None = None,
        output_dim: int | None = None,
        halting_config: StickBreakingConfig | None = None,
    ) -> LayerConfig:
        return LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=ActivationOptions.DISABLED,
            residual_flag=False,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=halting_config,
            shared_halting_flag=False,
            layer_model_config=AdditiveFeatureLastConfig(
                increment=increment,
            ),
        )

    def stack_block_config(
        self,
        increment: float = 1.0,
        num_layers: int = 2,
        halting_config: StickBreakingConfig | None = None,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=2,
            hidden_dim=3,
            output_dim=4,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            layer_config=self.layer_block_config(
                increment=increment,
                halting_config=halting_config,
            ),
        )

    def gate_config(self, value: float = 0.0) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=2,
            hidden_dim=2,
            output_dim=2,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                activation=ActivationOptions.DISABLED,
                residual_flag=False,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                shared_halting_flag=False,
                layer_model_config=ConstantFeatureLastConfig(value=value),
            ),
        )

    def halting_gate_config(
        self,
        threshold: float,
        high_logit: float = 10.0,
        low_logit: float = -10.0,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=2,
            hidden_dim=2,
            output_dim=2,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                activation=ActivationOptions.DISABLED,
                residual_flag=False,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                shared_halting_flag=False,
                layer_model_config=ThresholdHaltingGateConfig(
                    threshold=threshold,
                    high_logit=high_logit,
                    low_logit=low_logit,
                ),
            ),
        )

    def halting_config(
        self,
        dim: int,
        gate_threshold: float,
        threshold: float = 0.99,
        high_logit: float = 10.0,
        low_logit: float = -10.0,
    ) -> StickBreakingConfig:
        return StickBreakingConfig(
            input_dim=dim,
            threshold=threshold,
            halting_dropout=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=self.halting_gate_config(
                threshold=gate_threshold,
                high_logit=high_logit,
                low_logit=low_logit,
            ),
        )

    def recurrent_config(
        self,
        dim: int = 4,
        max_steps: int = 3,
        block_config: LayerConfig | LayerStackConfig | None = None,
        gate_config: LayerStackConfig | None = None,
        halting_config: StickBreakingConfig | None = None,
    ) -> RecurrentLayerConfig:
        if block_config is None:
            block_config = self.layer_block_config()
        return RecurrentLayerConfig(
            input_dim=dim,
            output_dim=dim,
            max_steps=max_steps,
            block_config=block_config,
            gate_config=gate_config,
            halting_config=halting_config,
        )

    def test_public_exports_and_config_build_dispatch(self):
        import emperor.base.layer as layer_package

        expected_exports = [
            "RecurrentLayerConfig",
            "RecurrentLayer",
            "RecurrentLayerValidator",
        ]
        for name in expected_exports:
            with self.subTest(name=name):
                self.assertIn(name, layer_package.__all__)
                self.assertIsNotNone(getattr(layer_package, name))

        cfg = self.recurrent_config()
        model = cfg.build()

        self.assertIsInstance(model, RecurrentLayer)
        self.assertIsInstance(model.block_model, Layer)

    def test_validation_errors(self):
        dim = 4
        valid_block = self.layer_block_config()
        invalid_cases = [
            RecurrentLayerConfig(
                input_dim=dim,
                output_dim=dim,
                max_steps=1,
                block_config=None,
            ),
            RecurrentLayerConfig(
                input_dim=dim,
                output_dim=dim,
                max_steps=1,
                block_config=object(),
            ),
            RecurrentLayerConfig(
                input_dim=dim,
                output_dim=dim,
                max_steps=0,
                block_config=valid_block,
            ),
            RecurrentLayerConfig(
                input_dim=dim,
                output_dim=dim + 1,
                max_steps=1,
                block_config=valid_block,
            ),
        ]

        for cfg in invalid_cases:
            with self.subTest(cfg=cfg):
                with self.assertRaises((TypeError, ValueError)):
                    RecurrentLayer(cfg)

    def test_rejects_halting_without_layer_finalization_contract(self):
        dim = 4
        cfg = self.recurrent_config(
            dim=dim,
            halting_config=SoftHaltingConfig(
                input_dim=dim,
                threshold=0.99,
                halting_dropout=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=self.halting_gate_config(threshold=1.0),
            ),
        )

        with self.assertRaises(ValueError):
            RecurrentLayer(cfg)

    def test_forward_rejects_bad_hidden_rank_and_feature_dim(self):
        dim = 4
        model = RecurrentLayer(self.recurrent_config(dim=dim))
        invalid_hidden = [
            torch.randn(dim),
            torch.randn(2, dim + 1),
        ]

        for hidden in invalid_hidden:
            with self.subTest(shape=tuple(hidden.shape)):
                with self.assertRaises(ValueError):
                    model(LayerState(hidden=hidden))

    def test_runs_exact_max_steps_without_halting_and_reuses_block_instance(self):
        dim = 4
        max_steps = 5
        cfg = self.recurrent_config(
            dim=dim,
            max_steps=max_steps,
            block_config=self.layer_block_config(increment=1.0),
        )
        model = RecurrentLayer(cfg)
        hidden = torch.zeros(2, dim)

        result = model(LayerState(hidden=hidden))

        torch.testing.assert_close(result.hidden, torch.full_like(hidden, max_steps))
        self.assertEqual(model.block_model.model.call_count, max_steps)

    def test_layer_config_block_dimensions_are_overridden(self):
        dim = 5
        cfg = self.recurrent_config(
            dim=dim,
            block_config=self.layer_block_config(
                increment=1.0,
                input_dim=1,
                output_dim=2,
            ),
        )

        model = RecurrentLayer(cfg)

        self.assertEqual(model.block_model.input_dim, dim)
        self.assertEqual(model.block_model.output_dim, dim)
        self.assertEqual(model.block_model.model.input_dim, dim)
        self.assertEqual(model.block_model.model.output_dim, dim)

    def test_layer_stack_block_dimensions_are_overridden(self):
        dim = 5
        cfg = self.recurrent_config(
            dim=dim,
            block_config=self.stack_block_config(num_layers=2),
        )

        model = RecurrentLayer(cfg)
        layers = list(model.block_model)

        self.assertEqual(len(layers), 2)
        for layer in layers:
            with self.subTest(layer=layer):
                self.assertEqual(layer.input_dim, dim)
                self.assertEqual(layer.output_dim, dim)
                self.assertEqual(layer.model.input_dim, dim)
                self.assertEqual(layer.model.output_dim, dim)

    def test_recurrent_gate_interpolates_only_when_config_exists(self):
        dim = 3
        hidden = torch.ones(2, dim)
        block_config = self.layer_block_config(increment=2.0)

        without_gate = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=block_config,
            )
        )
        with_gate = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=block_config,
                gate_config=self.gate_config(value=0.0),
            )
        )

        ungated = without_gate(LayerState(hidden=hidden.clone()))
        gated = with_gate(LayerState(hidden=hidden.clone()))

        torch.testing.assert_close(ungated.hidden, torch.full_like(hidden, 3.0))
        torch.testing.assert_close(gated.hidden, torch.full_like(hidden, 2.0))

    def test_accepts_2d_and_3d_feature_last_hidden(self):
        dim = 4
        shapes = [(2, dim), (2, 3, dim)]

        for shape in shapes:
            with self.subTest(shape=shape):
                model = RecurrentLayer(
                    self.recurrent_config(
                        dim=dim,
                        max_steps=2,
                        block_config=self.layer_block_config(increment=1.5),
                    )
                )
                hidden = torch.zeros(*shape)
                result = model(LayerState(hidden=hidden))

                torch.testing.assert_close(result.hidden, torch.full_like(hidden, 3.0))

    def test_recurrent_halting_stops_early_when_all_positions_halt(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=5,
                block_config=self.layer_block_config(increment=1.0),
                halting_config=self.halting_config(dim=dim, gate_threshold=0.0),
            )
        )
        hidden = torch.zeros(3, dim)

        result = model(LayerState(hidden=hidden))

        self.assertEqual(model.block_model.model.call_count, 1)
        torch.testing.assert_close(result.hidden, torch.ones_like(hidden))

    def test_recurrent_halting_preserves_halted_positions(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=5,
                block_config=self.layer_block_config(increment=1.0),
                halting_config=self.halting_config(dim=dim, gate_threshold=2.0),
            )
        )
        hidden = torch.tensor([[0.0, 0.0], [10.0, 10.0]])

        result = model(LayerState(hidden=hidden))

        expected = torch.tensor([[2.0, 2.0], [11.0, 11.0]])
        self.assertEqual(model.block_model.model.call_count, 2)
        torch.testing.assert_close(result.hidden, expected)

    def test_recurrent_halting_respects_max_steps_when_not_all_positions_halt(self):
        dim = 2
        max_steps = 3
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=max_steps,
                block_config=self.layer_block_config(increment=1.0),
                halting_config=self.halting_config(dim=dim, gate_threshold=100.0),
            )
        )
        hidden = torch.zeros(2, dim)

        result = model(LayerState(hidden=hidden))

        self.assertEqual(model.block_model.model.call_count, max_steps)
        self.assertEqual(result.hidden.shape, hidden.shape)

    def test_recurrent_loss_adds_to_existing_loss(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=2,
                block_config=self.layer_block_config(increment=1.0),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=100.0,
                    threshold=1.0,
                ),
            )
        )
        existing_loss = torch.tensor(5.0)
        hidden = torch.zeros(2, dim)

        result = model(LayerState(hidden=hidden, loss=existing_loss))

        self.assertIsNotNone(result.loss)
        self.assertTrue(torch.all(result.loss > existing_loss))

    def test_wrapped_block_and_recurrent_halting_do_not_leak_halting_state(self):
        dim = 2
        sentinel_halting_state = DummyHaltingState(marker="outer")
        block_halting = self.halting_config(dim=dim, gate_threshold=0.0)
        recurrent_halting = self.halting_config(dim=dim, gate_threshold=0.0)
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=2,
                block_config=self.stack_block_config(
                    num_layers=2,
                    halting_config=block_halting,
                ),
                halting_config=recurrent_halting,
            )
        )
        state = LayerState(
            hidden=torch.zeros(2, dim),
            loss=torch.tensor(1.0),
            halting_state=sentinel_halting_state,
        )

        result = model(state)

        self.assertIs(result.halting_state, sentinel_halting_state)
        self.assertIsNotNone(result.loss)


if __name__ == "__main__":
    unittest.main()
