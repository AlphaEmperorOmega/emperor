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
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core.config import (
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
)
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.experts.model import MixtureOfExpertsModel
from emperor.halting.config import SoftHaltingConfig, StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.halting.utils.options.base import HaltingStateBase
from emperor.sampler.core.config import RouterConfig, SamplerConfig


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
class MismatchedShapeFeatureLastConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")

    def _registry_owner(self) -> type:
        return MismatchedShapeFeatureLastLayer


class MismatchedShapeFeatureLastLayer(Module):
    def __init__(
        self,
        cfg: MismatchedShapeFeatureLastConfig,
        overrides: MismatchedShapeFeatureLastConfig | None = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X[:1]


@dataclass
class TrainableScaleFeatureLastConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    scale: float | None = optional_field("Initial multiplicative scale.")

    def _registry_owner(self) -> type:
        return TrainableScaleFeatureLastLayer


class TrainableScaleFeatureLastLayer(Module):
    def __init__(
        self,
        cfg: TrainableScaleFeatureLastConfig,
        overrides: TrainableScaleFeatureLastConfig | None = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.scale = torch.nn.Parameter(torch.tensor(self.cfg.scale))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.input_dim != self.output_dim:
            raise ValueError("TrainableScaleFeatureLastLayer requires stable dimensions")
        return X * self.scale


@dataclass
class CustomStateBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    hidden_dim: int | None = optional_field("Internal feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    increment: float | None = optional_field("Value added to every hidden element.")

    def _registry_owner(self) -> type:
        return CustomStateBlock


class CustomStateBlock(Module):
    def __init__(
        self,
        cfg: CustomStateBlockConfig,
        overrides: CustomStateBlockConfig | None = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.hidden_dim = self.cfg.hidden_dim
        self.output_dim = self.cfg.output_dim
        self.increment = self.cfg.increment

    def forward(self, state: LayerState) -> LayerState:
        if state.hidden.shape[-1] != self.input_dim:
            raise ValueError("CustomStateBlock received the wrong input dimension")
        if self.input_dim != self.output_dim:
            raise ValueError("CustomStateBlock requires stable dimensions")
        state.hidden = state.hidden + self.increment
        return state


@dataclass
class MissingInputDimBlockConfig(ConfigBase):
    output_dim: int | None = optional_field("Output feature dimension.")

    def _registry_owner(self) -> type:
        return CustomStateBlock


@dataclass
class MissingOutputDimBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")

    def _registry_owner(self) -> type:
        return CustomStateBlock


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
        input_dim: int = 2,
        hidden_dim: int = 3,
        output_dim: int = 4,
        num_layers: int = 2,
        halting_config: StickBreakingConfig | None = None,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            layer_config=self.layer_block_config(
                increment=increment,
                halting_config=halting_config,
            ),
        )

    def moe_sampler_config(
        self,
        dim: int,
        top_k: int = 1,
        num_experts: int = 2,
    ) -> SamplerConfig:
        return SamplerConfig(
            top_k=top_k,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
            router_config=RouterConfig(
                input_dim=dim,
                num_experts=num_experts,
                noisy_topk_flag=False,
                model_config=LayerStackConfig(
                    input_dim=dim,
                    hidden_dim=dim,
                    output_dim=num_experts,
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
                        layer_model_config=ConstantFeatureLastConfig(value=1.0),
                    ),
                ),
            ),
        )

    def moe_expert_model_config(self, dim: int) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=self.layer_block_config(increment=1.0),
        )

    def moe_block_config(
        self,
        dim: int = 4,
        config_dim: int = 1,
    ) -> MixtureOfExpertsModelConfig:
        top_k = 1
        num_experts = 2
        sampler_config = self.moe_sampler_config(
            dim=dim,
            top_k=top_k,
            num_experts=num_experts,
        )
        return MixtureOfExpertsModelConfig(
            input_dim=config_dim,
            output_dim=config_dim,
            top_k=top_k,
            routing_initialization_mode=RoutingInitializationMode.DISABLED,
            sampler_config=None,
            stack_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=dim,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=MixtureOfExpertsLayerConfig(
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_flag=False,
                    dropout_probability=0.0,
                    gate_config=None,
                    halting_config=None,
                    shared_halting_flag=False,
                    layer_model_config=MixtureOfExpertsConfig(
                        input_dim=dim,
                        output_dim=dim,
                        top_k=top_k,
                        num_experts=num_experts,
                        capacity_factor=0.0,
                        dropped_token_behavior=DroppedTokenOptions.ZEROS,
                        compute_expert_mixture_flag=True,
                        weighted_parameters_flag=False,
                        weighting_position_option=(
                            ExpertWeightingPositionOptions.BEFORE_EXPERTS
                        ),
                        routing_initialization_mode=RoutingInitializationMode.LAYER,
                        sampler_config=sampler_config,
                        expert_model_config=self.moe_expert_model_config(dim),
                    ),
                ),
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
        block_config: ConfigBase | None = None,
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

    def test_init_stores_all_config_attributes_and_builds_optional_models(self):
        dim = 2
        cfg = self.recurrent_config(
            dim=dim,
            max_steps=4,
            block_config=self.layer_block_config(increment=1.5),
            gate_config=self.gate_config(value=0.25),
            halting_config=self.halting_config(dim=dim, gate_threshold=10.0),
        )

        model = RecurrentLayer(cfg)

        self.assertIsInstance(model, RecurrentLayer)
        self.assertEqual(model.input_dim, cfg.input_dim)
        self.assertEqual(model.output_dim, cfg.output_dim)
        self.assertEqual(model.max_steps, cfg.max_steps)
        self.assertEqual(model.block_config, cfg.block_config)
        self.assertEqual(model.gate_config, cfg.gate_config)
        self.assertEqual(model.halting_config, cfg.halting_config)
        self.assertIsInstance(model.block_model, Layer)
        self.assertIsInstance(model.gate_model, Layer)
        self.assertIsNotNone(model.halting_model)

    def test_init_with_overrides(self):
        cfg = self.recurrent_config(
            dim=2,
            max_steps=2,
            block_config=self.layer_block_config(increment=1.0),
        )
        override_block = self.layer_block_config(increment=3.0)
        override_gate = self.gate_config(value=0.5)
        overrides = RecurrentLayerConfig(
            input_dim=3,
            output_dim=3,
            max_steps=5,
            block_config=override_block,
            gate_config=override_gate,
        )

        model = RecurrentLayer(cfg, overrides)

        self.assertEqual(model.input_dim, 3)
        self.assertEqual(model.output_dim, 3)
        self.assertEqual(model.max_steps, 5)
        self.assertEqual(model.block_config, override_block)
        self.assertEqual(model.gate_config, override_gate)
        self.assertEqual(model.halting_config, cfg.halting_config)
        self.assertEqual(model.block_model.model.increment, 3.0)
        self.assertEqual(model.block_model.input_dim, 3)
        self.assertEqual(model.block_model.output_dim, 3)
        self.assertIsInstance(model.gate_model, Layer)

    def test_validation_errors(self):
        dim = 4
        valid_block = self.layer_block_config()
        nested_gate_config = self.gate_config()
        nested_gate_config.layer_config.gate_config = self.gate_config()
        invalid_cases = [
            (
                "block_config_none",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    block_config=None,
                ),
                ValueError,
            ),
            (
                "invalid_block_type",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    block_config=object(),
                ),
                TypeError,
            ),
            (
                "block_missing_input_dim",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    block_config=MissingInputDimBlockConfig(output_dim=dim),
                ),
                TypeError,
            ),
            (
                "block_missing_output_dim",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    block_config=MissingOutputDimBlockConfig(input_dim=dim),
                ),
                TypeError,
            ),
            (
                "max_steps_zero",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=0,
                    block_config=valid_block,
                ),
                ValueError,
            ),
            (
                "non_int_max_steps",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps="3",
                    block_config=valid_block,
                ),
                TypeError,
            ),
            (
                "mismatched_recurrent_dimensions",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim + 1,
                    max_steps=1,
                    block_config=valid_block,
                ),
                ValueError,
            ),
            (
                "invalid_gate_config",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    block_config=valid_block,
                    gate_config=object(),
                ),
                TypeError,
            ),
            (
                "nested_gate_config",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    block_config=valid_block,
                    gate_config=nested_gate_config,
                ),
                ValueError,
            ),
            (
                "invalid_halting_config",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    block_config=valid_block,
                    halting_config=object(),
                ),
                TypeError,
            ),
            (
                "unsupported_halting_finalization_contract",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    block_config=valid_block,
                    halting_config=SoftHaltingConfig(
                        input_dim=dim,
                        threshold=0.99,
                        halting_dropout=0.0,
                        hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                        halting_gate_config=self.halting_gate_config(threshold=1.0),
                    ),
                ),
                ValueError,
            ),
        ]

        for name, cfg, expected_exception in invalid_cases:
            with self.subTest(name=name):
                with self.assertRaises(expected_exception):
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

    def test_forward_input_validation_errors(self):
        dim = 4
        model = RecurrentLayer(self.recurrent_config(dim=dim))
        invalid_cases = [
            ("non_layer_state", torch.zeros(2, dim), TypeError),
            ("hidden_rank_lt_2", LayerState(hidden=torch.randn(dim)), ValueError),
            (
                "wrong_feature_dim",
                LayerState(hidden=torch.randn(2, dim + 1)),
                ValueError,
            ),
        ]

        for name, state, expected_exception in invalid_cases:
            with self.subTest(name=name):
                with self.assertRaises(expected_exception):
                    model(state)

        bad_block = self.layer_block_config(
            increment=1.0,
            input_dim=dim,
            output_dim=dim,
        )
        bad_block.layer_model_config = MismatchedShapeFeatureLastConfig()
        bad_model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=bad_block,
            )
        )

        with self.subTest(name="candidate_shape_mismatch"):
            with self.assertRaises(ValueError):
                bad_model(LayerState(hidden=torch.zeros(2, dim)))

    def test_resolve_block_overrides_for_supported_block_configs(self):
        dim = 5
        cases = [
            (
                "layer",
                self.layer_block_config(input_dim=1, output_dim=2),
                LayerConfig,
                {"input_dim": dim, "output_dim": dim},
            ),
            (
                "stack",
                self.stack_block_config(num_layers=2),
                LayerStackConfig,
                {"input_dim": dim, "output_dim": dim},
            ),
            (
                "moe",
                self.moe_block_config(dim=dim, config_dim=1),
                MixtureOfExpertsModelConfig,
                {"input_dim": dim, "output_dim": dim},
            ),
        ]

        for name, block_config, expected_type, expected_fields in cases:
            with self.subTest(name=name):
                model = RecurrentLayer(
                    self.recurrent_config(
                        dim=dim,
                        block_config=block_config,
                    )
                )
                overrides = model._RecurrentLayer__resolve_block_overrides()

                self.assertIsInstance(overrides, expected_type)
                for field_name, expected_value in expected_fields.items():
                    self.assertEqual(getattr(overrides, field_name), expected_value)
                if isinstance(overrides, LayerStackConfig):
                    self.assertIsNone(overrides.hidden_dim)

    def test_custom_config_base_block_builds_through_recurrent_layer(self):
        dim = 4
        original_hidden_dim = 9
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=2,
                block_config=CustomStateBlockConfig(
                    input_dim=1,
                    hidden_dim=original_hidden_dim,
                    output_dim=2,
                    increment=1.25,
                ),
            )
        )
        hidden = torch.zeros(2, dim)

        result = model(LayerState(hidden=hidden))

        self.assertIsInstance(model.block_model, CustomStateBlock)
        self.assertEqual(model.block_model.input_dim, dim)
        self.assertEqual(model.block_model.hidden_dim, original_hidden_dim)
        self.assertEqual(model.block_model.output_dim, dim)
        torch.testing.assert_close(result.hidden, torch.full_like(hidden, 2.5))

    def test_preserve_halted_hidden_broadcasts_mask_for_2d_and_3d_hidden(self):
        model = RecurrentLayer(self.recurrent_config(dim=3))
        cases = [
            (
                "2d",
                torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
                torch.tensor([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]),
                torch.tensor([True, False]),
                torch.tensor([[1.0, 1.0, 1.0], [20.0, 20.0, 20.0]]),
            ),
            (
                "3d",
                torch.arange(12.0).view(2, 2, 3),
                torch.full((2, 2, 3), 100.0),
                torch.tensor([[True, False], [False, True]]),
                torch.tensor(
                    [
                        [[0.0, 1.0, 2.0], [100.0, 100.0, 100.0]],
                        [[100.0, 100.0, 100.0], [9.0, 10.0, 11.0]],
                    ]
                ),
            ),
        ]

        for name, previous_hidden, candidate_hidden, halt_mask, expected in cases:
            with self.subTest(name=name):
                result = model._RecurrentLayer__preserve_halted_hidden(
                    previous_hidden,
                    candidate_hidden,
                    halt_mask,
                )

                torch.testing.assert_close(result, expected)

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

    def test_recurrent_trainable_block_receives_gradients_across_steps(self):
        dim = 3
        max_steps = 3
        cfg = self.recurrent_config(
            dim=dim,
            max_steps=max_steps,
            block_config=LayerConfig(
                activation=ActivationOptions.DISABLED,
                residual_flag=False,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                shared_halting_flag=False,
                layer_model_config=TrainableScaleFeatureLastConfig(scale=0.5),
            ),
        )
        model = RecurrentLayer(cfg)
        hidden = torch.ones(2, dim)

        result = model(LayerState(hidden=hidden))
        result.hidden.sum().backward()

        scale_grad = model.block_model.model.scale.grad
        self.assertIsNotNone(scale_grad)
        self.assertTrue(torch.any(scale_grad.abs() > 0))

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
        original_hidden_dim = 3
        cfg = self.recurrent_config(
            dim=dim,
            block_config=self.stack_block_config(
                hidden_dim=original_hidden_dim,
                num_layers=2,
            ),
        )

        model = RecurrentLayer(cfg)
        layers = list(model.block_model)

        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0].input_dim, dim)
        self.assertEqual(layers[0].output_dim, original_hidden_dim)
        self.assertEqual(layers[0].model.input_dim, dim)
        self.assertEqual(layers[0].model.output_dim, original_hidden_dim)
        self.assertEqual(layers[-1].input_dim, original_hidden_dim)
        self.assertEqual(layers[-1].output_dim, dim)
        self.assertEqual(layers[-1].model.input_dim, original_hidden_dim)
        self.assertEqual(layers[-1].model.output_dim, dim)

    def test_mixture_of_experts_model_block_dimensions_are_overridden(self):
        dim = 4
        cfg = self.recurrent_config(
            dim=dim,
            max_steps=2,
            block_config=self.moe_block_config(dim=dim, config_dim=1),
        )

        model = RecurrentLayer(cfg)
        result = model(LayerState(hidden=torch.zeros(3, dim)))

        self.assertIsInstance(model.block_model, MixtureOfExpertsModel)
        self.assertEqual(model.block_model.input_dim, dim)
        self.assertEqual(model.block_model.output_dim, dim)
        self.assertEqual(result.hidden.shape, (3, dim))

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
                    input_dim=dim,
                    hidden_dim=dim,
                    output_dim=dim,
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
