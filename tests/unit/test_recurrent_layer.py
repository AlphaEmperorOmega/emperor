import unittest
from dataclasses import dataclass

import torch

from emperor.attention import AttentionLayerState
from emperor.config import ConfigBase, optional_field
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
)
from emperor.experts._model import MixtureOfExpertsModel
from emperor.halting import (
    HaltingHiddenStateModeOptions,
    HaltingStateBase,
    SoftHaltingConfig,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    LayerState,
    RecurrentLayer,
    RecurrentLayerConfig,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.layers._composition.gate import LayerGate
from emperor.layers._recurrent import _RecurrentState
from emperor.linears import LinearLayerConfig
from emperor.memory import (
    DynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
)
from emperor.nn import Module
from emperor.sampler import RouterConfig, SamplerConfig


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
class StateSpyBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    increment: float | None = optional_field("Value added to every hidden element.")

    def _registry_owner(self) -> type:
        return StateSpyBlock


class StateSpyBlock(Module):
    def __init__(
        self,
        cfg: StateSpyBlockConfig,
        overrides: StateSpyBlockConfig | None = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.increment = self.cfg.increment
        self.received_states = []
        self.received_hidden_inputs = []

    def forward(self, state: LayerState) -> LayerState:
        self.received_states.append(state)
        self.received_hidden_inputs.append(state.hidden.detach().clone())
        if state.hidden.shape[-1] != self.input_dim:
            raise ValueError("StateSpyBlock received the wrong input dimension")
        state.hidden = state.hidden + self.increment
        return state


@dataclass
class LossAccumulatingBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    increment: float | None = optional_field("Value added to every hidden element.")
    per_step_loss: float | None = optional_field("Auxiliary loss added per step.")

    def _registry_owner(self) -> type:
        return LossAccumulatingBlock


class LossAccumulatingBlock(Module):
    def __init__(
        self,
        cfg: LossAccumulatingBlockConfig,
        overrides: LossAccumulatingBlockConfig | None = None,
    ):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.increment = self.cfg.increment
        self.per_step_loss = self.cfg.per_step_loss

    def forward(self, state: LayerState) -> LayerState:
        if state.hidden.shape[-1] != self.input_dim:
            raise ValueError("LossAccumulatingBlock received the wrong input dimension")
        state.hidden = state.hidden + self.increment
        loss = torch.tensor(
            self.per_step_loss,
            dtype=state.hidden.dtype,
            device=state.hidden.device,
        )
        state.loss = loss if state.loss is None else state.loss + loss
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


class RecordingTransform(torch.nn.Module):
    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        super().__init__()
        self.scale = scale
        self.offset = offset
        self.inputs: list[torch.Tensor] = []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.inputs.append(X.detach().clone())
        return X * self.scale + self.offset


class AddConstantMemory(torch.nn.Module):
    def __init__(self, value: float, position: MemoryPositionOptions):
        super().__init__()
        self.value = value
        self.memory_position_option = position
        self.inputs: list[torch.Tensor] = []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.inputs.append(X.detach().clone())
        return X + self.value


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
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=halting_config,
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
                        residual_config=None,
                        dropout_probability=0.0,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        gate_config=None,
                        halting_config=None,
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
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
                    dropout_probability=0.0,
                    gate_config=None,
                    halting_config=None,
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
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                layer_model_config=ConstantFeatureLastConfig(value=value),
            ),
        )

    def trainable_gate_config(self, dim: int) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=dim,
                output_dim=dim,
                activation=ActivationOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    bias_flag=True,
                ),
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
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
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
        gate_config: LayerStackConfig | GateConfig | None = None,
        gate_option: LayerGateOptions | None = None,
        gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID,
        residual_connection_option: ResidualConnectionOptions | None = None,
        halting_config: StickBreakingConfig | None = None,
        memory_config: DynamicMemoryConfig | None = None,
        recurrent_layer_norm_position: LayerNormPositionOptions = (
            LayerNormPositionOptions.DISABLED
        ),
        residual_model_config: LinearLayerConfig | None = None,
    ) -> RecurrentLayerConfig:
        if block_config is None:
            block_config = self.layer_block_config()
        return RecurrentLayerConfig(
            input_dim=dim,
            output_dim=dim,
            max_steps=max_steps,
            recurrent_layer_norm_position=recurrent_layer_norm_position,
            block_config=block_config,
            gate_config=self.recurrent_gate_config(
                gate_config,
                gate_option,
                gate_activation,
            ),
            residual_config=None
            if residual_connection_option is None
            else ResidualConfig(
                option=residual_connection_option, model_config=residual_model_config
            ),
            halting_config=halting_config,
            memory_config=memory_config,
        )

    def recurrent_gate_config(
        self,
        model_config: LayerStackConfig | GateConfig | None,
        option: LayerGateOptions | None,
        activation: ActivationOptions | None = ActivationOptions.SIGMOID,
    ) -> GateConfig | None:
        if isinstance(model_config, GateConfig):
            return model_config
        if model_config is None:
            if option is None:
                return None
            model_config = self.gate_config()
        if option is None:
            option = LayerGateOptions.MULTIPLIER
        return GateConfig(
            model_config=model_config,
            option=option,
            activation=activation,
        )

    def test_public_exports_and_config_build_dispatch(self):
        import emperor.layers as layer_package

        expected_exports = [
            "RecurrentLayerConfig",
            "RecurrentLayer",
        ]
        for name in expected_exports:
            with self.subTest(name=name):
                self.assertIn(name, layer_package.__all__)
                self.assertIsNotNone(getattr(layer_package, name))
        self.assertNotIn("RecurrentLayerValidator", layer_package.__all__)

        cfg = self.recurrent_config()
        model = cfg.build()

        self.assertIsInstance(model, RecurrentLayer)
        self.assertIsInstance(model.block_model, Layer)

    def test_gate_config_registry_owner_and_build_return_recurrent_gate(self):
        gate_config = GateConfig(
            model_config=self.gate_config(),
            option=LayerGateOptions.MULTIPLIER,
        )

        self.assertIs(gate_config._registry_owner(), LayerGate)
        built_gate = gate_config.build()
        self.assertIsInstance(built_gate, LayerGate)
        self.assertEqual(built_gate.option, LayerGateOptions.MULTIPLIER)

    def test_gate_config_build_rejects_missing_option(self):
        gate_config = GateConfig(model_config=self.gate_config())

        with self.assertRaisesRegex(ValueError, "GateConfig.option.*LayerGateOptions"):
            gate_config.build()

    def test_init_stores_all_config_attributes_and_builds_optional_models(self):
        dim = 2
        cfg = self.recurrent_config(
            dim=dim,
            max_steps=4,
            block_config=self.layer_block_config(increment=1.5),
            gate_config=self.gate_config(value=0.25),
            residual_connection_option=ResidualConnectionOptions.WEIGHTED_BLEND,
            halting_config=self.halting_config(dim=dim, gate_threshold=10.0),
        )

        model = RecurrentLayer(cfg)

        self.assertIsInstance(model, RecurrentLayer)
        self.assertEqual(model.input_dim, cfg.input_dim)
        self.assertEqual(model.output_dim, cfg.output_dim)
        self.assertEqual(model.max_steps, cfg.max_steps)
        self.assertEqual(
            model.recurrent_layer_norm_position,
            cfg.recurrent_layer_norm_position,
        )
        self.assertEqual(model.block_config, cfg.block_config)
        self.assertEqual(model.gate_config, cfg.gate_config)
        self.assertIsNotNone(model.recurrent_gate)
        self.assertEqual(model.recurrent_gate.gate_dim, cfg.output_dim)
        self.assertEqual(model.recurrent_gate.option, LayerGateOptions.MULTIPLIER)
        self.assertEqual(
            model.residual_config.option,
            ResidualConnectionOptions.WEIGHTED_BLEND,
        )
        self.assertEqual(model.halting_config, cfg.halting_config)
        self.assertIsInstance(model.block_model, Layer)
        self.assertIsInstance(model.recurrent_gate.model, LayerStack)
        self.assertIsNotNone(model.residual_connection)
        self.assertIsNotNone(model.halting_model)
        self.assertIsNone(model.recurrent_layer_norm_module)

    def test_recurrent_layer_norm_disabled_does_not_create_or_apply_module(self):
        dim = 3
        hidden = torch.arange(6.0).view(2, dim)
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=self.layer_block_config(increment=2.0),
                recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            )
        )

        result = model(LayerState(hidden=hidden.clone()))

        self.assertIsNone(model.recurrent_layer_norm_module)
        torch.testing.assert_close(result.hidden, hidden + 2.0)

    def test_recurrent_layer_norm_before_normalizes_block_input(self):
        dim = 3
        hidden = torch.zeros(2, dim)
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=StateSpyBlockConfig(
                    input_dim=1,
                    output_dim=2,
                    increment=1.0,
                ),
                recurrent_layer_norm_position=LayerNormPositionOptions.BEFORE,
            )
        )
        transform = RecordingTransform(offset=10.0)
        model.recurrent_layer_norm_module = transform

        result = model(LayerState(hidden=hidden.clone()))

        self.assertEqual(len(transform.inputs), 1)
        torch.testing.assert_close(transform.inputs[0], hidden)
        torch.testing.assert_close(
            model.block_model.received_hidden_inputs[0],
            hidden + 10.0,
        )
        torch.testing.assert_close(result.hidden, hidden + 11.0)

    def test_recurrent_layer_norm_default_normalizes_after_block_and_memory(self):
        dim = 3
        hidden = torch.zeros(2, dim)
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=StateSpyBlockConfig(
                    input_dim=1,
                    output_dim=2,
                    increment=1.0,
                ),
                recurrent_layer_norm_position=LayerNormPositionOptions.DEFAULT,
            )
        )
        transform = RecordingTransform(offset=10.0)
        model.recurrent_layer_norm_module = transform

        result = model(LayerState(hidden=hidden.clone()))

        self.assertEqual(len(transform.inputs), 1)
        torch.testing.assert_close(
            model.block_model.received_hidden_inputs[0],
            hidden,
        )
        torch.testing.assert_close(transform.inputs[0], hidden + 1.0)
        torch.testing.assert_close(result.hidden, hidden + 11.0)

    def test_recurrent_layer_norm_after_normalizes_after_recurrent_controllers(self):
        dim = 3
        hidden = torch.zeros(2, dim)
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=self.layer_block_config(increment=1.0),
                gate_config=self.gate_config(value=2.0),
                gate_option=LayerGateOptions.ADDITION,
                gate_activation=None,
                recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
            )
        )
        transform = RecordingTransform(scale=10.0)
        model.recurrent_layer_norm_module = transform

        result = model(LayerState(hidden=hidden.clone()))

        self.assertEqual(len(transform.inputs), 1)
        torch.testing.assert_close(transform.inputs[0], torch.full_like(hidden, 3.0))
        torch.testing.assert_close(result.hidden, torch.full_like(hidden, 30.0))

    def test_recurrent_layer_norm_after_normalizes_before_halting_update(self):
        class RecordingHalting:
            def __init__(self):
                self.received_hidden = None

            def update_halting_state(self, previous_state, hidden):
                self.received_hidden = hidden.detach().clone()
                return previous_state, hidden + 1.0

        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
            )
        )
        transform = RecordingTransform(offset=10.0)
        halting = RecordingHalting()
        model.recurrent_layer_norm_module = transform
        model.halting_model = halting
        previous_hidden = torch.zeros(2, dim)
        candidate_hidden = torch.full_like(previous_hidden, 3.0)
        recurrent_state = _RecurrentState(
            hidden=candidate_hidden,
            loss=None,
            context_state=LayerState(hidden=previous_hidden.clone()),
            halting_state=None,
        )

        result = model._RecurrentLayer__run_controllers(
            recurrent_state,
            previous_hidden,
        )

        torch.testing.assert_close(transform.inputs[0], candidate_hidden)
        torch.testing.assert_close(halting.received_hidden, candidate_hidden + 10.0)
        torch.testing.assert_close(result.hidden, candidate_hidden + 11.0)

    def test_recurrent_controllers_apply_gate_residual_norm_before_halting(self):
        class RecordingGate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.received_hidden = None

            def forward(self, hidden: torch.Tensor) -> torch.Tensor:
                self.received_hidden = hidden.detach().clone()
                return hidden + 2.0

        class RecordingResidual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.received_current = None
                self.received_previous = None

            def forward(
                self,
                current: torch.Tensor,
                previous: torch.Tensor,
            ) -> torch.Tensor:
                self.received_current = current.detach().clone()
                self.received_previous = previous.detach().clone()
                return current + 3.0 * previous

        class RecordingHalting:
            def __init__(self, next_state):
                self.next_state = next_state
                self.received_state = None
                self.received_hidden = None

            def update_halting_state(self, previous_state, hidden):
                self.received_state = previous_state
                self.received_hidden = hidden.detach().clone()
                return self.next_state, hidden + 7.0

        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
            )
        )
        gate = RecordingGate()
        residual = RecordingResidual()
        layer_norm = RecordingTransform(scale=5.0)
        next_halting_state = DummyHaltingState(marker="next")
        halting = RecordingHalting(next_halting_state)
        model.recurrent_gate = gate
        model.residual_connection = residual
        model.recurrent_layer_norm_module = layer_norm
        model.halting_model = halting
        candidate_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        previous_hidden = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        previous_halting_state = DummyHaltingState(marker="previous")
        existing_loss = torch.tensor(2.5)
        recurrent_state = _RecurrentState(
            hidden=candidate_hidden,
            loss=existing_loss,
            context_state=LayerState(hidden=previous_hidden.clone()),
            halting_state=previous_halting_state,
        )

        result = model._RecurrentLayer__run_controllers(
            recurrent_state,
            previous_hidden,
        )

        gated_hidden = candidate_hidden + 2.0
        residual_hidden = gated_hidden + 3.0 * previous_hidden
        normalized_hidden = residual_hidden * 5.0
        expected_hidden = normalized_hidden + 7.0
        torch.testing.assert_close(gate.received_hidden, candidate_hidden)
        torch.testing.assert_close(residual.received_current, gated_hidden)
        torch.testing.assert_close(residual.received_previous, previous_hidden)
        torch.testing.assert_close(layer_norm.inputs[0], residual_hidden)
        self.assertIs(halting.received_state, previous_halting_state)
        torch.testing.assert_close(halting.received_hidden, normalized_hidden)
        torch.testing.assert_close(result.hidden, expected_hidden)
        self.assertIs(result.loss, existing_loss)
        self.assertIs(result.halting_state, next_halting_state)

    def test_recurrent_layer_norm_parameters_receive_gradients_when_enabled(self):
        dim = 3
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=self.layer_block_config(increment=1.0),
                recurrent_layer_norm_position=LayerNormPositionOptions.DEFAULT,
            )
        )
        hidden = torch.tensor([[1.0, 2.0, 4.0], [2.0, 5.0, 9.0]])

        result = model(LayerState(hidden=hidden))
        result.hidden.sum().backward()

        parameters = list(model.recurrent_layer_norm_module.parameters())
        nonzero_gradients = [
            parameter.grad
            for parameter in parameters
            if parameter.grad is not None and torch.any(parameter.grad.abs() > 0)
        ]
        self.assertTrue(len(nonzero_gradients) > 0)

    def test_recurrent_gate_and_weighted_residual_receive_gradients(self):
        dim = 3
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=self.layer_block_config(increment=1.0),
                gate_config=self.trainable_gate_config(dim),
                gate_option=LayerGateOptions.MULTIPLIER,
                gate_activation=None,
                residual_connection_option=ResidualConnectionOptions.WEIGHTED_BLEND,
            )
        )
        gate_layer = model.recurrent_gate.model[0]
        with torch.no_grad():
            gate_layer.model.weight_params.fill_(0.1)
            gate_layer.model.bias_params.fill_(0.2)
        hidden = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 0.5, 1.5]],
            requires_grad=True,
        )

        result = model(LayerState(hidden=hidden))
        result.hidden.sum().backward()

        gate_gradients = [
            parameter.grad
            for parameter in model.recurrent_gate.model.parameters()
            if parameter.requires_grad
        ]
        nonzero_gate_gradients = [
            gradient
            for gradient in gate_gradients
            if gradient is not None and torch.any(gradient.abs() > 0)
        ]
        residual_gradient = model.residual_connection.raw_weight.grad
        self.assertTrue(len(nonzero_gate_gradients) > 0)
        self.assertIsNotNone(residual_gradient)
        self.assertTrue(torch.any(residual_gradient.abs() > 0))
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.any(hidden.grad.abs() > 0))

    def test_recurrent_memory_builds_and_applies_at_configured_position(self):
        dim = 3
        hidden = torch.zeros(2, dim)
        cases = [
            (
                MemoryPositionOptions.BEFORE_AFFINE,
                hidden,
                torch.full_like(hidden, 2.0),
            ),
            (
                MemoryPositionOptions.AFTER_AFFINE,
                torch.full_like(hidden, 1.0),
                hidden,
            ),
        ]

        for position, expected_memory_input, expected_block_input in cases:
            with self.subTest(position=position):
                memory_config = GatedResidualDynamicMemoryConfig(
                    input_dim=dim,
                    output_dim=dim,
                    memory_position_option=position,
                    test_time_training_learning_rate=None,
                    test_time_training_num_inner_steps=None,
                    model_config=self.trainable_gate_config(dim),
                )
                model = RecurrentLayer(
                    self.recurrent_config(
                        dim=dim,
                        max_steps=1,
                        block_config=StateSpyBlockConfig(
                            input_dim=1,
                            output_dim=2,
                            increment=1.0,
                        ),
                        memory_config=memory_config,
                    )
                )
                self.assertIsInstance(
                    model.memory_model,
                    memory_config._registry_owner(),
                )
                memory = AddConstantMemory(2.0, position)
                model.memory_model = memory

                result = model(LayerState(hidden=hidden.clone()))

                self.assertEqual(len(memory.inputs), 1)
                torch.testing.assert_close(memory.inputs[0], expected_memory_input)
                torch.testing.assert_close(
                    model.block_model.received_hidden_inputs[0],
                    expected_block_input,
                )
                torch.testing.assert_close(
                    result.hidden,
                    torch.full_like(hidden, 3.0),
                )

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
            gate_config=self.recurrent_gate_config(override_gate, None),
            residual_config=ResidualConfig(option=ResidualConnectionOptions.RESIDUAL),
        )

        model = RecurrentLayer(cfg, overrides)

        self.assertEqual(model.input_dim, 3)
        self.assertEqual(model.output_dim, 3)
        self.assertEqual(model.max_steps, 5)
        self.assertEqual(model.block_config, override_block)
        self.assertEqual(model.gate_config.model_config, override_gate)
        self.assertEqual(
            model.residual_config.option,
            ResidualConnectionOptions.RESIDUAL,
        )
        self.assertEqual(model.halting_config, cfg.halting_config)
        self.assertEqual(model.block_model.model.increment, 3.0)
        self.assertEqual(model.block_model.input_dim, 3)
        self.assertEqual(model.block_model.output_dim, 3)
        self.assertIsInstance(model.recurrent_gate.model, LayerStack)

    def test_validation_errors(self):
        dim = 4
        valid_block = self.layer_block_config()
        nested_gate_config = self.gate_config()
        nested_gate_config.layer_config.gate_config = GateConfig(
            model_config=self.gate_config()
        )
        invalid_cases = [
            (
                "block_config_none",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=None,
                    residual_config=None,
                ),
                ValueError,
            ),
            (
                "invalid_block_type",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=object(),
                    residual_config=None,
                ),
                TypeError,
            ),
            (
                "block_missing_input_dim",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=MissingInputDimBlockConfig(output_dim=dim),
                    residual_config=None,
                ),
                TypeError,
            ),
            (
                "block_missing_output_dim",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=MissingOutputDimBlockConfig(input_dim=dim),
                    residual_config=None,
                ),
                TypeError,
            ),
            (
                "max_steps_zero",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=0,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    residual_config=None,
                ),
                ValueError,
            ),
            (
                "non_int_max_steps",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps="3",
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    residual_config=None,
                ),
                TypeError,
            ),
            (
                "mismatched_recurrent_dimensions",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim + 1,
                    max_steps=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    residual_config=None,
                ),
                ValueError,
            ),
            (
                "missing_recurrent_layer_norm_position",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    block_config=valid_block,
                    residual_config=None,
                ),
                ValueError,
            ),
            (
                "invalid_recurrent_layer_norm_position",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    recurrent_layer_norm_position=object(),
                    block_config=valid_block,
                    residual_config=None,
                ),
                TypeError,
            ),
            (
                "invalid_gate_config",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    gate_config=object(),
                    residual_config=None,
                ),
                TypeError,
            ),
            (
                "invalid_gate_option",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    gate_config=GateConfig(option=object()),
                    residual_config=None,
                ),
                TypeError,
            ),
            (
                "invalid_residual_connection_option",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    residual_config=ResidualConfig(option=object()),
                ),
                TypeError,
            ),
            (
                "nested_gate_config",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    gate_config=GateConfig(
                        model_config=nested_gate_config,
                        option=LayerGateOptions.MULTIPLIER,
                    ),
                    residual_config=None,
                ),
                ValueError,
            ),
            (
                "invalid_halting_config",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    halting_config=object(),
                    residual_config=None,
                ),
                TypeError,
            ),
            (
                "unsupported_halting_finalization_contract",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
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

    def test_block_dimensions_are_overridden_for_supported_block_configs(self):
        dim = 5
        cases = [
            (
                "layer",
                self.layer_block_config(input_dim=1, output_dim=2),
            ),
            (
                "stack",
                self.stack_block_config(num_layers=2),
            ),
            (
                "moe",
                self.moe_block_config(dim=dim, config_dim=1),
            ),
        ]

        for name, block_config in cases:
            with self.subTest(name=name):
                model = RecurrentLayer(
                    self.recurrent_config(
                        dim=dim,
                        block_config=block_config,
                    )
                )
                block_model = model.block_model

                self.assertEqual(block_model.input_dim, dim)
                self.assertEqual(block_model.output_dim, dim)
                self.assertEqual(block_model.cfg.input_dim, dim)
                self.assertEqual(block_model.cfg.output_dim, dim)
                if isinstance(block_model, LayerStack):
                    self.assertEqual(block_model.hidden_dim, block_config.hidden_dim)

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

    def test_recurrent_block_preserves_attention_state_and_gate_gets_plain_state(self):
        class SpyGate(Module):
            def __init__(self):
                super().__init__()
                self.received_state = None
                self.received_hidden = None

            def forward(self, state: LayerState) -> LayerState:
                self.received_state = state
                self.received_hidden = state.hidden
                state.hidden = torch.zeros_like(state.hidden)
                return state

        dim = 3
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=StateSpyBlockConfig(
                    input_dim=1,
                    output_dim=2,
                    increment=1.0,
                ),
                gate_config=self.gate_config(),
            )
        )
        gate = SpyGate()
        model.recurrent_gate.model = gate
        hidden = torch.zeros(2, dim)
        key_padding_mask = torch.tensor([[False, True], [False, False]])
        attention_mask = torch.zeros(2, 2)
        existing_loss = torch.tensor(4.0)
        state = AttentionLayerState(
            hidden=hidden,
            loss=existing_loss,
            halting_state=object(),
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )

        result = model(state)

        block_state = model.block_model.received_states[0]
        gate_state = gate.received_state
        self.assertIs(result, state)
        self.assertIsInstance(block_state, AttentionLayerState)
        self.assertIs(type(gate_state), LayerState)
        torch.testing.assert_close(gate.received_hidden, hidden + 1.0)
        self.assertIs(block_state.key_padding_mask, key_padding_mask)
        self.assertIs(block_state.attention_mask, attention_mask)
        self.assertFalse(hasattr(gate_state, "key_padding_mask"))
        self.assertFalse(hasattr(gate_state, "attention_mask"))
        self.assertIs(block_state.loss, existing_loss)
        self.assertIsNone(block_state.halting_state)
        self.assertIsNone(gate_state.loss)
        self.assertIsNone(gate_state.halting_state)

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
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
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

    def test_recurrent_gate_scales_only_when_config_exists(self):
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
                gate_config=self.gate_config(value=0.5),
                gate_activation=None,
            )
        )

        ungated = without_gate(LayerState(hidden=hidden.clone()))
        gated = with_gate(LayerState(hidden=hidden.clone()))

        torch.testing.assert_close(ungated.hidden, torch.full_like(hidden, 3.0))
        torch.testing.assert_close(gated.hidden, torch.full_like(hidden, 1.5))

    def test_recurrent_gate_option_none_is_rejected_when_gate_config_is_provided(self):
        gate_config = GateConfig(
            model_config=self.gate_config(),
            option=None,
        )

        with self.assertRaisesRegex(
            ValueError,
            "RecurrentLayerConfig.gate_config.option.*LayerGateOptions.*MULTIPLIER",
        ):
            RecurrentLayer(self.recurrent_config(gate_config=gate_config))

    def test_recurrent_residual_config_none_disables_residuals(self):
        config = self.recurrent_config()
        config.residual_config = None

        model = RecurrentLayer(config)

        self.assertIsNone(model.residual_config)
        self.assertIsNone(model.residual_connection)

    def test_recurrent_residual_options_apply_between_steps(self):
        dim = 3
        hidden = torch.ones(2, dim)
        block_config = self.layer_block_config(increment=2.0)
        data_dependent_model_config = LinearLayerConfig(bias_flag=True)
        cases = [
            (
                None,
                None,
                torch.full_like(hidden, 3.0),
            ),
            (
                ResidualConnectionOptions.RESIDUAL,
                None,
                torch.full_like(hidden, 4.0),
            ),
            (
                ResidualConnectionOptions.WEIGHTED_RESIDUAL,
                None,
                torch.full_like(hidden, 1.0),
            ),
            (
                ResidualConnectionOptions.WEIGHTED_BLEND,
                None,
                torch.full_like(hidden, 2.8),
            ),
            (
                ResidualConnectionOptions.WEIGHTED_RESIDUAL,
                data_dependent_model_config,
                torch.full_like(hidden, 1.0),
            ),
            (
                ResidualConnectionOptions.WEIGHTED_BLEND,
                data_dependent_model_config,
                torch.full_like(hidden, 2.8),
            ),
        ]

        for option, residual_model_config, expected in cases:
            with self.subTest(
                option=option,
                data_dependent=residual_model_config is not None,
            ):
                model = RecurrentLayer(
                    self.recurrent_config(
                        dim=dim,
                        max_steps=1,
                        block_config=block_config,
                        residual_connection_option=option,
                        residual_model_config=residual_model_config,
                    )
                )

                result = model(LayerState(hidden=hidden.clone()))

                torch.testing.assert_close(result.hidden, expected)

    def test_recurrent_data_dependent_residual_builds_configured_model(self):
        dim = 3
        residual_model_config = LinearLayerConfig(
            input_dim=99,
            output_dim=99,
            bias_flag=True,
        )

        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                residual_connection_option=ResidualConnectionOptions.WEIGHTED_BLEND,
                residual_model_config=residual_model_config,
            )
        )

        residual_connection = model.residual_connection
        self.assertEqual(residual_connection.model_config, residual_model_config)
        self.assertIsNot(residual_connection.model_config, residual_model_config)
        self.assertEqual(residual_connection.model.input_dim, dim * 2)
        self.assertEqual(residual_connection.model.output_dim, dim)

    def test_missing_recurrent_gate_config_bypasses_recurrent_gate(self):
        dim = 3
        hidden = torch.ones(2, dim)
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=self.layer_block_config(increment=2.0),
            )
        )
        self.assertIsNone(model.recurrent_gate)

        result = model(LayerState(hidden=hidden.clone()))

        torch.testing.assert_close(result.hidden, torch.full_like(hidden, 3.0))

    def test_recurrent_gate_options_apply_expected_formula(self):
        dim = 3
        previous = torch.ones(2, dim)
        candidate = previous + 2.0
        gate_logits = torch.full_like(previous, 1.0)
        cases = [
            (LayerGateOptions.MULTIPLIER, None, gate_logits * candidate),
            (
                LayerGateOptions.MULTIPLIER,
                ActivationOptions.SIGMOID,
                torch.sigmoid(gate_logits) * candidate,
            ),
            (
                LayerGateOptions.MULTIPLIER,
                ActivationOptions.TANH,
                torch.tanh(gate_logits) * candidate,
            ),
            (LayerGateOptions.ADDITION, None, candidate + gate_logits),
            (
                LayerGateOptions.ADDITION,
                ActivationOptions.SIGMOID,
                candidate + torch.sigmoid(gate_logits),
            ),
            (
                LayerGateOptions.ADDITION,
                ActivationOptions.TANH,
                candidate + torch.tanh(gate_logits),
            ),
        ]

        for option, gate_activation, expected in cases:
            with self.subTest(option=option, gate_activation=gate_activation):
                model = RecurrentLayer(
                    self.recurrent_config(
                        dim=dim,
                        max_steps=1,
                        block_config=self.layer_block_config(increment=2.0),
                        gate_config=self.gate_config(value=1.0),
                        gate_option=option,
                        gate_activation=gate_activation,
                    )
                )

                result = model(LayerState(hidden=previous.clone()))

                torch.testing.assert_close(result.hidden, expected)

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

    def test_recurrent_halting_preserves_3d_halted_positions_with_controllers(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=4,
                block_config=self.layer_block_config(increment=1.0),
                gate_config=self.gate_config(value=0.0),
                gate_option=LayerGateOptions.ADDITION,
                gate_activation=None,
                residual_connection_option=ResidualConnectionOptions.RESIDUAL,
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=3.0,
                    high_logit=20.0,
                    low_logit=-20.0,
                ),
                recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
            )
        )
        layer_norm = RecordingTransform()
        model.recurrent_layer_norm_module = layer_norm
        model.eval()
        hidden = torch.tensor(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ]
        )

        result = model(LayerState(hidden=hidden.clone()))

        expected = torch.full_like(hidden, 3.0)
        self.assertEqual(model.block_model.model.call_count, 2)
        self.assertEqual(len(layer_norm.inputs), 2)
        torch.testing.assert_close(result.hidden, expected, rtol=1e-5, atol=1e-5)

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

    def test_recurrent_loss_accumulates_block_and_halting_loss_exactly(self):
        class FakeMaskHaltingState:
            def __init__(self, halt_mask: torch.Tensor):
                self.halt_mask = halt_mask

        class VectorLossHalting:
            def __init__(self, loss: torch.Tensor):
                self.loss = loss
                self.update_calls = 0
                self.finalize_calls = 0
                self.finalize_input = None

            def update_halting_state(self, previous_state, hidden):
                self.update_calls += 1
                if previous_state is None:
                    previous_state = FakeMaskHaltingState(
                        torch.zeros(
                            hidden.shape[:-1],
                            dtype=torch.bool,
                            device=hidden.device,
                        )
                    )
                return previous_state, hidden

            def finalize_weighted_accumulation(self, state, current_hidden):
                self.finalize_calls += 1
                self.finalize_input = current_hidden.detach().clone()
                return current_hidden + 10.0, self.loss.to(current_hidden)

        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=2,
                block_config=LossAccumulatingBlockConfig(
                    input_dim=dim,
                    output_dim=dim,
                    increment=1.0,
                    per_step_loss=0.25,
                ),
            )
        )
        halting = VectorLossHalting(torch.tensor([2.0, 4.0]))
        model.halting_model = halting
        existing_loss = torch.tensor(5.0)
        hidden = torch.zeros(2, dim)

        result = model(LayerState(hidden=hidden, loss=existing_loss))

        expected_hidden = torch.full_like(hidden, 12.0)
        expected_loss = torch.tensor(8.5)
        self.assertEqual(halting.update_calls, 2)
        self.assertEqual(halting.finalize_calls, 1)
        torch.testing.assert_close(halting.finalize_input, torch.full_like(hidden, 2.0))
        torch.testing.assert_close(result.hidden, expected_hidden)
        torch.testing.assert_close(result.loss, expected_loss)

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
