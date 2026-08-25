import unittest
from dataclasses import dataclass, replace
from unittest.mock import patch

import torch

from emperor.attention import AttentionLayerState
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    DynamicDepthOptions,
    GeneratorDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
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
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    HaltingStateBase,
    HaltingUsageTrackerManager,
    SoftHalting,
    SoftHaltingConfig,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    AttentionResidualConfig,
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
    RowLayout,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.layers._composition.gate import LayerGate
from emperor.layers._composition.recurrent.runtime.execution import RecurrentExecution
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentBranchExecutionPlan,
    RecurrentNestedSmoothHandoffExecutionPlan,
)
from emperor.layers._composition.recurrent.runtime.residual_schedule import (
    DepthwiseRecurrentResidualSchedule,
    RecurrentResidualSchedule,
    SharedRecurrentResidualSchedule,
)
from emperor.layers._composition.recurrent.validation import (
    RecurrentExecutionValidator,
    RecurrentResidualScheduleValidator,
)
from emperor.layers._composition.residual.base import (
    ResidualConnectionAbstract,
    ResidualRuntimeRequirement,
    ResidualState,
)
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
class DepthwiseTestResidualConfig(ResidualConfig):
    def _registry_owner(self) -> type:
        return DepthwiseTestResidual


class DepthwiseTestResidual(ResidualConnectionAbstract):
    RUNTIME_REQUIREMENTS = frozenset(
        {ResidualRuntimeRequirement.DEPTH_SPECIFIC_CONNECTIONS}
    )

    def __init__(
        self,
        cfg: DepthwiseTestResidualConfig,
        overrides: DepthwiseTestResidualConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.offset = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        *,
        residual_state=None,
        row_layout=None,
    ) -> torch.Tensor:
        return current + self.offset


class TestRecurrentResidualScheduleValidatorAdapter(unittest.TestCase):
    def test_schedule_exposes_validator_adapter(self):
        self.assertIs(
            RecurrentResidualSchedule.VALIDATOR,
            RecurrentResidualScheduleValidator,
        )

    def test_construction_dispatches_through_substituted_validator(self):
        class RejectingValidator(RecurrentResidualScheduleValidator):
            @classmethod
            def validate(cls, schedule):
                raise RuntimeError("substituted schedule validator was called")

        class RejectingSchedule(SharedRecurrentResidualSchedule):
            VALIDATOR = RejectingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted schedule validator was called",
        ):
            RejectingSchedule(1)

    def test_transition_lookup_dispatches_through_substituted_validator(self):
        class RejectingValidator(RecurrentResidualScheduleValidator):
            @staticmethod
            def validate_transition_index(schedule, transition_index):
                raise RuntimeError("substituted transition validator was called")

        class RejectingSchedule(SharedRecurrentResidualSchedule):
            VALIDATOR = RejectingValidator

        schedule = RejectingSchedule(1)
        primary_connection = AdditiveResidualConfig().build()

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted transition validator was called",
        ):
            schedule.connection_for_transition(primary_connection, 0)

    def test_depthwise_construction_dispatches_through_substituted_validator(self):
        class RejectingValidator(RecurrentResidualScheduleValidator):
            @staticmethod
            def validate_subsequent_connections(schedule):
                raise RuntimeError("substituted depthwise validator was called")

        class RejectingSchedule(DepthwiseRecurrentResidualSchedule):
            VALIDATOR = RejectingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted depthwise validator was called",
        ):
            RejectingSchedule(1, ())

    def test_validation_error_contracts_are_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "transition_count must be a positive integer",
        ):
            SharedRecurrentResidualSchedule(0)

        schedule = SharedRecurrentResidualSchedule(2)
        primary_connection = AdditiveResidualConfig().build()
        with self.assertRaisesRegex(
            IndexError,
            "transition_index must identify a configured recurrent transition",
        ):
            schedule.connection_for_transition(primary_connection, 2)

        with self.assertRaisesRegex(
            ValueError,
            "subsequent_connections must contain one connection",
        ):
            DepthwiseRecurrentResidualSchedule(2, ())

    def test_schedule_forks_state_without_a_gradient_lifecycle(self):
        class ForkOnlyState(ResidualState):
            def __init__(self, source):
                self.source = source
                self.branch_fork_count = 0

            def fork(self):
                self.branch_fork_count += 1
                return type(self)(self.source)

        schedule = SharedRecurrentResidualSchedule(1)
        source = torch.tensor([[1.0, 2.0]], requires_grad=True)
        state = ForkOnlyState(source)

        forked_state = schedule.fork_state(state)

        self.assertEqual(state.branch_fork_count, 1)
        self.assertIsNot(forked_state, state)
        self.assertIs(forked_state.source, source)
        self.assertTrue(forked_state.source.requires_grad)
        self.assertFalse(hasattr(schedule, "fork_state_at_gradient_boundary"))


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
        self.grad_modes: list[bool] = []
        self.outputs: list[torch.Tensor] = []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.grad_modes.append(torch.is_grad_enabled())
        if self.input_dim != self.output_dim:
            raise ValueError(
                "TrainableScaleFeatureLastLayer requires stable dimensions"
            )
        output = X * self.scale
        self.outputs.append(output)
        return output


@dataclass(frozen=True)
class _ForwardGradientSnapshot:
    output: torch.Tensor
    input_gradient: torch.Tensor | None
    parameter_gradients: dict[str, torch.Tensor | None]


def _capture_forward_gradients(
    model: RecurrentLayer,
    input_values: torch.Tensor,
) -> _ForwardGradientSnapshot:
    model_input = input_values.detach().clone().requires_grad_()
    output = model(LayerState(hidden=model_input)).hidden
    named_parameters = tuple(model.named_parameters())
    gradients = torch.autograd.grad(
        output.sum(),
        (model_input, *(parameter for _, parameter in named_parameters)),
        allow_unused=True,
    )
    return _ForwardGradientSnapshot(
        output=output.detach(),
        input_gradient=(None if gradients[0] is None else gradients[0].detach()),
        parameter_gradients={
            name: None if gradient is None else gradient.detach()
            for (name, _), gradient in zip(
                named_parameters,
                gradients[1:],
                strict=True,
            )
        },
    )


@dataclass
class IdentityStateBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")

    def _registry_owner(self) -> type:
        return IdentityStateBlock


class IdentityStateBlock(Module):
    def __init__(
        self,
        cfg: IdentityStateBlockConfig,
        overrides: IdentityStateBlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)

    def forward(self, state: LayerState) -> LayerState:
        return state


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
        self.grad_modes: list[bool] = []

    def forward(self, state: LayerState) -> LayerState:
        self.received_states.append(state)
        self.received_hidden_inputs.append(state.hidden.detach().clone())
        self.grad_modes.append(torch.is_grad_enabled())
        if state.hidden.shape[-1] != self.input_dim:
            raise ValueError("StateSpyBlock received the wrong input dimension")
        state.hidden = state.hidden + self.increment
        return state


@dataclass
class StochasticStateBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")

    def _registry_owner(self) -> type:
        return StochasticStateBlock


class StochasticStateBlock(Module):
    def __init__(
        self,
        cfg: StochasticStateBlockConfig,
        overrides: StochasticStateBlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.call_count = 0

    def forward(self, state: LayerState) -> LayerState:
        self.call_count += 1
        state.hidden = state.hidden * self.scale + torch.rand_like(state.hidden)
        return state


@dataclass
class FailingStochasticStateBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    fail_on_transition_step: int | None = optional_field(
        "One-based mutable transition step that raises."
    )

    def _registry_owner(self) -> type:
        return FailingStochasticStateBlock


class FailingStochasticStateBlock(Module):
    def __init__(
        self,
        cfg: FailingStochasticStateBlockConfig,
        overrides: FailingStochasticStateBlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("transition_step", torch.zeros(()))

    def forward(self, state: LayerState) -> LayerState:
        self.transition_step.add_(1.0)
        state.hidden = state.hidden * self.scale + torch.rand_like(state.hidden)
        if int(self.transition_step.item()) == self.cfg.fail_on_transition_step:
            raise RuntimeError("target transition failed")
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


@dataclass
class LegacyHaltingConfig(HaltingConfig):
    def _registry_owner(self) -> type:
        return LegacyHalting


class LegacyHalting(Module):
    @classmethod
    def implements_halting_interface(cls) -> bool:
        return True

    def __init__(
        self,
        cfg: LegacyHaltingConfig,
        overrides: LegacyHaltingConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)

    def update_halting_state(self, previous_state, model_hidden_state):
        state = DummyHaltingState(marker="legacy")
        state.halt_mask = torch.zeros(
            model_hidden_state.shape[:-1],
            dtype=torch.bool,
            device=model_hidden_state.device,
        )
        return state, model_hidden_state

    def finalize_weighted_accumulation(self, state, current_hidden):
        return current_hidden, current_hidden.new_zeros(())


@dataclass
class StochasticHaltingState(HaltingStateBase):
    update_count: int


@dataclass
class StochasticHaltingConfig(HaltingConfig):
    def _registry_owner(self) -> type:
        return StochasticHalting


class StochasticHalting(Module):
    supports_minimum_step_delay = True

    @classmethod
    def implements_halting_interface(cls) -> bool:
        return True

    def __init__(
        self,
        cfg: StochasticHaltingConfig,
        overrides: StochasticHaltingConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.register_buffer("update_step", torch.zeros(()))

    def update_halting_state(
        self,
        previous_state: StochasticHaltingState | None,
        model_hidden_state: torch.Tensor,
    ) -> tuple[StochasticHaltingState, torch.Tensor]:
        self.update_step.add_(1.0)
        update_count = 1 if previous_state is None else previous_state.update_count + 1
        state = StochasticHaltingState(update_count=update_count)
        state.halt_mask = torch.zeros(
            model_hidden_state.shape[:-1],
            dtype=torch.bool,
            device=model_hidden_state.device,
        )
        return state, model_hidden_state + torch.rand_like(model_hidden_state)

    def finalize_weighted_accumulation(
        self,
        state: StochasticHaltingState,
        current_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return current_hidden, current_hidden.new_zeros(())


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
        halting_config: HaltingConfig | None = None,
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

    def trainable_scale_block_config(
        self,
        *,
        dim: int,
        scale: float,
    ) -> LayerConfig:
        return LayerConfig(
            input_dim=dim,
            output_dim=dim,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=TrainableScaleFeatureLastConfig(scale=scale),
        )

    def adaptive_parameter_generator_config(self, dim: int) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=False,
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

    def adaptive_parameter_block_config(self, dim: int) -> LayerConfig:
        generator_config = self.adaptive_parameter_generator_config(dim)
        decay_config = {
            "decay_schedule": WeightDecayScheduleOptions.EXPONENTIAL,
            "decay_rate": 0.1,
            "decay_warmup_batches": 0,
            "model_config": generator_config,
        }
        adaptive_linear_config = AdaptiveLinearLayerConfig(
            input_dim=dim,
            output_dim=dim,
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                input_dim=dim,
                output_dim=dim,
                grouping_scope=AdaptiveParameterGroupingScopeOptions.DISABLED,
                weight_config=SingleModelDynamicWeightConfig(
                    input_dim=dim,
                    output_dim=dim,
                    generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
                    normalization_option=WeightNormalizationOptions.DISABLED,
                    normalization_position_option=(
                        WeightNormalizationPositionOptions.DISABLED
                    ),
                    **decay_config,
                ),
                bias_config=GeneratorDynamicBiasConfig(
                    input_dim=dim,
                    output_dim=dim,
                    **decay_config,
                ),
                model_config=generator_config,
            ),
        )
        return LayerConfig(
            input_dim=dim,
            output_dim=dim,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=adaptive_linear_config,
        )

    def stack_block_config(
        self,
        increment: float = 1.0,
        input_dim: int = 2,
        hidden_dim: int = 3,
        output_dim: int = 4,
        num_layers: int = 2,
        halting_config: HaltingConfig | None = None,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=True,
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
                    apply_output_postprocessing_flag=False,
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
            apply_output_postprocessing_flag=False,
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
            routing_initialization_mode=RoutingInitializationMode.LAYER,
            sampler_config=None,
            stack_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=dim,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_postprocessing_flag=False,
                layer_config=MixtureOfExpertsLayerConfig(
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None,
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
            apply_output_postprocessing_flag=False,
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
            apply_output_postprocessing_flag=False,
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
            apply_output_postprocessing_flag=False,
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
        ponder_cost_weight: float | None = 1.0,
        min_steps: int | None = 1,
    ) -> StickBreakingConfig:
        return StickBreakingConfig(
            input_dim=dim,
            threshold=threshold,
            ponder_cost_weight=ponder_cost_weight,
            min_steps=min_steps,
            dropout_probability=0.0,
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
        no_gradient_transition_count: int | None = None,
        initial_iterations: int | None = None,
        gradient_transition_count: int | None = None,
        iteration_increment: int | None = None,
        forward_calls_before_iteration_increment: int | None = None,
        reinject_original_hidden_flag: bool | None = None,
        smooth_iteration_growth_flag: bool | None = None,
        block_config: ConfigBase | None = None,
        gate_config: LayerStackConfig | GateConfig | None = None,
        gate_option: LayerGateOptions | None = None,
        gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID,
        residual_connection_option: type[ResidualConfig] | None = None,
        halting_config: StickBreakingConfig | None = None,
        memory_config: DynamicMemoryConfig | None = None,
        recurrent_layer_norm_position: LayerNormPositionOptions = (
            LayerNormPositionOptions.DISABLED
        ),
        residual_model_config: LinearLayerConfig | None = None,
    ) -> RecurrentLayerConfig:
        if block_config is None:
            block_config = self.layer_block_config()
        if initial_iterations is None:
            initial_iterations = max_steps
        if iteration_increment is None:
            iteration_increment = 1
        if forward_calls_before_iteration_increment is None:
            forward_calls_before_iteration_increment = 1
        return RecurrentLayerConfig(
            input_dim=dim,
            output_dim=dim,
            max_steps=max_steps,
            no_gradient_transition_count=no_gradient_transition_count,
            initial_iterations=initial_iterations,
            gradient_transition_count=gradient_transition_count,
            iteration_increment=iteration_increment,
            forward_calls_before_iteration_increment=forward_calls_before_iteration_increment,
            reinject_original_hidden_flag=reinject_original_hidden_flag,
            smooth_iteration_growth_flag=smooth_iteration_growth_flag,
            recurrent_layer_norm_position=recurrent_layer_norm_position,
            block_config=block_config,
            gate_config=self.recurrent_gate_config(
                gate_config,
                gate_option,
                gate_activation,
            ),
            residual_config=(
                None
                if residual_connection_option is None
                else residual_connection_option(
                    **(
                        {}
                        if residual_model_config is None
                        else {"model_config": residual_model_config}
                    )
                )
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

        self.assertIn("RecurrentLayerConfig", layer_package.__all__)
        self.assertIsNotNone(layer_package.RecurrentLayerConfig)
        self.assertIn("RecurrentLayer", layer_package.__all__)
        self.assertIs(layer_package.RecurrentLayer, RecurrentLayer)
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
            residual_connection_option=WeightedBlendResidualConfig,
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
            type(model.residual_config),
            WeightedBlendResidualConfig,
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

    def test_omitted_recurrent_layer_norm_is_disabled(self):
        dim = 3
        hidden = torch.arange(6.0).view(2, dim)
        model = RecurrentLayer(
            RecurrentLayerConfig(
                input_dim=dim,
                output_dim=dim,
                max_steps=1,
                initial_iterations=1,
                iteration_increment=1,
                forward_calls_before_iteration_increment=1,
                block_config=self.layer_block_config(increment=2.0),
            )
        )

        result = model(LayerState(hidden=hidden.clone()))

        self.assertIs(
            model.recurrent_layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )
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

    def test_reinjection_precedes_before_norm_and_memory_before_block(self):
        dim = 3
        hidden = torch.ones(2, dim)
        memory_config = GatedResidualDynamicMemoryConfig(
            input_dim=dim,
            output_dim=dim,
            memory_position_option=MemoryPositionOptions.BEFORE_AFFINE,
            test_time_training_learning_rate=None,
            test_time_training_num_inner_steps=None,
            model_config=self.trainable_gate_config(dim),
        )
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                reinject_original_hidden_flag=True,
                block_config=StateSpyBlockConfig(
                    input_dim=1,
                    output_dim=2,
                    increment=1.0,
                ),
                memory_config=memory_config,
                recurrent_layer_norm_position=LayerNormPositionOptions.BEFORE,
            )
        )
        layer_norm = RecordingTransform(offset=10.0)
        memory = AddConstantMemory(
            20.0,
            MemoryPositionOptions.BEFORE_AFFINE,
        )
        model.recurrent_layer_norm_module = layer_norm
        model.memory_model = memory

        result = model(LayerState(hidden=hidden))

        reinjected_hidden = hidden + hidden
        normalized_hidden = reinjected_hidden + 10.0
        torch.testing.assert_close(layer_norm.inputs[0], reinjected_hidden)
        torch.testing.assert_close(memory.inputs[0], normalized_hidden)
        torch.testing.assert_close(
            model.block_model.received_hidden_inputs[0],
            normalized_hidden + 20.0,
        )
        torch.testing.assert_close(result.hidden, normalized_hidden + 21.0)

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

    def test_recurrent_layer_norm_after_normalizes_after_controllers(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=StateSpyBlockConfig(
                    input_dim=dim,
                    output_dim=dim,
                    increment=3.0,
                ),
                recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
            )
        )
        transform = RecordingTransform(offset=10.0)
        model.recurrent_layer_norm_module = transform
        previous_hidden = torch.zeros(2, dim)
        candidate_hidden = torch.full_like(previous_hidden, 3.0)

        result = model(LayerState(hidden=previous_hidden.clone()))

        torch.testing.assert_close(transform.inputs[0], candidate_hidden)
        torch.testing.assert_close(result.hidden, candidate_hidden + 10.0)

    def test_recurrent_controllers_apply_gate_residual_and_norm_in_order(self):
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
                *,
                residual_state=None,
                row_layout=None,
            ) -> torch.Tensor:
                self.received_current = current.detach().clone()
                self.received_previous = previous.detach().clone()
                return current + 3.0 * previous

        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=1,
                block_config=StateSpyBlockConfig(
                    input_dim=dim,
                    output_dim=dim,
                    increment=1.0,
                ),
                recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
            )
        )
        gate = RecordingGate()
        residual = RecordingResidual()
        layer_norm = RecordingTransform(scale=5.0)
        model.recurrent_gate = gate
        model.residual_connection = residual
        model.recurrent_layer_norm_module = layer_norm
        previous_hidden = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        candidate_hidden = previous_hidden + 1.0
        previous_halting_state = DummyHaltingState(marker="previous")
        existing_loss = torch.tensor(2.5)
        input_state = LayerState(
            hidden=previous_hidden.clone(),
            loss=existing_loss,
            halting_state=previous_halting_state,
        )

        result = model(input_state)

        gated_hidden = candidate_hidden + 2.0
        residual_hidden = gated_hidden + 3.0 * previous_hidden
        normalized_hidden = residual_hidden * 5.0
        expected_hidden = normalized_hidden
        torch.testing.assert_close(gate.received_hidden, candidate_hidden)
        torch.testing.assert_close(residual.received_current, gated_hidden)
        torch.testing.assert_close(residual.received_previous, previous_hidden)
        torch.testing.assert_close(layer_norm.inputs[0], residual_hidden)
        torch.testing.assert_close(result.hidden, expected_hidden)
        self.assertIs(result.loss, existing_loss)
        self.assertIs(result.halting_state, previous_halting_state)

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
                residual_connection_option=WeightedBlendResidualConfig,
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
            initial_iterations=5,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            reinject_original_hidden_flag=True,
            block_config=override_block,
            gate_config=self.recurrent_gate_config(override_gate, None),
            residual_config=AdditiveResidualConfig(),
        )

        model = RecurrentLayer(cfg, overrides)

        self.assertEqual(model.input_dim, 3)
        self.assertEqual(model.output_dim, 3)
        self.assertEqual(model.max_steps, 5)
        self.assertTrue(model.reinject_original_hidden_flag)
        self.assertEqual(model.block_config, override_block)
        self.assertEqual(model.gate_config.model_config, override_gate)
        self.assertEqual(
            type(model.residual_config),
            AdditiveResidualConfig,
        )
        self.assertEqual(model.halting_config, cfg.halting_config)
        self.assertEqual(model.block_model.model.increment, 3.0)
        self.assertEqual(model.block_model.input_dim, 3)
        self.assertEqual(model.block_model.output_dim, 3)
        self.assertIsInstance(model.recurrent_gate.model, LayerStack)

    def test_reinject_original_hidden_flag_rejects_non_boolean_values(self):
        for invalid_value in (1, "true", object()):
            with self.subTest(invalid_value=invalid_value):
                cfg = self.recurrent_config()
                cfg.reinject_original_hidden_flag = invalid_value

                with self.assertRaisesRegex(
                    TypeError,
                    "reinject_original_hidden_flag must be bool or None",
                ):
                    cfg.build()

    def test_smooth_iteration_growth_requires_an_explicit_gradient_window(self):
        config = self.recurrent_config(
            max_steps=4,
            initial_iterations=2,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "smooth iteration growth requires either gradient_transition_count "
            "or explicit no_gradient_transition_count=0",
        ):
            config.build()

    def test_smooth_iteration_growth_flag_rejects_non_boolean_values(self):
        for invalid_value in (1, "true", object()):
            with self.subTest(invalid_value=invalid_value):
                config = self.recurrent_config()
                config.smooth_iteration_growth_flag = invalid_value

                with self.assertRaisesRegex(
                    TypeError,
                    "smooth_iteration_growth_flag must be bool or None",
                ):
                    config.build()

    def test_smooth_iteration_growth_rejects_unsupported_schedule_shapes(self):
        invalid_cases = (
            (
                "multi-depth increment",
                {"iteration_increment": 2},
                "iteration_increment to equal 1",
            ),
            (
                "unit cadence",
                {"forward_calls_before_iteration_increment": 1},
                "even integer greater than or equal to 2",
            ),
            (
                "odd cadence",
                {"forward_calls_before_iteration_increment": 3},
                "even integer greater than or equal to 2",
            ),
            (
                "fixed no-gradient prefix",
                {"no_gradient_transition_count": 0},
                "mutually exclusive",
            ),
        )

        for name, overrides, message in invalid_cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                self.recurrent_config(
                    max_steps=4,
                    initial_iterations=2,
                    gradient_transition_count=2,
                    smooth_iteration_growth_flag=True,
                    **overrides,
                ).build()

    def test_smooth_iteration_growth_rejects_stateful_dynamic_memory(self):
        dim = 2
        memory_config = GatedResidualDynamicMemoryConfig(
            input_dim=dim,
            output_dim=dim,
            memory_position_option=MemoryPositionOptions.BEFORE_AFFINE,
            test_time_training_learning_rate=None,
            test_time_training_num_inner_steps=None,
            model_config=self.trainable_gate_config(dim),
        )

        with self.assertRaisesRegex(
            ValueError,
            "smooth iteration growth does not support memory_config",
        ):
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                initial_iterations=2,
                gradient_transition_count=2,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=True,
                memory_config=memory_config,
            ).build()

    def test_non_default_minimum_requires_halting_delay_support(self):
        config = self.recurrent_config(
            max_steps=3,
            halting_config=LegacyHaltingConfig(min_steps=2),
        )

        with self.assertRaisesRegex(
            ValueError,
            "min_steps.*minimum-step delay support",
        ):
            config.build()

    def test_default_minimum_preserves_legacy_halting_strategies(self):
        model = self.recurrent_config(
            dim=2,
            max_steps=2,
            block_config=self.layer_block_config(increment=1.0),
            halting_config=LegacyHaltingConfig(),
        ).build()

        result = model(LayerState(hidden=torch.zeros(1, 2)))

        torch.testing.assert_close(result.hidden, torch.full((1, 2), 2.0))

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
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
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
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
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
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
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
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
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
                    initial_iterations=0,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
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
                    initial_iterations="3",
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    residual_config=None,
                ),
                TypeError,
            ),
            (
                "boolean_no_gradient_transition_count",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=2,
                    initial_iterations=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
                    no_gradient_transition_count=True,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    residual_config=None,
                ),
                TypeError,
            ),
            (
                "negative_no_gradient_transition_count",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=2,
                    initial_iterations=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
                    no_gradient_transition_count=-1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    residual_config=None,
                ),
                ValueError,
            ),
            (
                "no_gradient_transition_count_leaves_no_gradient_steps",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=2,
                    initial_iterations=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
                    no_gradient_transition_count=2,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    residual_config=None,
                ),
                ValueError,
            ),
            (
                "mismatched_recurrent_dimensions",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim + 1,
                    max_steps=1,
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
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
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
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
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
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
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    gate_config=GateConfig(option=object()),
                    residual_config=None,
                ),
                TypeError,
            ),
            (
                "abstract_residual_config",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    residual_config=ResidualConfig(),
                ),
                ValueError,
            ),
            (
                "nested_gate_config",
                RecurrentLayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    max_steps=1,
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
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
                    initial_iterations=1,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=1,
                    recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                    block_config=valid_block,
                    halting_config=object(),
                    residual_config=None,
                ),
                TypeError,
            ),
        ]

        for name, cfg, expected_exception in invalid_cases:
            with self.subTest(name=name):
                cfg.initial_iterations = 1
                cfg.iteration_increment = 1
                cfg.forward_calls_before_iteration_increment = 1
                with self.assertRaises(expected_exception):
                    RecurrentLayer(cfg)

    def test_recurrent_layer_composes_soft_through_the_halting_interface(self):
        dim = 4
        cfg = self.recurrent_config(
            dim=dim,
            halting_config=SoftHaltingConfig(
                input_dim=dim,
                threshold=0.99,
                ponder_cost_weight=1.0,
                min_steps=1,
                dropout_probability=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=self.halting_gate_config(threshold=1.0),
            ),
        )
        model = RecurrentLayer(cfg).eval()
        state = LayerState(hidden=torch.zeros(2, dim))

        result = model(state)

        self.assertIsInstance(model.halting_model, SoftHalting)
        self.assertEqual(result.hidden.shape, state.hidden.shape)
        self.assertIsNotNone(result.loss)
        self.assertTrue(torch.isfinite(result.loss).item())

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

    def test_reinjection_preserves_attention_metadata_layout_and_loss(self):
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
                reinject_original_hidden_flag=True,
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
        hidden = torch.arange(6, dtype=torch.float64).reshape(2, dim)
        key_padding_mask = torch.tensor([[False, True], [False, False]])
        attention_mask = torch.zeros(2, 2)
        row_layout = RowLayout.rows(
            2,
            context_sharing_restricted=False,
        )
        existing_loss = torch.tensor(4.0)
        state = AttentionLayerState(
            hidden=hidden,
            loss=existing_loss,
            halting_state=object(),
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
            row_layout=row_layout,
        )

        result = model(state)

        block_state = model.block_model.received_states[0]
        gate_state = gate.received_state
        self.assertIs(result, state)
        self.assertIsInstance(block_state, AttentionLayerState)
        self.assertIs(type(gate_state), LayerState)
        torch.testing.assert_close(gate.received_hidden, hidden + hidden + 1.0)
        self.assertEqual(result.hidden.shape, hidden.shape)
        self.assertEqual(result.hidden.dtype, hidden.dtype)
        self.assertEqual(result.hidden.device, hidden.device)
        self.assertIs(block_state.key_padding_mask, key_padding_mask)
        self.assertIs(block_state.attention_mask, attention_mask)
        self.assertIs(block_state.row_layout, row_layout)
        self.assertIs(result.row_layout, row_layout)
        self.assertFalse(hasattr(gate_state, "key_padding_mask"))
        self.assertFalse(hasattr(gate_state, "attention_mask"))
        self.assertIs(block_state.loss, existing_loss)
        self.assertIsNone(block_state.halting_state)
        self.assertIsNone(gate_state.loss)
        self.assertIsNone(gate_state.halting_state)

    def test_smooth_handoff_preserves_caller_metadata_layout_and_dtype(self):
        dim = 3
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                initial_iterations=2,
                gradient_transition_count=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=True,
                block_config=StateSpyBlockConfig(
                    input_dim=dim,
                    output_dim=dim,
                    increment=1.0,
                ),
            )
        ).to(dtype=torch.float64)
        model.recurrent_iteration_schedule.load_state_dict(
            {"forward_call_progress": torch.tensor(4, dtype=torch.long)},
            strict=True,
        )
        hidden = torch.arange(6, dtype=torch.float64).reshape(2, dim)
        key_padding_mask = torch.tensor([[False, True], [False, False]])
        attention_mask = torch.zeros(2, 2)
        row_layout = RowLayout.rows(2, context_sharing_restricted=False)
        existing_loss = torch.tensor(4.0, dtype=torch.float64)
        owner_halting_state = object()
        state = AttentionLayerState(
            hidden=hidden,
            loss=existing_loss,
            halting_state=owner_halting_state,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
            row_layout=row_layout,
        )

        result = model(state)

        self.assertIs(result, state)
        self.assertEqual(result.hidden.shape, hidden.shape)
        self.assertEqual(result.hidden.dtype, torch.float64)
        self.assertEqual(result.hidden.device, hidden.device)
        self.assertIs(result.loss, existing_loss)
        self.assertIs(result.halting_state, owner_halting_state)
        self.assertIs(result.key_padding_mask, key_padding_mask)
        self.assertIs(result.attention_mask, attention_mask)
        self.assertIs(result.row_layout, row_layout)
        self.assertTrue(
            all(
                transition_state.row_layout is row_layout
                for transition_state in model.block_model.received_states
            )
        )

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

    def test_standard_checkpoint_keeps_direct_block_model_paths(self):
        config = self.recurrent_config(
            dim=2,
            max_steps=2,
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
        recurrent = RecurrentLayer(config)
        inputs = torch.ones(1, 2)
        expected = recurrent(LayerState(hidden=inputs.clone())).hidden

        checkpoint = recurrent.state_dict()

        self.assertEqual(
            set(checkpoint),
            {
                "recurrent_iteration_schedule.forward_call_progress",
                "block_model.model.scale",
            },
        )
        restored = RecurrentLayer(config)
        restored.load_state_dict(checkpoint, strict=True)
        actual = restored(LayerState(hidden=inputs.clone())).hidden
        torch.testing.assert_close(actual, expected)

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

    def test_reinjects_the_original_hidden_before_every_recurrent_block(self):
        recurrent = RecurrentLayerConfig(
            input_dim=1,
            output_dim=1,
            max_steps=3,
            initial_iterations=3,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            no_gradient_transition_count=None,
            reinject_original_hidden_flag=True,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=StateSpyBlockConfig(
                input_dim=1,
                output_dim=1,
                increment=1.0,
            ),
            gate_config=None,
            residual_config=None,
            halting_config=None,
            memory_config=None,
        ).build()

        result = recurrent(LayerState(hidden=torch.ones(1, 1)))

        recorded_inputs = recurrent.block_model.received_hidden_inputs
        self.assertEqual(len(recorded_inputs), 3)
        for actual, expected in zip(
            recorded_inputs,
            (2.0, 4.0, 6.0),
            strict=True,
        ):
            torch.testing.assert_close(actual, torch.full_like(actual, expected))
        torch.testing.assert_close(result.hidden, torch.tensor([[7.0]]))

    def test_disabled_and_omitted_reinjection_preserve_identical_outputs(self):
        hidden = torch.tensor([[1.0, 2.0]])
        omitted = RecurrentLayer(self.recurrent_config(dim=2, max_steps=3))
        disabled = RecurrentLayer(
            self.recurrent_config(
                dim=2,
                max_steps=3,
                reinject_original_hidden_flag=False,
            )
        )

        omitted_result = omitted(LayerState(hidden=hidden.clone()))
        disabled_result = disabled(LayerState(hidden=hidden.clone()))

        self.assertFalse(omitted.reinject_original_hidden_flag)
        self.assertFalse(disabled.reinject_original_hidden_flag)
        self.assertTrue(torch.equal(omitted_result.hidden, disabled_result.hidden))

    def test_reinjection_does_not_change_the_residual_reference_hidden(self):
        recurrent = RecurrentLayer(
            self.recurrent_config(
                dim=1,
                max_steps=2,
                reinject_original_hidden_flag=True,
                block_config=StateSpyBlockConfig(
                    input_dim=1,
                    output_dim=1,
                    increment=1.0,
                ),
                residual_connection_option=AdditiveResidualConfig,
            )
        )

        result = recurrent(LayerState(hidden=torch.ones(1, 1)))

        recorded_inputs = recurrent.block_model.received_hidden_inputs
        torch.testing.assert_close(recorded_inputs[0], torch.tensor([[2.0]]))
        torch.testing.assert_close(recorded_inputs[1], torch.tensor([[5.0]]))
        torch.testing.assert_close(result.hidden, torch.tensor([[10.0]]))

    def test_no_gradient_transition_count_detaches_only_the_configured_prefix(self):
        dim = 3
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=4,
                no_gradient_transition_count=2,
                reinject_original_hidden_flag=True,
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
        )

        result = model(LayerState(hidden=torch.ones(2, dim)))
        result.hidden.sum().backward()

        self.assertEqual(
            model.block_model.model.grad_modes,
            [False, False, True, True],
        )
        self.assertIsNotNone(model.block_model.model.scale.grad)

    def test_scheduled_iterations_keep_a_fixed_gradient_suffix_as_depth_grows(self):
        dim = 3
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=10,
                initial_iterations=2,
                gradient_transition_count=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=1,
                block_config=self.trainable_scale_block_config(
                    dim=dim,
                    scale=0.5,
                ),
            )
        )

        for expected_active_iterations in range(2, 11):
            with self.subTest(active_iterations=expected_active_iterations):
                model.block_model.model.grad_modes.clear()

                model(LayerState(hidden=torch.ones(2, dim)))

                self.assertEqual(
                    model.block_model.model.grad_modes,
                    [False] * (expected_active_iterations - 2) + [True, True],
                )

        schedule = model.recurrent_iteration_schedule
        self.assertEqual(schedule.active_iterations, schedule.maximum_iterations)
        self.assertEqual(schedule.no_gradient_transition_count, 8)

    def test_iteration_schedule_runtime_values_belong_to_schedule_module(
        self,
    ) -> None:
        model = self.recurrent_config(max_steps=5, initial_iterations=2).build()
        schedule = model.recurrent_iteration_schedule

        expected_values = {
            "iteration_unit": "transition",
            "maximum_iterations": 5,
            "active_iterations": 2,
            "maximum_transition_count": 5,
            "active_transition_count": 2,
            "complete": False,
        }
        for attribute_name, expected_value in expected_values.items():
            with self.subTest(attribute_name=attribute_name):
                self.assertEqual(getattr(schedule, attribute_name), expected_value)
                self.assertFalse(hasattr(model, attribute_name))
        self.assertEqual(schedule.snapshot().forward_call_progress, 0)
        self.assertFalse(hasattr(model, "schedule_forward_call_progress"))

    def test_initial_iterations_is_required_by_validator(self) -> None:
        config = self.recurrent_config()
        config.initial_iterations = None

        with self.assertRaisesRegex(
            ValueError,
            "initial_iterations is required for RecurrentLayerConfig",
        ):
            config.build()

    def test_iteration_schedule_advances_on_forward_call_boundaries_and_caps_at_capacity(
        self,
    ) -> None:
        model = RecurrentLayerConfig(
            input_dim=4,
            output_dim=4,
            max_steps=7,
            initial_iterations=2,
            iteration_increment=2,
            forward_calls_before_iteration_increment=3,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=self.layer_block_config(),
        ).build()

        executed_iterations = []
        for _ in range(10):
            call_count_before = model.block_model.model.call_count
            model(LayerState(hidden=torch.zeros(1, 4)))
            executed_iterations.append(
                model.block_model.model.call_count - call_count_before
            )

        self.assertEqual(executed_iterations, [2, 2, 2, 4, 4, 4, 6, 6, 6, 7])
        schedule = model.recurrent_iteration_schedule
        self.assertEqual(schedule.active_iterations, 7)
        self.assertEqual(schedule.snapshot().forward_call_progress, 9)
        self.assertTrue(schedule.complete)

    def test_failed_forward_does_not_advance_iteration_schedule(self) -> None:
        model = RecurrentLayer(
            self.recurrent_config(
                max_steps=10,
                initial_iterations=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=1,
            )
        )

        with self.assertRaisesRegex(ValueError, "last dimension must be 4"):
            model(LayerState(hidden=torch.ones(1, 3)))

        schedule = model.recurrent_iteration_schedule
        self.assertEqual(schedule.snapshot().forward_call_progress, 0)
        self.assertEqual(schedule.active_iterations, 2)

    def test_recurrent_execution_commits_and_records_one_successful_forward(
        self,
    ) -> None:
        model = self.recurrent_config(
            max_steps=2,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
        ).build()
        schedule = model.recurrent_iteration_schedule
        initial_loss = torch.tensor(3.0)
        state = LayerState(hidden=torch.zeros(1, 4), loss=initial_loss)

        result = RecurrentExecution().execute(model, state, schedule)

        self.assertIs(result, state)
        torch.testing.assert_close(result.hidden, torch.ones(1, 4))
        self.assertIs(result.loss, initial_loss)
        self.assertEqual(schedule.snapshot().forward_call_progress, 1)
        self.assertEqual(schedule.active_iterations, 2)

    def test_recurrent_execution_rejects_non_module_adapters_consistently(self) -> None:
        stable_model = self.recurrent_config(
            max_steps=2,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
        ).build()
        smooth_model = self.recurrent_config(
            max_steps=2,
            initial_iterations=1,
            gradient_transition_count=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=True,
        ).build()
        smooth_model.recurrent_iteration_schedule.forward_call_progress.fill_(4)

        for schedule in (
            stable_model.recurrent_iteration_schedule,
            smooth_model.recurrent_iteration_schedule,
        ):
            with (
                self.subTest(plan_type=type(schedule.execution_plan()).__name__),
                self.assertRaisesRegex(
                    TypeError,
                    "Recurrent Execution Adapter must be an nn.Module",
                ),
            ):
                RecurrentExecution().execute(
                    object(),
                    LayerState(hidden=torch.zeros(1, 4)),
                    schedule,
                )

    def test_recurrent_execution_dispatches_through_substituted_validator(
        self,
    ) -> None:
        class RejectingValidator(RecurrentExecutionValidator):
            @staticmethod
            def validate_adapter_is_module(adapter: object) -> None:
                raise RuntimeError("substituted execution validator was called")

        class RejectingExecution(RecurrentExecution):
            VALIDATOR = RejectingValidator

        model = self.recurrent_config(max_steps=1, initial_iterations=1).build()

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted execution validator was called",
        ):
            RejectingExecution().execute(
                model,
                LayerState(hidden=torch.zeros(1, 4)),
                model.recurrent_iteration_schedule,
            )

    def test_schedule_recording_failure_rolls_back_complete_smooth_handoff(
        self,
    ) -> None:
        dim = 2
        model = self.recurrent_config(
            dim=dim,
            max_steps=3,
            initial_iterations=2,
            gradient_transition_count=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=True,
            block_config=FailingStochasticStateBlockConfig(
                input_dim=dim,
                output_dim=dim,
                fail_on_transition_step=999,
            ),
        ).build()
        schedule = model.recurrent_iteration_schedule
        schedule.load_state_dict(
            {"forward_call_progress": torch.tensor(4, dtype=torch.long)},
            strict=True,
        )
        original_record_success = schedule.record_successful_forward
        initial_hidden = torch.ones(1, dim)
        initial_loss = torch.tensor(3.0)
        state = LayerState(hidden=initial_hidden, loss=initial_loss)
        transition_step_buffer = model.block_model.transition_step
        initial_transition_step = transition_step_buffer.clone()
        torch.manual_seed(41)
        expected_next_random_value = torch.rand(())
        torch.manual_seed(41)

        def record_then_fail() -> None:
            original_record_success()
            raise RuntimeError("schedule recording failed")

        with (
            patch.object(
                schedule,
                "record_successful_forward",
                side_effect=record_then_fail,
            ),
            self.assertRaisesRegex(RuntimeError, "schedule recording failed"),
        ):
            model(state)

        self.assertIs(state.hidden, initial_hidden)
        self.assertIs(state.loss, initial_loss)
        self.assertEqual(schedule.snapshot().forward_call_progress, 4)
        self.assertIs(model.block_model.transition_step, transition_step_buffer)
        torch.testing.assert_close(
            model.block_model.transition_step,
            initial_transition_step,
        )
        torch.testing.assert_close(torch.rand(()), expected_next_random_value)

    def test_transition_failure_does_not_commit_or_advance_iteration_schedule(
        self,
    ) -> None:
        class FailingTransition(Module):
            def forward(self, _state: LayerState) -> LayerState:
                raise RuntimeError("transition failed")

        model = self.recurrent_config(
            max_steps=2,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
        ).build()
        model.block_model = FailingTransition()
        schedule = model.recurrent_iteration_schedule
        initial_hidden = torch.ones(1, 4)
        initial_loss = torch.tensor(3.0)
        state = LayerState(hidden=initial_hidden, loss=initial_loss)

        with self.assertRaisesRegex(RuntimeError, "transition failed"):
            model(state)

        self.assertIs(state.hidden, initial_hidden)
        self.assertIs(state.loss, initial_loss)
        self.assertEqual(schedule.snapshot().forward_call_progress, 0)
        self.assertEqual(schedule.active_iterations, 1)

    def test_failed_smooth_handoff_rolls_back_runtime_state_and_rng(self) -> None:
        dim = 2
        model = self.recurrent_config(
            dim=dim,
            max_steps=4,
            initial_iterations=3,
            gradient_transition_count=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=True,
            block_config=FailingStochasticStateBlockConfig(
                input_dim=dim,
                output_dim=dim,
                fail_on_transition_step=4,
            ),
        ).build()
        schedule = model.recurrent_iteration_schedule
        schedule.load_state_dict(
            {"forward_call_progress": torch.tensor(4, dtype=torch.long)},
            strict=True,
        )
        starting_hidden = torch.ones(1, dim)
        starting_loss = torch.tensor(3.0)
        state = LayerState(hidden=starting_hidden, loss=starting_loss)
        starting_schedule_progress_buffer = schedule.forward_call_progress
        starting_transition_step_buffer = model.block_model.transition_step
        starting_transition_step = model.block_model.transition_step.clone()
        torch.manual_seed(47)
        expected_next_random_value = torch.rand(())
        torch.manual_seed(47)

        with self.assertRaisesRegex(RuntimeError, "target transition failed"):
            model(state)
        actual_next_random_value = torch.rand(())

        self.assertIs(state.hidden, starting_hidden)
        self.assertIs(state.loss, starting_loss)
        self.assertIs(
            schedule.forward_call_progress,
            starting_schedule_progress_buffer,
        )
        self.assertIs(
            model.block_model.transition_step,
            starting_transition_step_buffer,
        )
        self.assertEqual(schedule.snapshot().forward_call_progress, 4)
        torch.testing.assert_close(
            model.block_model.transition_step,
            starting_transition_step,
        )
        torch.testing.assert_close(
            actual_next_random_value,
            expected_next_random_value,
        )

    def test_failed_smooth_handoff_restores_captured_cuda_rng_state(self) -> None:
        dim = 2
        model = self.recurrent_config(
            dim=dim,
            max_steps=4,
            initial_iterations=3,
            gradient_transition_count=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=True,
            block_config=FailingStochasticStateBlockConfig(
                input_dim=dim,
                output_dim=dim,
                fail_on_transition_step=1,
            ),
        ).build()
        model.recurrent_iteration_schedule.load_state_dict(
            {"forward_call_progress": torch.tensor(4, dtype=torch.long)},
            strict=True,
        )
        captured_cuda_rng_states = [torch.tensor([7, 11], dtype=torch.uint8)]

        with (
            patch.object(torch.cuda, "is_initialized", return_value=True),
            patch.object(
                torch.cuda,
                "get_rng_state_all",
                return_value=captured_cuda_rng_states,
            ),
            patch.object(torch.cuda, "set_rng_state_all") as restore_cuda_rng,
            self.assertRaisesRegex(RuntimeError, "target transition failed"),
        ):
            model(LayerState(hidden=torch.ones(1, dim)))

        restore_cuda_rng.assert_called_once_with(captured_cuda_rng_states)

    def test_iteration_schedule_counts_training_and_inference_forwards(self) -> None:
        model = RecurrentLayer(
            self.recurrent_config(
                max_steps=3,
                initial_iterations=1,
                iteration_increment=1,
                forward_calls_before_iteration_increment=1,
            )
        ).eval()

        with torch.inference_mode():
            model(LayerState(hidden=torch.ones(1, 4)))
        schedule = model.recurrent_iteration_schedule
        self.assertEqual(schedule.snapshot().forward_call_progress, 1)
        self.assertEqual(schedule.active_iterations, 2)

        model.train()
        model(LayerState(hidden=torch.ones(1, 4)))
        self.assertEqual(schedule.snapshot().forward_call_progress, 2)
        self.assertEqual(schedule.active_iterations, 3)

    def test_iteration_schedule_progress_round_trips_through_a_checkpoint(self):
        config = RecurrentLayerConfig(
            input_dim=4,
            output_dim=4,
            max_steps=7,
            initial_iterations=2,
            iteration_increment=2,
            forward_calls_before_iteration_increment=3,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=self.layer_block_config(),
        )
        model = config.build()
        for _ in range(3):
            model(LayerState(hidden=torch.zeros(1, 4)))

        checkpoint = model.state_dict()
        restored = config.build()
        restored.load_state_dict(checkpoint, strict=True)

        restored_schedule = restored.recurrent_iteration_schedule
        self.assertEqual(restored_schedule.snapshot().forward_call_progress, 3)
        self.assertEqual(restored_schedule.active_iterations, 4)
        call_count_before = restored.block_model.model.call_count
        restored(LayerState(hidden=torch.zeros(1, 4)))
        self.assertEqual(restored.block_model.model.call_count - call_count_before, 4)

    def test_smooth_checkpoint_restores_the_exact_next_forward_and_gradients(self):
        dim = 2
        config = self.recurrent_config(
            dim=dim,
            max_steps=3,
            initial_iterations=2,
            gradient_transition_count=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=True,
            block_config=self.trainable_scale_block_config(
                dim=dim,
                scale=0.5,
            ),
        )

        for progress in (3, 4, 5, 6):
            with self.subTest(progress=progress):
                source = config.build()
                source.recurrent_iteration_schedule.load_state_dict(
                    {
                        "forward_call_progress": torch.tensor(
                            progress,
                            dtype=torch.long,
                        )
                    },
                    strict=True,
                )
                checkpoint = source.state_dict()
                restored = config.build()
                restored.load_state_dict(checkpoint, strict=True)
                source_input = torch.ones(1, dim, requires_grad=True)
                restored_input = torch.ones(1, dim, requires_grad=True)

                source_output = source(LayerState(hidden=source_input)).hidden
                restored_output = restored(LayerState(hidden=restored_input)).hidden
                source_output.sum().backward()
                restored_output.sum().backward()

                torch.testing.assert_close(restored_output, source_output)
                source_parameters = dict(source.named_parameters())
                restored_parameters = dict(restored.named_parameters())
                self.assertEqual(source_parameters.keys(), restored_parameters.keys())
                for parameter_name in source_parameters:
                    with self.subTest(
                        progress=progress,
                        parameter_name=parameter_name,
                    ):
                        torch.testing.assert_close(
                            restored_parameters[parameter_name].grad,
                            source_parameters[parameter_name].grad,
                        )
                if source_input.grad is None:
                    self.assertIsNone(restored_input.grad)
                else:
                    torch.testing.assert_close(
                        restored_input.grad,
                        source_input.grad,
                    )
                self.assertEqual(
                    restored.recurrent_iteration_schedule.snapshot(),
                    source.recurrent_iteration_schedule.snapshot(),
                )

    def test_legacy_checkpoint_missing_only_schedule_progress_loads_strictly(self):
        config = RecurrentLayerConfig(
            input_dim=4,
            output_dim=4,
            max_steps=7,
            initial_iterations=2,
            iteration_increment=2,
            forward_calls_before_iteration_increment=3,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=self.layer_block_config(),
        )
        legacy_checkpoint = config.build().state_dict()
        legacy_checkpoint.pop("recurrent_iteration_schedule.forward_call_progress")

        restored = config.build()
        restored.load_state_dict(legacy_checkpoint, strict=True)

        restored_schedule = restored.recurrent_iteration_schedule
        self.assertEqual(restored_schedule.snapshot().forward_call_progress, 0)
        self.assertEqual(restored_schedule.active_iterations, 2)

    def test_complete_schedule_target_ignores_loaded_schedule_progress(self) -> None:
        scheduled_config = self.recurrent_config(
            max_steps=3,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
        )
        scheduled = scheduled_config.build()
        scheduled(LayerState(hidden=torch.ones(1, 4)))

        complete = self.recurrent_config(max_steps=3).build()
        complete.load_state_dict(scheduled.state_dict(), strict=True)

        complete_schedule = complete.recurrent_iteration_schedule
        self.assertEqual(complete_schedule.snapshot().forward_call_progress, 0)
        self.assertEqual(complete_schedule.active_iterations, 3)
        self.assertTrue(complete_schedule.complete)

    def test_checkpoint_progress_is_validated_and_clamped_to_saturation(self):
        config = RecurrentLayerConfig(
            input_dim=4,
            output_dim=4,
            max_steps=7,
            initial_iterations=2,
            iteration_increment=2,
            forward_calls_before_iteration_increment=3,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=self.layer_block_config(),
        )
        checkpoint = config.build().state_dict()
        progress_key = "recurrent_iteration_schedule.forward_call_progress"
        checkpoint[progress_key] = torch.tensor(999, dtype=torch.long)

        restored = config.build()
        restored.load_state_dict(checkpoint, strict=True)

        restored_schedule = restored.recurrent_iteration_schedule
        self.assertEqual(restored_schedule.snapshot().forward_call_progress, 9)
        self.assertEqual(restored_schedule.active_iterations, 7)
        checkpoint[progress_key] = torch.tensor(-1, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            config.build().load_state_dict(checkpoint, strict=True)

        malformed_progress = (
            (torch.tensor([1], dtype=torch.long), ValueError, "scalar Tensor"),
            (torch.tensor(1.0), TypeError, "torch.long dtype"),
            ("1", TypeError, "must be a Tensor"),
        )
        for value, error_type, message in malformed_progress:
            with self.subTest(value=value):
                checkpoint[progress_key] = value
                with self.assertRaisesRegex(error_type, message):
                    config.build().load_state_dict(checkpoint, strict=True)

    def test_gradient_transition_count_is_static_for_a_complete_iteration_schedule(
        self,
    ):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=5,
                gradient_transition_count=2,
                block_config=self.trainable_scale_block_config(
                    dim=dim,
                    scale=0.5,
                ),
            )
        )

        model(LayerState(hidden=torch.ones(1, dim)))

        self.assertEqual(model.recurrent_iteration_schedule.active_iterations, 5)
        self.assertEqual(
            model.block_model.model.grad_modes,
            [False, False, False, True, True],
        )

    def test_smooth_growth_interpolates_exact_source_and_target_depth_outputs(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                initial_iterations=2,
                gradient_transition_count=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=True,
                block_config=self.layer_block_config(increment=1.0),
            )
        ).eval()
        hidden = torch.zeros(1, dim)

        for _ in range(4):
            stable_result = model(LayerState(hidden=hidden.clone()))
            torch.testing.assert_close(stable_result.hidden, hidden + 2.0)

        self.assertEqual(
            model.recurrent_iteration_schedule.snapshot().transition_weight,
            0.5,
        )
        model.block_model.model.call_count = 0

        transition_result = model(LayerState(hidden=hidden.clone()))

        torch.testing.assert_close(transition_result.hidden, hidden + 2.5)
        self.assertEqual(model.block_model.model.call_count, 4)

        model.block_model.model.call_count = 0
        endpoint_result = model(LayerState(hidden=hidden.clone()))
        self.assertEqual(
            model.recurrent_iteration_schedule.snapshot().settled_iterations,
            3,
        )
        model.block_model.model.call_count = 0
        stable_target_result = model(LayerState(hidden=hidden.clone()))

        torch.testing.assert_close(endpoint_result.hidden, hidden + 3.0)
        torch.testing.assert_close(stable_target_result.hidden, endpoint_result.hidden)
        self.assertEqual(model.block_model.model.call_count, 3)

    def test_full_gradient_smooth_growth_matches_one_nested_chain_oracle(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                initial_iterations=2,
                no_gradient_transition_count=0,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=True,
                block_config=self.trainable_scale_block_config(
                    dim=dim,
                    scale=0.5,
                ),
            )
        ).eval()
        schedule = model.recurrent_iteration_schedule
        schedule.forward_call_progress.fill_(4)
        model_input = torch.ones(1, dim, requires_grad=True)
        transition_model = model.block_model.model

        result = model(LayerState(hidden=model_input))
        input_gradient, parameter_gradient = torch.autograd.grad(
            result.hidden.sum(),
            (model_input, transition_model.scale),
        )

        oracle_input = torch.ones(1, dim, requires_grad=True)
        oracle_scale = transition_model.scale.detach().clone().requires_grad_()
        depth_one = oracle_input * oracle_scale
        depth_two = depth_one * oracle_scale
        depth_three = depth_two * oracle_scale
        oracle_output = 0.5 * depth_two + 0.5 * depth_three
        oracle_input_gradient, oracle_parameter_gradient = torch.autograd.grad(
            oracle_output.sum(),
            (oracle_input, oracle_scale),
        )

        torch.testing.assert_close(result.hidden, oracle_output)
        torch.testing.assert_close(input_gradient, oracle_input_gradient)
        torch.testing.assert_close(parameter_gradient, oracle_parameter_gradient)
        self.assertEqual(len(transition_model.outputs), 3)
        self.assertEqual(transition_model.grad_modes, [True, True, True])

    def test_full_gradient_smooth_growth_matches_gradient_oracles_at_endpoints(
        self,
    ) -> None:
        dim = 2
        for transition_weight in (0.0, 1.0):
            with self.subTest(transition_weight=transition_weight):
                model = RecurrentLayer(
                    self.recurrent_config(
                        dim=dim,
                        max_steps=3,
                        initial_iterations=2,
                        no_gradient_transition_count=0,
                        iteration_increment=1,
                        forward_calls_before_iteration_increment=4,
                        smooth_iteration_growth_flag=True,
                        block_config=self.trainable_scale_block_config(
                            dim=dim,
                            scale=0.5,
                        ),
                    )
                ).eval()
                schedule = model.recurrent_iteration_schedule
                execution_plan = RecurrentNestedSmoothHandoffExecutionPlan(
                    source_branch=RecurrentBranchExecutionPlan(2, 0),
                    target_branch=RecurrentBranchExecutionPlan(3, 0),
                    transition_weight=transition_weight,
                )
                model_input = torch.ones(1, dim, requires_grad=True)
                transition_model = model.block_model.model

                with patch.object(
                    schedule,
                    "execution_plan",
                    return_value=execution_plan,
                ):
                    result = model(LayerState(hidden=model_input))
                input_gradient, parameter_gradient = torch.autograd.grad(
                    result.hidden.sum(),
                    (model_input, transition_model.scale),
                )

                oracle_input = torch.ones(1, dim, requires_grad=True)
                oracle_scale = transition_model.scale.detach().clone().requires_grad_()
                depth_one = oracle_input * oracle_scale
                depth_two = depth_one * oracle_scale
                depth_three = depth_two * oracle_scale
                oracle_output = (
                    1.0 - transition_weight
                ) * depth_two + transition_weight * depth_three
                oracle_input_gradient, oracle_parameter_gradient = torch.autograd.grad(
                    oracle_output.sum(),
                    (oracle_input, oracle_scale),
                )

                torch.testing.assert_close(result.hidden, oracle_output)
                torch.testing.assert_close(input_gradient, oracle_input_gradient)
                torch.testing.assert_close(
                    parameter_gradient, oracle_parameter_gradient
                )
                self.assertEqual(transition_model.grad_modes, [True, True, True])

    def test_full_gradient_smooth_growth_accumulates_caller_loss_once(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                initial_iterations=2,
                no_gradient_transition_count=0,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=True,
                block_config=LossAccumulatingBlockConfig(
                    input_dim=dim,
                    output_dim=dim,
                    increment=1.0,
                    per_step_loss=2.0,
                ),
            )
        ).eval()
        model.recurrent_iteration_schedule.forward_call_progress.fill_(4)
        incoming_loss = torch.tensor(7.0, requires_grad=True)

        result = model(
            LayerState(
                hidden=torch.zeros(1, dim),
                loss=incoming_loss,
            )
        )

        torch.testing.assert_close(result.loss, torch.tensor(12.0))
        result.loss.backward()
        torch.testing.assert_close(incoming_loss.grad, torch.tensor(1.0))

    def test_full_gradient_smooth_growth_matches_attention_residual_oracle(
        self,
    ) -> None:
        dim = 2

        def build_model(depth: int, *, smooth: bool) -> RecurrentLayer:
            return RecurrentLayer(
                self.recurrent_config(
                    dim=dim,
                    max_steps=3,
                    initial_iterations=depth,
                    no_gradient_transition_count=0,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4,
                    smooth_iteration_growth_flag=smooth,
                    block_config=self.trainable_scale_block_config(
                        dim=dim,
                        scale=1.2,
                    ),
                    residual_connection_option=AttentionResidualConfig,
                )
            ).eval()

        smooth_model = build_model(2, smooth=True)
        with torch.no_grad():
            for parameter_index, (parameter_name, parameter) in enumerate(
                smooth_model.named_parameters(),
                start=1,
            ):
                if parameter_name.endswith("query"):
                    parameter.fill_(0.1 * parameter_index)
                elif parameter_name.endswith("key_norm.weight"):
                    parameter.fill_(1.0 + 0.05 * parameter_index)
        source_model = build_model(2, smooth=False)
        target_model = build_model(3, smooth=False)
        source_model.load_state_dict(smooth_model.state_dict(), strict=True)
        target_model.load_state_dict(smooth_model.state_dict(), strict=True)
        smooth_model.recurrent_iteration_schedule.forward_call_progress.fill_(4)
        input_values = torch.tensor([[1.0, 2.0]])

        source = _capture_forward_gradients(source_model, input_values)
        target = _capture_forward_gradients(target_model, input_values)
        smooth = _capture_forward_gradients(smooth_model, input_values)

        torch.testing.assert_close(
            smooth.output,
            0.5 * source.output + 0.5 * target.output,
        )
        self.assertIsNotNone(source.input_gradient)
        self.assertIsNotNone(target.input_gradient)
        self.assertIsNotNone(smooth.input_gradient)
        torch.testing.assert_close(
            smooth.input_gradient,
            0.5 * source.input_gradient + 0.5 * target.input_gradient,
        )
        for parameter_name, smooth_gradient in smooth.parameter_gradients.items():
            with self.subTest(parameter_name=parameter_name):
                source_gradient = source.parameter_gradients[parameter_name]
                target_gradient = target.parameter_gradients[parameter_name]
                if source_gradient is None and target_gradient is None:
                    self.assertIsNone(smooth_gradient)
                    continue
                expected_gradient = sum(
                    0.5 * gradient
                    for gradient in (source_gradient, target_gradient)
                    if gradient is not None
                )
                torch.testing.assert_close(smooth_gradient, expected_gradient)

        self.assertEqual(len(smooth_model.block_model.model.outputs), 3)
        self.assertEqual(
            smooth_model.block_model.model.grad_modes,
            [True, True, True],
        )

    def test_full_gradient_smooth_growth_stops_at_the_realized_source_depth(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                initial_iterations=2,
                no_gradient_transition_count=0,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=True,
                block_config=self.layer_block_config(increment=1.0),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=2.0,
                    high_logit=20.0,
                    low_logit=-20.0,
                    min_steps=1,
                ),
            )
        ).eval()
        schedule = model.recurrent_iteration_schedule
        schedule.forward_call_progress.fill_(4)

        result = model(LayerState(hidden=torch.zeros(1, dim)))

        self.assertEqual(model.block_model.model.call_count, 2)
        self.assertEqual(schedule.snapshot().forward_call_progress, 5)
        torch.testing.assert_close(result.hidden, torch.full((1, dim), 2.0))

    def test_full_gradient_smooth_growth_returns_an_earlier_realized_depth(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                initial_iterations=2,
                no_gradient_transition_count=0,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=True,
                block_config=self.layer_block_config(increment=1.0),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=1.0,
                    high_logit=20.0,
                    low_logit=-20.0,
                    min_steps=1,
                ),
            )
        ).eval()
        schedule = model.recurrent_iteration_schedule
        schedule.forward_call_progress.fill_(4)

        result = model(LayerState(hidden=torch.zeros(1, dim)))

        self.assertEqual(model.block_model.model.call_count, 1)
        self.assertEqual(schedule.snapshot().forward_call_progress, 5)
        torch.testing.assert_close(result.hidden, torch.ones(1, dim))

    def test_full_gradient_smooth_growth_preserves_halted_rows_in_mixed_batches(
        self,
    ) -> None:
        dim = 2

        def build_model(depth: int, *, smooth: bool) -> RecurrentLayer:
            return RecurrentLayer(
                self.recurrent_config(
                    dim=dim,
                    max_steps=3 if smooth else depth,
                    initial_iterations=2 if smooth else depth,
                    no_gradient_transition_count=0,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4 if smooth else 1,
                    smooth_iteration_growth_flag=smooth,
                    block_config=self.layer_block_config(increment=1.0),
                    halting_config=self.halting_config(
                        dim=dim,
                        gate_threshold=1.5,
                        high_logit=20.0,
                        low_logit=-20.0,
                        min_steps=1,
                    ),
                )
            ).eval()

        smooth_model = build_model(2, smooth=True)
        smooth_model.recurrent_iteration_schedule.forward_call_progress.fill_(4)
        source_model = build_model(2, smooth=False)
        target_model = build_model(3, smooth=False)
        hidden = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])

        source_result = source_model(LayerState(hidden=hidden.clone()))
        target_result = target_model(LayerState(hidden=hidden.clone()))
        smooth_result = smooth_model(LayerState(hidden=hidden.clone()))

        expected_hidden = 0.5 * source_result.hidden + 0.5 * target_result.hidden
        torch.testing.assert_close(smooth_result.hidden, expected_hidden)
        torch.testing.assert_close(smooth_result.hidden[0], source_result.hidden[0])
        self.assertEqual(smooth_model.block_model.model.call_count, 3)

    def test_failed_full_gradient_smooth_handoff_rolls_back_state_and_rng(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                initial_iterations=2,
                no_gradient_transition_count=0,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=True,
                block_config=FailingStochasticStateBlockConfig(
                    input_dim=dim,
                    output_dim=dim,
                    fail_on_transition_step=3,
                ),
            )
        ).eval()
        schedule = model.recurrent_iteration_schedule
        schedule.forward_call_progress.fill_(4)
        initial_hidden = torch.ones(1, dim)
        initial_loss = torch.tensor(3.0)
        state = LayerState(hidden=initial_hidden, loss=initial_loss)
        initial_transition_step = model.block_model.transition_step.clone()
        torch.manual_seed(53)
        expected_next_random_value = torch.rand(())
        torch.manual_seed(53)
        captured_cuda_rng_states = [torch.tensor([13, 17], dtype=torch.uint8)]

        with (
            patch.object(torch.cuda, "is_initialized", return_value=True),
            patch.object(
                torch.cuda,
                "get_rng_state_all",
                return_value=captured_cuda_rng_states,
            ),
            patch.object(torch.cuda, "set_rng_state_all") as restore_cuda_rng,
            self.assertRaisesRegex(RuntimeError, "target transition failed"),
        ):
            model(state)

        self.assertIs(state.hidden, initial_hidden)
        self.assertIs(state.loss, initial_loss)
        self.assertEqual(schedule.snapshot().forward_call_progress, 4)
        torch.testing.assert_close(
            model.block_model.transition_step,
            initial_transition_step,
        )
        torch.testing.assert_close(torch.rand(()), expected_next_random_value)
        restore_cuda_rng.assert_called_once_with(captured_cuda_rng_states)

    def test_smooth_growth_interpolates_legacy_gradient_window_oracles(self):
        dim = 2

        def build_model(depth: int, *, smooth: bool) -> RecurrentLayer:
            return RecurrentLayer(
                self.recurrent_config(
                    dim=dim,
                    max_steps=3 if smooth else depth,
                    initial_iterations=2 if smooth else depth,
                    gradient_transition_count=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4 if smooth else 1,
                    smooth_iteration_growth_flag=smooth,
                    block_config=self.trainable_scale_block_config(
                        dim=dim,
                        scale=0.5,
                    ),
                )
            ).eval()

        smooth_model = build_model(2, smooth=True)
        for _ in range(4):
            smooth_model(LayerState(hidden=torch.ones(1, dim)))
        source_model = build_model(2, smooth=False)
        target_model = build_model(3, smooth=False)

        source_output = source_model(LayerState(hidden=torch.ones(1, dim))).hidden
        source_gradient = torch.autograd.grad(
            source_output.sum(),
            source_model.block_model.model.scale,
        )[0]
        target_output = target_model(LayerState(hidden=torch.ones(1, dim))).hidden
        target_gradient = torch.autograd.grad(
            target_output.sum(),
            target_model.block_model.model.scale,
        )[0]

        smooth_output = smooth_model(LayerState(hidden=torch.ones(1, dim))).hidden
        smooth_gradient = torch.autograd.grad(
            smooth_output.sum(),
            smooth_model.block_model.model.scale,
        )[0]

        torch.testing.assert_close(
            smooth_output,
            0.5 * source_output + 0.5 * target_output,
        )
        torch.testing.assert_close(
            smooth_gradient,
            0.5 * source_gradient + 0.5 * target_gradient,
        )

    def test_smooth_growth_endpoints_match_legacy_output_and_gradients(self):
        dim = 2

        def build_model(depth: int, *, smooth: bool) -> RecurrentLayer:
            return RecurrentLayer(
                self.recurrent_config(
                    dim=dim,
                    max_steps=3 if smooth else depth,
                    initial_iterations=2 if smooth else depth,
                    gradient_transition_count=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4 if smooth else 1,
                    smooth_iteration_growth_flag=smooth,
                    block_config=self.trainable_scale_block_config(
                        dim=dim,
                        scale=0.5,
                    ),
                )
            ).eval()

        for transition_weight in (0.0, 0.5, 1.0):
            with self.subTest(transition_weight=transition_weight):
                source_model = build_model(2, smooth=False)
                target_model = build_model(3, smooth=False)
                smooth_model = build_model(2, smooth=True)
                schedule = smooth_model.recurrent_iteration_schedule
                schedule.load_state_dict(
                    {"forward_call_progress": torch.tensor(4, dtype=torch.long)},
                    strict=True,
                )
                execution_plan = replace(
                    schedule.execution_plan(),
                    transition_weight=transition_weight,
                )

                source_input = torch.ones(1, dim, requires_grad=True)
                target_input = torch.ones(1, dim, requires_grad=True)
                smooth_input = torch.ones(1, dim, requires_grad=True)
                source_output = source_model(LayerState(hidden=source_input)).hidden
                target_output = target_model(LayerState(hidden=target_input)).hidden
                with patch.object(
                    schedule,
                    "execution_plan",
                    return_value=execution_plan,
                ):
                    smooth_output = smooth_model(LayerState(hidden=smooth_input)).hidden

                source_parameter_gradient, source_input_gradient = torch.autograd.grad(
                    source_output.sum(),
                    (source_model.block_model.model.scale, source_input),
                    allow_unused=True,
                )
                target_parameter_gradient, target_input_gradient = torch.autograd.grad(
                    target_output.sum(),
                    (target_model.block_model.model.scale, target_input),
                    allow_unused=True,
                )
                smooth_parameter_gradient, smooth_input_gradient = torch.autograd.grad(
                    smooth_output.sum(),
                    (smooth_model.block_model.model.scale, smooth_input),
                    allow_unused=True,
                )

                torch.testing.assert_close(
                    smooth_output,
                    (1.0 - transition_weight) * source_output
                    + transition_weight * target_output,
                )
                torch.testing.assert_close(
                    smooth_parameter_gradient,
                    (1.0 - transition_weight) * source_parameter_gradient
                    + transition_weight * target_parameter_gradient,
                )
                expected_input_gradient = (
                    source_input_gradient
                    if transition_weight == 0.0
                    else target_input_gradient
                    if transition_weight == 1.0
                    else (1.0 - transition_weight) * source_input_gradient
                )
                if expected_input_gradient is None:
                    self.assertIsNone(smooth_input_gradient)
                else:
                    torch.testing.assert_close(
                        smooth_input_gradient,
                        expected_input_gradient,
                    )

    def test_smooth_growth_weights_each_logical_gradient_contribution(self):
        dim = 2
        transition_weight = 0.25
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                initial_iterations=2,
                gradient_transition_count=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=True,
                block_config=self.trainable_scale_block_config(
                    dim=dim,
                    scale=1.0,
                ),
            )
        ).eval()
        schedule = model.recurrent_iteration_schedule
        schedule.load_state_dict(
            {"forward_call_progress": torch.tensor(4, dtype=torch.long)},
            strict=True,
        )
        execution_plan = replace(
            schedule.execution_plan(),
            transition_weight=transition_weight,
        )
        transition_model = model.block_model.model
        transition_model.outputs.clear()

        with patch.object(
            schedule,
            "execution_plan",
            return_value=execution_plan,
        ):
            output = model(
                LayerState(hidden=torch.ones(1, dim, requires_grad=True))
            ).hidden

        retained_gradients = torch.autograd.grad(
            output.sum(),
            tuple(transition_model.outputs),
            allow_unused=True,
        )
        expected_weights = (
            1.0 - transition_weight,
            1.0 - transition_weight,
            transition_weight,
            transition_weight,
        )

        self.assertEqual(len(retained_gradients), len(expected_weights))
        for transition_index, (gradient, expected_weight) in enumerate(
            zip(retained_gradients, expected_weights, strict=True)
        ):
            with self.subTest(transition_index=transition_index):
                torch.testing.assert_close(
                    gradient,
                    torch.full((1, dim), expected_weight),
                )

    def test_smooth_growth_blends_only_branch_local_auxiliary_loss_deltas(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                initial_iterations=2,
                gradient_transition_count=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=True,
                block_config=LossAccumulatingBlockConfig(
                    input_dim=dim,
                    output_dim=dim,
                    increment=1.0,
                    per_step_loss=2.0,
                ),
            )
        ).eval()
        for _ in range(4):
            model(LayerState(hidden=torch.zeros(1, dim)))
        incoming_loss = torch.tensor(5.0, requires_grad=True)

        result = model(
            LayerState(
                hidden=torch.zeros(1, dim),
                loss=incoming_loss,
            )
        )

        torch.testing.assert_close(result.loss, torch.tensor(10.0))
        result.loss.backward()
        torch.testing.assert_close(incoming_loss.grad, torch.tensor(1.0))

    def test_smooth_growth_loss_endpoints_match_legacy_branches(self):
        dim = 2

        def build_model(depth: int, *, smooth: bool) -> RecurrentLayer:
            return RecurrentLayer(
                self.recurrent_config(
                    dim=dim,
                    max_steps=3 if smooth else depth,
                    initial_iterations=2 if smooth else depth,
                    gradient_transition_count=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4 if smooth else 1,
                    smooth_iteration_growth_flag=smooth,
                    block_config=LossAccumulatingBlockConfig(
                        input_dim=dim,
                        output_dim=dim,
                        increment=1.0,
                        per_step_loss=2.0,
                    ),
                )
            ).eval()

        for transition_weight in (0.0, 0.5, 1.0):
            with self.subTest(transition_weight=transition_weight):
                source_model = build_model(2, smooth=False)
                target_model = build_model(3, smooth=False)
                smooth_model = build_model(2, smooth=True)
                schedule = smooth_model.recurrent_iteration_schedule
                schedule.load_state_dict(
                    {"forward_call_progress": torch.tensor(4, dtype=torch.long)},
                    strict=True,
                )
                execution_plan = replace(
                    schedule.execution_plan(),
                    transition_weight=transition_weight,
                )
                source_incoming_loss = torch.tensor(5.0, requires_grad=True)
                target_incoming_loss = torch.tensor(5.0, requires_grad=True)
                smooth_incoming_loss = torch.tensor(5.0, requires_grad=True)

                source_loss = source_model(
                    LayerState(
                        hidden=torch.zeros(1, dim),
                        loss=source_incoming_loss,
                    )
                ).loss
                target_loss = target_model(
                    LayerState(
                        hidden=torch.zeros(1, dim),
                        loss=target_incoming_loss,
                    )
                ).loss
                with patch.object(
                    schedule,
                    "execution_plan",
                    return_value=execution_plan,
                ):
                    smooth_loss = smooth_model(
                        LayerState(
                            hidden=torch.zeros(1, dim),
                            loss=smooth_incoming_loss,
                        )
                    ).loss

                self.assertIsNotNone(source_loss)
                self.assertIsNotNone(target_loss)
                self.assertIsNotNone(smooth_loss)
                torch.testing.assert_close(
                    smooth_loss,
                    (1.0 - transition_weight) * source_loss
                    + transition_weight * target_loss,
                )
                smooth_incoming_gradient = torch.autograd.grad(
                    smooth_loss,
                    smooth_incoming_loss,
                )[0]
                torch.testing.assert_close(
                    smooth_incoming_gradient,
                    torch.tensor(1.0),
                )

    def test_smooth_branch_loss_blend_preserves_the_zero_weight_endpoint(self):
        common_loss = torch.tensor(2.0)
        source_loss = torch.tensor(3.0)
        target_loss = torch.tensor(5.0)

        blended_loss = RecurrentLayer._blend_recurrent_branch_losses(
            common_loss,
            source_loss,
            target_loss,
            0.0,
        )

        self.assertIs(blended_loss, source_loss)

    def test_smooth_growth_blends_independent_legacy_halting_oracles(self):
        dim = 2

        def build_model(depth: int, *, smooth: bool) -> RecurrentLayer:
            return RecurrentLayer(
                self.recurrent_config(
                    dim=dim,
                    max_steps=3 if smooth else depth,
                    initial_iterations=2 if smooth else depth,
                    gradient_transition_count=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4 if smooth else 1,
                    smooth_iteration_growth_flag=smooth,
                    block_config=self.layer_block_config(increment=1.0),
                    halting_config=self.halting_config(
                        dim=dim,
                        gate_threshold=2.5,
                        high_logit=20.0,
                        low_logit=-20.0,
                        min_steps=2,
                    ),
                )
            ).eval()

        hidden = torch.zeros(1, dim)
        smooth_model = build_model(2, smooth=True)
        for _ in range(4):
            smooth_model(LayerState(hidden=hidden.clone()))
        source_result = build_model(2, smooth=False)(LayerState(hidden=hidden.clone()))
        target_result = build_model(3, smooth=False)(LayerState(hidden=hidden.clone()))

        smooth_result = smooth_model(LayerState(hidden=hidden.clone()))

        torch.testing.assert_close(
            smooth_result.hidden,
            0.5 * source_result.hidden + 0.5 * target_result.hidden,
        )
        torch.testing.assert_close(
            smooth_result.loss,
            0.5 * source_result.loss + 0.5 * target_result.loss,
        )

    def test_smooth_growth_matches_independent_soft_halting_oracles(self):
        dim = 2

        def build_model(depth: int, *, smooth: bool) -> RecurrentLayer:
            return RecurrentLayer(
                self.recurrent_config(
                    dim=dim,
                    max_steps=3 if smooth else depth,
                    initial_iterations=2 if smooth else depth,
                    gradient_transition_count=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4 if smooth else 1,
                    smooth_iteration_growth_flag=smooth,
                    block_config=self.layer_block_config(increment=1.0),
                    halting_config=SoftHaltingConfig(
                        input_dim=dim,
                        threshold=0.99,
                        ponder_cost_weight=1.0,
                        min_steps=2,
                        dropout_probability=0.0,
                        hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                        halting_gate_config=self.halting_gate_config(
                            threshold=2.5,
                            high_logit=20.0,
                            low_logit=-20.0,
                        ),
                    ),
                )
            ).eval()

        smooth_model = build_model(2, smooth=True)
        smooth_model.recurrent_iteration_schedule.load_state_dict(
            {"forward_call_progress": torch.tensor(4, dtype=torch.long)},
            strict=True,
        )
        source_model = build_model(2, smooth=False)
        target_model = build_model(3, smooth=False)
        source_model.halting_model.load_state_dict(
            smooth_model.halting_model.state_dict(),
            strict=True,
        )
        target_model.halting_model.load_state_dict(
            smooth_model.halting_model.state_dict(),
            strict=True,
        )
        hidden = torch.zeros(1, dim)

        source_result = source_model(LayerState(hidden=hidden.clone()))
        target_result = target_model(LayerState(hidden=hidden.clone()))
        smooth_result = smooth_model(LayerState(hidden=hidden.clone()))

        torch.testing.assert_close(
            smooth_result.hidden,
            0.5 * source_result.hidden + 0.5 * target_result.hidden,
        )
        torch.testing.assert_close(
            smooth_result.loss,
            0.5 * source_result.loss + 0.5 * target_result.loss,
        )

    def test_smooth_growth_keeps_source_halting_rng_branch_local(self) -> None:
        dim = 2

        def build_model(depth: int, *, smooth: bool) -> RecurrentLayer:
            return RecurrentLayer(
                self.recurrent_config(
                    dim=dim,
                    max_steps=4 if smooth else depth,
                    initial_iterations=3 if smooth else depth,
                    gradient_transition_count=2,
                    forward_calls_before_iteration_increment=4,
                    smooth_iteration_growth_flag=smooth,
                    block_config=self.layer_block_config(increment=1.0),
                    halting_config=StochasticHaltingConfig(min_steps=2),
                )
            )

        smooth_model = build_model(3, smooth=True)
        for _ in range(4):
            smooth_model(LayerState(hidden=torch.zeros(1, dim)))
        starting_update_step = smooth_model.halting_model.update_step.clone()
        source_model = build_model(3, smooth=False)
        target_model = build_model(4, smooth=False)
        source_model.halting_model.load_state_dict(
            smooth_model.halting_model.state_dict()
        )
        target_model.halting_model.load_state_dict(
            smooth_model.halting_model.state_dict()
        )

        torch.manual_seed(31)
        source_result = source_model(LayerState(hidden=torch.zeros(1, dim)))
        torch.manual_seed(31)
        target_result = target_model(LayerState(hidden=torch.zeros(1, dim)))
        expected_next_random_value = torch.rand(())
        torch.manual_seed(31)
        smooth_result = smooth_model(LayerState(hidden=torch.zeros(1, dim)))
        actual_next_random_value = torch.rand(())

        torch.testing.assert_close(
            smooth_result.hidden,
            0.5 * source_result.hidden + 0.5 * target_result.hidden,
        )
        torch.testing.assert_close(
            smooth_model.halting_model.update_step,
            starting_update_step + 2.0,
        )
        torch.testing.assert_close(
            smooth_model.halting_model.update_step,
            target_model.halting_model.update_step,
        )
        torch.testing.assert_close(
            actual_next_random_value,
            expected_next_random_value,
        )

    def test_smooth_growth_forks_partial_attention_residual_history(self):
        dim = 2

        def build_model(depth: int, *, smooth: bool) -> RecurrentLayer:
            config = self.recurrent_config(
                dim=dim,
                max_steps=4 if smooth else depth,
                initial_iterations=2 if smooth else depth,
                gradient_transition_count=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4 if smooth else 1,
                smooth_iteration_growth_flag=smooth,
                block_config=self.layer_block_config(increment=1.0),
                residual_connection_option=AttentionResidualConfig,
            )
            config.residual_config.block_size = 2
            return RecurrentLayer(config).eval()

        smooth_model = build_model(3, smooth=True)
        hidden = torch.tensor([[1.0, 2.0]])
        for _ in range(8):
            smooth_model(LayerState(hidden=hidden.clone()))
        source_model = build_model(3, smooth=False)
        target_model = build_model(4, smooth=False)
        source_output = source_model(LayerState(hidden=hidden.clone())).hidden
        target_output = target_model(LayerState(hidden=hidden.clone())).hidden
        smooth_model.block_model.model.call_count = 0

        smooth_output = smooth_model(LayerState(hidden=hidden.clone())).hidden

        torch.testing.assert_close(
            smooth_output,
            0.5 * source_output + 0.5 * target_output,
        )
        self.assertEqual(smooth_model.block_model.model.call_count, 5)

    def test_attention_residual_smooth_endpoint_matches_settled_target_gradients(
        self,
    ) -> None:
        dim = 2

        def build_model() -> RecurrentLayer:
            return RecurrentLayer(
                self.recurrent_config(
                    dim=dim,
                    max_steps=4,
                    initial_iterations=2,
                    gradient_transition_count=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4,
                    smooth_iteration_growth_flag=True,
                    block_config=self.trainable_scale_block_config(
                        dim=dim,
                        scale=1.2,
                    ),
                    residual_connection_option=AttentionResidualConfig,
                )
            ).eval()

        for endpoint_progress, settled_progress in ((5, 6), (9, 10)):
            with self.subTest(
                endpoint_progress=endpoint_progress,
                settled_progress=settled_progress,
            ):
                endpoint_model = build_model()
                with torch.no_grad():
                    for parameter_index, (parameter_name, parameter) in enumerate(
                        endpoint_model.named_parameters(),
                        start=1,
                    ):
                        if parameter_name.endswith("query"):
                            parameter.copy_(
                                torch.linspace(
                                    0.1 * parameter_index,
                                    -0.05 * parameter_index,
                                    parameter.numel(),
                                ).reshape_as(parameter)
                            )
                        elif parameter_name.endswith("key_norm.weight"):
                            parameter.copy_(
                                torch.linspace(
                                    0.7 + 0.02 * parameter_index,
                                    1.3 + 0.02 * parameter_index,
                                    parameter.numel(),
                                ).reshape_as(parameter)
                            )
                settled_model = build_model()
                settled_model.load_state_dict(
                    endpoint_model.state_dict(),
                    strict=True,
                )
                endpoint_model.recurrent_iteration_schedule.forward_call_progress.fill_(
                    endpoint_progress
                )
                settled_model.recurrent_iteration_schedule.forward_call_progress.fill_(
                    settled_progress
                )
                input_values = torch.tensor([[1.0, 2.0]])

                endpoint = _capture_forward_gradients(endpoint_model, input_values)
                settled = _capture_forward_gradients(settled_model, input_values)

                torch.testing.assert_close(endpoint.output, settled.output)
                self.assertIsNotNone(endpoint.input_gradient)
                self.assertIsNotNone(settled.input_gradient)
                torch.testing.assert_close(
                    endpoint.input_gradient,
                    settled.input_gradient,
                )
                self.assertEqual(
                    endpoint.parameter_gradients.keys(),
                    settled.parameter_gradients.keys(),
                )
                for parameter_name in endpoint.parameter_gradients:
                    endpoint_gradient = endpoint.parameter_gradients[parameter_name]
                    settled_gradient = settled.parameter_gradients[parameter_name]
                    with self.subTest(parameter_name=parameter_name):
                        if endpoint_gradient is None or settled_gradient is None:
                            self.assertIs(endpoint_gradient, settled_gradient)
                        else:
                            torch.testing.assert_close(
                                endpoint_gradient,
                                settled_gradient,
                            )

    def test_smooth_growth_matches_each_attention_residual_router_gradient(self):
        dim = 2

        def build_model(
            initial_iterations: int,
            *,
            smooth: bool,
        ) -> RecurrentLayer:
            return RecurrentLayer(
                self.recurrent_config(
                    dim=dim,
                    max_steps=4,
                    initial_iterations=initial_iterations,
                    gradient_transition_count=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4,
                    smooth_iteration_growth_flag=smooth,
                    block_config=self.trainable_scale_block_config(
                        dim=dim,
                        scale=1.2,
                    ),
                    residual_connection_option=AttentionResidualConfig,
                )
            ).eval()

        smooth_model = build_model(2, smooth=True)
        with torch.no_grad():
            for parameter_index, (parameter_name, parameter) in enumerate(
                smooth_model.named_parameters(),
                start=1,
            ):
                if parameter_name.endswith("query"):
                    parameter.fill_(0.1 * parameter_index)
                elif parameter_name.endswith("key_norm.weight"):
                    parameter.fill_(1.0 + 0.05 * parameter_index)
        source_model = build_model(2, smooth=False)
        target_model = build_model(3, smooth=False)
        source_model.load_state_dict(smooth_model.state_dict(), strict=True)
        target_model.load_state_dict(smooth_model.state_dict(), strict=True)
        smooth_model.recurrent_iteration_schedule.load_state_dict(
            {"forward_call_progress": torch.tensor(4, dtype=torch.long)},
            strict=True,
        )
        input_values = torch.tensor([[1.0, 2.0]])

        source = _capture_forward_gradients(source_model, input_values)
        target = _capture_forward_gradients(target_model, input_values)
        smooth = _capture_forward_gradients(smooth_model, input_values)

        torch.testing.assert_close(
            smooth.output,
            0.5 * source.output + 0.5 * target.output,
        )
        self.assertIsNotNone(source.input_gradient)
        self.assertIsNotNone(target.input_gradient)
        self.assertIsNotNone(smooth.input_gradient)
        torch.testing.assert_close(
            smooth.input_gradient,
            0.5 * source.input_gradient + 0.5 * target.input_gradient,
        )
        self.assertEqual(
            smooth.parameter_gradients.keys(),
            source.parameter_gradients.keys(),
        )
        self.assertEqual(
            smooth.parameter_gradients.keys(),
            target.parameter_gradients.keys(),
        )
        for parameter_name, smooth_gradient in smooth.parameter_gradients.items():
            with self.subTest(parameter_name=parameter_name):
                source_gradient = source.parameter_gradients[parameter_name]
                target_gradient = target.parameter_gradients[parameter_name]
                expected_gradient = None
                if source_gradient is not None:
                    expected_gradient = 0.5 * source_gradient
                if target_gradient is not None:
                    weighted_target_gradient = 0.5 * target_gradient
                    expected_gradient = (
                        weighted_target_gradient
                        if expected_gradient is None
                        else expected_gradient + weighted_target_gradient
                    )
                if expected_gradient is None:
                    self.assertIsNone(smooth_gradient)
                else:
                    torch.testing.assert_close(
                        smooth_gradient,
                        expected_gradient,
                    )

        provisional_router_name = (
            "recurrent_residual_schedule.subsequent_connections.1.query"
        )
        self.assertIsNone(source.parameter_gradients[provisional_router_name])
        target_router_gradient = target.parameter_gradients[provisional_router_name]
        self.assertIsNotNone(target_router_gradient)
        self.assertTrue(target_router_gradient.ne(0).any())

    def test_smooth_growth_shares_adaptive_boundary_and_commits_target_progress(
        self,
    ) -> None:
        torch.manual_seed(7)
        dim = 2

        def build_model(depth: int, *, smooth: bool) -> RecurrentLayer:
            return RecurrentLayer(
                self.recurrent_config(
                    dim=dim,
                    max_steps=4 if smooth else depth,
                    initial_iterations=3 if smooth else depth,
                    gradient_transition_count=2,
                    iteration_increment=1,
                    forward_calls_before_iteration_increment=4 if smooth else 1,
                    smooth_iteration_growth_flag=smooth,
                    block_config=self.adaptive_parameter_block_config(dim),
                )
            )

        smooth_model = build_model(3, smooth=True).eval()
        schedule_input = torch.tensor([[0.25, -0.5]])
        for _ in range(4):
            smooth_model(LayerState(hidden=schedule_input.clone()))

        source_model = build_model(3, smooth=False)
        target_model = build_model(4, smooth=False)
        source_model.block_model.load_state_dict(
            smooth_model.block_model.state_dict(),
            strict=True,
        )
        target_model.block_model.load_state_dict(
            smooth_model.block_model.state_dict(),
            strict=True,
        )
        smooth_model.train()
        source_model.train()
        target_model.train()

        source_input = schedule_input.clone().requires_grad_()
        target_input = schedule_input.clone().requires_grad_()
        smooth_input = schedule_input.clone().requires_grad_()
        source_result = source_model(LayerState(hidden=source_input))
        target_result = target_model(LayerState(hidden=target_input))

        adaptive_linear = smooth_model.block_model.model
        adaptive_weight = adaptive_linear.adaptive_behaviour.weight_model
        adaptive_bias = adaptive_linear.adaptive_behaviour.bias_model
        schedule_progress_buffer = (
            smooth_model.recurrent_iteration_schedule.forward_call_progress
        )
        adaptive_decay_step_buffer = adaptive_weight.decay_step
        with (
            patch.object(
                adaptive_linear,
                "forward",
                wraps=adaptive_linear.forward,
            ) as adaptive_forward,
            patch.object(
                adaptive_weight,
                "forward",
                wraps=adaptive_weight.forward,
            ) as adaptive_weight_forward,
            patch.object(
                adaptive_bias,
                "forward",
                wraps=adaptive_bias.forward,
            ) as adaptive_bias_forward,
        ):
            smooth_result = smooth_model(LayerState(hidden=smooth_input))

        torch.testing.assert_close(adaptive_weight.decay_step, torch.tensor([4.0]))
        self.assertIs(
            smooth_model.recurrent_iteration_schedule.forward_call_progress,
            schedule_progress_buffer,
        )
        self.assertIs(adaptive_weight.decay_step, adaptive_decay_step_buffer)
        self.assertEqual(adaptive_forward.call_count, 5)
        self.assertEqual(adaptive_weight_forward.call_count, 5)
        self.assertEqual(adaptive_bias_forward.call_count, 5)
        torch.testing.assert_close(
            smooth_result.hidden,
            0.5 * source_result.hidden + 0.5 * target_result.hidden,
        )

        source_parameters = dict(source_model.block_model.named_parameters())
        target_parameters = dict(target_model.block_model.named_parameters())
        smooth_parameters = dict(smooth_model.block_model.named_parameters())
        parameter_names = tuple(smooth_parameters)
        source_gradients = torch.autograd.grad(
            source_result.hidden.sum(),
            tuple(source_parameters[name] for name in parameter_names),
            allow_unused=True,
            retain_graph=True,
        )
        target_gradients = torch.autograd.grad(
            target_result.hidden.sum(),
            tuple(target_parameters[name] for name in parameter_names),
            allow_unused=True,
            retain_graph=True,
        )
        smooth_gradients = torch.autograd.grad(
            smooth_result.hidden.sum(),
            tuple(smooth_parameters[name] for name in parameter_names),
            allow_unused=True,
            retain_graph=True,
        )
        for name, source_gradient, target_gradient, smooth_gradient in zip(
            parameter_names,
            source_gradients,
            target_gradients,
            smooth_gradients,
            strict=True,
        ):
            with self.subTest(parameter=name):
                if smooth_gradient is None:
                    self.assertIsNone(source_gradient)
                    self.assertIsNone(target_gradient)
                    continue
                expected_gradient = torch.zeros_like(smooth_gradient)
                if source_gradient is not None:
                    expected_gradient = expected_gradient + 0.5 * source_gradient
                if target_gradient is not None:
                    expected_gradient = expected_gradient + 0.5 * target_gradient
                torch.testing.assert_close(smooth_gradient, expected_gradient)

        smooth_input_gradient = torch.autograd.grad(
            smooth_result.hidden.sum(),
            smooth_input,
            allow_unused=True,
        )[0]
        source_input_gradient = torch.autograd.grad(
            source_result.hidden.sum(),
            source_input,
            allow_unused=True,
        )[0]
        target_input_gradient = torch.autograd.grad(
            target_result.hidden.sum(),
            target_input,
            allow_unused=True,
        )[0]
        self.assertIsNone(smooth_input_gradient)
        self.assertIsNone(source_input_gradient)
        self.assertIsNone(target_input_gradient)

    def test_smooth_growth_restores_rng_before_committing_target_branch(self) -> None:
        dim = 2

        def build_model(depth: int, *, smooth: bool) -> RecurrentLayer:
            return RecurrentLayer(
                self.recurrent_config(
                    dim=dim,
                    max_steps=4 if smooth else depth,
                    initial_iterations=3 if smooth else depth,
                    gradient_transition_count=2,
                    forward_calls_before_iteration_increment=4,
                    smooth_iteration_growth_flag=smooth,
                    block_config=StochasticStateBlockConfig(
                        input_dim=dim,
                        output_dim=dim,
                    ),
                )
            )

        smooth_model = build_model(3, smooth=True)
        for _ in range(4):
            smooth_model(LayerState(hidden=torch.ones(1, dim)))
        source_model = build_model(3, smooth=False)
        target_model = build_model(4, smooth=False)
        source_model.block_model.load_state_dict(smooth_model.block_model.state_dict())
        target_model.block_model.load_state_dict(smooth_model.block_model.state_dict())

        torch.manual_seed(23)
        source_result = source_model(LayerState(hidden=torch.ones(1, dim)))
        source_gradient = torch.autograd.grad(
            source_result.hidden.sum(),
            source_model.block_model.scale,
        )[0]
        torch.manual_seed(23)
        target_result = target_model(LayerState(hidden=torch.ones(1, dim)))
        target_gradient = torch.autograd.grad(
            target_result.hidden.sum(),
            target_model.block_model.scale,
        )[0]
        expected_next_random_value = torch.rand(())

        torch.manual_seed(23)
        smooth_model.block_model.call_count = 0
        smooth_result = smooth_model(LayerState(hidden=torch.ones(1, dim)))
        smooth_gradient = torch.autograd.grad(
            smooth_result.hidden.sum(),
            smooth_model.block_model.scale,
        )[0]
        actual_next_random_value = torch.rand(())

        torch.testing.assert_close(
            smooth_result.hidden,
            0.5 * source_result.hidden + 0.5 * target_result.hidden,
        )
        torch.testing.assert_close(
            smooth_gradient,
            0.5 * source_gradient + 0.5 * target_gradient,
        )
        torch.testing.assert_close(
            actual_next_random_value,
            expected_next_random_value,
        )
        self.assertEqual(smooth_model.block_model.call_count, 5)

    def test_recurrent_step_controls_reject_ambiguous_or_invalid_windows(self):
        invalid_cases = (
            (
                "both gradient window forms",
                {
                    "max_steps": 5,
                    "no_gradient_transition_count": 1,
                    "gradient_transition_count": 2,
                },
                ValueError,
                "mutually exclusive",
            ),
            (
                "gradient suffix exceeds initial depth",
                {
                    "max_steps": 10,
                    "initial_iterations": 2,
                    "gradient_transition_count": 3,
                    "iteration_increment": 1,
                    "forward_calls_before_iteration_increment": 1,
                },
                ValueError,
                "less than or equal to the minimum active transition count",
            ),
            (
                "initial depth exceeds capacity",
                {
                    "max_steps": 4,
                    "initial_iterations": 5,
                    "iteration_increment": 1,
                    "forward_calls_before_iteration_increment": 1,
                },
                ValueError,
                "initial_iterations must be less than or equal to the variant's maximum",
            ),
            (
                "zero schedule increment",
                {
                    "max_steps": 4,
                    "initial_iterations": 2,
                    "iteration_increment": 0,
                    "forward_calls_before_iteration_increment": 1,
                },
                ValueError,
                "iteration_increment must be greater than or equal to 1",
            ),
        )

        for name, overrides, error_type, message in invalid_cases:
            with self.subTest(name=name), self.assertRaisesRegex(error_type, message):
                self.recurrent_config(**overrides).build()

    def test_fixed_gradient_suffix_requires_a_matching_halting_floor(self):
        config = self.recurrent_config(
            dim=2,
            max_steps=5,
            gradient_transition_count=2,
            halting_config=self.halting_config(
                dim=2,
                gate_threshold=0.0,
                min_steps=1,
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "halting_config.min_steps.*required update count",
        ):
            config.build()

    def test_fixed_gradient_suffix_executes_fully_with_matching_halting_floor(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=5,
                gradient_transition_count=2,
                block_config=self.trainable_scale_block_config(
                    dim=dim,
                    scale=0.5,
                ),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=0.0,
                    min_steps=2,
                ),
            )
        ).eval()

        model(LayerState(hidden=torch.ones(1, dim)))

        self.assertEqual(
            model.block_model.model.grad_modes,
            [False, False, False, True, True],
        )

    def test_reinjection_refines_no_grad_prefix_and_reconnects_original_input(self):
        model = RecurrentLayer(
            self.recurrent_config(
                dim=1,
                max_steps=3,
                no_gradient_transition_count=2,
                reinject_original_hidden_flag=True,
                block_config=StateSpyBlockConfig(
                    input_dim=1,
                    output_dim=1,
                    increment=1.0,
                ),
            )
        )
        original_hidden = torch.ones(1, 1, requires_grad=True)

        result = model(LayerState(hidden=original_hidden))
        result.hidden.sum().backward()

        self.assertEqual(model.block_model.grad_modes, [False, False, True])
        for actual, expected in zip(
            model.block_model.received_hidden_inputs,
            (2.0, 4.0, 6.0),
            strict=True,
        ):
            torch.testing.assert_close(actual, torch.full_like(actual, expected))
        torch.testing.assert_close(result.hidden, torch.tensor([[7.0]]))
        torch.testing.assert_close(
            original_hidden.grad, torch.ones_like(original_hidden)
        )

    def test_gradient_boundary_detaches_an_identity_block_output(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=2,
                no_gradient_transition_count=1,
                block_config=IdentityStateBlockConfig(
                    input_dim=dim,
                    output_dim=dim,
                ),
            )
        )
        inputs = torch.ones(2, dim, requires_grad=True)

        result = model(LayerState(hidden=inputs))

        self.assertIsNot(result.hidden, inputs)
        self.assertFalse(result.hidden.requires_grad)
        self.assertIsNone(result.hidden.grad_fn)
        self.assertTrue(inputs.requires_grad)

    def test_halting_starts_after_the_detached_prefix(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=5,
                no_gradient_transition_count=2,
                block_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    gate_config=None,
                    halting_config=None,
                    layer_model_config=TrainableScaleFeatureLastConfig(scale=0.5),
                ),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=0.0,
                ),
            )
        ).eval()

        result = model(LayerState(hidden=torch.ones(2, dim)))
        result.hidden.sum().backward()

        self.assertEqual(
            model.block_model.model.grad_modes,
            [False, False, True],
        )
        self.assertIsNotNone(model.block_model.model.scale.grad)

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
        self.assertIsNone(model.recurrent_residual_schedule)

    def test_recurrent_attention_residual_routes_across_step_history(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=2,
                block_config=self.layer_block_config(increment=2.0),
                residual_connection_option=AttentionResidualConfig,
            )
        )
        hidden = torch.ones(1, dim)

        result = model(LayerState(hidden=hidden))

        torch.testing.assert_close(
            result.hidden,
            torch.full_like(hidden, 8.0 / 3.0),
        )

    def test_recurrent_uses_custom_depthwise_residual_schedule(self):
        max_steps = 3
        model = RecurrentLayer(
            self.recurrent_config(
                dim=2,
                max_steps=max_steps,
                initial_iterations=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=1,
                block_config=self.layer_block_config(increment=0.0),
                residual_connection_option=DepthwiseTestResidualConfig,
            )
        )
        schedule = model.recurrent_residual_schedule

        self.assertIsInstance(schedule, DepthwiseRecurrentResidualSchedule)
        connections = tuple(
            schedule.connection_for_transition(
                model.residual_connection,
                transition_index,
            )
            for transition_index in range(max_steps)
        )
        self.assertEqual(len(connections), max_steps)
        self.assertEqual(len({id(connection) for connection in connections}), max_steps)
        with torch.no_grad():
            for offset, connection in enumerate(connections, start=1):
                connection.offset.fill_(offset)

        initial_result = model(LayerState(hidden=torch.zeros(1, 2)))
        capacity_result = model(LayerState(hidden=torch.zeros(1, 2)))

        torch.testing.assert_close(initial_result.hidden, torch.full((1, 2), 3.0))
        torch.testing.assert_close(capacity_result.hidden, torch.full((1, 2), 6.0))

    def test_recurrent_attention_residual_checkpoint_owns_one_router_per_step(self):
        dim = 2
        max_steps = 3
        config = self.recurrent_config(
            dim=dim,
            max_steps=max_steps,
            block_config=self.layer_block_config(increment=1.0),
            residual_connection_option=AttentionResidualConfig,
        )
        model = RecurrentLayer(config)
        query_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.endswith("query")
        ]

        self.assertEqual(len(query_parameters), max_steps)
        self.assertEqual(
            len({id(parameter) for parameter in query_parameters}),
            max_steps,
        )
        with torch.no_grad():
            for step, query in enumerate(query_parameters, start=1):
                query.fill_(step / 10)
        hidden = torch.tensor([[1.0, -2.0]])
        expected = model(LayerState(hidden=hidden.clone())).hidden
        checkpoint = model.state_dict()

        restored = RecurrentLayer(config)
        restored.load_state_dict(checkpoint, strict=True)
        actual = restored(LayerState(hidden=hidden.clone())).hidden

        self.assertEqual(set(restored.state_dict()), set(checkpoint))
        torch.testing.assert_close(actual, expected)

    def test_pairwise_recurrent_residual_checkpoint_paths_remain_compatible(self):
        config = self.recurrent_config(
            dim=2,
            max_steps=3,
            block_config=self.layer_block_config(increment=1.0),
            residual_connection_option=WeightedBlendResidualConfig,
        )
        model = RecurrentLayer(config)
        hidden = torch.tensor([[1.0, -2.0]])
        expected = model(LayerState(hidden=hidden.clone())).hidden

        checkpoint = model.state_dict()

        self.assertEqual(
            set(checkpoint),
            {
                "recurrent_iteration_schedule.forward_call_progress",
                "residual_connection.raw_weight",
            },
        )
        restored = RecurrentLayer(config)
        restored.load_state_dict(checkpoint, strict=True)
        actual = restored(LayerState(hidden=hidden.clone())).hidden
        torch.testing.assert_close(actual, expected)

    def test_recurrent_attention_residual_starts_fresh_history_per_forward(self):
        config = self.recurrent_config(
            dim=2,
            max_steps=3,
            block_config=self.layer_block_config(increment=1.0),
            residual_connection_option=AttentionResidualConfig,
        )
        reused_model = RecurrentLayer(config)
        fresh_model = RecurrentLayer(config)
        reused_model(LayerState(hidden=torch.tensor([[1.0, -2.0]])))
        next_hidden = torch.tensor([[0.5, 3.0], [2.0, -1.0]])

        reused_result = reused_model(LayerState(hidden=next_hidden.clone()))
        fresh_result = fresh_model(LayerState(hidden=next_hidden.clone()))

        torch.testing.assert_close(reused_result.hidden, fresh_result.hidden)

    def test_recurrent_attention_residual_backpropagates_through_all_routers(self):
        model = RecurrentLayer(
            self.recurrent_config(
                dim=2,
                max_steps=3,
                block_config=self.layer_block_config(increment=1.0),
                residual_connection_option=AttentionResidualConfig,
            )
        )
        router_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.endswith("query") or name.endswith("key_norm.weight")
        ]
        with torch.no_grad():
            for parameter in router_parameters:
                parameter.copy_(torch.tensor([0.25, -0.15]))
        hidden = torch.tensor(
            [[1.0, -2.0], [0.5, 3.0]],
            requires_grad=True,
        )

        result = model(LayerState(hidden=hidden))
        result.hidden.square().sum().backward()

        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(torch.count_nonzero(hidden.grad).item(), 0)
        self.assertEqual(len(router_parameters), model.max_steps * 2)
        for parameter in router_parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

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
                AdditiveResidualConfig,
                None,
                torch.full_like(hidden, 4.0),
            ),
            (
                WeightedResidualConfig,
                None,
                torch.full_like(hidden, 1.0),
            ),
            (
                WeightedBlendResidualConfig,
                None,
                torch.full_like(hidden, 2.8),
            ),
            (
                WeightedResidualConfig,
                data_dependent_model_config,
                torch.full_like(hidden, 1.0),
            ),
            (
                WeightedBlendResidualConfig,
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
                residual_connection_option=WeightedBlendResidualConfig,
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

    def test_default_minimum_preserves_first_transition_halting(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=5,
                block_config=self.layer_block_config(increment=1.0),
                halting_config=self.halting_config(dim=dim, gate_threshold=0.0),
            )
        ).eval()
        hidden = torch.zeros(3, dim)

        result = model(LayerState(hidden=hidden))

        self.assertFalse(hasattr(model, "min_steps"))
        self.assertEqual(model.halting_model.min_steps, 1)
        self.assertEqual(model.block_model.model.call_count, 1)
        torch.testing.assert_close(result.hidden, torch.ones_like(hidden))

    def test_always_halt_waits_for_three_functional_transitions(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=5,
                block_config=self.layer_block_config(increment=1.0),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=0.0,
                    min_steps=3,
                ),
            )
        ).eval()
        hidden = torch.zeros(3, dim)

        result = model(LayerState(hidden=hidden))

        self.assertEqual(model.block_model.model.call_count, 3)
        torch.testing.assert_close(result.hidden, torch.full_like(hidden, 3.0))

    def test_mandatory_trainable_transitions_affect_output_and_task_gradient(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=5,
                block_config=self.trainable_scale_block_config(
                    dim=dim,
                    scale=0.5,
                ),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=0.0,
                    high_logit=100.0,
                    low_logit=-100.0,
                    min_steps=3,
                ),
            )
        ).eval()

        result = model(LayerState(hidden=torch.ones(1, dim)))
        result.hidden.sum().backward()

        trainable_block = model.block_model.model
        self.assertEqual(trainable_block.grad_modes, [True, True, True])
        torch.testing.assert_close(result.hidden, torch.full((1, dim), 0.125))
        self.assertIsNotNone(trainable_block.scale.grad)
        self.assertGreater(trainable_block.scale.grad.abs().item(), 0.0)

    def test_halting_can_continue_from_minimum_until_the_maximum(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=5,
                block_config=self.layer_block_config(increment=1.0),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=100.0,
                    min_steps=3,
                ),
            )
        ).eval()

        result = model(LayerState(hidden=torch.zeros(1, dim)))

        self.assertEqual(model.block_model.model.call_count, 5)
        torch.testing.assert_close(result.hidden, torch.full((1, dim), 5.0))

    def test_equal_minimum_and_maximum_produces_fixed_depth_execution(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                block_config=self.layer_block_config(increment=1.0),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=0.0,
                    high_logit=100.0,
                    low_logit=-100.0,
                    min_steps=3,
                ),
            )
        ).eval()

        result = model(LayerState(hidden=torch.zeros(1, dim)))

        self.assertEqual(model.block_model.model.call_count, 3)
        torch.testing.assert_close(result.hidden, torch.full((1, dim), 3.0))

    def test_minimum_and_no_gradient_constraints_both_gate_halting(self):
        cases = (
            (1, 3, [False, False, False, True]),
            (4, 1, [False, True, True, True, True]),
        )

        for min_steps, no_gradient_count, expected_gradient_modes in cases:
            with self.subTest(
                min_steps=min_steps,
                no_gradient_transition_count=no_gradient_count,
            ):
                dim = 2
                model = RecurrentLayer(
                    self.recurrent_config(
                        dim=dim,
                        max_steps=5,
                        no_gradient_transition_count=no_gradient_count,
                        block_config=self.trainable_scale_block_config(
                            dim=dim,
                            scale=0.5,
                        ),
                        halting_config=self.halting_config(
                            dim=dim,
                            gate_threshold=0.0,
                            high_logit=100.0,
                            low_logit=-100.0,
                            min_steps=min_steps,
                        ),
                    )
                ).eval()

                model(LayerState(hidden=torch.ones(1, dim)))

                self.assertEqual(
                    model.block_model.model.grad_modes,
                    expected_gradient_modes,
                )

    def test_runtime_controls_do_not_change_checkpoint_namespaces(self):
        dim = 2
        default_model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                gate_config=self.trainable_gate_config(dim),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=0.0,
                ),
            )
        )
        controlled_model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                initial_iterations=2,
                gradient_transition_count=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=1,
                gate_config=self.trainable_gate_config(dim),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=0.0,
                    ponder_cost_weight=0.5,
                    min_steps=3,
                ),
            )
        )
        self.assertEqual(
            tuple(default_model.state_dict()),
            tuple(controlled_model.state_dict()),
        )
        controlled_model.load_state_dict(default_model.state_dict(), strict=True)
        default_model.load_state_dict(controlled_model.state_dict(), strict=True)

    def test_halting_monitor_reports_realized_depth_and_optional_ponder_cost(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=5,
                block_config=self.layer_block_config(increment=1.0),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=4.0,
                    high_logit=20.0,
                    low_logit=-20.0,
                    ponder_cost_weight=0.5,
                    min_steps=3,
                ),
            )
        ).eval()
        tracker_manager = HaltingUsageTrackerManager()
        tracker = tracker_manager.attach(model.halting_model)

        result = model(LayerState(hidden=torch.zeros(1, dim)))

        self.assertEqual(model.block_model.model.call_count, 4)
        torch.testing.assert_close(tracker.last_step_count, torch.tensor(4.0))
        torch.testing.assert_close(tracker.last_raw_ponder_loss, torch.tensor(1.0))
        torch.testing.assert_close(
            tracker.last_effective_ponder_loss,
            torch.tensor(0.5),
        )
        torch.testing.assert_close(result.loss, torch.tensor(0.5))
        tracker_manager.detach(model.halting_model)

    def test_smooth_growth_commits_only_target_branch_halting_usage(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=3,
                initial_iterations=2,
                gradient_transition_count=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=4,
                smooth_iteration_growth_flag=True,
                block_config=self.layer_block_config(increment=1.0),
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=2.5,
                    min_steps=2,
                ),
            )
        ).eval()
        for _ in range(4):
            model(LayerState(hidden=torch.zeros(1, dim)))
        tracker_manager = HaltingUsageTrackerManager()
        tracker = tracker_manager.attach(model.halting_model)

        result = model(LayerState(hidden=torch.zeros(1, dim)))

        torch.testing.assert_close(tracker.last_step_count, torch.tensor(2.0))
        torch.testing.assert_close(
            tracker.last_survival,
            torch.tensor([1.0, 0.0]),
        )
        self.assertTrue(torch.isfinite(result.hidden).all())
        tracker_manager.detach(model.halting_model)

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

    def test_recurrent_attention_residual_preserves_individually_halted_rows(self):
        dim = 2
        model = RecurrentLayer(
            self.recurrent_config(
                dim=dim,
                max_steps=4,
                block_config=self.layer_block_config(increment=1.0),
                residual_connection_option=AttentionResidualConfig,
                halting_config=self.halting_config(
                    dim=dim,
                    gate_threshold=2.0,
                    high_logit=20.0,
                    low_logit=-20.0,
                ),
            )
        )
        model.eval()
        hidden = torch.tensor([[0.0, 0.0], [10.0, 10.0]])

        result = model(LayerState(hidden=hidden))

        self.assertEqual(model.block_model.model.call_count, model.max_steps)
        torch.testing.assert_close(
            result.hidden[1],
            torch.tensor([10.5, 10.5]),
        )
        self.assertTrue(torch.isfinite(result.hidden).all())
        self.assertGreater(result.hidden[0, 0].item(), hidden[0, 0].item())

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
                residual_connection_option=AdditiveResidualConfig,
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
        class VectorLossHalting:
            def __init__(self, loss: torch.Tensor):
                self.loss = loss
                self.update_calls = 0
                self.finalize_calls = 0
                self.finalize_input = None

            def update_halting_state(
                self,
                previous_state,
                model_hidden_state,
            ):
                self.update_calls += 1
                leading_shape = model_hidden_state.shape[:-1]
                halt_mask = torch.zeros(
                    model_hidden_state.shape[:-1],
                    dtype=torch.bool,
                    device=model_hidden_state.device,
                )
                state = HaltingStateBase()
                state.output_hidden = model_hidden_state
                state.accumulated_hidden = torch.zeros_like(model_hidden_state)
                state.continuation_probability = model_hidden_state.new_ones(
                    leading_shape
                )
                state.halt_mask = halt_mask
                state.valid_mask = torch.ones_like(halt_mask)
                state.advanced_mask = state.valid_mask.clone()
                state.step_indices = model_hidden_state.new_zeros(leading_shape)
                return state, model_hidden_state

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
