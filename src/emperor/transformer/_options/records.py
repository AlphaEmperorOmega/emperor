from __future__ import annotations

from dataclasses import dataclass, replace

from emperor.halting import HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConnectionOptions,
)
from emperor.memory import (
    DynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
)


@dataclass(frozen=True)
class TransformerStackOptions:
    """Options for the outer encoder or decoder block stack."""

    num_layers: int = 3
    layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
    stack_gate_flag: bool = False
    stack_halting_flag: bool = False
    memory_flag: bool = False
    recurrent_flag: bool = False
    recurrent_gate_flag: bool = False
    recurrent_halting_flag: bool = False
    recurrent_max_steps: int = 2
    stack_residual_connection_option: ResidualConnectionOptions | None = None
    recurrent_residual_connection_option: ResidualConnectionOptions | None = None


@dataclass(frozen=True)
class SubmoduleStackOptions:
    """Resolved options for an attention projection or feed-forward stack."""

    hidden_dim: int = 128
    num_layers: int = 1
    last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
    apply_output_pipeline_flag: bool = False
    activation: ActivationOptions = ActivationOptions.DISABLED
    layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED
    residual_connection_option: ResidualConnectionOptions | None = None
    dropout_probability: float = 0.0
    bias_flag: bool = True


@dataclass(frozen=True)
class ControllerStackOptions:
    """Optional overrides for an independently configured controller stack."""

    independent_flag: bool = False
    hidden_dim: int | None = None
    num_layers: int | None = None
    last_layer_bias_option: LastLayerBiasOptions | None = None
    apply_output_pipeline_flag: bool | None = None
    activation: ActivationOptions | None = None
    layer_norm_position: LayerNormPositionOptions | None = None
    residual_connection_option: ResidualConnectionOptions | None = None
    dropout_probability: float | None = None
    bias_flag: bool | None = None


# GPT/BERT runtime modules historically call this inheritance payload a source.
SubmoduleStackSource = ControllerStackOptions


def resolve_controller_stack_options(
    source: ControllerStackOptions,
    defaults: SubmoduleStackOptions,
) -> SubmoduleStackOptions:
    """Resolve a controller stack, inheriting the primary stack by default."""

    if not source.independent_flag:
        return defaults
    return SubmoduleStackOptions(
        hidden_dim=(
            defaults.hidden_dim if source.hidden_dim is None else source.hidden_dim
        ),
        num_layers=(
            defaults.num_layers if source.num_layers is None else source.num_layers
        ),
        last_layer_bias_option=(
            defaults.last_layer_bias_option
            if source.last_layer_bias_option is None
            else source.last_layer_bias_option
        ),
        apply_output_pipeline_flag=(
            defaults.apply_output_pipeline_flag
            if source.apply_output_pipeline_flag is None
            else source.apply_output_pipeline_flag
        ),
        activation=(
            defaults.activation if source.activation is None else source.activation
        ),
        layer_norm_position=(
            defaults.layer_norm_position
            if source.layer_norm_position is None
            else source.layer_norm_position
        ),
        residual_connection_option=(
            defaults.residual_connection_option
            if source.residual_connection_option is None
            else source.residual_connection_option
        ),
        dropout_probability=(
            defaults.dropout_probability
            if source.dropout_probability is None
            else source.dropout_probability
        ),
        bias_flag=(
            defaults.bias_flag if source.bias_flag is None else source.bias_flag
        ),
    )


@dataclass(frozen=True)
class LayerControllerOptions:
    stack_gate_flag: bool = False
    gate_option: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
    gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID
    gate_stack_options: ControllerStackOptions = ControllerStackOptions()
    stack_halting_flag: bool = False
    halting_threshold: float = 0.99
    halting_dropout: float = 0.0
    halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        HaltingHiddenStateModeOptions.RAW
    )
    halting_stack_options: ControllerStackOptions = ControllerStackOptions()
    shared_gate_config: GateConfig | None = None

    @property
    def gate_stack_source(self) -> ControllerStackOptions:
        return self.gate_stack_options

    @property
    def halting_stack_source(self) -> ControllerStackOptions:
        return self.halting_stack_options


@dataclass(frozen=True)
class DynamicMemoryOptions:
    memory_flag: bool = False
    memory_option: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
    memory_position_option: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
    memory_test_time_training_learning_rate: float | None = None
    memory_test_time_training_num_inner_steps: int | None = None
    memory_stack_options: ControllerStackOptions = ControllerStackOptions()

    @property
    def memory_stack_source(self) -> ControllerStackOptions:
        return self.memory_stack_options


@dataclass(frozen=True)
class RecurrentControllerOptions:
    recurrent_flag: bool = False
    recurrent_max_steps: int = 2
    recurrent_layer_norm_position: LayerNormPositionOptions = (
        LayerNormPositionOptions.DISABLED
    )
    recurrent_gate_flag: bool = False
    recurrent_gate_option: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
    recurrent_gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID
    recurrent_gate_stack_options: ControllerStackOptions = ControllerStackOptions()
    recurrent_halting_flag: bool = False
    recurrent_halting_threshold: float = 0.99
    recurrent_halting_dropout: float = 0.0
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        HaltingHiddenStateModeOptions.RAW
    )
    recurrent_halting_stack_options: ControllerStackOptions = ControllerStackOptions()

    @property
    def recurrent_gate_stack_source(self) -> ControllerStackOptions:
        return self.recurrent_gate_stack_options

    @property
    def recurrent_halting_stack_source(self) -> ControllerStackOptions:
        return self.recurrent_halting_stack_options


_UNSET = object()


@dataclass(frozen=True, init=False)
class TransformerAttentionOptions:
    """All options for one Transformer attention path.

    The four original constructor arguments remain supported. ``num_layers`` is
    accepted as the GPT-style projection-depth argument, while the complete
    resolved stack is available through ``stack_options``.
    """

    num_heads: int
    projection_bias_flag: bool
    add_key_value_bias_flag: bool
    zero_attention_flag: bool
    num_layers: int
    stack_options: SubmoduleStackOptions
    layer_controller_options: LayerControllerOptions
    dynamic_memory_options: DynamicMemoryOptions
    recurrent_controller_options: RecurrentControllerOptions

    def __init__(
        self,
        num_heads: int = 4,
        projection_bias_flag: bool | object = _UNSET,
        add_key_value_bias_flag: bool = False,
        zero_attention_flag: bool = False,
        *,
        num_layers: int | object = _UNSET,
        stack_options: SubmoduleStackOptions | None = None,
        layer_controller_options: LayerControllerOptions | None = None,
        dynamic_memory_options: DynamicMemoryOptions | None = None,
        recurrent_controller_options: RecurrentControllerOptions | None = None,
    ) -> None:
        stack = stack_options or SubmoduleStackOptions()
        if projection_bias_flag is not _UNSET:
            stack = replace(stack, bias_flag=projection_bias_flag)
        if num_layers is not _UNSET:
            stack = replace(stack, num_layers=num_layers)
        object.__setattr__(self, "num_heads", num_heads)
        object.__setattr__(self, "projection_bias_flag", stack.bias_flag)
        object.__setattr__(self, "add_key_value_bias_flag", add_key_value_bias_flag)
        object.__setattr__(self, "zero_attention_flag", zero_attention_flag)
        object.__setattr__(self, "num_layers", stack.num_layers)
        object.__setattr__(self, "stack_options", stack)
        object.__setattr__(
            self,
            "layer_controller_options",
            layer_controller_options or LayerControllerOptions(),
        )
        object.__setattr__(
            self,
            "dynamic_memory_options",
            dynamic_memory_options or DynamicMemoryOptions(),
        )
        object.__setattr__(
            self,
            "recurrent_controller_options",
            recurrent_controller_options or RecurrentControllerOptions(),
        )

    @property
    def projection_stack_options(self) -> SubmoduleStackOptions:
        return self.stack_options

    @property
    def projection_layer_controller_options(self) -> LayerControllerOptions:
        return self.layer_controller_options

    @property
    def projection_dynamic_memory_options(self) -> DynamicMemoryOptions:
        return self.dynamic_memory_options

    @property
    def projection_recurrent_controller_options(
        self,
    ) -> RecurrentControllerOptions:
        return self.recurrent_controller_options


@dataclass(frozen=True, init=False)
class TransformerFeedForwardOptions:
    """All options for one Transformer feed-forward path."""

    hidden_dim: int
    num_layers: int
    bias_flag: bool
    stack_options: SubmoduleStackOptions
    layer_controller_options: LayerControllerOptions
    dynamic_memory_options: DynamicMemoryOptions
    recurrent_controller_options: RecurrentControllerOptions

    def __init__(
        self,
        hidden_dim: int | object = _UNSET,
        num_layers: int | object = _UNSET,
        *,
        bias_flag: bool | object = _UNSET,
        stack_options: SubmoduleStackOptions | None = None,
        layer_controller_options: LayerControllerOptions | None = None,
        dynamic_memory_options: DynamicMemoryOptions | None = None,
        recurrent_controller_options: RecurrentControllerOptions | None = None,
    ) -> None:
        stack = stack_options or SubmoduleStackOptions(
            hidden_dim=512,
            num_layers=2,
            activation=ActivationOptions.RELU,
            dropout_probability=0.1,
        )
        if hidden_dim is not _UNSET:
            stack = replace(stack, hidden_dim=hidden_dim)
        if num_layers is not _UNSET:
            stack = replace(stack, num_layers=num_layers)
        if bias_flag is not _UNSET:
            stack = replace(stack, bias_flag=bias_flag)
        object.__setattr__(self, "hidden_dim", stack.hidden_dim)
        object.__setattr__(self, "num_layers", stack.num_layers)
        object.__setattr__(self, "bias_flag", stack.bias_flag)
        object.__setattr__(self, "stack_options", stack)
        object.__setattr__(
            self,
            "layer_controller_options",
            layer_controller_options or LayerControllerOptions(),
        )
        object.__setattr__(
            self,
            "dynamic_memory_options",
            dynamic_memory_options or DynamicMemoryOptions(),
        )
        object.__setattr__(
            self,
            "recurrent_controller_options",
            recurrent_controller_options or RecurrentControllerOptions(),
        )


@dataclass(frozen=True)
class TransformerPathOptions:
    encoder_attention_options: TransformerAttentionOptions
    decoder_self_attention_options: TransformerAttentionOptions
    decoder_cross_attention_options: TransformerAttentionOptions
    encoder_feed_forward_options: TransformerFeedForwardOptions
    decoder_feed_forward_options: TransformerFeedForwardOptions
