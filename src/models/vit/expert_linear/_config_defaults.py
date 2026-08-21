from dataclasses import dataclass
from enum import Enum, auto
from types import ModuleType

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    ResidualConfig,
)
from models.vit.expert_linear.runtime_options import (
    DynamicMemoryOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    SubmoduleStackSource,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
    VitOutputOptions,
    VitPatchOptions,
    resolve_experts_submodule_stack_options,
)


class LinearRole(Enum):
    MAIN = auto()
    ATTENTION = auto()
    FEED_FORWARD = auto()


class ExpertStackRole(Enum):
    MAIN = auto()
    EXPERT = auto()
    ROUTER = auto()


class ExpertControlRole(Enum):
    MAIN = auto()
    EXPERT = auto()


class _ControllerStackRole(Enum):
    GATE = auto()
    HALTING = auto()
    MEMORY = auto()
    RECURRENT_GATE = auto()
    RECURRENT_HALTING = auto()


@dataclass(frozen=True, slots=True)
class _MainStackDefaults:
    bias_flag: bool
    layer_norm_position: LayerNormPositionOptions
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: type[ResidualConfig]
    residual_model_flag: bool
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool


@dataclass(frozen=True, slots=True)
class _MainControllerStackDefaults:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    dropout_probability: float | None
    bias_flag: bool | None


def vit_patch_options(config: ModuleType) -> VitPatchOptions:
    return VitPatchOptions(
        patch_size=config.IMAGE_PATCH_SIZE,
        input_channels=config.INPUT_CHANNELS,
        image_height=config.IMAGE_HEIGHT,
        dropout_probability=config.PATCH_DROPOUT_PROBABILITY,
        bias_flag=config.PATCH_BIAS_FLAG,
    )


def vit_positional_embedding_options(
    config: ModuleType,
) -> TransformerPositionalEmbeddingOptions:
    return TransformerPositionalEmbeddingOptions(
        option=config.POSITIONAL_EMBEDDING_OPTION,
        padding_idx=config.POSITIONAL_EMBEDDING_PADDING_IDX,
        auto_expand_flag=config.POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG,
    )


def vit_encoder_options(config: ModuleType) -> TransformerEncoderOptions:
    return TransformerEncoderOptions(
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.STACK_NUM_LAYERS,
        activation=config.STACK_ACTIVATION,
        dropout_probability=config.STACK_DROPOUT_PROBABILITY,
        layer_norm_position=config.LAYER_NORM_POSITION,
        causal_attention_mask_flag=False,
    )


def vit_attention_options(config: ModuleType) -> TransformerAttentionOptions:
    return TransformerAttentionOptions(
        num_heads=config.ATTN_NUM_HEADS,
        num_layers=config.ATTN_NUM_LAYERS,
        bias_flag=config.ATTN_BIAS_FLAG,
        add_key_value_bias_flag=config.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
    )


def vit_feed_forward_options(config: ModuleType) -> TransformerFeedForwardOptions:
    return TransformerFeedForwardOptions(
        num_layers=config.FF_NUM_LAYERS,
        bias_flag=config.FF_BIAS_FLAG,
    )


def vit_output_options(config: ModuleType) -> VitOutputOptions:
    return VitOutputOptions(bias_flag=config.OUTPUT_BIAS_FLAG)


def main_layer_stack_options(config: ModuleType) -> MainLayerStackOptions:
    defaults = _main_stack_defaults(config)
    return MainLayerStackOptions(
        bias_flag=defaults.bias_flag,
        layer_norm_position=defaults.layer_norm_position,
        num_layers=defaults.num_layers,
        activation=defaults.activation,
        residual_connection_option=defaults.residual_connection_option,
        residual_model_flag=defaults.residual_model_flag,
        dropout_probability=defaults.dropout_probability,
        last_layer_bias_option=defaults.last_layer_bias_option,
        apply_output_postprocessing_flag=defaults.apply_output_postprocessing_flag,
    )


def _main_stack_defaults(config: ModuleType) -> _MainStackDefaults:
    return _MainStackDefaults(
        bias_flag=config.STACK_BIAS_FLAG,
        layer_norm_position=config.LAYER_NORM_POSITION,
        num_layers=config.STACK_NUM_LAYERS,
        activation=config.STACK_ACTIVATION,
        residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.STACK_DROPOUT_PROBABILITY,
        last_layer_bias_option=config.STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_postprocessing_flag=config.STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    )


def linears_submodule_stack_options(
    config: ModuleType,
    role: LinearRole,
) -> SubmoduleStackOptions:
    if role is LinearRole.MAIN:
        return SubmoduleStackOptions(
            hidden_dim=config.SUBMODULE_STACK_HIDDEN_DIM,
            num_layers=config.SUBMODULE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=(
                config.SUBMODULE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.SUBMODULE_STACK_ACTIVATION,
            layer_norm_position=config.SUBMODULE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.SUBMODULE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.SUBMODULE_STACK_BIAS_FLAG,
        )
    if role is LinearRole.ATTENTION:
        return SubmoduleStackOptions(
            hidden_dim=config.ATTN_STACK_HIDDEN_DIM,
            num_layers=config.ATTN_NUM_LAYERS,
            last_layer_bias_option=config.ATTN_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.ATTN_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.ATTN_STACK_ACTIVATION,
            layer_norm_position=config.ATTN_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.ATTN_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ATTN_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ATTN_BIAS_FLAG,
        )
    return SubmoduleStackOptions(
        hidden_dim=config.FF_STACK_HIDDEN_DIM,
        num_layers=config.FF_NUM_LAYERS,
        last_layer_bias_option=config.FF_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_postprocessing_flag=config.FF_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
        activation=config.FF_STACK_ACTIVATION,
        layer_norm_position=config.FF_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.FF_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.FF_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.FF_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.FF_BIAS_FLAG,
    )


def _controller_stack_source(
    config: ModuleType,
    role: LinearRole,
    stack_role: _ControllerStackRole,
) -> SubmoduleStackSource:
    if role is LinearRole.MAIN:
        return _main_controller_stack_source(config, stack_role)
    if role is LinearRole.ATTENTION:
        return _attention_controller_stack_source(config, stack_role)
    return _feed_forward_controller_stack_source(config, stack_role)


def _main_controller_stack_source(
    config: ModuleType,
    stack_role: _ControllerStackRole,
) -> SubmoduleStackSource:
    defaults = _main_controller_stack_defaults(config, stack_role)
    return SubmoduleStackSource(
        independent_flag=defaults.independent_flag,
        hidden_dim=defaults.hidden_dim,
        num_layers=defaults.num_layers,
        last_layer_bias_option=defaults.last_layer_bias_option,
        apply_output_postprocessing_flag=defaults.apply_output_postprocessing_flag,
        activation=defaults.activation,
        layer_norm_position=defaults.layer_norm_position,
        residual_connection_option=defaults.residual_connection_option,
        residual_model_flag=defaults.residual_model_flag,
        dropout_probability=defaults.dropout_probability,
        bias_flag=defaults.bias_flag,
    )


def _main_controller_stack_defaults(
    config: ModuleType,
    stack_role: _ControllerStackRole,
) -> _MainControllerStackDefaults:
    if stack_role is _ControllerStackRole.GATE:
        return _MainControllerStackDefaults(
            independent_flag=config.GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.GATE_STACK_HIDDEN_DIM,
            num_layers=config.GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.GATE_STACK_ACTIVATION,
            layer_norm_position=config.GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.GATE_STACK_BIAS_FLAG,
        )
    if stack_role is _ControllerStackRole.HALTING:
        return _MainControllerStackDefaults(
            independent_flag=config.HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.HALTING_STACK_HIDDEN_DIM,
            num_layers=config.HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=(
                config.HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.HALTING_STACK_ACTIVATION,
            layer_norm_position=config.HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.HALTING_STACK_BIAS_FLAG,
        )
    if stack_role is _ControllerStackRole.MEMORY:
        return _MainControllerStackDefaults(
            independent_flag=config.MEMORY_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.MEMORY_STACK_HIDDEN_DIM,
            num_layers=config.MEMORY_STACK_NUM_LAYERS,
            last_layer_bias_option=config.MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.MEMORY_STACK_ACTIVATION,
            layer_norm_position=config.MEMORY_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.MEMORY_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.MEMORY_STACK_BIAS_FLAG,
        )
    if stack_role is _ControllerStackRole.RECURRENT_GATE:
        return _MainControllerStackDefaults(
            independent_flag=config.RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION),
            apply_output_postprocessing_flag=(
                config.RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=config.RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.RECURRENT_GATE_STACK_BIAS_FLAG,
        )
    return _MainControllerStackDefaults(
        independent_flag=config.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        hidden_dim=config.RECURRENT_HALTING_STACK_HIDDEN_DIM,
        num_layers=config.RECURRENT_HALTING_STACK_NUM_LAYERS,
        last_layer_bias_option=(config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION),
        apply_output_postprocessing_flag=(
            config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
        ),
        activation=config.RECURRENT_HALTING_STACK_ACTIVATION,
        layer_norm_position=config.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config.RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.RECURRENT_HALTING_STACK_BIAS_FLAG,
    )


def _attention_controller_stack_source(
    config: ModuleType,
    stack_role: _ControllerStackRole,
) -> SubmoduleStackSource:
    if stack_role is _ControllerStackRole.GATE:
        return SubmoduleStackSource(
            independent_flag=config.ATTN_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_GATE_STACK_HIDDEN_DIM,
            num_layers=config.ATTN_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ATTN_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=(
                config.ATTN_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.ATTN_GATE_STACK_ACTIVATION,
            layer_norm_position=config.ATTN_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.ATTN_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.ATTN_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ATTN_GATE_STACK_BIAS_FLAG,
        )
    if stack_role is _ControllerStackRole.HALTING:
        return SubmoduleStackSource(
            independent_flag=config.ATTN_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.ATTN_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ATTN_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=(
                config.ATTN_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.ATTN_HALTING_STACK_ACTIVATION,
            layer_norm_position=config.ATTN_HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.ATTN_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.ATTN_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ATTN_HALTING_STACK_BIAS_FLAG,
        )
    if stack_role is _ControllerStackRole.MEMORY:
        return SubmoduleStackSource(
            independent_flag=config.ATTN_MEMORY_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_MEMORY_STACK_HIDDEN_DIM,
            num_layers=config.ATTN_MEMORY_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ATTN_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=(
                config.ATTN_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.ATTN_MEMORY_STACK_ACTIVATION,
            layer_norm_position=config.ATTN_MEMORY_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.ATTN_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.ATTN_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_MEMORY_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ATTN_MEMORY_STACK_BIAS_FLAG,
        )
    if stack_role is _ControllerStackRole.RECURRENT_GATE:
        return SubmoduleStackSource(
            independent_flag=config.ATTN_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.ATTN_RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.ATTN_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config.ATTN_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.ATTN_RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=config.ATTN_RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.ATTN_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.ATTN_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=(config.ATTN_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY),
            bias_flag=config.ATTN_RECURRENT_GATE_STACK_BIAS_FLAG,
        )
    return SubmoduleStackSource(
        independent_flag=config.ATTN_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        hidden_dim=config.ATTN_RECURRENT_HALTING_STACK_HIDDEN_DIM,
        num_layers=config.ATTN_RECURRENT_HALTING_STACK_NUM_LAYERS,
        last_layer_bias_option=(
            config.ATTN_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
        ),
        apply_output_postprocessing_flag=(
            config.ATTN_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
        ),
        activation=config.ATTN_RECURRENT_HALTING_STACK_ACTIVATION,
        layer_norm_position=config.ATTN_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config.ATTN_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config.ATTN_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.ATTN_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.ATTN_RECURRENT_HALTING_STACK_BIAS_FLAG,
    )


def _feed_forward_controller_stack_source(
    config: ModuleType,
    stack_role: _ControllerStackRole,
) -> SubmoduleStackSource:
    if stack_role is _ControllerStackRole.GATE:
        return SubmoduleStackSource(
            independent_flag=config.FF_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_GATE_STACK_HIDDEN_DIM,
            num_layers=config.FF_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.FF_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=(
                config.FF_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.FF_GATE_STACK_ACTIVATION,
            layer_norm_position=config.FF_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.FF_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.FF_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.FF_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.FF_GATE_STACK_BIAS_FLAG,
        )
    if stack_role is _ControllerStackRole.HALTING:
        return SubmoduleStackSource(
            independent_flag=config.FF_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.FF_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=config.FF_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=(
                config.FF_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.FF_HALTING_STACK_ACTIVATION,
            layer_norm_position=config.FF_HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.FF_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.FF_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.FF_HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.FF_HALTING_STACK_BIAS_FLAG,
        )
    if stack_role is _ControllerStackRole.MEMORY:
        return SubmoduleStackSource(
            independent_flag=config.FF_MEMORY_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_MEMORY_STACK_HIDDEN_DIM,
            num_layers=config.FF_MEMORY_STACK_NUM_LAYERS,
            last_layer_bias_option=config.FF_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=(
                config.FF_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.FF_MEMORY_STACK_ACTIVATION,
            layer_norm_position=config.FF_MEMORY_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.FF_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.FF_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.FF_MEMORY_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.FF_MEMORY_STACK_BIAS_FLAG,
        )
    if stack_role is _ControllerStackRole.RECURRENT_GATE:
        return SubmoduleStackSource(
            independent_flag=config.FF_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.FF_RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.FF_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config.FF_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.FF_RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=config.FF_RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.FF_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.FF_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=(config.FF_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY),
            bias_flag=config.FF_RECURRENT_GATE_STACK_BIAS_FLAG,
        )
    return SubmoduleStackSource(
        independent_flag=config.FF_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        hidden_dim=config.FF_RECURRENT_HALTING_STACK_HIDDEN_DIM,
        num_layers=config.FF_RECURRENT_HALTING_STACK_NUM_LAYERS,
        last_layer_bias_option=(
            config.FF_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
        ),
        apply_output_postprocessing_flag=(
            config.FF_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
        ),
        activation=config.FF_RECURRENT_HALTING_STACK_ACTIVATION,
        layer_norm_position=config.FF_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config.FF_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config.FF_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.FF_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.FF_RECURRENT_HALTING_STACK_BIAS_FLAG,
    )


def linears_layer_controller_options(
    config: ModuleType,
    role: LinearRole,
) -> LayerControllerOptions:
    if role is LinearRole.MAIN:
        return LayerControllerOptions(
            stack_gate_flag=config.STACK_GATE_FLAG,
            gate_option=config.GATE_OPTION,
            gate_activation=config.GATE_ACTIVATION,
            gate_stack_source=_controller_stack_source(
                config, role, _ControllerStackRole.GATE
            ),
            stack_halting_flag=config.STACK_HALTING_FLAG,
            halting_option=config.HALTING_OPTION,
            halting_threshold=config.HALTING_THRESHOLD,
            halting_dropout=config.HALTING_DROPOUT,
            halting_hidden_state_mode=config.HALTING_HIDDEN_STATE_MODE,
            halting_stack_source=_controller_stack_source(
                config, role, _ControllerStackRole.HALTING
            ),
        )
    if role is LinearRole.ATTENTION:
        return LayerControllerOptions(
            stack_gate_flag=config.ATTN_STACK_GATE_FLAG,
            gate_option=config.ATTN_GATE_OPTION,
            gate_activation=config.ATTN_GATE_ACTIVATION,
            gate_stack_source=_controller_stack_source(
                config, role, _ControllerStackRole.GATE
            ),
            stack_halting_flag=config.ATTN_STACK_HALTING_FLAG,
            halting_option=config.ATTN_HALTING_OPTION,
            halting_threshold=config.ATTN_HALTING_THRESHOLD,
            halting_dropout=config.ATTN_HALTING_DROPOUT,
            halting_hidden_state_mode=config.ATTN_HALTING_HIDDEN_STATE_MODE,
            halting_stack_source=_controller_stack_source(
                config, role, _ControllerStackRole.HALTING
            ),
        )
    return LayerControllerOptions(
        stack_gate_flag=config.FF_STACK_GATE_FLAG,
        gate_option=config.FF_GATE_OPTION,
        gate_activation=config.FF_GATE_ACTIVATION,
        gate_stack_source=_controller_stack_source(
            config, role, _ControllerStackRole.GATE
        ),
        stack_halting_flag=config.FF_STACK_HALTING_FLAG,
        halting_option=config.FF_HALTING_OPTION,
        halting_threshold=config.FF_HALTING_THRESHOLD,
        halting_dropout=config.FF_HALTING_DROPOUT,
        halting_hidden_state_mode=config.FF_HALTING_HIDDEN_STATE_MODE,
        halting_stack_source=_controller_stack_source(
            config, role, _ControllerStackRole.HALTING
        ),
    )


def linears_dynamic_memory_options(
    config: ModuleType,
    role: LinearRole,
) -> DynamicMemoryOptions:
    if role is LinearRole.MAIN:
        return DynamicMemoryOptions(
            memory_flag=config.MEMORY_FLAG,
            memory_option=config.MEMORY_OPTION,
            memory_position_option=config.MEMORY_POSITION_OPTION,
            memory_test_time_training_learning_rate=(
                config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
            ),
            memory_test_time_training_num_inner_steps=(
                config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
            ),
            memory_stack_source=_controller_stack_source(
                config, role, _ControllerStackRole.MEMORY
            ),
        )
    if role is LinearRole.ATTENTION:
        return DynamicMemoryOptions(
            memory_flag=config.ATTN_MEMORY_FLAG,
            memory_option=config.ATTN_MEMORY_OPTION,
            memory_position_option=config.ATTN_MEMORY_POSITION_OPTION,
            memory_test_time_training_learning_rate=(
                config.ATTN_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
            ),
            memory_test_time_training_num_inner_steps=(
                config.ATTN_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
            ),
            memory_stack_source=_controller_stack_source(
                config, role, _ControllerStackRole.MEMORY
            ),
        )
    return DynamicMemoryOptions(
        memory_flag=config.FF_MEMORY_FLAG,
        memory_option=config.FF_MEMORY_OPTION,
        memory_position_option=config.FF_MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate=(
            config.FF_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        ),
        memory_test_time_training_num_inner_steps=(
            config.FF_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
        ),
        memory_stack_source=_controller_stack_source(
            config, role, _ControllerStackRole.MEMORY
        ),
    )


def linears_recurrent_controller_options(
    config: ModuleType,
    role: LinearRole,
) -> RecurrentControllerOptions:
    if role is LinearRole.MAIN:
        return RecurrentControllerOptions(
            recurrent_flag=config.RECURRENT_FLAG,
            recurrent_max_steps=config.RECURRENT_MAX_STEPS,
            recurrent_initial_iterations=config.RECURRENT_INITIAL_ITERATIONS,
            recurrent_gradient_transition_count=(
                config.RECURRENT_GRADIENT_TRANSITION_COUNT
            ),
            recurrent_no_gradient_transition_count=(
                config.RECURRENT_NO_GRADIENT_TRANSITION_COUNT
            ),
            recurrent_iteration_increment=config.RECURRENT_ITERATION_INCREMENT,
            recurrent_forward_calls_before_iteration_increment=(
                config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT
            ),
            recurrent_smooth_iteration_growth_flag=(
                config.RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG
            ),
            recurrent_layer_norm_position=config.RECURRENT_LAYER_NORM_POSITION,
            recurrent_stack_gate_flag=config.RECURRENT_STACK_GATE_FLAG,
            recurrent_gate_option=config.RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=_controller_stack_source(
                config, role, _ControllerStackRole.RECURRENT_GATE
            ),
            recurrent_stack_halting_flag=config.RECURRENT_STACK_HALTING_FLAG,
            recurrent_halting_option=config.RECURRENT_HALTING_OPTION,
            recurrent_halting_threshold=config.RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=(
                config.RECURRENT_HALTING_HIDDEN_STATE_MODE
            ),
            recurrent_halting_stack_source=_controller_stack_source(
                config, role, _ControllerStackRole.RECURRENT_HALTING
            ),
        )
    if role is LinearRole.ATTENTION:
        return RecurrentControllerOptions(
            recurrent_flag=config.ATTN_RECURRENT_FLAG,
            recurrent_max_steps=config.ATTN_RECURRENT_MAX_STEPS,
            recurrent_initial_iterations=2,
            recurrent_gradient_transition_count=None,
            recurrent_no_gradient_transition_count=None,
            recurrent_iteration_increment=1,
            recurrent_forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=config.ATTN_RECURRENT_LAYER_NORM_POSITION,
            recurrent_stack_gate_flag=config.ATTN_RECURRENT_STACK_GATE_FLAG,
            recurrent_gate_option=config.ATTN_RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.ATTN_RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=_controller_stack_source(
                config, role, _ControllerStackRole.RECURRENT_GATE
            ),
            recurrent_stack_halting_flag=config.ATTN_RECURRENT_STACK_HALTING_FLAG,
            recurrent_halting_option=config.ATTN_RECURRENT_HALTING_OPTION,
            recurrent_halting_threshold=config.ATTN_RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.ATTN_RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=(
                config.ATTN_RECURRENT_HALTING_HIDDEN_STATE_MODE
            ),
            recurrent_halting_stack_source=_controller_stack_source(
                config, role, _ControllerStackRole.RECURRENT_HALTING
            ),
        )
    return RecurrentControllerOptions(
        recurrent_flag=config.FF_RECURRENT_FLAG,
        recurrent_max_steps=config.FF_RECURRENT_MAX_STEPS,
        recurrent_initial_iterations=2,
        recurrent_gradient_transition_count=None,
        recurrent_no_gradient_transition_count=None,
        recurrent_iteration_increment=1,
        recurrent_forward_calls_before_iteration_increment=1,
        recurrent_layer_norm_position=config.FF_RECURRENT_LAYER_NORM_POSITION,
        recurrent_stack_gate_flag=config.FF_RECURRENT_STACK_GATE_FLAG,
        recurrent_gate_option=config.FF_RECURRENT_GATE_OPTION,
        recurrent_gate_activation=config.FF_RECURRENT_GATE_ACTIVATION,
        recurrent_gate_stack_source=_controller_stack_source(
            config, role, _ControllerStackRole.RECURRENT_GATE
        ),
        recurrent_stack_halting_flag=config.FF_RECURRENT_STACK_HALTING_FLAG,
        recurrent_halting_option=config.FF_RECURRENT_HALTING_OPTION,
        recurrent_halting_threshold=config.FF_RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout=config.FF_RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode=(
            config.FF_RECURRENT_HALTING_HIDDEN_STATE_MODE
        ),
        recurrent_halting_stack_source=_controller_stack_source(
            config, role, _ControllerStackRole.RECURRENT_HALTING
        ),
    )


def experts_submodule_stack_options(
    config: ModuleType, role: ExpertStackRole
) -> ExpertsSubmoduleStackOptions:
    if role is ExpertStackRole.MAIN:
        return ExpertsSubmoduleStackOptions(
            hidden_dim=config.SUBMODULE_STACK_HIDDEN_DIM,
            num_layers=config.SUBMODULE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=(
                config.SUBMODULE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.SUBMODULE_STACK_ACTIVATION,
            layer_norm_position=config.SUBMODULE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.SUBMODULE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.SUBMODULE_STACK_BIAS_FLAG,
        )
    if role is ExpertStackRole.EXPERT:
        return ExpertsSubmoduleStackOptions(
            hidden_dim=config.EXPERT_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_STACK_NUM_LAYERS,
            last_layer_bias_option=config.EXPERT_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.EXPERT_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.EXPERT_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.EXPERT_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.EXPERT_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_BIAS_FLAG,
        )
    return ExpertsSubmoduleStackOptions(
        hidden_dim=config.ROUTER_STACK_HIDDEN_DIM,
        num_layers=config.ROUTER_STACK_NUM_LAYERS,
        last_layer_bias_option=config.ROUTER_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_postprocessing_flag=config.ROUTER_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
        activation=config.ROUTER_STACK_ACTIVATION,
        layer_norm_position=config.ROUTER_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.ROUTER_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.ROUTER_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.ROUTER_BIAS_FLAG,
    )


def experts_stack_options(
    config: ModuleType, *, hidden_dim: int
) -> ExpertsStackOptions:
    defaults = _main_stack_defaults(config)
    return ExpertsStackOptions(
        hidden_dim=hidden_dim,
        bias_flag=defaults.bias_flag,
        layer_norm_position=defaults.layer_norm_position,
        num_layers=defaults.num_layers,
        activation=defaults.activation,
        residual_connection_option=defaults.residual_connection_option,
        residual_model_flag=defaults.residual_model_flag,
        dropout_probability=defaults.dropout_probability,
        last_layer_bias_option=defaults.last_layer_bias_option,
        apply_output_postprocessing_flag=defaults.apply_output_postprocessing_flag,
    )


def experts_expert_stack_options(
    config: ModuleType,
    defaults: ExpertsSubmoduleStackOptions,
) -> ExpertsSubmoduleStackOptions:
    return resolve_experts_submodule_stack_options(
        defaults,
        layer_norm_position=config.EXPERT_STACK_LAYER_NORM_POSITION,
        apply_output_postprocessing_flag=config.EXPERT_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    )


def experts_submodule_stack_source(
    config: ModuleType,
    role: ExpertControlRole,
    stack_role: _ControllerStackRole,
) -> ExpertsSubmoduleStackSource:
    if role is ExpertControlRole.MAIN:
        return _main_experts_submodule_stack_source(config, stack_role)
    return _expert_submodule_stack_source(config, stack_role)


def _main_experts_submodule_stack_source(
    config: ModuleType, stack_role: _ControllerStackRole
) -> ExpertsSubmoduleStackSource:
    defaults = _main_controller_stack_defaults(config, stack_role)
    return ExpertsSubmoduleStackSource(
        independent_flag=defaults.independent_flag,
        hidden_dim=defaults.hidden_dim,
        num_layers=defaults.num_layers,
        last_layer_bias_option=defaults.last_layer_bias_option,
        apply_output_postprocessing_flag=defaults.apply_output_postprocessing_flag,
        activation=defaults.activation,
        layer_norm_position=defaults.layer_norm_position,
        residual_connection_option=defaults.residual_connection_option,
        residual_model_flag=defaults.residual_model_flag,
        dropout_probability=defaults.dropout_probability,
        bias_flag=defaults.bias_flag,
    )


def _expert_submodule_stack_source(
    config: ModuleType, stack_role: _ControllerStackRole
) -> ExpertsSubmoduleStackSource:
    if stack_role is _ControllerStackRole.GATE:
        return ExpertsSubmoduleStackSource(
            independent_flag=config.EXPERT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.EXPERT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=(
                config.EXPERT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.EXPERT_GATE_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.EXPERT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.EXPERT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_GATE_STACK_BIAS_FLAG,
        )
    if stack_role is _ControllerStackRole.HALTING:
        return ExpertsSubmoduleStackSource(
            independent_flag=config.EXPERT_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(config.EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION),
            apply_output_postprocessing_flag=(
                config.EXPERT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.EXPERT_HALTING_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.EXPERT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.EXPERT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_HALTING_STACK_BIAS_FLAG,
        )
    if stack_role is _ControllerStackRole.MEMORY:
        return ExpertsSubmoduleStackSource(
            independent_flag=config.EXPERT_MEMORY_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_MEMORY_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_MEMORY_STACK_NUM_LAYERS,
            last_layer_bias_option=config.EXPERT_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=(
                config.EXPERT_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.EXPERT_MEMORY_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_MEMORY_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.EXPERT_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.EXPERT_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_MEMORY_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_MEMORY_STACK_BIAS_FLAG,
        )
    if stack_role is _ControllerStackRole.RECURRENT_GATE:
        return ExpertsSubmoduleStackSource(
            independent_flag=config.EXPERT_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.EXPERT_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config.EXPERT_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config.EXPERT_RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(
                config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config.EXPERT_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=config.EXPERT_RECURRENT_GATE_STACK_BIAS_FLAG,
        )
    return ExpertsSubmoduleStackSource(
        independent_flag=config.EXPERT_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        hidden_dim=config.EXPERT_RECURRENT_HALTING_STACK_HIDDEN_DIM,
        num_layers=config.EXPERT_RECURRENT_HALTING_STACK_NUM_LAYERS,
        last_layer_bias_option=(
            config.EXPERT_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
        ),
        apply_output_postprocessing_flag=(
            config.EXPERT_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
        ),
        activation=config.EXPERT_RECURRENT_HALTING_STACK_ACTIVATION,
        layer_norm_position=config.EXPERT_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=(config.EXPERT_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY),
        bias_flag=config.EXPERT_RECURRENT_HALTING_STACK_BIAS_FLAG,
    )


def experts_layer_controller_options(
    config: ModuleType, role: ExpertControlRole
) -> ExpertsLayerControllerOptions:
    if role is ExpertControlRole.EXPERT:
        return ExpertsLayerControllerOptions(
            stack_gate_flag=config.EXPERT_STACK_GATE_FLAG,
            gate_option=config.EXPERT_GATE_OPTION,
            gate_activation=config.EXPERT_GATE_ACTIVATION,
            gate_stack_source=experts_submodule_stack_source(
                config, role, _ControllerStackRole.GATE
            ),
            stack_halting_flag=config.EXPERT_STACK_HALTING_FLAG,
            halting_option=config.EXPERT_HALTING_OPTION,
            halting_threshold=config.EXPERT_HALTING_THRESHOLD,
            halting_dropout=config.EXPERT_HALTING_DROPOUT,
            halting_hidden_state_mode=config.EXPERT_HALTING_HIDDEN_STATE_MODE,
            halting_stack_source=experts_submodule_stack_source(
                config, role, _ControllerStackRole.HALTING
            ),
            halting_output_dim=config.EXPERT_HALTING_OUTPUT_DIM,
        )
    return ExpertsLayerControllerOptions(
        stack_gate_flag=config.STACK_GATE_FLAG,
        gate_option=config.GATE_OPTION,
        gate_activation=config.GATE_ACTIVATION,
        gate_stack_source=experts_submodule_stack_source(
            config, role, _ControllerStackRole.GATE
        ),
        stack_halting_flag=config.STACK_HALTING_FLAG,
        halting_option=config.HALTING_OPTION,
        halting_threshold=config.HALTING_THRESHOLD,
        halting_dropout=config.HALTING_DROPOUT,
        halting_hidden_state_mode=config.HALTING_HIDDEN_STATE_MODE,
        halting_stack_source=experts_submodule_stack_source(
            config, role, _ControllerStackRole.HALTING
        ),
        halting_output_dim=config.HALTING_OUTPUT_DIM,
    )


def experts_dynamic_memory_options(
    config: ModuleType, role: ExpertControlRole
) -> ExpertsDynamicMemoryOptions:
    if role is ExpertControlRole.EXPERT:
        return ExpertsDynamicMemoryOptions(
            memory_flag=config.EXPERT_MEMORY_FLAG,
            memory_option=config.EXPERT_MEMORY_OPTION,
            memory_position_option=config.EXPERT_MEMORY_POSITION_OPTION,
            memory_test_time_training_learning_rate=(
                config.EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
            ),
            memory_test_time_training_num_inner_steps=(
                config.EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
            ),
            memory_stack_source=experts_submodule_stack_source(
                config, role, _ControllerStackRole.MEMORY
            ),
        )
    return ExpertsDynamicMemoryOptions(
        memory_flag=config.MEMORY_FLAG,
        memory_option=config.MEMORY_OPTION,
        memory_position_option=config.MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate=(
            config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        ),
        memory_test_time_training_num_inner_steps=(
            config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
        ),
        memory_stack_source=experts_submodule_stack_source(
            config, role, _ControllerStackRole.MEMORY
        ),
    )


def experts_recurrent_controller_options(
    config: ModuleType, role: ExpertControlRole
) -> ExpertsRecurrentControllerOptions:
    if role is ExpertControlRole.EXPERT:
        return ExpertsRecurrentControllerOptions(
            recurrent_flag=config.EXPERT_RECURRENT_FLAG,
            recurrent_max_steps=config.EXPERT_RECURRENT_MAX_STEPS,
            recurrent_initial_iterations=2,
            recurrent_gradient_transition_count=None,
            recurrent_no_gradient_transition_count=None,
            recurrent_iteration_increment=1,
            recurrent_forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=config.EXPERT_RECURRENT_LAYER_NORM_POSITION,
            recurrent_stack_gate_flag=config.EXPERT_RECURRENT_STACK_GATE_FLAG,
            recurrent_gate_option=config.EXPERT_RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.EXPERT_RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=experts_submodule_stack_source(
                config, role, _ControllerStackRole.RECURRENT_GATE
            ),
            recurrent_stack_halting_flag=config.EXPERT_RECURRENT_STACK_HALTING_FLAG,
            recurrent_halting_option=config.EXPERT_RECURRENT_HALTING_OPTION,
            recurrent_halting_threshold=config.EXPERT_RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.EXPERT_RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=(
                config.EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE
            ),
            recurrent_halting_stack_source=experts_submodule_stack_source(
                config, role, _ControllerStackRole.RECURRENT_HALTING
            ),
        )
    return ExpertsRecurrentControllerOptions(
        recurrent_flag=config.RECURRENT_FLAG,
        recurrent_max_steps=config.RECURRENT_MAX_STEPS,
        recurrent_initial_iterations=config.RECURRENT_INITIAL_ITERATIONS,
        recurrent_gradient_transition_count=(
            config.RECURRENT_GRADIENT_TRANSITION_COUNT
        ),
        recurrent_no_gradient_transition_count=(
            config.RECURRENT_NO_GRADIENT_TRANSITION_COUNT
        ),
        recurrent_iteration_increment=config.RECURRENT_ITERATION_INCREMENT,
        recurrent_forward_calls_before_iteration_increment=(
            config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT
        ),
        recurrent_layer_norm_position=config.RECURRENT_LAYER_NORM_POSITION,
        recurrent_stack_gate_flag=config.RECURRENT_STACK_GATE_FLAG,
        recurrent_gate_option=config.RECURRENT_GATE_OPTION,
        recurrent_gate_activation=config.RECURRENT_GATE_ACTIVATION,
        recurrent_gate_stack_source=experts_submodule_stack_source(
            config, role, _ControllerStackRole.RECURRENT_GATE
        ),
        recurrent_stack_halting_flag=config.RECURRENT_STACK_HALTING_FLAG,
        recurrent_halting_option=config.RECURRENT_HALTING_OPTION,
        recurrent_halting_threshold=config.RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout=config.RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode=(
            config.RECURRENT_HALTING_HIDDEN_STATE_MODE
        ),
        recurrent_halting_stack_source=experts_submodule_stack_source(
            config, role, _ControllerStackRole.RECURRENT_HALTING
        ),
    )


def experts_mixture_options(config: ModuleType) -> ExpertsMixtureOptions:
    return ExpertsMixtureOptions(
        top_k=config.TOP_K,
        num_experts=config.NUM_EXPERTS,
        capacity_factor=config.CAPACITY_FACTOR,
        dropped_token_behavior=config.DROPPED_TOKEN_BEHAVIOR,
        compute_expert_mixture_flag=config.COMPUTE_EXPERT_MIXTURE_FLAG,
        weighted_parameters_flag=config.WEIGHTED_PARAMETERS_FLAG,
        weighting_position_option=config.WEIGHTING_POSITION_OPTION,
        routing_initialization_mode=config.ROUTING_INITIALIZATION_MODE,
    )


def experts_sampler_options(config: ModuleType) -> ExpertsSamplerOptions:
    return ExpertsSamplerOptions(
        threshold=config.SAMPLER_THRESHOLD,
        filter_above_threshold=config.SAMPLER_FILTER_ABOVE_THRESHOLD,
        num_topk_samples=config.SAMPLER_NUM_TOPK_SAMPLES,
        normalize_probabilities_flag=config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
        noisy_topk_flag=config.SAMPLER_NOISY_TOPK_FLAG,
        coefficient_of_variation_loss_weight=config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
        switch_loss_weight=config.SAMPLER_SWITCH_LOSS_WEIGHT,
        zero_centred_loss_weight=config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
        mutual_information_loss_weight=config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT,
    )


def experts_router_options(config: ModuleType) -> ExpertsRouterOptions:
    return ExpertsRouterOptions(noisy_topk_flag=config.ROUTER_NOISY_TOPK_FLAG)
