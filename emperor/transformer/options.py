from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Any

from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.memory.config import DynamicMemoryConfig, GatedResidualDynamicMemoryConfig
from emperor.memory.options import MemoryPositionOptions


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
    stack_residual_connection_option: ResidualConnectionOptions = (
        ResidualConnectionOptions.DISABLED
    )
    recurrent_residual_connection_option: ResidualConnectionOptions = (
        ResidualConnectionOptions.DISABLED
    )


@dataclass(frozen=True)
class SubmoduleStackOptions:
    """Resolved options for an attention projection or feed-forward stack."""

    hidden_dim: int = 128
    num_layers: int = 1
    last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
    apply_output_pipeline_flag: bool = False
    activation: ActivationOptions = ActivationOptions.DISABLED
    layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED
    residual_connection_option: ResidualConnectionOptions = (
        ResidualConnectionOptions.DISABLED
    )
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


_STACK_FIELDS = (
    "hidden_dim",
    "num_layers",
    "last_layer_bias_option",
    "apply_output_pipeline_flag",
    "activation",
    "layer_norm_position",
    "residual_connection_option",
    "dropout_probability",
    "bias_flag",
)
_CONTROLLER_STACK_FIELDS = ("independent_flag", *_STACK_FIELDS)


def _controller_stack_from_config(
    config: ModuleType,
    prefix: str,
) -> ControllerStackOptions:
    return ControllerStackOptions(
        independent_flag=getattr(config, f"{prefix}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config, f"{prefix}_HIDDEN_DIM"),
        num_layers=getattr(config, f"{prefix}_NUM_LAYERS"),
        last_layer_bias_option=getattr(config, f"{prefix}_LAST_LAYER_BIAS_OPTION"),
        apply_output_pipeline_flag=getattr(
            config, f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG"
        ),
        activation=getattr(config, f"{prefix}_ACTIVATION"),
        layer_norm_position=getattr(config, f"{prefix}_LAYER_NORM_POSITION"),
        residual_connection_option=getattr(
            config, f"{prefix}_RESIDUAL_CONNECTION_OPTION"
        ),
        dropout_probability=getattr(config, f"{prefix}_DROPOUT_PROBABILITY"),
        bias_flag=getattr(config, f"{prefix}_BIAS_FLAG"),
    )


def _layer_controller_from_config(
    config: ModuleType,
    prefix: str,
) -> LayerControllerOptions:
    return LayerControllerOptions(
        stack_gate_flag=getattr(config, f"{prefix}_GATE_FLAG"),
        gate_option=getattr(config, f"{prefix}_GATE_OPTION"),
        gate_activation=getattr(config, f"{prefix}_GATE_ACTIVATION"),
        gate_stack_options=_controller_stack_from_config(
            config, f"{prefix}_GATE_STACK"
        ),
        stack_halting_flag=getattr(config, f"{prefix}_HALTING_FLAG"),
        halting_threshold=getattr(config, f"{prefix}_HALTING_THRESHOLD"),
        halting_dropout=getattr(config, f"{prefix}_HALTING_DROPOUT"),
        halting_hidden_state_mode=getattr(
            config, f"{prefix}_HALTING_HIDDEN_STATE_MODE"
        ),
        halting_stack_options=_controller_stack_from_config(
            config, f"{prefix}_HALTING_STACK"
        ),
    )


def _memory_from_config(config: ModuleType, prefix: str) -> DynamicMemoryOptions:
    return DynamicMemoryOptions(
        memory_flag=getattr(config, f"{prefix}_MEMORY_FLAG"),
        memory_option=getattr(config, f"{prefix}_MEMORY_OPTION"),
        memory_position_option=getattr(config, f"{prefix}_MEMORY_POSITION_OPTION"),
        memory_test_time_training_learning_rate=getattr(
            config, f"{prefix}_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE"
        ),
        memory_test_time_training_num_inner_steps=getattr(
            config, f"{prefix}_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS"
        ),
        memory_stack_options=_controller_stack_from_config(
            config, f"{prefix}_MEMORY_STACK"
        ),
    )


def _recurrent_from_config(
    config: ModuleType,
    prefix: str,
) -> RecurrentControllerOptions:
    return RecurrentControllerOptions(
        recurrent_flag=getattr(config, f"{prefix}_RECURRENT_FLAG"),
        recurrent_max_steps=getattr(config, f"{prefix}_RECURRENT_MAX_STEPS"),
        recurrent_layer_norm_position=getattr(
            config, f"{prefix}_RECURRENT_LAYER_NORM_POSITION"
        ),
        recurrent_gate_flag=getattr(config, f"{prefix}_RECURRENT_GATE_FLAG"),
        recurrent_gate_option=getattr(config, f"{prefix}_RECURRENT_GATE_OPTION"),
        recurrent_gate_activation=getattr(
            config, f"{prefix}_RECURRENT_GATE_ACTIVATION"
        ),
        recurrent_gate_stack_options=_controller_stack_from_config(
            config, f"{prefix}_RECURRENT_GATE_STACK"
        ),
        recurrent_halting_flag=getattr(config, f"{prefix}_RECURRENT_HALTING_FLAG"),
        recurrent_halting_threshold=getattr(
            config, f"{prefix}_RECURRENT_HALTING_THRESHOLD"
        ),
        recurrent_halting_dropout=getattr(
            config, f"{prefix}_RECURRENT_HALTING_DROPOUT"
        ),
        recurrent_halting_hidden_state_mode=getattr(
            config, f"{prefix}_RECURRENT_HALTING_HIDDEN_STATE_MODE"
        ),
        recurrent_halting_stack_options=_controller_stack_from_config(
            config, f"{prefix}_RECURRENT_HALTING_STACK"
        ),
    )


def attention_options_from_config(
    config: ModuleType,
) -> TransformerAttentionOptions:
    stack = SubmoduleStackOptions(
        hidden_dim=config.ATTN_STACK_HIDDEN_DIM,
        num_layers=config.ATTN_NUM_LAYERS,
        last_layer_bias_option=config.ATTN_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.ATTN_STACK_ACTIVATION,
        layer_norm_position=config.ATTN_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.ATTN_STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config.ATTN_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.ATTN_BIAS_FLAG,
    )
    return TransformerAttentionOptions(
        num_heads=config.ATTN_NUM_HEADS,
        add_key_value_bias_flag=config.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
        zero_attention_flag=config.ATTN_ZERO_ATTENTION_FLAG,
        stack_options=stack,
        layer_controller_options=_layer_controller_from_config(config, "ATTN"),
        dynamic_memory_options=_memory_from_config(config, "ATTN"),
        recurrent_controller_options=_recurrent_from_config(config, "ATTN"),
    )


def feed_forward_options_from_config(
    config: ModuleType,
) -> TransformerFeedForwardOptions:
    stack = SubmoduleStackOptions(
        hidden_dim=config.FF_STACK_HIDDEN_DIM,
        num_layers=config.FF_NUM_LAYERS,
        last_layer_bias_option=config.FF_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.FF_STACK_ACTIVATION,
        layer_norm_position=config.FF_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.FF_STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config.FF_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.FF_BIAS_FLAG,
    )
    return TransformerFeedForwardOptions(
        stack_options=stack,
        layer_controller_options=_layer_controller_from_config(config, "FF"),
        dynamic_memory_options=_memory_from_config(config, "FF"),
        recurrent_controller_options=_recurrent_from_config(config, "FF"),
    )


def _path_field_map(*, attention: bool) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    if attention:
        mapping.update(
            {
                "num_heads": ("path", "num_heads"),
                "add_key_value_bias_flag": (
                    "path",
                    "add_key_value_bias_flag",
                ),
                "zero_attention_flag": ("path", "zero_attention_flag"),
                "num_layers": ("stack", "num_layers"),
                "bias_flag": ("stack", "bias_flag"),
            }
        )
    else:
        mapping.update(
            {
                "num_layers": ("stack", "num_layers"),
                "bias_flag": ("stack", "bias_flag"),
            }
        )
    for field in _STACK_FIELDS:
        if field not in {"num_layers", "bias_flag"}:
            mapping[f"stack_{field}"] = ("stack", field)
    mapping.update(
        {
            "gate_flag": ("controller", "stack_gate_flag"),
            "gate_option": ("controller", "gate_option"),
            "gate_activation": ("controller", "gate_activation"),
            "halting_flag": ("controller", "stack_halting_flag"),
            "halting_threshold": ("controller", "halting_threshold"),
            "halting_dropout": ("controller", "halting_dropout"),
            "halting_hidden_state_mode": (
                "controller",
                "halting_hidden_state_mode",
            ),
            "memory_flag": ("memory", "memory_flag"),
            "memory_option": ("memory", "memory_option"),
            "memory_position_option": ("memory", "memory_position_option"),
            "memory_test_time_training_learning_rate": (
                "memory",
                "memory_test_time_training_learning_rate",
            ),
            "memory_test_time_training_num_inner_steps": (
                "memory",
                "memory_test_time_training_num_inner_steps",
            ),
            "recurrent_flag": ("recurrent", "recurrent_flag"),
            "recurrent_max_steps": ("recurrent", "recurrent_max_steps"),
            "recurrent_layer_norm_position": (
                "recurrent",
                "recurrent_layer_norm_position",
            ),
            "recurrent_gate_flag": ("recurrent", "recurrent_gate_flag"),
            "recurrent_gate_option": ("recurrent", "recurrent_gate_option"),
            "recurrent_gate_activation": (
                "recurrent",
                "recurrent_gate_activation",
            ),
            "recurrent_halting_flag": (
                "recurrent",
                "recurrent_halting_flag",
            ),
            "recurrent_halting_threshold": (
                "recurrent",
                "recurrent_halting_threshold",
            ),
            "recurrent_halting_dropout": (
                "recurrent",
                "recurrent_halting_dropout",
            ),
            "recurrent_halting_hidden_state_mode": (
                "recurrent",
                "recurrent_halting_hidden_state_mode",
            ),
        }
    )
    for role, component, field_name in (
        ("gate", "gate_stack", "gate_stack_options"),
        ("halting", "halting_stack", "halting_stack_options"),
        ("memory", "memory_stack", "memory_stack_options"),
        (
            "recurrent_gate",
            "recurrent_gate_stack",
            "recurrent_gate_stack_options",
        ),
        (
            "recurrent_halting",
            "recurrent_halting_stack",
            "recurrent_halting_stack_options",
        ),
    ):
        for field in _CONTROLLER_STACK_FIELDS:
            mapping[f"{role}_stack_{field}"] = (
                component,
                f"{field_name}.{field}",
            )
    return mapping


_ATTENTION_FIELD_MAP = _path_field_map(attention=True)
_FEED_FORWARD_FIELD_MAP = _path_field_map(attention=False)


def _replace_nested(source: Any, dotted_field: str, value: Any) -> Any:
    outer_field, inner_field = dotted_field.split(".", 1)
    return replace(
        source,
        **{outer_field: replace(getattr(source, outer_field), **{inner_field: value})},
    )


def _apply_path_updates(
    options: TransformerAttentionOptions | TransformerFeedForwardOptions,
    updates: dict[str, Any],
    *,
    attention: bool,
) -> TransformerAttentionOptions | TransformerFeedForwardOptions:
    field_map = _ATTENTION_FIELD_MAP if attention else _FEED_FORWARD_FIELD_MAP
    path = options
    stack = path.stack_options
    controller = path.layer_controller_options
    memory = path.dynamic_memory_options
    recurrent = path.recurrent_controller_options
    for suffix, value in updates.items():
        component, field = field_map[suffix]
        if component == "path":
            path = replace(path, **{field: value})
        elif component == "stack":
            if attention and field == "num_layers":
                path = replace(path, num_layers=value)
                stack = path.stack_options
            elif attention and field == "bias_flag":
                path = replace(path, projection_bias_flag=value)
                stack = path.stack_options
            elif not attention and field in {
                "hidden_dim",
                "num_layers",
                "bias_flag",
            }:
                path = replace(path, **{field: value})
                stack = path.stack_options
            else:
                stack = replace(stack, **{field: value})
        elif component == "controller":
            controller = replace(controller, **{field: value})
        elif component == "memory":
            memory = replace(memory, **{field: value})
        elif component == "recurrent":
            recurrent = replace(recurrent, **{field: value})
        elif component == "gate_stack":
            controller = _replace_nested(controller, field, value)
        elif component == "halting_stack":
            controller = _replace_nested(controller, field, value)
        elif component == "memory_stack":
            memory = _replace_nested(memory, field, value)
        elif component == "recurrent_gate_stack":
            recurrent = _replace_nested(recurrent, field, value)
        elif component == "recurrent_halting_stack":
            recurrent = _replace_nested(recurrent, field, value)
    return replace(
        path,
        stack_options=stack,
        layer_controller_options=controller,
        dynamic_memory_options=memory,
        recurrent_controller_options=recurrent,
    )


def _values_match(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        result = left == right
        return result if isinstance(result, bool) else bool(result)
    except (TypeError, ValueError):
        return False


def _normalize_legacy_aliases(values: MutableMapping[str, Any]) -> None:
    aliases = {
        "attn_projection_bias_flag": "attn_bias_flag",
        "feed_forward_hidden_dim": "ff_stack_hidden_dim",
        "feed_forward_num_layers": "ff_num_layers",
        "encoder_attn_projection_bias_flag": "encoder_attn_bias_flag",
        "decoder_self_attn_projection_bias_flag": ("decoder_self_attn_bias_flag"),
        "decoder_cross_attn_projection_bias_flag": ("decoder_cross_attn_bias_flag"),
        "encoder_feed_forward_hidden_dim": "encoder_ff_stack_hidden_dim",
        "encoder_feed_forward_num_layers": "encoder_ff_num_layers",
        "decoder_feed_forward_hidden_dim": "decoder_ff_stack_hidden_dim",
        "decoder_feed_forward_num_layers": "decoder_ff_num_layers",
    }
    for legacy, canonical in aliases.items():
        if legacy not in values:
            continue
        legacy_value = values.pop(legacy)
        if canonical in values and not _values_match(values[canonical], legacy_value):
            raise ValueError(
                f"Conflicting values for canonical option {canonical!r} and "
                f"legacy alias {legacy!r}."
            )
        values.setdefault(canonical, legacy_value)


def _pop_updates(
    values: MutableMapping[str, Any],
    prefix: str,
    field_map: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    updates = {}
    for suffix in field_map:
        key = f"{prefix}{suffix}"
        if key in values:
            updates[suffix] = values.pop(key)
    return updates


def resolve_transformer_path_options(
    values: MutableMapping[str, Any],
    defaults: TransformerPathOptions,
) -> TransformerPathOptions:
    """Consume and resolve flat attention/feed-forward path options.

    Resolution is defaults, then unscoped broadcast values, then a scoped path
    override. Legacy feed-forward and projection-bias names are normalized first.
    Unrecognized values are intentionally left in ``values`` for the package-local
    runtime resolver to validate.
    """

    _normalize_legacy_aliases(values)
    encoder_attention = values.pop(
        "encoder_attention_options", defaults.encoder_attention_options
    )
    decoder_self_attention = values.pop(
        "decoder_self_attention_options", defaults.decoder_self_attention_options
    )
    decoder_cross_attention = values.pop(
        "decoder_cross_attention_options", defaults.decoder_cross_attention_options
    )
    encoder_feed_forward = values.pop(
        "encoder_feed_forward_options", defaults.encoder_feed_forward_options
    )
    decoder_feed_forward = values.pop(
        "decoder_feed_forward_options", defaults.decoder_feed_forward_options
    )

    attention_broadcast = _pop_updates(values, "attn_", _ATTENTION_FIELD_MAP)
    feed_forward_broadcast = _pop_updates(values, "ff_", _FEED_FORWARD_FIELD_MAP)
    encoder_attention = _apply_path_updates(
        encoder_attention, attention_broadcast, attention=True
    )
    decoder_self_attention = _apply_path_updates(
        decoder_self_attention, attention_broadcast, attention=True
    )
    decoder_cross_attention = _apply_path_updates(
        decoder_cross_attention, attention_broadcast, attention=True
    )
    encoder_feed_forward = _apply_path_updates(
        encoder_feed_forward, feed_forward_broadcast, attention=False
    )
    decoder_feed_forward = _apply_path_updates(
        decoder_feed_forward, feed_forward_broadcast, attention=False
    )

    encoder_attention = _apply_path_updates(
        encoder_attention,
        _pop_updates(values, "encoder_attn_", _ATTENTION_FIELD_MAP),
        attention=True,
    )
    decoder_self_attention = _apply_path_updates(
        decoder_self_attention,
        _pop_updates(values, "decoder_self_attn_", _ATTENTION_FIELD_MAP),
        attention=True,
    )
    decoder_cross_attention = _apply_path_updates(
        decoder_cross_attention,
        _pop_updates(values, "decoder_cross_attn_", _ATTENTION_FIELD_MAP),
        attention=True,
    )
    encoder_feed_forward = _apply_path_updates(
        encoder_feed_forward,
        _pop_updates(values, "encoder_ff_", _FEED_FORWARD_FIELD_MAP),
        attention=False,
    )
    decoder_feed_forward = _apply_path_updates(
        decoder_feed_forward,
        _pop_updates(values, "decoder_ff_", _FEED_FORWARD_FIELD_MAP),
        attention=False,
    )
    return TransformerPathOptions(
        encoder_attention_options=encoder_attention,
        decoder_self_attention_options=decoder_self_attention,
        decoder_cross_attention_options=decoder_cross_attention,
        encoder_feed_forward_options=encoder_feed_forward,
        decoder_feed_forward_options=decoder_feed_forward,
    )


def expand_transformer_path_locks(locks: dict[str, Any]) -> dict[str, Any]:
    """Expand an unscoped preset lock to every affected Transformer path."""

    expanded = dict(locks)
    for key, value in tuple(locks.items()):
        if key.startswith("attn_"):
            suffix = key[len("attn_") :]
            for prefix in (
                "encoder_attn_",
                "decoder_self_attn_",
                "decoder_cross_attn_",
            ):
                expanded[f"{prefix}{suffix}"] = value
            if suffix == "bias_flag":
                for prefix in (
                    "encoder_attn_",
                    "decoder_self_attn_",
                    "decoder_cross_attn_",
                ):
                    expanded[f"{prefix}projection_bias_flag"] = value
        elif key.startswith("ff_"):
            suffix = key[len("ff_") :]
            expanded[f"encoder_ff_{suffix}"] = value
            expanded[f"decoder_ff_{suffix}"] = value
        elif key == "feed_forward_hidden_dim":
            expanded["encoder_feed_forward_hidden_dim"] = value
            expanded["decoder_feed_forward_hidden_dim"] = value
            expanded["encoder_ff_stack_hidden_dim"] = value
            expanded["decoder_ff_stack_hidden_dim"] = value
        elif key == "feed_forward_num_layers":
            expanded["encoder_feed_forward_num_layers"] = value
            expanded["decoder_feed_forward_num_layers"] = value
            expanded["encoder_ff_num_layers"] = value
            expanded["decoder_ff_num_layers"] = value
    return expanded


__all__ = [
    "ControllerStackOptions",
    "DynamicMemoryOptions",
    "LayerControllerOptions",
    "RecurrentControllerOptions",
    "SubmoduleStackOptions",
    "SubmoduleStackSource",
    "TransformerAttentionOptions",
    "TransformerFeedForwardOptions",
    "TransformerPathOptions",
    "TransformerStackOptions",
    "attention_options_from_config",
    "expand_transformer_path_locks",
    "feed_forward_options_from_config",
    "resolve_controller_stack_options",
    "resolve_transformer_path_options",
]
