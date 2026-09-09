from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Protocol, TypeGuard, TypeVar

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDepthOptions,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GroupingConfig,
    LowRankFactorSourceOptions,
    MaskDimensionOptions,
    SummaryNormalizationOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.config import ConfigBase
from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from models.neuron.linear_adaptive import config
from models.neuron.linear_adaptive._hidden._grouping import grouping_from_fields

_PACKAGE = "models.neuron.linear_adaptive._hidden"
_ValueT = TypeVar("_ValueT")
_ValueT_co = TypeVar("_ValueT_co", covariant=True)
_EnumT = TypeVar("_EnumT", bound=Enum)
_ImplementationT = TypeVar("_ImplementationT")


def _is_float(value: object) -> TypeGuard[int | float]:
    return type(value) in (int, float)


def _is_implementation(
    value: object,
    base_type: type[_ImplementationT],
) -> TypeGuard[type[_ImplementationT]]:
    return isinstance(value, type) and issubclass(value, base_type)


def _type_name(expected: object) -> str:
    return str(expected).replace("<class '", "").replace("'>", "")


def _implementation_type_name(base_type: type[object]) -> str:
    return f"type[{base_type.__module__}.{base_type.__qualname__}]"


def _instance_type_name(instance_type: type[object]) -> str:
    return f"{instance_type.__module__}.{instance_type.__qualname__}"


def _raise_type_error(key: str, value: object, expected: str) -> None:
    raise TypeError(
        f"{_PACKAGE}: runtime key {key!r} has type {type(value).__name__}; "
        f"expected {expected}"
    )


class _ValueRule(Protocol[_ValueT_co]):
    expected: str

    def accepts(self, value: object) -> bool: ...

    def read(self, key: str, value: object) -> _ValueT_co: ...


@dataclass(frozen=True, slots=True)
class _BooleanRule:
    expected: str = "bool"

    def accepts(self, value: object) -> bool:
        return type(value) is bool

    def read(self, key: str, value: object) -> bool:
        if type(value) is not bool:
            _raise_type_error(key, value, self.expected)
        return value


@dataclass(frozen=True, slots=True)
class _IntegerRule:
    expected: str = "int"

    def accepts(self, value: object) -> bool:
        return type(value) is int

    def read(self, key: str, value: object) -> int:
        if type(value) is not int:
            _raise_type_error(key, value, self.expected)
        return value


@dataclass(frozen=True, slots=True)
class _FloatRule:
    expected: str = "float"

    def accepts(self, value: object) -> bool:
        return _is_float(value)

    def read(self, key: str, value: object) -> float:
        if not _is_float(value):
            _raise_type_error(key, value, self.expected)
        return value


@dataclass(frozen=True, slots=True)
class _EnumRule(Generic[_EnumT]):
    enum_type: type[_EnumT]

    @property
    def expected(self) -> str:
        return _type_name(self.enum_type)

    def accepts(self, value: object) -> bool:
        return isinstance(value, self.enum_type)

    def read(self, key: str, value: object) -> _EnumT:
        if not isinstance(value, self.enum_type):
            _raise_type_error(key, value, self.expected)
        return value


@dataclass(frozen=True, slots=True)
class _ImplementationRule(Generic[_ImplementationT]):
    base_type: type[_ImplementationT]

    @property
    def expected(self) -> str:
        return _implementation_type_name(self.base_type)

    def accepts(self, value: object) -> bool:
        return _is_implementation(value, self.base_type)

    def read(self, key: str, value: object) -> type[_ImplementationT]:
        if not _is_implementation(value, self.base_type):
            _raise_type_error(key, value, self.expected)
        return value


@dataclass(frozen=True, slots=True)
class _InstanceRule(Generic[_ValueT]):
    instance_type: type[_ValueT]

    @property
    def expected(self) -> str:
        return _instance_type_name(self.instance_type)

    def accepts(self, value: object) -> bool:
        return isinstance(value, self.instance_type)

    def read(self, key: str, value: object) -> _ValueT:
        if not isinstance(value, self.instance_type):
            _raise_type_error(key, value, self.expected)
        return value


@dataclass(frozen=True, slots=True)
class _OptionalRule(Generic[_ValueT]):
    inner: _ValueRule[_ValueT]

    @property
    def expected(self) -> str:
        return f"{self.inner.expected} | None"

    def accepts(self, value: object) -> bool:
        return value is None or self.inner.accepts(value)

    def read(self, key: str, value: object) -> _ValueT | None:
        if value is None:
            return None
        return self.inner.read(key, value)


@dataclass(frozen=True, slots=True)
class _OptionalEnumRule(Generic[_EnumT]):
    enum_type: type[_EnumT]

    @property
    def expected(self) -> str:
        return f"{self.enum_type.__module__}.{self.enum_type.__qualname__} | None"

    def accepts(self, value: object) -> bool:
        return value is None or isinstance(value, self.enum_type)

    def read(self, key: str, value: object) -> _EnumT | None:
        if value is None:
            return None
        if not isinstance(value, self.enum_type):
            _raise_type_error(key, value, self.expected)
        return value


@dataclass(frozen=True, slots=True)
class RuntimeField(Generic[_ValueT]):
    key: str
    default: _ValueT
    rule: _ValueRule[_ValueT]

    @property
    def expected(self) -> str:
        return self.rule.expected

    def accepts(self, value: object) -> bool:
        return self.rule.accepts(value)

    def read(self, value: object) -> _ValueT:
        return self.rule.read(self.key, value)


class _BoundaryField(Protocol):
    key: str
    expected: str

    def accepts(self, value: object) -> bool: ...


_BOOLEAN = _BooleanRule()
_INTEGER = _IntegerRule()
_FLOAT = _FloatRule()


def _boolean_field(key: str, default: bool) -> RuntimeField[bool]:
    return RuntimeField(key, default, _BOOLEAN)


def _optional_boolean_field(
    key: str,
    default: bool | None,
) -> RuntimeField[bool | None]:
    return RuntimeField(key, default, _OptionalRule(_BOOLEAN))


def _integer_field(key: str, default: int) -> RuntimeField[int]:
    return RuntimeField(key, default, _INTEGER)


def _optional_integer_field(
    key: str,
    default: int | None,
) -> RuntimeField[int | None]:
    return RuntimeField(key, default, _OptionalRule(_INTEGER))


def _float_field(key: str, default: float) -> RuntimeField[float]:
    return RuntimeField(key, default, _FLOAT)


def _optional_float_field(
    key: str,
    default: float | None,
) -> RuntimeField[float | None]:
    return RuntimeField(key, default, _OptionalRule(_FLOAT))


def _enum_field(
    key: str,
    default: _EnumT,
    enum_type: type[_EnumT],
) -> RuntimeField[_EnumT]:
    return RuntimeField(key, default, _EnumRule(enum_type))


def _optional_enum_field(
    key: str,
    default: _EnumT | None,
    enum_type: type[_EnumT],
) -> RuntimeField[_EnumT | None]:
    return RuntimeField(key, default, _OptionalEnumRule(enum_type))


def _implementation_field(
    key: str,
    default: type[_ImplementationT],
    base_type: type[_ImplementationT],
) -> RuntimeField[type[_ImplementationT]]:
    return RuntimeField(key, default, _ImplementationRule(base_type))


def _optional_implementation_field(
    key: str,
    default: type[_ImplementationT] | None,
    base_type: type[_ImplementationT],
) -> RuntimeField[type[_ImplementationT] | None]:
    return RuntimeField(key, default, _OptionalRule(_ImplementationRule(base_type)))


def _optional_instance_field(
    key: str,
    default: _ValueT | None,
    instance_type: type[_ValueT],
) -> RuntimeField[_ValueT | None]:
    return RuntimeField(key, default, _OptionalRule(_InstanceRule(instance_type)))


@dataclass(slots=True)
class RuntimeOverrideReader:
    values: Mapping[str, object]
    fields: dict[str, _BoundaryField] = field(default_factory=dict, init=False)

    def read(self, runtime_field: RuntimeField[_ValueT]) -> _ValueT:
        self.fields[runtime_field.key] = runtime_field
        value = self.values.get(runtime_field.key, runtime_field.default)
        return runtime_field.read(value)


@dataclass(frozen=True, slots=True)
class RuntimeDimensionsValues:
    batch_size: int
    learning_rate: float
    input_dim: int
    hidden_dim: int
    output_dim: int


@dataclass(frozen=True, slots=True)
class RuntimeDimensionsFields:
    batch_size: RuntimeField[int]
    learning_rate: RuntimeField[float]
    input_dim: RuntimeField[int]
    hidden_dim: RuntimeField[int]
    output_dim: RuntimeField[int]


@dataclass(frozen=True, slots=True)
class StackValues:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float
    bias_flag: bool


@dataclass(frozen=True, slots=True)
class StackFields:
    hidden_dim: RuntimeField[int]
    num_layers: RuntimeField[int]
    last_layer_bias_option: RuntimeField[LastLayerBiasOptions]
    apply_output_postprocessing_flag: RuntimeField[bool]
    activation: RuntimeField[ActivationOptions]
    layer_norm_position: RuntimeField[LayerNormPositionOptions]
    normalization: RuntimeField[NormalizationOptions] = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    residual_connection_option: RuntimeField[type[ResidualConfig] | None]
    residual_model_flag: RuntimeField[bool]
    residual_block_size: RuntimeField[int | None]
    residual_rms_norm_epsilon: RuntimeField[float | None]
    dropout_probability: RuntimeField[float]
    bias_flag: RuntimeField[bool]


@dataclass(frozen=True, slots=True)
class OptionalStackValues:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    normalization: NormalizationOptions | None = field(default=None, kw_only=True)
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float | None
    bias_flag: bool | None


@dataclass(frozen=True, slots=True)
class OptionalStackFields:
    independent_flag: RuntimeField[bool]
    hidden_dim: RuntimeField[int | None]
    num_layers: RuntimeField[int | None]
    last_layer_bias_option: RuntimeField[LastLayerBiasOptions | None]
    apply_output_postprocessing_flag: RuntimeField[bool | None]
    activation: RuntimeField[ActivationOptions | None]
    layer_norm_position: RuntimeField[LayerNormPositionOptions | None]
    normalization: RuntimeField[NormalizationOptions | None] = field(
        default=None, kw_only=True
    )
    residual_connection_option: RuntimeField[type[ResidualConfig] | None]
    residual_model_flag: RuntimeField[bool]
    residual_block_size: RuntimeField[int | None]
    residual_rms_norm_epsilon: RuntimeField[float | None]
    dropout_probability: RuntimeField[float | None]
    bias_flag: RuntimeField[bool | None]


@dataclass(frozen=True, slots=True)
class ControlValues:
    stack_gate_flag: bool
    gate_option: LayerGateOptions | None
    gate_activation: ActivationOptions | None
    shared_gate_config: GateConfig | None
    stack_halting_flag: bool
    halting_option: type[HaltingConfig]
    halting_threshold: float
    halting_dropout: float
    halting_hidden_state_mode: HaltingHiddenStateModeOptions
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    memory_learning_rate: float | None
    memory_num_inner_steps: int | None
    recurrent_flag: bool
    recurrent_max_steps: int
    recurrent_initial_iterations: int
    recurrent_gradient_transition_count: int | None
    recurrent_no_gradient_transition_count: int | None
    recurrent_iteration_increment: int
    recurrent_forward_calls_before_iteration_increment: int
    recurrent_smooth_iteration_growth_flag: bool
    recurrent_layer_norm_position: LayerNormPositionOptions
    recurrent_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    recurrent_stack_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_stack_halting_flag: bool
    recurrent_halting_option: type[HaltingConfig]
    recurrent_halting_threshold: float
    recurrent_halting_dropout: float
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions


@dataclass(frozen=True, slots=True)
class ControlFields:
    stack_gate_flag: RuntimeField[bool]
    gate_option: RuntimeField[LayerGateOptions | None]
    gate_activation: RuntimeField[ActivationOptions | None]
    shared_gate_config: RuntimeField[GateConfig | None]
    stack_halting_flag: RuntimeField[bool]
    halting_option: RuntimeField[type[HaltingConfig]]
    halting_threshold: RuntimeField[float]
    halting_dropout: RuntimeField[float]
    halting_hidden_state_mode: RuntimeField[HaltingHiddenStateModeOptions]
    memory_flag: RuntimeField[bool]
    memory_option: RuntimeField[type[DynamicMemoryConfig]]
    memory_position_option: RuntimeField[MemoryPositionOptions]
    memory_learning_rate: RuntimeField[float | None]
    memory_num_inner_steps: RuntimeField[int | None]
    recurrent_flag: RuntimeField[bool]
    recurrent_max_steps: RuntimeField[int]
    recurrent_initial_iterations: RuntimeField[int]
    recurrent_gradient_transition_count: RuntimeField[int | None]
    recurrent_no_gradient_transition_count: RuntimeField[int | None]
    recurrent_iteration_increment: RuntimeField[int]
    recurrent_forward_calls_before_iteration_increment: RuntimeField[int]
    recurrent_smooth_iteration_growth_flag: RuntimeField[bool]
    recurrent_layer_norm_position: RuntimeField[LayerNormPositionOptions]
    recurrent_normalization: RuntimeField[NormalizationOptions] = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    recurrent_stack_gate_flag: RuntimeField[bool]
    recurrent_gate_option: RuntimeField[LayerGateOptions | None]
    recurrent_gate_activation: RuntimeField[ActivationOptions | None]
    recurrent_stack_halting_flag: RuntimeField[bool]
    recurrent_halting_option: RuntimeField[type[HaltingConfig]]
    recurrent_halting_threshold: RuntimeField[float]
    recurrent_halting_dropout: RuntimeField[float]
    recurrent_halting_hidden_state_mode: RuntimeField[HaltingHiddenStateModeOptions]


@dataclass(frozen=True, slots=True)
class WeightValues:
    enabled: bool
    option: type[DynamicWeightConfig] | None
    generator_depth: DynamicDepthOptions
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    normalization_option: WeightNormalizationOptions
    normalization_position_option: WeightNormalizationPositionOptions
    bank_expansion_factor: BankExpansionFactorOptions


@dataclass(frozen=True, slots=True)
class WeightFields:
    enabled: RuntimeField[bool]
    option: RuntimeField[type[DynamicWeightConfig] | None]
    generator_depth: RuntimeField[DynamicDepthOptions]
    decay_schedule: RuntimeField[WeightDecayScheduleOptions]
    decay_rate: RuntimeField[float]
    decay_warmup_batches: RuntimeField[int]
    normalization_option: RuntimeField[WeightNormalizationOptions]
    normalization_position_option: RuntimeField[WeightNormalizationPositionOptions]
    bank_expansion_factor: RuntimeField[BankExpansionFactorOptions]


@dataclass(frozen=True, slots=True)
class BiasValues:
    enabled: bool
    option: type[DynamicBiasConfig] | None
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions


@dataclass(frozen=True, slots=True)
class BiasFields:
    enabled: RuntimeField[bool]
    option: RuntimeField[type[DynamicBiasConfig] | None]
    decay_schedule: RuntimeField[WeightDecayScheduleOptions]
    decay_rate: RuntimeField[float]
    decay_warmup_batches: RuntimeField[int]
    bank_expansion_factor: RuntimeField[BankExpansionFactorOptions]


@dataclass(frozen=True, slots=True)
class DiagonalValues:
    enabled: bool
    option: type[DynamicDiagonalConfig] | None


@dataclass(frozen=True, slots=True)
class DiagonalFields:
    enabled: RuntimeField[bool]
    option: RuntimeField[type[DynamicDiagonalConfig] | None]


@dataclass(frozen=True, slots=True)
class MaskValues:
    enabled: bool
    row_mask_option: type[AxisMaskConfig] | None
    threshold: float
    floor: float
    transition_width: float
    surrogate_scale: float
    dimension_option: MaskDimensionOptions


@dataclass(frozen=True, slots=True)
class MaskFields:
    enabled: RuntimeField[bool]
    row_mask_option: RuntimeField[type[AxisMaskConfig] | None]
    threshold: RuntimeField[float]
    floor: RuntimeField[float]
    transition_width: RuntimeField[float]
    surrogate_scale: RuntimeField[float]
    dimension_option: RuntimeField[MaskDimensionOptions]


@dataclass(frozen=True, slots=True)
class ProjectionValues:
    weight_option: type[DynamicWeightConfig] | None
    generator_depth: DynamicDepthOptions
    weight_decay_schedule: WeightDecayScheduleOptions
    weight_decay_rate: float
    weight_decay_warmup_batches: int
    weight_normalization_option: WeightNormalizationOptions
    weight_normalization_position_option: WeightNormalizationPositionOptions
    weight_bank_expansion_factor: BankExpansionFactorOptions
    bias_option: type[DynamicBiasConfig] | None
    bias_decay_schedule: WeightDecayScheduleOptions
    bias_decay_rate: float
    bias_decay_warmup_batches: int
    bias_bank_expansion_factor: BankExpansionFactorOptions
    diagonal_option: type[DynamicDiagonalConfig] | None
    row_mask_option: type[AxisMaskConfig] | None
    mask_threshold: float
    mask_floor: float
    mask_transition_width: float
    mask_surrogate_scale: float
    mask_dimension_option: MaskDimensionOptions


@dataclass(frozen=True, slots=True)
class ProjectionFields:
    weight_option: RuntimeField[type[DynamicWeightConfig] | None]
    generator_depth: RuntimeField[DynamicDepthOptions]
    weight_decay_schedule: RuntimeField[WeightDecayScheduleOptions]
    weight_decay_rate: RuntimeField[float]
    weight_decay_warmup_batches: RuntimeField[int]
    weight_normalization_option: RuntimeField[WeightNormalizationOptions]
    weight_normalization_position_option: RuntimeField[
        WeightNormalizationPositionOptions
    ]
    weight_bank_expansion_factor: RuntimeField[BankExpansionFactorOptions]
    bias_option: RuntimeField[type[DynamicBiasConfig] | None]
    bias_decay_schedule: RuntimeField[WeightDecayScheduleOptions]
    bias_decay_rate: RuntimeField[float]
    bias_decay_warmup_batches: RuntimeField[int]
    bias_bank_expansion_factor: RuntimeField[BankExpansionFactorOptions]
    diagonal_option: RuntimeField[type[DynamicDiagonalConfig] | None]
    row_mask_option: RuntimeField[type[AxisMaskConfig] | None]
    mask_threshold: RuntimeField[float]
    mask_floor: RuntimeField[float]
    mask_transition_width: RuntimeField[float]
    mask_surrogate_scale: RuntimeField[float]
    mask_dimension_option: RuntimeField[MaskDimensionOptions]


@dataclass(frozen=True, slots=True)
class GenerationValues:
    weight_input_factor_source: LowRankFactorSourceOptions | None
    weight_output_factor_source: LowRankFactorSourceOptions | None
    weight_mixture_num_experts: int | None
    bias_mixture_num_experts: int | None
    weight_mixture_top_k: int | None
    bias_mixture_top_k: int | None
    weight_mixture_normalize_probabilities_flag: bool | None
    bias_mixture_normalize_probabilities_flag: bool | None
    weight_input_factor_generator_stack: OptionalStackValues
    weight_output_factor_generator_stack: OptionalStackValues
    weight_coefficient_generator_stack: OptionalStackValues
    weight_mixture_router_generator_stack: OptionalStackValues
    bias_mixture_router_generator_stack: OptionalStackValues


@dataclass(frozen=True, slots=True)
class RuntimeDefaultValues:
    grouping: GroupingConfig | None
    input_grouping: GroupingConfig | None
    output_grouping: GroupingConfig | None
    generation: GenerationValues
    input_generation: GenerationValues
    output_generation: GenerationValues
    dimensions: RuntimeDimensionsValues
    main_stack: StackValues
    submodule_stack: StackValues
    adaptive_generator_stack: StackValues
    residual_stack: OptionalStackValues
    gate_stack: OptionalStackValues
    halting_stack: OptionalStackValues
    memory_stack: OptionalStackValues
    recurrent_gate_stack: OptionalStackValues
    recurrent_halting_stack: OptionalStackValues
    weight_generator_stack: OptionalStackValues
    bias_generator_stack: OptionalStackValues
    diagonal_generator_stack: OptionalStackValues
    mask_generator_stack: OptionalStackValues
    control: ControlValues
    weight: WeightValues
    bias: BiasValues
    diagonal: DiagonalValues
    mask: MaskValues
    input_projection: ProjectionValues
    output_projection: ProjectionValues


_DIMENSION_FIELDS = RuntimeDimensionsFields(
    batch_size=_integer_field("batch_size", config.BATCH_SIZE),
    learning_rate=_float_field("learning_rate", config.LEARNING_RATE),
    input_dim=_integer_field("input_dim", config.INPUT_DIM),
    hidden_dim=_integer_field("hidden_dim", config.HIDDEN_DIM),
    output_dim=_integer_field("output_dim", config.OUTPUT_DIM),
)
_MAIN_STACK_FIELDS = StackFields(
    hidden_dim=_integer_field("hidden_dim", config.HIDDEN_DIM),
    num_layers=_integer_field("stack_num_layers", config.STACK_NUM_LAYERS),
    last_layer_bias_option=_enum_field(
        "stack_last_layer_bias_option",
        config.STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_boolean_field(
        "stack_apply_output_postprocessing_flag",
        config.STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    activation=_enum_field(
        "stack_activation", config.STACK_ACTIVATION, ActivationOptions
    ),
    layer_norm_position=_enum_field(
        "layer_norm_position",
        config.LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_enum_field(
        "normalization",
        config.NORMALIZATION,
        NormalizationOptions,
    ),
    residual_connection_option=_optional_implementation_field(
        "stack_residual_connection_option",
        config.STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "stack_residual_block_size", config.STACK_RESIDUAL_BLOCK_SIZE
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "stack_residual_rms_norm_epsilon", config.STACK_RESIDUAL_RMS_NORM_EPSILON
    ),
    residual_model_flag=_boolean_field(
        "stack_residual_model_flag", config.STACK_RESIDUAL_MODEL_FLAG
    ),
    dropout_probability=_float_field(
        "stack_dropout_probability", config.STACK_DROPOUT_PROBABILITY
    ),
    bias_flag=_boolean_field("stack_bias_flag", config.STACK_BIAS_FLAG),
)
_SUBMODULE_STACK_FIELDS = StackFields(
    hidden_dim=_integer_field(
        "submodule_stack_hidden_dim", config.SUBMODULE_STACK_HIDDEN_DIM
    ),
    num_layers=_integer_field(
        "submodule_stack_num_layers", config.SUBMODULE_STACK_NUM_LAYERS
    ),
    last_layer_bias_option=_enum_field(
        "submodule_stack_last_layer_bias_option",
        config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_boolean_field(
        "submodule_stack_apply_output_postprocessing_flag",
        config.SUBMODULE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    activation=_enum_field(
        "submodule_stack_activation",
        config.SUBMODULE_STACK_ACTIVATION,
        ActivationOptions,
    ),
    layer_norm_position=_enum_field(
        "submodule_stack_layer_norm_position",
        config.SUBMODULE_STACK_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_enum_field(
        "submodule_stack_normalization",
        config.SUBMODULE_STACK_NORMALIZATION,
        NormalizationOptions,
    ),
    residual_connection_option=_optional_implementation_field(
        "submodule_stack_residual_connection_option",
        config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "submodule_stack_residual_block_size",
        config.SUBMODULE_STACK_RESIDUAL_BLOCK_SIZE,
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "submodule_stack_residual_rms_norm_epsilon",
        config.SUBMODULE_STACK_RESIDUAL_RMS_NORM_EPSILON,
    ),
    residual_model_flag=_boolean_field(
        "submodule_stack_residual_model_flag",
        config.SUBMODULE_STACK_RESIDUAL_MODEL_FLAG,
    ),
    dropout_probability=_float_field(
        "submodule_stack_dropout_probability",
        config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
    ),
    bias_flag=_boolean_field(
        "submodule_stack_bias_flag", config.SUBMODULE_STACK_BIAS_FLAG
    ),
)
_ADAPTIVE_GENERATOR_STACK_FIELDS = StackFields(
    hidden_dim=_integer_field(
        "adaptive_generator_stack_hidden_dim",
        config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
    ),
    num_layers=_integer_field(
        "adaptive_generator_stack_num_layers",
        config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
    ),
    last_layer_bias_option=_enum_field(
        "adaptive_generator_stack_last_layer_bias_option",
        config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_boolean_field(
        "adaptive_generator_stack_apply_output_postprocessing_flag",
        config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    activation=_enum_field(
        "adaptive_generator_stack_activation",
        config.ADAPTIVE_GENERATOR_STACK_ACTIVATION,
        ActivationOptions,
    ),
    layer_norm_position=_enum_field(
        "adaptive_generator_stack_layer_norm_position",
        config.ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_enum_field(
        "adaptive_generator_stack_normalization",
        config.ADAPTIVE_GENERATOR_STACK_NORMALIZATION,
        NormalizationOptions,
    ),
    residual_connection_option=_optional_implementation_field(
        "adaptive_generator_stack_residual_connection_option",
        config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "adaptive_generator_stack_residual_block_size",
        config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "adaptive_generator_stack_residual_rms_norm_epsilon",
        config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
    ),
    residual_model_flag=_boolean_field(
        "adaptive_generator_stack_residual_model_flag",
        config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
    ),
    dropout_probability=_float_field(
        "adaptive_generator_stack_dropout_probability",
        config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
    ),
    bias_flag=_boolean_field(
        "adaptive_generator_stack_bias_flag",
        config.ADAPTIVE_GENERATOR_STACK_BIAS_FLAG,
    ),
)


_RESIDUAL_STACK_FIELDS = OptionalStackFields(
    independent_flag=_boolean_field(
        "residual_stack_independent_flag", config.RESIDUAL_STACK_INDEPENDENT_FLAG
    ),
    hidden_dim=_optional_integer_field(
        "residual_stack_hidden_dim", config.RESIDUAL_STACK_HIDDEN_DIM
    ),
    layer_norm_position=_optional_enum_field(
        "residual_stack_layer_norm_position",
        config.RESIDUAL_STACK_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_optional_enum_field(
        "residual_stack_normalization",
        config.RESIDUAL_STACK_NORMALIZATION,
        NormalizationOptions,
    ),
    num_layers=_optional_integer_field(
        "residual_stack_num_layers", config.RESIDUAL_STACK_NUM_LAYERS
    ),
    activation=_optional_enum_field(
        "residual_stack_activation",
        config.RESIDUAL_STACK_ACTIVATION,
        ActivationOptions,
    ),
    residual_connection_option=_optional_implementation_field(
        "residual_stack_residual_connection_option",
        config.RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "residual_stack_residual_block_size", config.RESIDUAL_STACK_RESIDUAL_BLOCK_SIZE
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "residual_stack_residual_rms_norm_epsilon",
        config.RESIDUAL_STACK_RESIDUAL_RMS_NORM_EPSILON,
    ),
    residual_model_flag=_boolean_field(
        "residual_stack_residual_model_flag",
        config.RESIDUAL_STACK_RESIDUAL_MODEL_FLAG,
    ),
    dropout_probability=_optional_float_field(
        "residual_stack_dropout_probability",
        config.RESIDUAL_STACK_DROPOUT_PROBABILITY,
    ),
    last_layer_bias_option=_optional_enum_field(
        "residual_stack_last_layer_bias_option",
        config.RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_optional_boolean_field(
        "residual_stack_apply_output_postprocessing_flag",
        config.RESIDUAL_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    bias_flag=_optional_boolean_field(
        "residual_stack_bias_flag", config.RESIDUAL_STACK_BIAS_FLAG
    ),
)
_GATE_STACK_FIELDS = OptionalStackFields(
    independent_flag=_boolean_field(
        "gate_stack_independent_flag", config.GATE_STACK_INDEPENDENT_FLAG
    ),
    hidden_dim=_optional_integer_field(
        "gate_stack_hidden_dim", config.GATE_STACK_HIDDEN_DIM
    ),
    layer_norm_position=_optional_enum_field(
        "gate_stack_layer_norm_position",
        config.GATE_STACK_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_optional_enum_field(
        "gate_stack_normalization",
        config.GATE_STACK_NORMALIZATION,
        NormalizationOptions,
    ),
    num_layers=_optional_integer_field(
        "gate_stack_num_layers", config.GATE_STACK_NUM_LAYERS
    ),
    activation=_optional_enum_field(
        "gate_stack_activation", config.GATE_STACK_ACTIVATION, ActivationOptions
    ),
    residual_connection_option=_optional_implementation_field(
        "gate_stack_residual_connection_option",
        config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "gate_stack_residual_block_size", config.GATE_STACK_RESIDUAL_BLOCK_SIZE
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "gate_stack_residual_rms_norm_epsilon",
        config.GATE_STACK_RESIDUAL_RMS_NORM_EPSILON,
    ),
    residual_model_flag=_boolean_field(
        "gate_stack_residual_model_flag", config.GATE_STACK_RESIDUAL_MODEL_FLAG
    ),
    dropout_probability=_optional_float_field(
        "gate_stack_dropout_probability", config.GATE_STACK_DROPOUT_PROBABILITY
    ),
    last_layer_bias_option=_optional_enum_field(
        "gate_stack_last_layer_bias_option",
        config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_optional_boolean_field(
        "gate_stack_apply_output_postprocessing_flag",
        config.GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    bias_flag=_optional_boolean_field(
        "gate_stack_bias_flag", config.GATE_STACK_BIAS_FLAG
    ),
)
_HALTING_STACK_FIELDS = OptionalStackFields(
    independent_flag=_boolean_field(
        "halting_stack_independent_flag", config.HALTING_STACK_INDEPENDENT_FLAG
    ),
    hidden_dim=_optional_integer_field(
        "halting_stack_hidden_dim", config.HALTING_STACK_HIDDEN_DIM
    ),
    layer_norm_position=_optional_enum_field(
        "halting_stack_layer_norm_position",
        config.HALTING_STACK_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_optional_enum_field(
        "halting_stack_normalization",
        config.HALTING_STACK_NORMALIZATION,
        NormalizationOptions,
    ),
    num_layers=_optional_integer_field(
        "halting_stack_num_layers", config.HALTING_STACK_NUM_LAYERS
    ),
    activation=_optional_enum_field(
        "halting_stack_activation",
        config.HALTING_STACK_ACTIVATION,
        ActivationOptions,
    ),
    residual_connection_option=_optional_implementation_field(
        "halting_stack_residual_connection_option",
        config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "halting_stack_residual_block_size", config.HALTING_STACK_RESIDUAL_BLOCK_SIZE
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "halting_stack_residual_rms_norm_epsilon",
        config.HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON,
    ),
    residual_model_flag=_boolean_field(
        "halting_stack_residual_model_flag",
        config.HALTING_STACK_RESIDUAL_MODEL_FLAG,
    ),
    dropout_probability=_optional_float_field(
        "halting_stack_dropout_probability",
        config.HALTING_STACK_DROPOUT_PROBABILITY,
    ),
    last_layer_bias_option=_optional_enum_field(
        "halting_stack_last_layer_bias_option",
        config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_optional_boolean_field(
        "halting_stack_apply_output_postprocessing_flag",
        config.HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    bias_flag=_optional_boolean_field(
        "halting_stack_bias_flag", config.HALTING_STACK_BIAS_FLAG
    ),
)
_MEMORY_STACK_FIELDS = OptionalStackFields(
    independent_flag=_boolean_field(
        "memory_stack_independent_flag", config.MEMORY_STACK_INDEPENDENT_FLAG
    ),
    hidden_dim=_optional_integer_field(
        "memory_stack_hidden_dim", config.MEMORY_STACK_HIDDEN_DIM
    ),
    layer_norm_position=_optional_enum_field(
        "memory_stack_layer_norm_position",
        config.MEMORY_STACK_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_optional_enum_field(
        "memory_stack_normalization",
        config.MEMORY_STACK_NORMALIZATION,
        NormalizationOptions,
    ),
    num_layers=_optional_integer_field(
        "memory_stack_num_layers", config.MEMORY_STACK_NUM_LAYERS
    ),
    activation=_optional_enum_field(
        "memory_stack_activation", config.MEMORY_STACK_ACTIVATION, ActivationOptions
    ),
    residual_connection_option=_optional_implementation_field(
        "memory_stack_residual_connection_option",
        config.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "memory_stack_residual_block_size", config.MEMORY_STACK_RESIDUAL_BLOCK_SIZE
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "memory_stack_residual_rms_norm_epsilon",
        config.MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON,
    ),
    residual_model_flag=_boolean_field(
        "memory_stack_residual_model_flag", config.MEMORY_STACK_RESIDUAL_MODEL_FLAG
    ),
    dropout_probability=_optional_float_field(
        "memory_stack_dropout_probability", config.MEMORY_STACK_DROPOUT_PROBABILITY
    ),
    last_layer_bias_option=_optional_enum_field(
        "memory_stack_last_layer_bias_option",
        config.MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_optional_boolean_field(
        "memory_stack_apply_output_postprocessing_flag",
        config.MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    bias_flag=_optional_boolean_field(
        "memory_stack_bias_flag", config.MEMORY_STACK_BIAS_FLAG
    ),
)
_RECURRENT_GATE_STACK_FIELDS = OptionalStackFields(
    independent_flag=_boolean_field(
        "recurrent_gate_stack_independent_flag",
        config.RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
    ),
    hidden_dim=_optional_integer_field(
        "recurrent_gate_stack_hidden_dim",
        config.RECURRENT_GATE_STACK_HIDDEN_DIM,
    ),
    layer_norm_position=_optional_enum_field(
        "recurrent_gate_stack_layer_norm_position",
        config.RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_optional_enum_field(
        "recurrent_gate_stack_normalization",
        config.RECURRENT_GATE_STACK_NORMALIZATION,
        NormalizationOptions,
    ),
    num_layers=_optional_integer_field(
        "recurrent_gate_stack_num_layers",
        config.RECURRENT_GATE_STACK_NUM_LAYERS,
    ),
    activation=_optional_enum_field(
        "recurrent_gate_stack_activation",
        config.RECURRENT_GATE_STACK_ACTIVATION,
        ActivationOptions,
    ),
    residual_connection_option=_optional_implementation_field(
        "recurrent_gate_stack_residual_connection_option",
        config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "recurrent_gate_stack_residual_block_size",
        config.RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE,
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "recurrent_gate_stack_residual_rms_norm_epsilon",
        config.RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON,
    ),
    residual_model_flag=_boolean_field(
        "recurrent_gate_stack_residual_model_flag",
        config.RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
    ),
    dropout_probability=_optional_float_field(
        "recurrent_gate_stack_dropout_probability",
        config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
    ),
    last_layer_bias_option=_optional_enum_field(
        "recurrent_gate_stack_last_layer_bias_option",
        config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_optional_boolean_field(
        "recurrent_gate_stack_apply_output_postprocessing_flag",
        config.RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    bias_flag=_optional_boolean_field(
        "recurrent_gate_stack_bias_flag",
        config.RECURRENT_GATE_STACK_BIAS_FLAG,
    ),
)
_RECURRENT_HALTING_STACK_FIELDS = OptionalStackFields(
    independent_flag=_boolean_field(
        "recurrent_halting_stack_independent_flag",
        config.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
    ),
    hidden_dim=_optional_integer_field(
        "recurrent_halting_stack_hidden_dim",
        config.RECURRENT_HALTING_STACK_HIDDEN_DIM,
    ),
    layer_norm_position=_optional_enum_field(
        "recurrent_halting_stack_layer_norm_position",
        config.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_optional_enum_field(
        "recurrent_halting_stack_normalization",
        config.RECURRENT_HALTING_STACK_NORMALIZATION,
        NormalizationOptions,
    ),
    num_layers=_optional_integer_field(
        "recurrent_halting_stack_num_layers",
        config.RECURRENT_HALTING_STACK_NUM_LAYERS,
    ),
    activation=_optional_enum_field(
        "recurrent_halting_stack_activation",
        config.RECURRENT_HALTING_STACK_ACTIVATION,
        ActivationOptions,
    ),
    residual_connection_option=_optional_implementation_field(
        "recurrent_halting_stack_residual_connection_option",
        config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "recurrent_halting_stack_residual_block_size",
        config.RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE,
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "recurrent_halting_stack_residual_rms_norm_epsilon",
        config.RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON,
    ),
    residual_model_flag=_boolean_field(
        "recurrent_halting_stack_residual_model_flag",
        config.RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
    ),
    dropout_probability=_optional_float_field(
        "recurrent_halting_stack_dropout_probability",
        config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
    ),
    last_layer_bias_option=_optional_enum_field(
        "recurrent_halting_stack_last_layer_bias_option",
        config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_optional_boolean_field(
        "recurrent_halting_stack_apply_output_postprocessing_flag",
        config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    bias_flag=_optional_boolean_field(
        "recurrent_halting_stack_bias_flag",
        config.RECURRENT_HALTING_STACK_BIAS_FLAG,
    ),
)
_WEIGHT_GENERATOR_STACK_FIELDS = OptionalStackFields(
    independent_flag=_boolean_field(
        "weight_generator_stack_independent_flag",
        config.WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG,
    ),
    hidden_dim=_optional_integer_field(
        "weight_generator_stack_hidden_dim",
        config.WEIGHT_GENERATOR_STACK_HIDDEN_DIM,
    ),
    layer_norm_position=_optional_enum_field(
        "weight_generator_stack_layer_norm_position",
        config.WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_optional_enum_field(
        "weight_generator_stack_normalization",
        config.WEIGHT_GENERATOR_STACK_NORMALIZATION,
        NormalizationOptions,
    ),
    num_layers=_optional_integer_field(
        "weight_generator_stack_num_layers",
        config.WEIGHT_GENERATOR_STACK_NUM_LAYERS,
    ),
    activation=_optional_enum_field(
        "weight_generator_stack_activation",
        config.WEIGHT_GENERATOR_STACK_ACTIVATION,
        ActivationOptions,
    ),
    residual_connection_option=_optional_implementation_field(
        "weight_generator_stack_residual_connection_option",
        config.WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "weight_generator_stack_residual_block_size",
        config.WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "weight_generator_stack_residual_rms_norm_epsilon",
        config.WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
    ),
    residual_model_flag=_boolean_field(
        "weight_generator_stack_residual_model_flag",
        config.WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
    ),
    dropout_probability=_optional_float_field(
        "weight_generator_stack_dropout_probability",
        config.WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY,
    ),
    last_layer_bias_option=_optional_enum_field(
        "weight_generator_stack_last_layer_bias_option",
        config.WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_optional_boolean_field(
        "weight_generator_stack_apply_output_postprocessing_flag",
        config.WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    bias_flag=_optional_boolean_field(
        "weight_generator_stack_bias_flag",
        config.WEIGHT_GENERATOR_STACK_BIAS_FLAG,
    ),
)
_BIAS_GENERATOR_STACK_FIELDS = OptionalStackFields(
    independent_flag=_boolean_field(
        "bias_generator_stack_independent_flag",
        config.BIAS_GENERATOR_STACK_INDEPENDENT_FLAG,
    ),
    hidden_dim=_optional_integer_field(
        "bias_generator_stack_hidden_dim",
        config.BIAS_GENERATOR_STACK_HIDDEN_DIM,
    ),
    layer_norm_position=_optional_enum_field(
        "bias_generator_stack_layer_norm_position",
        config.BIAS_GENERATOR_STACK_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_optional_enum_field(
        "bias_generator_stack_normalization",
        config.BIAS_GENERATOR_STACK_NORMALIZATION,
        NormalizationOptions,
    ),
    num_layers=_optional_integer_field(
        "bias_generator_stack_num_layers",
        config.BIAS_GENERATOR_STACK_NUM_LAYERS,
    ),
    activation=_optional_enum_field(
        "bias_generator_stack_activation",
        config.BIAS_GENERATOR_STACK_ACTIVATION,
        ActivationOptions,
    ),
    residual_connection_option=_optional_implementation_field(
        "bias_generator_stack_residual_connection_option",
        config.BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "bias_generator_stack_residual_block_size",
        config.BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "bias_generator_stack_residual_rms_norm_epsilon",
        config.BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
    ),
    residual_model_flag=_boolean_field(
        "bias_generator_stack_residual_model_flag",
        config.BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
    ),
    dropout_probability=_optional_float_field(
        "bias_generator_stack_dropout_probability",
        config.BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY,
    ),
    last_layer_bias_option=_optional_enum_field(
        "bias_generator_stack_last_layer_bias_option",
        config.BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_optional_boolean_field(
        "bias_generator_stack_apply_output_postprocessing_flag",
        config.BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    bias_flag=_optional_boolean_field(
        "bias_generator_stack_bias_flag",
        config.BIAS_GENERATOR_STACK_BIAS_FLAG,
    ),
)
_DIAGONAL_GENERATOR_STACK_FIELDS = OptionalStackFields(
    independent_flag=_boolean_field(
        "diagonal_generator_stack_independent_flag",
        config.DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG,
    ),
    hidden_dim=_optional_integer_field(
        "diagonal_generator_stack_hidden_dim",
        config.DIAGONAL_GENERATOR_STACK_HIDDEN_DIM,
    ),
    layer_norm_position=_optional_enum_field(
        "diagonal_generator_stack_layer_norm_position",
        config.DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_optional_enum_field(
        "diagonal_generator_stack_normalization",
        config.DIAGONAL_GENERATOR_STACK_NORMALIZATION,
        NormalizationOptions,
    ),
    num_layers=_optional_integer_field(
        "diagonal_generator_stack_num_layers",
        config.DIAGONAL_GENERATOR_STACK_NUM_LAYERS,
    ),
    activation=_optional_enum_field(
        "diagonal_generator_stack_activation",
        config.DIAGONAL_GENERATOR_STACK_ACTIVATION,
        ActivationOptions,
    ),
    residual_connection_option=_optional_implementation_field(
        "diagonal_generator_stack_residual_connection_option",
        config.DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "diagonal_generator_stack_residual_block_size",
        config.DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "diagonal_generator_stack_residual_rms_norm_epsilon",
        config.DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
    ),
    residual_model_flag=_boolean_field(
        "diagonal_generator_stack_residual_model_flag",
        config.DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
    ),
    dropout_probability=_optional_float_field(
        "diagonal_generator_stack_dropout_probability",
        config.DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY,
    ),
    last_layer_bias_option=_optional_enum_field(
        "diagonal_generator_stack_last_layer_bias_option",
        config.DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_optional_boolean_field(
        "diagonal_generator_stack_apply_output_postprocessing_flag",
        config.DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    bias_flag=_optional_boolean_field(
        "diagonal_generator_stack_bias_flag",
        config.DIAGONAL_GENERATOR_STACK_BIAS_FLAG,
    ),
)
_MASK_GENERATOR_STACK_FIELDS = OptionalStackFields(
    independent_flag=_boolean_field(
        "mask_generator_stack_independent_flag",
        config.MASK_GENERATOR_STACK_INDEPENDENT_FLAG,
    ),
    hidden_dim=_optional_integer_field(
        "mask_generator_stack_hidden_dim",
        config.MASK_GENERATOR_STACK_HIDDEN_DIM,
    ),
    layer_norm_position=_optional_enum_field(
        "mask_generator_stack_layer_norm_position",
        config.MASK_GENERATOR_STACK_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    normalization=_optional_enum_field(
        "mask_generator_stack_normalization",
        config.MASK_GENERATOR_STACK_NORMALIZATION,
        NormalizationOptions,
    ),
    num_layers=_optional_integer_field(
        "mask_generator_stack_num_layers",
        config.MASK_GENERATOR_STACK_NUM_LAYERS,
    ),
    activation=_optional_enum_field(
        "mask_generator_stack_activation",
        config.MASK_GENERATOR_STACK_ACTIVATION,
        ActivationOptions,
    ),
    residual_connection_option=_optional_implementation_field(
        "mask_generator_stack_residual_connection_option",
        config.MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        ResidualConfig,
    ),
    residual_block_size=_optional_integer_field(
        "mask_generator_stack_residual_block_size",
        config.MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
    ),
    residual_rms_norm_epsilon=_optional_float_field(
        "mask_generator_stack_residual_rms_norm_epsilon",
        config.MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
    ),
    residual_model_flag=_boolean_field(
        "mask_generator_stack_residual_model_flag",
        config.MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
    ),
    dropout_probability=_optional_float_field(
        "mask_generator_stack_dropout_probability",
        config.MASK_GENERATOR_STACK_DROPOUT_PROBABILITY,
    ),
    last_layer_bias_option=_optional_enum_field(
        "mask_generator_stack_last_layer_bias_option",
        config.MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        LastLayerBiasOptions,
    ),
    apply_output_postprocessing_flag=_optional_boolean_field(
        "mask_generator_stack_apply_output_postprocessing_flag",
        config.MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
    ),
    bias_flag=_optional_boolean_field(
        "mask_generator_stack_bias_flag",
        config.MASK_GENERATOR_STACK_BIAS_FLAG,
    ),
)
_CONTROL_FIELDS = ControlFields(
    stack_gate_flag=_boolean_field("stack_gate_flag", config.STACK_GATE_FLAG),
    gate_option=_optional_enum_field(
        "gate_option", config.GATE_OPTION, LayerGateOptions
    ),
    gate_activation=_optional_enum_field(
        "gate_activation", config.GATE_ACTIVATION, ActivationOptions
    ),
    shared_gate_config=_optional_instance_field("shared_gate_config", None, GateConfig),
    stack_halting_flag=_boolean_field("stack_halting_flag", config.STACK_HALTING_FLAG),
    halting_option=_implementation_field(
        "halting_option", config.HALTING_OPTION, HaltingConfig
    ),
    halting_threshold=_float_field("halting_threshold", config.HALTING_THRESHOLD),
    halting_dropout=_float_field("halting_dropout", config.HALTING_DROPOUT),
    halting_hidden_state_mode=_enum_field(
        "halting_hidden_state_mode",
        config.HALTING_HIDDEN_STATE_MODE,
        HaltingHiddenStateModeOptions,
    ),
    memory_flag=_boolean_field("memory_flag", config.MEMORY_FLAG),
    memory_option=_implementation_field(
        "memory_option", config.MEMORY_OPTION, DynamicMemoryConfig
    ),
    memory_position_option=_enum_field(
        "memory_position_option",
        config.MEMORY_POSITION_OPTION,
        MemoryPositionOptions,
    ),
    memory_learning_rate=_optional_float_field(
        "memory_test_time_training_learning_rate",
        config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
    ),
    memory_num_inner_steps=_optional_integer_field(
        "memory_test_time_training_num_inner_steps",
        config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
    ),
    recurrent_flag=_boolean_field("recurrent_flag", config.RECURRENT_FLAG),
    recurrent_max_steps=_integer_field(
        "recurrent_max_steps", config.RECURRENT_MAX_STEPS
    ),
    recurrent_initial_iterations=_integer_field(
        "recurrent_initial_iterations", config.RECURRENT_INITIAL_ITERATIONS
    ),
    recurrent_gradient_transition_count=_optional_integer_field(
        "recurrent_gradient_transition_count",
        config.RECURRENT_GRADIENT_TRANSITION_COUNT,
    ),
    recurrent_no_gradient_transition_count=_optional_integer_field(
        "recurrent_no_gradient_transition_count",
        config.RECURRENT_NO_GRADIENT_TRANSITION_COUNT,
    ),
    recurrent_iteration_increment=_integer_field(
        "recurrent_iteration_increment", config.RECURRENT_ITERATION_INCREMENT
    ),
    recurrent_forward_calls_before_iteration_increment=_integer_field(
        "recurrent_forward_calls_before_iteration_increment",
        config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT,
    ),
    recurrent_smooth_iteration_growth_flag=_boolean_field(
        "recurrent_smooth_iteration_growth_flag",
        config.RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG,
    ),
    recurrent_layer_norm_position=_enum_field(
        "recurrent_layer_norm_position",
        config.RECURRENT_LAYER_NORM_POSITION,
        LayerNormPositionOptions,
    ),
    recurrent_normalization=_enum_field(
        "recurrent_normalization",
        config.RECURRENT_NORMALIZATION,
        NormalizationOptions,
    ),
    recurrent_stack_gate_flag=_boolean_field(
        "recurrent_stack_gate_flag", config.RECURRENT_STACK_GATE_FLAG
    ),
    recurrent_gate_option=_optional_enum_field(
        "recurrent_gate_option", config.RECURRENT_GATE_OPTION, LayerGateOptions
    ),
    recurrent_gate_activation=_optional_enum_field(
        "recurrent_gate_activation",
        config.RECURRENT_GATE_ACTIVATION,
        ActivationOptions,
    ),
    recurrent_stack_halting_flag=_boolean_field(
        "recurrent_stack_halting_flag", config.RECURRENT_STACK_HALTING_FLAG
    ),
    recurrent_halting_option=_implementation_field(
        "recurrent_halting_option",
        config.RECURRENT_HALTING_OPTION,
        HaltingConfig,
    ),
    recurrent_halting_threshold=_float_field(
        "recurrent_halting_threshold", config.RECURRENT_HALTING_THRESHOLD
    ),
    recurrent_halting_dropout=_float_field(
        "recurrent_halting_dropout", config.RECURRENT_HALTING_DROPOUT
    ),
    recurrent_halting_hidden_state_mode=_enum_field(
        "recurrent_halting_hidden_state_mode",
        config.RECURRENT_HALTING_HIDDEN_STATE_MODE,
        HaltingHiddenStateModeOptions,
    ),
)
_WEIGHT_FIELDS = WeightFields(
    enabled=_boolean_field("weight_option_flag", config.WEIGHT_OPTION_FLAG),
    option=_optional_implementation_field(
        "weight_option", config.WEIGHT_OPTION, DynamicWeightConfig
    ),
    generator_depth=_enum_field(
        "generator_depth", config.GENERATOR_DEPTH, DynamicDepthOptions
    ),
    decay_schedule=_enum_field(
        "weight_decay_schedule",
        config.WEIGHT_DECAY_SCHEDULE,
        WeightDecayScheduleOptions,
    ),
    decay_rate=_float_field("weight_decay_rate", config.WEIGHT_DECAY_RATE),
    decay_warmup_batches=_integer_field(
        "weight_decay_warmup_batches", config.WEIGHT_DECAY_WARMUP_BATCHES
    ),
    normalization_option=_enum_field(
        "weight_normalization_option",
        config.WEIGHT_NORMALIZATION_OPTION,
        WeightNormalizationOptions,
    ),
    normalization_position_option=_enum_field(
        "weight_normalization_position_option",
        config.WEIGHT_NORMALIZATION_POSITION_OPTION,
        WeightNormalizationPositionOptions,
    ),
    bank_expansion_factor=_enum_field(
        "weight_bank_expansion_factor",
        config.WEIGHT_BANK_EXPANSION_FACTOR,
        BankExpansionFactorOptions,
    ),
)
_BIAS_FIELDS = BiasFields(
    enabled=_boolean_field("bias_option_flag", config.BIAS_OPTION_FLAG),
    option=_optional_implementation_field(
        "bias_option", config.BIAS_OPTION, DynamicBiasConfig
    ),
    decay_schedule=_enum_field(
        "bias_decay_schedule",
        config.BIAS_DECAY_SCHEDULE,
        WeightDecayScheduleOptions,
    ),
    decay_rate=_float_field("bias_decay_rate", config.BIAS_DECAY_RATE),
    decay_warmup_batches=_integer_field(
        "bias_decay_warmup_batches", config.BIAS_DECAY_WARMUP_BATCHES
    ),
    bank_expansion_factor=_enum_field(
        "bias_bank_expansion_factor",
        config.BIAS_BANK_EXPANSION_FACTOR,
        BankExpansionFactorOptions,
    ),
)
_DIAGONAL_FIELDS = DiagonalFields(
    enabled=_boolean_field("diagonal_option_flag", config.DIAGONAL_OPTION_FLAG),
    option=_optional_implementation_field(
        "diagonal_option", config.DIAGONAL_OPTION, DynamicDiagonalConfig
    ),
)
_MASK_FIELDS = MaskFields(
    enabled=_boolean_field("mask_option_flag", config.MASK_OPTION_FLAG),
    row_mask_option=_optional_implementation_field(
        "row_mask_option", config.ROW_MASK_OPTION, AxisMaskConfig
    ),
    threshold=_float_field("mask_threshold", config.MASK_THRESHOLD),
    floor=_float_field("mask_floor", config.MASK_FLOOR),
    transition_width=_float_field(
        "mask_transition_width", config.MASK_TRANSITION_WIDTH
    ),
    surrogate_scale=_float_field("mask_surrogate_scale", config.MASK_SURROGATE_SCALE),
    dimension_option=_enum_field(
        "mask_dimension_option", config.MASK_DIMENSION_OPTION, MaskDimensionOptions
    ),
)
_INPUT_PROJECTION_FIELDS = ProjectionFields(
    weight_option=_optional_implementation_field(
        "input_layer_weight_option",
        config.INPUT_LAYER_WEIGHT_OPTION,
        DynamicWeightConfig,
    ),
    generator_depth=_enum_field(
        "input_layer_generator_depth",
        config.INPUT_LAYER_GENERATOR_DEPTH,
        DynamicDepthOptions,
    ),
    weight_decay_schedule=_enum_field(
        "input_layer_weight_decay_schedule",
        config.INPUT_LAYER_WEIGHT_DECAY_SCHEDULE,
        WeightDecayScheduleOptions,
    ),
    weight_decay_rate=_float_field(
        "input_layer_weight_decay_rate", config.INPUT_LAYER_WEIGHT_DECAY_RATE
    ),
    weight_decay_warmup_batches=_integer_field(
        "input_layer_weight_decay_warmup_batches",
        config.INPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES,
    ),
    weight_normalization_option=_enum_field(
        "input_layer_weight_normalization_option",
        config.INPUT_LAYER_WEIGHT_NORMALIZATION_OPTION,
        WeightNormalizationOptions,
    ),
    weight_normalization_position_option=_enum_field(
        "input_layer_weight_normalization_position_option",
        config.INPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION,
        WeightNormalizationPositionOptions,
    ),
    weight_bank_expansion_factor=_enum_field(
        "input_layer_weight_bank_expansion_factor",
        config.INPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR,
        BankExpansionFactorOptions,
    ),
    bias_option=_optional_implementation_field(
        "input_layer_bias_option",
        config.INPUT_LAYER_BIAS_OPTION,
        DynamicBiasConfig,
    ),
    bias_decay_schedule=_enum_field(
        "input_layer_bias_decay_schedule",
        config.INPUT_LAYER_BIAS_DECAY_SCHEDULE,
        WeightDecayScheduleOptions,
    ),
    bias_decay_rate=_float_field(
        "input_layer_bias_decay_rate", config.INPUT_LAYER_BIAS_DECAY_RATE
    ),
    bias_decay_warmup_batches=_integer_field(
        "input_layer_bias_decay_warmup_batches",
        config.INPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES,
    ),
    bias_bank_expansion_factor=_enum_field(
        "input_layer_bias_bank_expansion_factor",
        config.INPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR,
        BankExpansionFactorOptions,
    ),
    diagonal_option=_optional_implementation_field(
        "input_layer_diagonal_option",
        config.INPUT_LAYER_DIAGONAL_OPTION,
        DynamicDiagonalConfig,
    ),
    row_mask_option=_optional_implementation_field(
        "input_layer_row_mask_option",
        config.INPUT_LAYER_ROW_MASK_OPTION,
        AxisMaskConfig,
    ),
    mask_threshold=_float_field(
        "input_layer_mask_threshold", config.INPUT_LAYER_MASK_THRESHOLD
    ),
    mask_floor=_float_field("input_layer_mask_floor", config.INPUT_LAYER_MASK_FLOOR),
    mask_transition_width=_float_field(
        "input_layer_mask_transition_width",
        config.INPUT_LAYER_MASK_TRANSITION_WIDTH,
    ),
    mask_surrogate_scale=_float_field(
        "input_layer_mask_surrogate_scale", config.INPUT_LAYER_MASK_SURROGATE_SCALE
    ),
    mask_dimension_option=_enum_field(
        "input_layer_mask_dimension_option",
        config.INPUT_LAYER_MASK_DIMENSION_OPTION,
        MaskDimensionOptions,
    ),
)
_OUTPUT_PROJECTION_FIELDS = ProjectionFields(
    weight_option=_optional_implementation_field(
        "output_layer_weight_option",
        config.OUTPUT_LAYER_WEIGHT_OPTION,
        DynamicWeightConfig,
    ),
    generator_depth=_enum_field(
        "output_layer_generator_depth",
        config.OUTPUT_LAYER_GENERATOR_DEPTH,
        DynamicDepthOptions,
    ),
    weight_decay_schedule=_enum_field(
        "output_layer_weight_decay_schedule",
        config.OUTPUT_LAYER_WEIGHT_DECAY_SCHEDULE,
        WeightDecayScheduleOptions,
    ),
    weight_decay_rate=_float_field(
        "output_layer_weight_decay_rate", config.OUTPUT_LAYER_WEIGHT_DECAY_RATE
    ),
    weight_decay_warmup_batches=_integer_field(
        "output_layer_weight_decay_warmup_batches",
        config.OUTPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES,
    ),
    weight_normalization_option=_enum_field(
        "output_layer_weight_normalization_option",
        config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_OPTION,
        WeightNormalizationOptions,
    ),
    weight_normalization_position_option=_enum_field(
        "output_layer_weight_normalization_position_option",
        config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION,
        WeightNormalizationPositionOptions,
    ),
    weight_bank_expansion_factor=_enum_field(
        "output_layer_weight_bank_expansion_factor",
        config.OUTPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR,
        BankExpansionFactorOptions,
    ),
    bias_option=_optional_implementation_field(
        "output_layer_bias_option",
        config.OUTPUT_LAYER_BIAS_OPTION,
        DynamicBiasConfig,
    ),
    bias_decay_schedule=_enum_field(
        "output_layer_bias_decay_schedule",
        config.OUTPUT_LAYER_BIAS_DECAY_SCHEDULE,
        WeightDecayScheduleOptions,
    ),
    bias_decay_rate=_float_field(
        "output_layer_bias_decay_rate", config.OUTPUT_LAYER_BIAS_DECAY_RATE
    ),
    bias_decay_warmup_batches=_integer_field(
        "output_layer_bias_decay_warmup_batches",
        config.OUTPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES,
    ),
    bias_bank_expansion_factor=_enum_field(
        "output_layer_bias_bank_expansion_factor",
        config.OUTPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR,
        BankExpansionFactorOptions,
    ),
    diagonal_option=_optional_implementation_field(
        "output_layer_diagonal_option",
        config.OUTPUT_LAYER_DIAGONAL_OPTION,
        DynamicDiagonalConfig,
    ),
    row_mask_option=_optional_implementation_field(
        "output_layer_row_mask_option",
        config.OUTPUT_LAYER_ROW_MASK_OPTION,
        AxisMaskConfig,
    ),
    mask_threshold=_float_field(
        "output_layer_mask_threshold", config.OUTPUT_LAYER_MASK_THRESHOLD
    ),
    mask_floor=_float_field("output_layer_mask_floor", config.OUTPUT_LAYER_MASK_FLOOR),
    mask_transition_width=_float_field(
        "output_layer_mask_transition_width",
        config.OUTPUT_LAYER_MASK_TRANSITION_WIDTH,
    ),
    mask_surrogate_scale=_float_field(
        "output_layer_mask_surrogate_scale",
        config.OUTPUT_LAYER_MASK_SURROGATE_SCALE,
    ),
    mask_dimension_option=_enum_field(
        "output_layer_mask_dimension_option",
        config.OUTPUT_LAYER_MASK_DIMENSION_OPTION,
        MaskDimensionOptions,
    ),
)


def _read_dimensions(reader: RuntimeOverrideReader) -> RuntimeDimensionsValues:
    fields = _DIMENSION_FIELDS
    return RuntimeDimensionsValues(
        batch_size=reader.read(fields.batch_size),
        learning_rate=reader.read(fields.learning_rate),
        input_dim=reader.read(fields.input_dim),
        hidden_dim=reader.read(fields.hidden_dim),
        output_dim=reader.read(fields.output_dim),
    )


def _read_stack(reader: RuntimeOverrideReader, fields: StackFields) -> StackValues:
    return StackValues(
        hidden_dim=reader.read(fields.hidden_dim),
        num_layers=reader.read(fields.num_layers),
        last_layer_bias_option=reader.read(fields.last_layer_bias_option),
        apply_output_postprocessing_flag=reader.read(
            fields.apply_output_postprocessing_flag
        ),
        activation=reader.read(fields.activation),
        layer_norm_position=reader.read(fields.layer_norm_position),
        normalization=reader.read(fields.normalization),
        residual_connection_option=reader.read(fields.residual_connection_option),
        residual_block_size=reader.read(fields.residual_block_size),
        residual_rms_norm_epsilon=reader.read(fields.residual_rms_norm_epsilon),
        residual_model_flag=reader.read(fields.residual_model_flag),
        dropout_probability=reader.read(fields.dropout_probability),
        bias_flag=reader.read(fields.bias_flag),
    )


def _read_optional_stack(
    reader: RuntimeOverrideReader,
    fields: OptionalStackFields,
) -> OptionalStackValues:
    return OptionalStackValues(
        independent_flag=reader.read(fields.independent_flag),
        hidden_dim=reader.read(fields.hidden_dim),
        num_layers=reader.read(fields.num_layers),
        last_layer_bias_option=reader.read(fields.last_layer_bias_option),
        apply_output_postprocessing_flag=reader.read(
            fields.apply_output_postprocessing_flag
        ),
        activation=reader.read(fields.activation),
        layer_norm_position=reader.read(fields.layer_norm_position),
        normalization=reader.read(fields.normalization),
        residual_connection_option=reader.read(fields.residual_connection_option),
        residual_block_size=reader.read(fields.residual_block_size),
        residual_rms_norm_epsilon=reader.read(fields.residual_rms_norm_epsilon),
        residual_model_flag=reader.read(fields.residual_model_flag),
        dropout_probability=reader.read(fields.dropout_probability),
        bias_flag=reader.read(fields.bias_flag),
    )


def _read_control(reader: RuntimeOverrideReader) -> ControlValues:
    fields = _CONTROL_FIELDS
    return ControlValues(
        stack_gate_flag=reader.read(fields.stack_gate_flag),
        gate_option=reader.read(fields.gate_option),
        gate_activation=reader.read(fields.gate_activation),
        shared_gate_config=reader.read(fields.shared_gate_config),
        stack_halting_flag=reader.read(fields.stack_halting_flag),
        halting_option=reader.read(fields.halting_option),
        halting_threshold=reader.read(fields.halting_threshold),
        halting_dropout=reader.read(fields.halting_dropout),
        halting_hidden_state_mode=reader.read(fields.halting_hidden_state_mode),
        memory_flag=reader.read(fields.memory_flag),
        memory_option=reader.read(fields.memory_option),
        memory_position_option=reader.read(fields.memory_position_option),
        memory_learning_rate=reader.read(fields.memory_learning_rate),
        memory_num_inner_steps=reader.read(fields.memory_num_inner_steps),
        recurrent_flag=reader.read(fields.recurrent_flag),
        recurrent_max_steps=reader.read(fields.recurrent_max_steps),
        recurrent_initial_iterations=reader.read(fields.recurrent_initial_iterations),
        recurrent_gradient_transition_count=reader.read(
            fields.recurrent_gradient_transition_count
        ),
        recurrent_no_gradient_transition_count=reader.read(
            fields.recurrent_no_gradient_transition_count
        ),
        recurrent_iteration_increment=reader.read(fields.recurrent_iteration_increment),
        recurrent_forward_calls_before_iteration_increment=reader.read(
            fields.recurrent_forward_calls_before_iteration_increment
        ),
        recurrent_smooth_iteration_growth_flag=reader.read(
            fields.recurrent_smooth_iteration_growth_flag
        ),
        recurrent_layer_norm_position=reader.read(fields.recurrent_layer_norm_position),
        recurrent_normalization=reader.read(fields.recurrent_normalization),
        recurrent_stack_gate_flag=reader.read(fields.recurrent_stack_gate_flag),
        recurrent_gate_option=reader.read(fields.recurrent_gate_option),
        recurrent_gate_activation=reader.read(fields.recurrent_gate_activation),
        recurrent_stack_halting_flag=reader.read(fields.recurrent_stack_halting_flag),
        recurrent_halting_option=reader.read(fields.recurrent_halting_option),
        recurrent_halting_threshold=reader.read(fields.recurrent_halting_threshold),
        recurrent_halting_dropout=reader.read(fields.recurrent_halting_dropout),
        recurrent_halting_hidden_state_mode=reader.read(
            fields.recurrent_halting_hidden_state_mode
        ),
    )


def _read_weight(reader: RuntimeOverrideReader) -> WeightValues:
    fields = _WEIGHT_FIELDS
    return WeightValues(
        enabled=reader.read(fields.enabled),
        option=reader.read(fields.option),
        generator_depth=reader.read(fields.generator_depth),
        decay_schedule=reader.read(fields.decay_schedule),
        decay_rate=reader.read(fields.decay_rate),
        decay_warmup_batches=reader.read(fields.decay_warmup_batches),
        normalization_option=reader.read(fields.normalization_option),
        normalization_position_option=reader.read(fields.normalization_position_option),
        bank_expansion_factor=reader.read(fields.bank_expansion_factor),
    )


def _read_bias(reader: RuntimeOverrideReader) -> BiasValues:
    fields = _BIAS_FIELDS
    return BiasValues(
        enabled=reader.read(fields.enabled),
        option=reader.read(fields.option),
        decay_schedule=reader.read(fields.decay_schedule),
        decay_rate=reader.read(fields.decay_rate),
        decay_warmup_batches=reader.read(fields.decay_warmup_batches),
        bank_expansion_factor=reader.read(fields.bank_expansion_factor),
    )


def _read_diagonal(reader: RuntimeOverrideReader) -> DiagonalValues:
    fields = _DIAGONAL_FIELDS
    return DiagonalValues(
        enabled=reader.read(fields.enabled),
        option=reader.read(fields.option),
    )


def _read_mask(reader: RuntimeOverrideReader) -> MaskValues:
    fields = _MASK_FIELDS
    return MaskValues(
        enabled=reader.read(fields.enabled),
        row_mask_option=reader.read(fields.row_mask_option),
        threshold=reader.read(fields.threshold),
        floor=reader.read(fields.floor),
        transition_width=reader.read(fields.transition_width),
        surrogate_scale=reader.read(fields.surrogate_scale),
        dimension_option=reader.read(fields.dimension_option),
    )


def _read_projection(
    reader: RuntimeOverrideReader,
    fields: ProjectionFields,
) -> ProjectionValues:
    return ProjectionValues(
        weight_option=reader.read(fields.weight_option),
        generator_depth=reader.read(fields.generator_depth),
        weight_decay_schedule=reader.read(fields.weight_decay_schedule),
        weight_decay_rate=reader.read(fields.weight_decay_rate),
        weight_decay_warmup_batches=reader.read(fields.weight_decay_warmup_batches),
        weight_normalization_option=reader.read(fields.weight_normalization_option),
        weight_normalization_position_option=reader.read(
            fields.weight_normalization_position_option
        ),
        weight_bank_expansion_factor=reader.read(fields.weight_bank_expansion_factor),
        bias_option=reader.read(fields.bias_option),
        bias_decay_schedule=reader.read(fields.bias_decay_schedule),
        bias_decay_rate=reader.read(fields.bias_decay_rate),
        bias_decay_warmup_batches=reader.read(fields.bias_decay_warmup_batches),
        bias_bank_expansion_factor=reader.read(fields.bias_bank_expansion_factor),
        diagonal_option=reader.read(fields.diagonal_option),
        row_mask_option=reader.read(fields.row_mask_option),
        mask_threshold=reader.read(fields.mask_threshold),
        mask_floor=reader.read(fields.mask_floor),
        mask_transition_width=reader.read(fields.mask_transition_width),
        mask_surrogate_scale=reader.read(fields.mask_surrogate_scale),
        mask_dimension_option=reader.read(fields.mask_dimension_option),
    )


def _generation_stack_fields(prefix: str) -> OptionalStackFields:
    return OptionalStackFields(
        independent_flag=_boolean_field(
            f"{prefix}_independent_flag",
            getattr(config, f"{prefix.upper()}_INDEPENDENT_FLAG"),
        ),
        hidden_dim=_optional_integer_field(
            f"{prefix}_hidden_dim",
            getattr(config, f"{prefix.upper()}_HIDDEN_DIM"),
        ),
        layer_norm_position=_optional_enum_field(
            f"{prefix}_layer_norm_position",
            getattr(config, f"{prefix.upper()}_LAYER_NORM_POSITION"),
            LayerNormPositionOptions,
        ),
        normalization=_optional_enum_field(
            f"{prefix}_normalization",
            getattr(config, f"{prefix.upper()}_NORMALIZATION"),
            NormalizationOptions,
        ),
        num_layers=_optional_integer_field(
            f"{prefix}_num_layers",
            getattr(config, f"{prefix.upper()}_NUM_LAYERS"),
        ),
        activation=_optional_enum_field(
            f"{prefix}_activation",
            getattr(config, f"{prefix.upper()}_ACTIVATION"),
            ActivationOptions,
        ),
        residual_connection_option=_optional_implementation_field(
            f"{prefix}_residual_connection_option",
            getattr(config, f"{prefix.upper()}_RESIDUAL_CONNECTION_OPTION"),
            ResidualConfig,
        ),
        residual_block_size=_optional_integer_field(
            f"{prefix}_residual_block_size",
            getattr(config, f"{prefix.upper()}_RESIDUAL_BLOCK_SIZE"),
        ),
        residual_rms_norm_epsilon=_optional_float_field(
            f"{prefix}_residual_rms_norm_epsilon",
            getattr(config, f"{prefix.upper()}_RESIDUAL_RMS_NORM_EPSILON"),
        ),
        residual_model_flag=_boolean_field(
            f"{prefix}_residual_model_flag",
            getattr(config, f"{prefix.upper()}_RESIDUAL_MODEL_FLAG"),
        ),
        dropout_probability=_optional_float_field(
            f"{prefix}_dropout_probability",
            getattr(config, f"{prefix.upper()}_DROPOUT_PROBABILITY"),
        ),
        last_layer_bias_option=_optional_enum_field(
            f"{prefix}_last_layer_bias_option",
            getattr(config, f"{prefix.upper()}_LAST_LAYER_BIAS_OPTION"),
            LastLayerBiasOptions,
        ),
        apply_output_postprocessing_flag=_optional_boolean_field(
            f"{prefix}_apply_output_postprocessing_flag",
            getattr(config, f"{prefix.upper()}_APPLY_OUTPUT_POSTPROCESSING_FLAG"),
        ),
        bias_flag=_optional_boolean_field(
            f"{prefix}_bias_flag",
            getattr(config, f"{prefix.upper()}_BIAS_FLAG"),
        ),
    )


def _read_generation(
    reader: RuntimeOverrideReader, prefix: str = ""
) -> GenerationValues:
    return GenerationValues(
        weight_input_factor_source=reader.read(
            _optional_enum_field(
                f"{prefix}weight_input_factor_source",
                getattr(config, f"{prefix.upper()}WEIGHT_INPUT_FACTOR_SOURCE"),
                LowRankFactorSourceOptions,
            )
        ),
        weight_output_factor_source=reader.read(
            _optional_enum_field(
                f"{prefix}weight_output_factor_source",
                getattr(config, f"{prefix.upper()}WEIGHT_OUTPUT_FACTOR_SOURCE"),
                LowRankFactorSourceOptions,
            )
        ),
        weight_mixture_num_experts=reader.read(
            _optional_integer_field(
                f"{prefix}weight_mixture_num_experts",
                getattr(config, f"{prefix.upper()}WEIGHT_MIXTURE_NUM_EXPERTS"),
            )
        ),
        bias_mixture_num_experts=reader.read(
            _optional_integer_field(
                f"{prefix}bias_mixture_num_experts",
                getattr(config, f"{prefix.upper()}BIAS_MIXTURE_NUM_EXPERTS"),
            )
        ),
        weight_mixture_top_k=reader.read(
            _optional_integer_field(
                f"{prefix}weight_mixture_top_k",
                getattr(config, f"{prefix.upper()}WEIGHT_MIXTURE_TOP_K"),
            )
        ),
        bias_mixture_top_k=reader.read(
            _optional_integer_field(
                f"{prefix}bias_mixture_top_k",
                getattr(config, f"{prefix.upper()}BIAS_MIXTURE_TOP_K"),
            )
        ),
        weight_mixture_normalize_probabilities_flag=reader.read(
            _optional_boolean_field(
                f"{prefix}weight_mixture_normalize_probabilities_flag",
                getattr(
                    config,
                    f"{prefix.upper()}WEIGHT_MIXTURE_NORMALIZE_PROBABILITIES_FLAG",
                ),
            )
        ),
        bias_mixture_normalize_probabilities_flag=reader.read(
            _optional_boolean_field(
                f"{prefix}bias_mixture_normalize_probabilities_flag",
                getattr(
                    config,
                    f"{prefix.upper()}BIAS_MIXTURE_NORMALIZE_PROBABILITIES_FLAG",
                ),
            )
        ),
        weight_input_factor_generator_stack=_read_optional_stack(
            reader,
            _generation_stack_fields(f"{prefix}weight_input_factor_generator_stack"),
        ),
        weight_output_factor_generator_stack=_read_optional_stack(
            reader,
            _generation_stack_fields(f"{prefix}weight_output_factor_generator_stack"),
        ),
        weight_coefficient_generator_stack=_read_optional_stack(
            reader,
            _generation_stack_fields(f"{prefix}weight_coefficient_generator_stack"),
        ),
        weight_mixture_router_generator_stack=_read_optional_stack(
            reader,
            _generation_stack_fields(f"{prefix}weight_mixture_router_generator_stack"),
        ),
        bias_mixture_router_generator_stack=_read_optional_stack(
            reader,
            _generation_stack_fields(f"{prefix}bias_mixture_router_generator_stack"),
        ),
    )


def _read_grouping(reader: RuntimeOverrideReader, prefix: str = ""):
    values = {}
    key = prefix + "grouping_scope"
    values["grouping_scope"] = reader.read(
        _optional_enum_field(
            key, getattr(config, key.upper()), AdaptiveParameterGroupingScopeOptions
        )
    )
    key = prefix + "group_count"
    values["group_count"] = reader.read(
        _optional_integer_field(key, getattr(config, key.upper()))
    )
    key = prefix + "chunk_size"
    values["chunk_size"] = reader.read(
        _optional_integer_field(key, getattr(config, key.upper()))
    )
    key = prefix + "grouping_sequence_length"
    values["grouping_sequence_length"] = reader.read(
        _optional_integer_field(key, getattr(config, key.upper()))
    )
    key = prefix + "grouping_input_order"
    values["grouping_input_order"] = reader.read(
        _optional_enum_field(
            key, getattr(config, key.upper()), AdaptiveParameterInputOrderOptions
        )
    )
    key = prefix + "grouping_method"
    values["grouping_method"] = reader.read(
        _optional_implementation_field(
            key, getattr(config, key.upper()), GroupingConfig
        )
    )
    key = prefix + "grouping_summary_normalization"
    values["grouping_summary_normalization"] = reader.read(
        _optional_enum_field(
            key,
            getattr(config, key.upper()),
            SummaryNormalizationOptions,
        )
    )
    key = prefix + "grouping_model_config"
    values["grouping_model_config"] = reader.read(
        _optional_instance_field(key, getattr(config, key.upper()), ConfigBase)
    )
    key = prefix + "grouping_attention_hidden_dim"
    values["grouping_attention_hidden_dim"] = reader.read(
        _optional_integer_field(key, getattr(config, key.upper()))
    )
    key = prefix + "grouping_rms_norm_epsilon"
    values["grouping_rms_norm_epsilon"] = reader.read(
        _optional_float_field(key, getattr(config, key.upper()))
    )
    return grouping_from_fields(values, prefix=prefix or "main")


def _read_runtime_values(reader: RuntimeOverrideReader) -> RuntimeDefaultValues:
    return RuntimeDefaultValues(
        grouping=_read_grouping(reader),
        input_grouping=_read_grouping(reader, "input_layer_"),
        output_grouping=_read_grouping(reader, "output_layer_"),
        generation=_read_generation(reader),
        input_generation=_read_generation(reader, "input_layer_"),
        output_generation=_read_generation(reader, "output_layer_"),
        dimensions=_read_dimensions(reader),
        main_stack=_read_stack(reader, _MAIN_STACK_FIELDS),
        submodule_stack=_read_stack(reader, _SUBMODULE_STACK_FIELDS),
        adaptive_generator_stack=_read_stack(reader, _ADAPTIVE_GENERATOR_STACK_FIELDS),
        residual_stack=_read_optional_stack(reader, _RESIDUAL_STACK_FIELDS),
        gate_stack=_read_optional_stack(reader, _GATE_STACK_FIELDS),
        halting_stack=_read_optional_stack(reader, _HALTING_STACK_FIELDS),
        memory_stack=_read_optional_stack(reader, _MEMORY_STACK_FIELDS),
        recurrent_gate_stack=_read_optional_stack(reader, _RECURRENT_GATE_STACK_FIELDS),
        recurrent_halting_stack=_read_optional_stack(
            reader, _RECURRENT_HALTING_STACK_FIELDS
        ),
        weight_generator_stack=_read_optional_stack(
            reader, _WEIGHT_GENERATOR_STACK_FIELDS
        ),
        bias_generator_stack=_read_optional_stack(reader, _BIAS_GENERATOR_STACK_FIELDS),
        diagonal_generator_stack=_read_optional_stack(
            reader, _DIAGONAL_GENERATOR_STACK_FIELDS
        ),
        mask_generator_stack=_read_optional_stack(reader, _MASK_GENERATOR_STACK_FIELDS),
        control=_read_control(reader),
        weight=_read_weight(reader),
        bias=_read_bias(reader),
        diagonal=_read_diagonal(reader),
        mask=_read_mask(reader),
        input_projection=_read_projection(reader, _INPUT_PROJECTION_FIELDS),
        output_projection=_read_projection(reader, _OUTPUT_PROJECTION_FIELDS),
    )


def _catalog_fields() -> dict[str, _BoundaryField]:
    reader = RuntimeOverrideReader({})
    _read_runtime_values(reader)
    return reader.fields


_FIELDS = _catalog_fields()


def _normalize_overrides(overrides: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for supplied_key, value in overrides.items():
        if not isinstance(supplied_key, str):
            raise TypeError(
                f"{_PACKAGE}: runtime override keys must be str, got "
                f"{type(supplied_key).__name__}"
            )
        runtime_field = _FIELDS.get(supplied_key)
        if runtime_field is None:
            accepted = ", ".join(sorted(_FIELDS))
            raise ValueError(
                f"{_PACKAGE}: unknown runtime key {supplied_key!r}; "
                f"accepted keys: {accepted}"
            )
        if not runtime_field.accepts(value):
            _raise_type_error(supplied_key, value, runtime_field.expected)
        normalized[supplied_key] = value
    return normalized


def runtime_default_values(
    overrides: Mapping[str, object] | None,
) -> RuntimeDefaultValues:
    normalized = _normalize_overrides(overrides or {})
    return _read_runtime_values(RuntimeOverrideReader(normalized))
