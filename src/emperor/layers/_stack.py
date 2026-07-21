from copy import deepcopy
from typing import TYPE_CHECKING

from torch.nn import ModuleList

from emperor.layers._config import (
    GateConfig,
    LayerConfig,
    LayerStackConfig,
)
from emperor.layers._layer import Layer
from emperor.layers._options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.layers._support import LayerModuleBase
from emperor.layers._validation import LayerStackValidator

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.halting import HaltingConfig
    from emperor.layers._state import LayerState
    from emperor.memory import DynamicMemoryConfig


class LayerStack(LayerModuleBase):
    VALIDATOR = LayerStackValidator

    SHARED_INPUT_OUTPUT_DIM = 1
    SEPARATE_INPUT_OUTPUT_DIM = 2
    _supports_rectangular_gate = False

    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "layer_stack_config", cfg)
        self.cfg: LayerStackConfig = self._override_config(config, overrides)
        self.VALIDATOR.validate(self)

        self.input_dim: int = self.cfg.input_dim
        self.hidden_dim: int = self.cfg.hidden_dim
        self.output_dim: int = self.cfg.output_dim
        self.num_layers: int = self.cfg.num_layers
        self.apply_output_pipeline_flag: bool = self.cfg.apply_output_pipeline_flag
        self.last_layer_bias_option: LastLayerBiasOptions = (
            self.cfg.last_layer_bias_option
        )
        self.layer_config: LayerConfig = self.cfg.layer_config
        self.shared_gate_config: GateConfig | None = self.cfg.shared_gate_config
        self.shared_halting_config: HaltingConfig | None = (
            self.cfg.shared_halting_config
        )
        self.shared_memory_config: DynamicMemoryConfig | None = (
            self.cfg.shared_memory_config
        )
        self.layers = self.__build_layer_stack()

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]

    def __len__(self) -> int:
        return len(self.layers)

    def __build_layer_stack(self) -> ModuleList:
        dimensions = self._layer_dimensions()
        stack_layers = self.__create_stack_layers(dimensions)
        self.__maybe_share_gate_model(stack_layers)
        self.__maybe_share_halting_model(stack_layers)
        self.__maybe_share_memory_model(stack_layers)

        self._initialize_parameters(*stack_layers)
        return ModuleList(stack_layers)

    def _layer_dimensions(self) -> list[tuple[int, int]]:
        dimensions: list[tuple[int, int]] = []
        boundary_layer_count = self.__add_initial_layer_dimensions(dimensions)
        self.__add_hidden_layer_dimensions(dimensions, boundary_layer_count)
        self.__add_output_layer_dimensions(dimensions)
        return dimensions

    def __add_initial_layer_dimensions(
        self,
        dimensions: list[tuple[int, int]],
    ) -> int:
        if self.__requires_input_projection():
            dimensions.append((self.input_dim, self.hidden_dim))
            return self.SEPARATE_INPUT_OUTPUT_DIM
        return self.SHARED_INPUT_OUTPUT_DIM

    def __requires_input_projection(self) -> bool:
        input_dimension_differs_from_hidden_dimension = (
            self.input_dim != self.hidden_dim
        )
        stack_has_multiple_layers = self.num_layers > 1
        return (
            input_dimension_differs_from_hidden_dimension and stack_has_multiple_layers
        )

    def __add_hidden_layer_dimensions(
        self,
        dimensions: list[tuple[int, int]],
        boundary_layer_count: int,
    ) -> None:
        hidden_layer_count = self.num_layers - boundary_layer_count
        for _ in range(hidden_layer_count):
            dimensions.append((self.hidden_dim, self.hidden_dim))

    def __add_output_layer_dimensions(
        self,
        dimensions: list[tuple[int, int]],
    ) -> None:
        output_layer_input_dim = (
            self.hidden_dim if self.num_layers > 1 else self.input_dim
        )
        dimensions.append((output_layer_input_dim, self.output_dim))

    def __create_stack_layers(
        self,
        dimensions: list[tuple[int, int]],
    ) -> list[Layer]:
        stack_layers: list[Layer] = []
        layer_count = len(dimensions)
        for layer_number, (input_dim, output_dim) in enumerate(dimensions, start=1):
            is_last_layer = layer_number == layer_count
            stack_layer = self.__create_layer(
                input_dim,
                output_dim,
                is_last_layer=is_last_layer,
            )
            stack_layers.append(stack_layer)
        return stack_layers

    def __create_layer(
        self,
        input_dim: int,
        output_dim: int,
        layer_overrides: LayerConfig | None = None,
        is_last_layer: bool = False,
    ) -> Layer:
        has_stable_dimension = input_dim == output_dim
        resolved_layer_config = self.__resolve_layer_config(
            input_dim, output_dim, layer_overrides, is_last_layer
        )
        self.__apply_layer_compatibility_overrides(
            resolved_layer_config,
            is_last_layer,
            has_stable_dimension,
        )
        layer = resolved_layer_config.build()
        if is_last_layer:
            layer.mark_as_last_layer()
        return layer

    def __resolve_layer_config(
        self,
        input_dim: int,
        output_dim: int,
        layer_overrides: LayerConfig | None,
        is_last_layer: bool,
    ) -> LayerConfig:
        dimension_overrides = self._resolve_config_overrides(
            self.layer_config,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        if is_last_layer:
            output_layer_overrides = self.__resolve_output_layer_overrides()
            layer_overrides = self.__merge_layer_override(
                layer_overrides,
                output_layer_overrides,
            )
        if layer_overrides is not None:
            dimension_overrides = self._override_config(
                dimension_overrides, layer_overrides
            )
        resolved_layer_config = self._override_config(
            self.layer_config, dimension_overrides
        )
        return resolved_layer_config

    def __resolve_output_layer_overrides(self) -> "LayerConfig | None":
        output_layer_overrides: LayerConfig | None = None
        if not self.apply_output_pipeline_flag:
            output_layer_overrides = LayerConfig(
                activation=ActivationOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
            )
        last_layer_bias_override = self.__resolve_last_layer_bias_override()
        output_layer_overrides = self.__merge_layer_override(
            output_layer_overrides,
            last_layer_bias_override,
        )
        return output_layer_overrides

    def __resolve_last_layer_bias_override(self) -> LayerConfig | None:
        if self.last_layer_bias_option == LastLayerBiasOptions.DEFAULT:
            return None
        if not hasattr(self.layer_config.layer_model_config, "bias_flag"):
            return None

        last_layer_model_config = deepcopy(self.layer_config.layer_model_config)
        match self.last_layer_bias_option:
            case LastLayerBiasOptions.DISABLED:
                last_layer_model_config.bias_flag = False
            case LastLayerBiasOptions.ENABLED:
                last_layer_model_config.bias_flag = True
            case _:
                raise ValueError(
                    "Unsupported last layer bias option "
                    f"{self.last_layer_bias_option} for LayerStack."
                )
        return LayerConfig(layer_model_config=last_layer_model_config)

    def __merge_layer_override(
        self,
        base_override: "LayerConfig | None",
        additional_override: "LayerConfig | None",
    ) -> "LayerConfig | None":
        if additional_override is None:
            return base_override
        if base_override is None:
            return additional_override
        return self._override_config(base_override, additional_override)

    def __apply_layer_compatibility_overrides(
        self,
        resolved_layer_config: LayerConfig,
        is_last_layer: bool,
        has_stable_dimension: bool,
    ) -> None:
        should_disable_residual = self.__should_disable_residual(
            is_last_layer, has_stable_dimension
        )
        if should_disable_residual:
            resolved_layer_config.residual_config = None
        if not has_stable_dimension and not self._supports_rectangular_gate:
            resolved_layer_config.gate_config = None

    def __should_disable_residual(
        self,
        is_last_layer: bool,
        has_stable_dimension: bool,
    ) -> bool:
        output_layer_pipeline_is_disabled = (
            is_last_layer and not self.apply_output_pipeline_flag
        )
        layer_dimensions_do_not_support_residual = not has_stable_dimension
        return (
            output_layer_pipeline_is_disabled
            or layer_dimensions_do_not_support_residual
        )

    def __maybe_share_gate_model(self, stack_layers: list[Layer]) -> None:
        if self.shared_gate_config is None:
            return
        shared_gate_model = self._build_from_config(
            self.shared_gate_config,
            gate_dim=self.output_dim,
        )
        for stack_layer in stack_layers:
            stack_layer.gate_config = self.shared_gate_config
            stack_layer.gate_model = shared_gate_model

    def __maybe_share_halting_model(self, stack_layers: list[Layer]) -> None:
        if self.shared_halting_config is None:
            return
        shared_halting_model = self._build_from_config(
            self.shared_halting_config,
            input_dim=self.output_dim,
        )
        for stack_layer in stack_layers:
            stack_layer.halting_model = shared_halting_model

    def __maybe_share_memory_model(self, stack_layers: list[Layer]) -> None:
        if self.shared_memory_config is None:
            return
        shared_memory_model = self._build_from_config(
            self.shared_memory_config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        for stack_layer in stack_layers:
            stack_layer.memory_model = shared_memory_model

    def forward(self, state: "LayerState") -> "LayerState":
        layer_state = state
        for stack_layer in self.layers:
            layer_state = stack_layer(layer_state)
        return layer_state
