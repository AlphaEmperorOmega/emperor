from copy import deepcopy
from typing import TYPE_CHECKING

from torch.nn import ModuleList

from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)

from ._validator import LayerStackValidator
from .base import LayerModuleBase
from .config import LayerConfig, LayerStackConfig
from .gate import GateConfig
from .layer import Layer
from .residual import ResidualConnectionOptions

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.halting.config import HaltingConfig
    from emperor.memory.config import DynamicMemoryConfig

    from .state import LayerState


class LayerStack(LayerModuleBase):
    VALIDATOR = LayerStackValidator

    SHARED_INPUT_OUTPUT_DIM = 1
    SEPARATE_INPUT_OUTPUT_DIM = 2

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
        layers = []
        layer_adjustment = self.__add_initial_layer(layers)
        self.__add_hidden_layers(layers, layer_adjustment)
        self.__add_output_layer(layers)
        self.__maybe_share_gate_model(layers)
        self.__maybe_share_halting_model(layers)
        self.__maybe_share_memory_model(layers)

        self._initialize_parameters(*layers)
        return ModuleList(layers)

    def __add_initial_layer(self, layers: list) -> int:
        if self.input_dim != self.hidden_dim and self.num_layers > 1:
            layer = self.__create_layer(self.input_dim, self.hidden_dim)
            layers.append(layer)
            return self.SEPARATE_INPUT_OUTPUT_DIM
        return self.SHARED_INPUT_OUTPUT_DIM

    def __add_hidden_layers(self, layers: list, layer_adjustment: int) -> None:
        for _ in range(self.num_layers - layer_adjustment):
            layer = self.__create_layer(self.hidden_dim, self.hidden_dim)
            layers.append(layer)

    def __add_output_layer(self, layers: list) -> None:
        layer_input_dim = self.hidden_dim if self.num_layers > 1 else self.input_dim
        overrides = self.__resolve_output_layer_overrides()
        layer = self.__create_layer(layer_input_dim, self.output_dim, overrides)
        layer.mark_as_last_layer()
        layers.append(layer)

    def __resolve_output_layer_overrides(self) -> "LayerConfig | None":
        overrides: LayerConfig | None = None
        if not self.apply_output_pipeline_flag:
            overrides = LayerConfig(
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
            )
        last_layer_bias = self.__resolve_last_layer_bias_override()
        overrides = self.__merge_layer_override(overrides, last_layer_bias)
        return overrides

    def __merge_layer_override(
        self,
        base: "LayerConfig | None",
        addition: "LayerConfig | None",
    ) -> "LayerConfig | None":
        if addition is None:
            return base
        if base is None:
            return addition
        return self._override_config(base, addition)

    def __resolve_last_layer_bias_override(self) -> LayerConfig | None:
        if self.last_layer_bias_option == LastLayerBiasOptions.DEFAULT:
            return None
        if not hasattr(self.layer_config.layer_model_config, "bias_flag"):
            return None

        model_config = deepcopy(self.layer_config.layer_model_config)
        match self.last_layer_bias_option:
            case LastLayerBiasOptions.DISABLED:
                model_config.bias_flag = False
            case LastLayerBiasOptions.ENABLED:
                model_config.bias_flag = True
        return LayerConfig(layer_model_config=model_config)

    def __create_layer(
        self,
        input_dim: int,
        output_dim: int,
        overrides: LayerConfig | None = None,
    ) -> Layer:
        has_stable_dimension = input_dim == output_dim
        residual_connection_option = (
            None if has_stable_dimension else ResidualConnectionOptions.DISABLED
        )
        dim_overrides = self._resolve_config_overrides(
            self.layer_config,
            input_dim=input_dim,
            output_dim=output_dim,
            residual_connection_option=residual_connection_option,
        )
        if overrides is not None:
            dim_overrides = self._override_config(dim_overrides, overrides)
        layer_config = self._override_config(self.layer_config, dim_overrides)
        if not has_stable_dimension:
            layer_config.gate_config = None
        return layer_config.build()

    def __maybe_share_gate_model(self, layers: list[Layer]) -> None:
        if self.shared_gate_config is None:
            return
        shared_gate = self._build_from_config(
            self.shared_gate_config,
            gate_dim=self.output_dim,
        )
        for layer in layers:
            layer.gate_config = self.shared_gate_config
            layer.gate_model = shared_gate

    def __maybe_share_halting_model(self, layers: list[Layer]) -> None:
        if self.shared_halting_config is None:
            return
        shared_halting_model = self._build_from_config(
            self.shared_halting_config,
            input_dim=self.output_dim,
        )
        for layer in layers:
            layer.halting_model = shared_halting_model

    def __maybe_share_memory_model(self, layers: list[Layer]) -> None:
        if self.shared_memory_config is None:
            return
        shared_memory_model = self._build_from_config(
            self.shared_memory_config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        for layer in layers:
            layer.memory_model = shared_memory_model

    def forward(self, state: "LayerState") -> "LayerState":
        for layer in self.layers:
            state = layer(state)
        return state
