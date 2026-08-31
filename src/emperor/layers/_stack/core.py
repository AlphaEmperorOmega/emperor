from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import Sequential

from emperor.layers._config import LayerStackConfig
from emperor.layers._layer import Layer
from emperor.layers._options import LastLayerBiasOptions
from emperor.layers._stack.builder import LayerStackBuilder
from emperor.layers._stack.shared_controllers import LayerStackSharedControllers
from emperor.layers._stack.topology import LayerStackTopology
from emperor.layers._stack.validation import LayerStackValidator
from emperor.layers._support import LayerModuleBase

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.layers._state import LayerState


class LayerStack(LayerModuleBase):
    VALIDATOR = LayerStackValidator

    _supports_rectangular_gate = False

    def __init__(
        self,
        cfg: LayerStackConfig | ModelConfig,
        overrides: LayerStackConfig | None = None,
    ):
        super().__init__()
        config = getattr(cfg, "layer_stack_config", cfg)
        self.cfg: LayerStackConfig = self._override_config(config, overrides)
        self.VALIDATOR.validate(self)
        self.__initialize_from_config()
        self.__initialize_delegates()
        self.layers = self.__build_layer_stack()

    def __initialize_from_config(self) -> None:
        self.input_dim: int = self.cfg.input_dim
        self.hidden_dim: int = self.cfg.hidden_dim
        self.output_dim: int = self.cfg.output_dim
        self.num_layers: int = self.cfg.num_layers
        self.apply_output_postprocessing_flag: bool = (
            self.cfg.apply_output_postprocessing_flag
        )
        self.last_layer_bias_option: LastLayerBiasOptions = (
            self.cfg.last_layer_bias_option
        )

    def __initialize_delegates(self) -> None:
        self.layer_dimension_resolver = LayerStackTopology(self.cfg)
        self.shared_controllers = LayerStackSharedControllers(self.cfg)
        self.stack_layer_builder = LayerStackBuilder(
            self.cfg,
            supports_rectangular_gate=self._supports_rectangular_gate,
        )

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]

    def __len__(self) -> int:
        return len(self.layers)

    def __build_layer_stack(self) -> Sequential:
        dimensions = self._layer_dimensions()
        stack_layers = self.stack_layer_builder.build_layer_stack(dimensions)
        self.shared_controllers.bind(stack_layers)

        self._initialize_parameters(*stack_layers)
        return Sequential(*stack_layers)

    def _layer_dimensions(self) -> tuple[tuple[int, int], ...]:
        return self.layer_dimension_resolver.resolve()

    def forward(self, state: LayerState) -> LayerState:
        with state.scoped_residual_state(None):
            return self.layers(state)
