from torch import Tensor
from emperor.augmentations.adaptive_parameters.core.weight.depth_mapper import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters.core.weight.base import (
    DynamicWeightAbstract,
)
from emperor.augmentations.adaptive_parameters.core.weight.config import (
    DualModelDynamicWeightConfig,
)


class DualModelDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: DualModelDynamicWeightConfig,
        overrides: DualModelDynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.normalization_option = self.cfg.normalization_option
        self.normalization_position_option = self.cfg.normalization_position_option
        self.input_model = self.__init_input_model()
        self.output_model = self.__init_output_model()

    def __init_input_model(self) -> DepthMappingLayerStack:
        overrides = DepthMappingHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        return self._init_model(overrides)

    def __init_output_model(self) -> DepthMappingLayerStack:
        overrides = DepthMappingHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self._init_model(overrides)

    def forward(
        self,
        weight_params: Tensor,
        X: Tensor,
    ) -> Tensor:
        input_vectors = self.input_model(X)
        output_vectors = self.output_model(X)
        outer_product = self._compute_outer_product(input_vectors, output_vectors)
        dynamic_params = self._compute_dynamic_weights(outer_product)
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + dynamic_params
