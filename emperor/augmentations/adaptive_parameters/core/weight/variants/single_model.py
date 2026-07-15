from torch import Tensor

from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicWeightValidator,
)
from emperor.augmentations.adaptive_parameters.core.weight.base import (
    DynamicWeightAbstract,
)
from emperor.augmentations.adaptive_parameters.core.weight.config import (
    SingleModelDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight.depth_mapper import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)


class SingleModelDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: SingleModelDynamicWeightConfig,
        overrides: SingleModelDynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.normalization_option = self.cfg.normalization_option
        self.normalization_position_option = self.cfg.normalization_position_option
        DynamicWeightValidator.validate_square_dimensions(self)
        self.model = self._init_model()

    def _init_model(self) -> DepthMappingLayerStack:
        model_overrides = DepthMappingHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        return super()._init_model(model_overrides)

    def forward(
        self,
        weight_params: Tensor,
        X: Tensor,
    ) -> Tensor:
        vectors = self.model(X)
        outer_product = self._compute_outer_product(vectors, vectors)
        dynamic_params = self._compute_dynamic_weights(outer_product)
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + dynamic_params
