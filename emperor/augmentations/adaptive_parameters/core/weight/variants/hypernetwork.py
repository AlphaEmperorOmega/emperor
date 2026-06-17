from torch import Tensor
from emperor.augmentations.adaptive_parameters.core.weight.depth_mapper import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters.core.weight.base import (
    DynamicWeightAbstract,
)
from emperor.augmentations.adaptive_parameters.core.weight.config import (
    HypernetworkDynamicWeightConfig,
)


class HypernetworkDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: HypernetworkDynamicWeightConfig,
        overrides: HypernetworkDynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)

        self.normalization_option = self.cfg.normalization_option
        self.model = self._init_model()

    def _init_model(self) -> DepthMappingLayerStack:
        model_overrides = DepthMappingHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim * self.output_dim,
        )
        return super()._init_model(model_overrides)

    def forward(
        self,
        weight_params: Tensor,
        X: Tensor,
    ) -> Tensor:
        logits = self.model(X)
        flat = self._apply_normalization_transform(logits)
        flat = self._compute_dynamic_weights(flat)
        update = flat.view(-1, self.input_dim, self.output_dim)
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + update
