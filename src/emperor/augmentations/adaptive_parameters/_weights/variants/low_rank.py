import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._weights.base import (
    DynamicWeightAbstract,
)
from emperor.augmentations.adaptive_parameters._weights.config import (
    LowRankDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)


class LowRankDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: LowRankDynamicWeightConfig,
        overrides: LowRankDynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.normalization_option = self.cfg.normalization_option
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
        input_lowrank_matrix = self.input_model(X)
        output_lowrank_matrix = self.output_model(X)
        input_matrix = self._apply_normalization_transform(input_lowrank_matrix)
        input_matrix_transposed = input_matrix.transpose(1, 2)
        output_matrix = self._apply_normalization_transform(output_lowrank_matrix)
        dynamic_params = torch.bmm(input_matrix_transposed, output_matrix)
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + dynamic_params
