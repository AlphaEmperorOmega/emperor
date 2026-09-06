import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._decay import DecayPolicy
from emperor.augmentations.adaptive_parameters._weights.config import (
    DynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters._weights.normalization import (
    WeightNormalizationPolicy,
)
from emperor.augmentations.adaptive_parameters._weights.validation import (
    DynamicWeightValidator,
)
from emperor.nn import Module


class DynamicWeightAbstract(Module):
    VALIDATOR = DynamicWeightValidator

    def __init__(
        self,
        cfg: "DynamicWeightConfig",
        overrides: "DynamicWeightConfig | None" = None,
    ):
        super().__init__()
        self.cfg: DynamicWeightConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.generator_depth = self.cfg.generator_depth
        self._decay_policy = DecayPolicy(self.cfg)
        self._normalization_policy = WeightNormalizationPolicy(self.cfg)

    def _init_model(
        self, overrides: "DepthMappingHandlerConfig"
    ) -> DepthMappingLayerStack:
        return DepthMappingLayerStack(self.cfg, overrides)

    def forward(
        self,
        weight_params: Tensor,
        X: Tensor,
    ) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__} must implement forward().")

    def _compute_dynamic_weights(self, outer_product: Tensor) -> Tensor:
        return outer_product.sum(dim=1)

    def _compute_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        input_vectors = self._normalization_policy.normalize_before_outer_product(
            input_vectors
        )
        output_vectors = self._normalization_policy.normalize_before_outer_product(
            output_vectors
        )
        outer_product = self._compute_raw_outer_product(input_vectors, output_vectors)
        return self._normalization_policy.normalize_after_outer_product(outer_product)

    def _compute_raw_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        return torch.einsum("bki,bkj->bkij", input_vectors, output_vectors)

    def _apply_normalization_transform(
        self,
        vectors: Tensor,
    ) -> Tensor:
        return self._normalization_policy(vectors)

    def _maybe_apply_weight_decay(self, weight_params: Tensor) -> Tensor:
        return self._decay_policy(weight_params)
