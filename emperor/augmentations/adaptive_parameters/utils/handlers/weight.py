import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from emperor.base.utils import Module
from emperor.augmentations.adaptive_parameters.options import WeightNormalizationOptions
from emperor.augmentations.adaptive_parameters.utils.handlers.parameter import (
    DepthMappingLayerStack,
)
from emperor.base.layer import (
    LayerStackConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.config import (
        AdaptiveParameterAugmentationConfig,
    )


class WeightHandlerAbstract(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._override_config(
            cfg, overrides
        )
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.normalization_option = self.cfg.weight_normalization
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.clamp_limit = nn.Parameter(torch.tensor(1.0))

    def _init_generator_model(
        self, overrides: "LayerStackConfig"
    ) -> DepthMappingLayerStack:
        return DepthMappingLayerStack(self.cfg, overrides)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        return weight_params

    def _compute_dynamic_weights(self, outer_product: Tensor) -> Tensor:
        return outer_product.sum(dim=1)

    def _compute_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        input_vectors = self._normalize_vectors(input_vectors)
        output_vectors = self._normalize_vectors(output_vectors)
        return torch.einsum("bki,bkj->bkij", input_vectors, output_vectors)

    def _normalize_vectors(
        self,
        vectors: Tensor,
    ) -> Tensor:
        match self.normalization_option:
            case WeightNormalizationOptions.CLAMP:
                return torch.clamp(vectors, -self.clamp_limit, self.clamp_limit)
            case WeightNormalizationOptions.L2_SCALE:
                return F.normalize(vectors, dim=-1) * self.scale
            case WeightNormalizationOptions.SOFT_CLAMP:
                return self.clamp_limit * torch.tanh(vectors / self.clamp_limit)
            case WeightNormalizationOptions.RMS:
                rms = vectors.pow(2).mean(dim=-1, keepdim=True).sqrt()
                return vectors / (rms + 1e-8) * self.scale
            case WeightNormalizationOptions.SIGMOID_SCALE:
                return (torch.sigmoid(vectors) * 2 - 1) * self.scale
            case WeightNormalizationOptions.DISABLED:
                return vectors
            case _:
                raise ValueError(
                    f"Unknown weight normalization option: {self.normalization_option}"
                )


class SingleModelWeightHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self.__init_model()

    def __init_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        return self._init_generator_model(overrides)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        vectors = self.model(logits)
        outer_product = self._compute_outer_product(vectors, vectors)
        dynamic_params = self._compute_dynamic_weights(outer_product)
        return weight_params + dynamic_params


class DualModelWeightHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.input_model = self.__init_input_model()
        self.output_model = self.__init_output_model()

    def __init_input_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        return self._init_generator_model(overrides)

    def __init_output_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self._init_generator_model(overrides)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        input_vectors = self.input_model(logits)
        output_vectors = self.output_model(logits)
        outer_product = self._compute_outer_product(input_vectors, output_vectors)
        dynamic_params = self._compute_dynamic_weights(outer_product)
        return weight_params + dynamic_params


class LowRankWeightHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.input_model = self.__init_input_model()
        self.output_model = self.__init_output_model()

    def __init_input_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        return self._init_generator_model(overrides)

    def __init_output_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self._init_generator_model(overrides)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        input_lowrank_matrix = self.input_model(logits)
        output_lowrank_matrix = self.output_model(logits)
        input_matrix = self._normalize_vectors(input_lowrank_matrix)
        input_matrix_transposed = input_matrix.transpose(1, 2)
        output_matrix = self._normalize_vectors(output_lowrank_matrix)
        dynamic_params = torch.bmm(input_matrix_transposed, output_matrix)
        return weight_params + dynamic_params


class WeightMaskHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.input_model = self.__init_input_model()
        self.output_model = self.__init_output_model()

    def __init_input_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        return self._init_generator_model(overrides)

    def __init_output_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self._init_generator_model(overrides)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        input_vectors = self.input_model(logits)
        output_vectors = self.output_model(logits)
        outer_product = self._compute_outer_product(input_vectors, output_vectors)
        mask = self._compute_dynamic_weights(outer_product)
        return weight_params * mask


class HypernetworkWeightHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self.__init_model()

    def __init_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim * self.output_dim,
        )
        return self._init_generator_model(overrides)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        flat = self._normalize_vectors(self.model(logits))
        flat = self._compute_dynamic_weights(flat)
        update = flat.view(-1, self.input_dim, self.output_dim)
        return weight_params + update


class WeightedBankWeightHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.bank_expansion_factor = self.cfg.weight_bank_size
        self.weight_bank = self._init_parameter_bank(
            (1, self.bank_expansion_factor * self.input_dim, self.output_dim)
        )
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.bank_expansion_factor * self.input_dim,
        )
        self.distribution_generator = self._init_generator_model(overrides)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        bank_logits = self.distribution_generator(logits)
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        bank_distribution_reshaped = bank_distribution.unsqueeze(dim=2)
        batched_weighted_bank = self.weight_bank * bank_distribution_reshaped
        split_weghts_by_factor = batched_weighted_bank.view(
            -1, self.input_dim, self.bank_expansion_factor, self.output_dim
        )
        return weight_params + split_weghts_by_factor.sum(dim=2)
