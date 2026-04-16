import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from dataclasses import dataclass, field
from emperor.base.utils import Module, ConfigBase
from emperor.augmentations.adaptive_parameters.options import (
    DynamicDepthOptions,
    DynamicWeightOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters.core.handlers.depth_mapper import (
    DepthMappingLayerStack,
)
from emperor.base.layer import (
    LayerStackConfig,
)


@dataclass
class WeightHandlerConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the weight transformation."},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the weight transformation."},
    )
    weight_option: DynamicWeightOptions | None = field(
        default=None,
        metadata={
            "help": "Selects the weight handler type for input-dependent weight adjustments."
        },
    )
    normalization: WeightNormalizationOptions | None = field(
        default=None,
        metadata={
            "help": "Normalization applied to vectors before the outer product computation."
        },
    )
    normalization_position: WeightNormalizationPositionOptions | None = field(
        default=None,
        metadata={
            "help": "Controls whether normalization is applied to the input vectors before or to the outer product after computation."
        },
    )
    generator_depth: DynamicDepthOptions | None = field(
        default=None,
        metadata={
            "help": "Depth of the generator network that produces input-dependent weight adjustments."
        },
    )
    bank_expansion_factor: int | None = field(
        default=None,
        metadata={
            "help": "Number of times default weight parameter bank will be scaled by for example (weight_bank_expansion_factor * input_dim, output_dim)"
        },
    )
    model_config: LayerStackConfig | None = field(
        default=None,
        metadata={
            "help": "Layer stack configuration for the internal generator network."
        },
    )
    decay_schedule: WeightDecayScheduleOptions | None = field(
        default=None,
        metadata={
            "help": "Schedule used to decay the base weight parameters over forward calls, eventually driving them to zero."
        },
    )
    decay_rate: float | None = field(
        default=None,
        metadata={
            "help": "Decay rate applied by the selected schedule. Interpretation depends on the schedule (exponent factor, linear slope, or multiplicative factor)."
        },
    )


class WeightHandlerAbstract(Module):
    def __init__(
        self,
        cfg: WeightHandlerConfig,
        overrides: WeightHandlerConfig | None = None,
    ):
        super().__init__()
        self.cfg: WeightHandlerConfig = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.generator_depth = self.cfg.generator_depth
        self.normalization_option = self.cfg.normalization
        self.normalization_position_option = self.cfg.normalization_position
        self.decay_schedule_option = self.cfg.decay_schedule
        self.decay_rate = self.cfg.decay_rate
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.clamp_limit = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("decay_step", torch.zeros(1))

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
        return self.__apply_normalization_by_position(
            input_vectors, output_vectors, self.normalization_position_option
        )

    def __apply_normalization_by_position(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
        position: WeightNormalizationPositionOptions,
    ) -> Tensor:
        match position:
            case WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT:
                return self._compute_prenormalized_outer_product(
                    input_vectors, output_vectors
                )
            case WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT:
                return self._compute_postnormalized_outer_product(
                    input_vectors, output_vectors
                )
            case WeightNormalizationPositionOptions.DISABLED:
                return self._compute_raw_outer_product(input_vectors, output_vectors)
            case _:
                raise ValueError(f"Unknown normalization position option: {position}")

    def _compute_prenormalized_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        input_vectors = self._apply_normalization_transform(input_vectors)
        output_vectors = self._apply_normalization_transform(output_vectors)
        return self._compute_raw_outer_product(input_vectors, output_vectors)

    def _compute_postnormalized_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        outer_product = self._compute_raw_outer_product(input_vectors, output_vectors)
        return self._apply_normalization_transform(outer_product)

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

    def _maybe_apply_weight_decay(self, weight_params: Tensor) -> Tensor:
        if (
            self.decay_schedule_option is None
            or self.decay_schedule_option == WeightDecayScheduleOptions.DISABLED
        ):
            return weight_params
        decay_factor = self.__compute_decay_factor_by_schedule(
            self.decay_schedule_option
        )
        self.decay_step += 1
        return weight_params * decay_factor

    def __compute_decay_factor_by_schedule(
        self,
        schedule: WeightDecayScheduleOptions,
    ) -> Tensor:
        step = self.decay_step
        rate = self.decay_rate
        match schedule:
            case WeightDecayScheduleOptions.EXPONENTIAL:
                return torch.exp(-rate * step)
            case WeightDecayScheduleOptions.LINEAR:
                return torch.clamp(1.0 - rate * step, min=0.0)
            case WeightDecayScheduleOptions.MULTIPLICATIVE:
                return torch.pow(torch.tensor(1.0 - rate), step)
            case _:
                raise ValueError(f"Unknown weight decay schedule option: {schedule}")


class SingleModelWeightHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: WeightHandlerConfig,
        overrides: WeightHandlerConfig | None = None,
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
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + dynamic_params


class DualModelWeightHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: WeightHandlerConfig,
        overrides: WeightHandlerConfig | None = None,
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
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + dynamic_params


class LowRankWeightHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: WeightHandlerConfig,
        overrides: WeightHandlerConfig | None = None,
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
        input_matrix = self._apply_normalization_transform(input_lowrank_matrix)
        input_matrix_transposed = input_matrix.transpose(1, 2)
        output_matrix = self._apply_normalization_transform(output_lowrank_matrix)
        dynamic_params = torch.bmm(input_matrix_transposed, output_matrix)
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + dynamic_params


class WeightMaskHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: WeightHandlerConfig,
        overrides: WeightHandlerConfig | None = None,
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
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params * mask


class HypernetworkWeightHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: WeightHandlerConfig,
        overrides: WeightHandlerConfig | None = None,
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
        flat = self._apply_normalization_transform(self.model(logits))
        flat = self._compute_dynamic_weights(flat)
        update = flat.view(-1, self.input_dim, self.output_dim)
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + update


class LayeredWeightedBankWeightHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: WeightHandlerConfig,
        overrides: WeightHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.bank_expansion_factor = self.cfg.bank_expansion_factor
        self.weight_bank = self._init_parameter_bank(
            (
                self.generator_depth,
                self.bank_expansion_factor * self.input_dim,
                self.output_dim,
            )
        )
        generator_overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.bank_expansion_factor * self.input_dim,
        )
        self.distribution_generator = self._init_generator_model(generator_overrides)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        bank_logits = self.distribution_generator(logits)
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        bank_distribution_reshaped = bank_distribution.unsqueeze(dim=3)
        batched_weighted_bank = self.weight_bank * bank_distribution_reshaped
        split_weights_by_factor = batched_weighted_bank.view(
            -1, self.input_dim, self.bank_expansion_factor, self.output_dim
        )
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + split_weights_by_factor.sum(dim=2)


class SoftWeightedBankWeightHandler(WeightHandlerAbstract):
    def __init__(
        self,
        cfg: WeightHandlerConfig,
        overrides: WeightHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)

        self.bank_expansion_factor = self.cfg.bank_expansion_factor
        self.weight_bank = self._init_parameter_bank(
            (
                self.generator_depth,
                self.input_dim,
                self.bank_expansion_factor,
                self.output_dim,
            )
        )

        generator_overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim * self.bank_expansion_factor,
        )
        self.distribution_generator = self._init_generator_model(generator_overrides)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        bank_logits = self.distribution_generator(logits)
        bank_logits = bank_logits.view(
            -1, self.generator_depth, self.input_dim, self.bank_expansion_factor
        )

        bank_distribution = torch.softmax(bank_logits, dim=-1)
        compressed_params = torch.einsum(
            "bdik,diko->bdio", bank_distribution, self.weight_bank
        )

        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + compressed_params
