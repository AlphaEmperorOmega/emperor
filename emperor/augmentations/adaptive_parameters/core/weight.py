import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from dataclasses import dataclass
from emperor.base.registry import subclass_registry
from emperor.base.utils import ConfigBase, Module, optional_field
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    DynamicWeightOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters.core.depth_mapper import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicWeightValidator,
)
from emperor.base.layer import (
    LayerStackConfig,
)


@dataclass
class DynamicWeightConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input dimensionality of the dynamic weight module."
    )
    output_dim: int | None = optional_field(
        "Output dimensionality of the dynamic weight module."
    )
    model_type: DynamicWeightOptions | None = optional_field(
        "Dynamic weight strategy used to generate input-dependent weight updates."
    )
    normalization_option: WeightNormalizationOptions | None = optional_field(
        "Normalization method applied during dynamic weight generation."
    )
    normalization_position_option: WeightNormalizationPositionOptions | None = (
        optional_field(
            "Specifies whether normalization is applied before the outer-product computation or after it."
        )
    )
    generator_depth: DynamicDepthOptions | None = optional_field(
        "Number of generator layers used to produce input-dependent weight updates."
    )
    bank_expansion_factor: BankExpansionFactorOptions | None = optional_field(
        "Expansion factor for bank-based weight strategies. Applies only to LAYERED_WEIGHTED_BANK and SOFT_WEIGHTED_BANK."
    )
    decay_schedule: WeightDecayScheduleOptions | None = optional_field(
        "Schedule used to decay the base weight parameters across forward passes."
    )
    decay_rate: float | None = optional_field(
        "Decay coefficient for the selected schedule. Its interpretation depends on the schedule type."
    )
    decay_warmup_batches: int | None = optional_field(
        "Number of initial batches to skip before applying weight decay."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Configuration for the internal generator network."
    )

    def _registry_owner(self) -> type:
        return DynamicWeightAbstract


@subclass_registry
class DynamicWeightAbstract(Module):
    def __init__(
        self,
        cfg: "DynamicWeightConfig",
        overrides: "DynamicWeightConfig | None" = None,
    ):
        super().__init__()
        self.cfg: DynamicWeightConfig = self._override_config(cfg, overrides)
        DynamicWeightValidator.validate(self)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.model_type = self.cfg.model_type
        self.generator_depth = self.cfg.generator_depth
        self.normalization_option = self.cfg.normalization_option
        self.normalization_position_option = self.cfg.normalization_position_option
        self.decay_schedule_option = self.cfg.decay_schedule
        self.decay_rate = self.cfg.decay_rate
        self.decay_warmup_batches = self.cfg.decay_warmup_batches or 0
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.clamp_limit = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("decay_step", torch.zeros(1))
        self.register_buffer("warmup_step", torch.zeros(1))

    def _init_model(self, overrides: "LayerStackConfig") -> DepthMappingLayerStack:
        return DepthMappingLayerStack(self.cfg, overrides)

    def forward(
        self,
        weight_params: Tensor,
        X: Tensor,
    ) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()."
        )

    def _compute_dynamic_weights(self, outer_product: Tensor) -> Tensor:
        return outer_product.sum(dim=1)

    def _compute_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        match self.normalization_position_option:
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
                raise ValueError(
                    "Unsupported normalization_position_option value: "
                    f"{self.normalization_position_option!r}."
                )

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
                    f"Unsupported normalization_option value: {self.normalization_option!r}."
                )

    def _maybe_apply_weight_decay(self, weight_params: Tensor) -> Tensor:
        if (
            self.decay_schedule_option is None
            or self.decay_schedule_option == WeightDecayScheduleOptions.DISABLED
        ):
            return weight_params
        if self.warmup_step < self.decay_warmup_batches:
            if self.training:
                self.warmup_step += 1
            return weight_params
        decay_factor = self.__compute_decay_factor_by_schedule(
            self.decay_schedule_option
        )
        if self.training:
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
                decay_base = step.new_tensor(1.0 - rate)
                return torch.pow(decay_base, step)
            case _:
                raise ValueError(f"Unsupported decay_schedule value: {schedule!r}.")


@DynamicWeightAbstract.register(DynamicWeightOptions.SINGLE_MODEL)
class SingleModelDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: DynamicWeightConfig,
        overrides: DynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        DynamicWeightValidator.validate_square_dimensions(self)
        self.model = self._init_model()

    def _init_model(self) -> DepthMappingLayerStack:
        model_overrides = LayerStackConfig(
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


@DynamicWeightAbstract.register(DynamicWeightOptions.DUAL_MODEL)
class DualModelDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: DynamicWeightConfig,
        overrides: DynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.input_model = self.__init_input_model()
        self.output_model = self.__init_output_model()

    def __init_input_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
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


@DynamicWeightAbstract.register(DynamicWeightOptions.LOW_RANK)
class LowRankDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: DynamicWeightConfig,
        overrides: DynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.input_model = self.__init_input_model()
        self.output_model = self.__init_output_model()

    def __init_input_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        return self._init_model(overrides)

    def __init_output_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
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


@DynamicWeightAbstract.register(DynamicWeightOptions.HYPERNETWORK)
class HypernetworkDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: DynamicWeightConfig,
        overrides: DynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model()

    def _init_model(self) -> DepthMappingLayerStack:
        model_overrides = LayerStackConfig(
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


@DynamicWeightAbstract.register(DynamicWeightOptions.LAYERED_WEIGHTED_BANK)
class LayeredWeightedBankDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: DynamicWeightConfig,
        overrides: DynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        DynamicWeightValidator.validate_bank_expansion_factor(self)
        self.bank_expansion_factor = self.cfg.bank_expansion_factor.value
        self.depth_value = self.generator_depth.value
        broadcast_batch = 1
        self.weight_bank = self._init_parameter_bank(
            (
                broadcast_batch,
                self.depth_value,
                self.bank_expansion_factor * self.input_dim,
                self.output_dim,
            )
        )
        self.model = self._init_model()

    def _init_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim * self.bank_expansion_factor,
        )
        return super()._init_model(overrides)

    def forward(
        self,
        weight_params: Tensor,
        X: Tensor,
    ) -> Tensor:
        bank_logits = self.model(X)
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        bank_distribution_reshaped = bank_distribution.unsqueeze(dim=-1)
        batched_weighted_bank = self.weight_bank * bank_distribution_reshaped
        split_weights_by_factor = batched_weighted_bank.view(
            -1,
            self.depth_value,
            self.input_dim,
            self.bank_expansion_factor,
            self.output_dim,
        )
        depth_and_expansion_reduced_weights = split_weights_by_factor.sum(dim=(1, 3))
        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        return decayed_weight_params + depth_and_expansion_reduced_weights


@DynamicWeightAbstract.register(DynamicWeightOptions.SOFT_WEIGHTED_BANK)
class SoftWeightedBankDynamicWeight(DynamicWeightAbstract):
    def __init__(
        self,
        cfg: DynamicWeightConfig,
        overrides: DynamicWeightConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        DynamicWeightValidator.validate_bank_expansion_factor(self)
        self.depth_value = self.generator_depth.value
        self.bank_expansion_factor = self.cfg.bank_expansion_factor.value
        self.weight_bank = self._init_parameter_bank(
            (
                self.depth_value,
                self.input_dim,
                self.bank_expansion_factor,
                self.output_dim,
            )
        )

        self.model = self._init_model()

    def _init_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim * self.bank_expansion_factor,
        )
        return super()._init_model(overrides)

    def forward(
        self,
        weight_params: Tensor,
        X: Tensor,
    ) -> Tensor:
        bank_logits = self.model(X)
        bank_logits = bank_logits.view(
            -1, self.depth_value, self.input_dim, self.bank_expansion_factor
        )

        bank_distribution = torch.softmax(bank_logits, dim=-1)
        compressed_params = torch.einsum(
            "bdik,diko->bdio", bank_distribution, self.weight_bank
        )

        decayed_weight_params = self._maybe_apply_weight_decay(weight_params)
        # TODO: return decayed_weight_params + compressed_params
        return decayed_weight_params.unsqueeze(0).expand(X.size(0), -1, -1)
