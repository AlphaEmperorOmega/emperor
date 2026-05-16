from torch import Tensor

from emperor.base.validator import ValidatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.sampler.model import SamplerModel
    from emperor.sampler.core.routers import RouterModel
    from emperor.sampler.core.samplers import (
        SamplerBase,
        SamplerSparse,
        SamplerTopk,
        SamplerFull,
)


class SamplerModelValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"router_config"}

    @staticmethod
    def validate(model: "SamplerModel") -> None:
        SamplerModelValidator.validate_required_fields(model.sampler_config)
        SamplerModelValidator.validate_field_types(model.sampler_config)
        SamplerModelValidator.validate_sampler_dimensions(model)
        SamplerModelValidator.validate_router_config(model.router_config)

    @staticmethod
    def validate_sampler_dimensions(model: "SamplerModel") -> None:
        SamplerBaseValidator.validate_positive_integer(
            "top_k", model.sampler_config.top_k
        )
        SamplerBaseValidator.validate_positive_integer(
            "num_experts", model.num_experts
        )
        if model.sampler_config.top_k > model.num_experts:
            raise ValueError(
                "top_k cannot exceed num_experts for SamplerModel, "
                f"received top_k={model.sampler_config.top_k}, "
                f"num_experts={model.num_experts}."
            )

    @staticmethod
    def validate_router_config(router_config) -> None:
        if router_config is None:
            return
        from emperor.sampler.core.config import RouterConfig

        if not isinstance(router_config, RouterConfig):
            raise TypeError(
                "router_config must be a RouterConfig for SamplerModel, "
                f"got {type(router_config).__name__}."
            )

    @staticmethod
    def validate_input_matrix(input_matrix) -> None:
        if not isinstance(input_matrix, Tensor):
            raise TypeError(
                "input_matrix must be a Tensor, "
                f"received {type(input_matrix).__name__}."
            )
        if input_matrix.dim() != 2:
            raise ValueError(
                "SamplerModel expects a 2D input tensor (batch_size, features), "
                f"received a {input_matrix.dim()}D tensor with shape "
                f"{tuple(input_matrix.shape)}."
            )


class RouterModelValidator(ValidatorBase):
    @staticmethod
    def validate(model: "RouterModel") -> None:
        RouterModelValidator.validate_required_fields(model.cfg)
        RouterModelValidator.validate_field_types(model.cfg)
        RouterModelValidator.validate_dimensions(
            input_dim=model.input_dim,
            num_experts=model.num_experts,
        )
        RouterModelValidator.validate_positive_integer(
            "input_dim", model.input_dim
        )
        RouterModelValidator.validate_positive_integer(
            "num_experts", model.num_experts
        )
        RouterModelValidator.validate_model_config(model.cfg.model_config)

    @staticmethod
    def validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, received {value!r}.")

    @staticmethod
    def validate_model_config(model_config: "LayerStackConfig") -> None:
        from emperor.base.layer import LayerStackConfig

        if not isinstance(model_config, LayerStackConfig):
            raise TypeError(
                "model_config must be a LayerStackConfig for RouterConfig, "
                f"got {type(model_config).__name__}."
            )

    @staticmethod
    def validate_input_batch(model: "RouterModel", input_batch) -> None:
        if not isinstance(input_batch, Tensor):
            raise TypeError(
                "RouterModel input_batch must be a Tensor, "
                f"received {type(input_batch).__name__}."
            )
        if input_batch.dim() != 2:
            raise ValueError(
                "RouterModel expects a 2D input tensor (batch_size, input_dim), "
                f"received a {input_batch.dim()}D tensor with shape "
                f"{tuple(input_batch.shape)}."
            )
        if input_batch.shape[-1] != model.input_dim:
            raise ValueError(
                "RouterModel input feature dimension must match input_dim, "
                f"received input_dim={model.input_dim} and input shape "
                f"{tuple(input_batch.shape)}."
            )


class SamplerBaseValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"router_config"}

    @staticmethod
    def validate(model: "SamplerBase") -> None:
        SamplerBaseValidator.validate_required_fields(model.cfg)
        SamplerBaseValidator.validate_field_types(model.cfg)
        SamplerBaseValidator.validate_positive_integer("top_k", model.top_k)
        SamplerBaseValidator.validate_positive_integer(
            "num_experts", model.num_experts
        )
        SamplerBaseValidator.validate_non_negative_integer(
            "num_topk_samples", model.num_topk_samples
        )
        SamplerBaseValidator.validate_probability("threshold", model.threshold)
        SamplerBaseValidator.validate_non_negative_float(
            "coefficient_of_variation_loss_weight",
            model.coefficient_of_variation_loss_weight,
        )
        SamplerBaseValidator.validate_non_negative_float(
            "switch_loss_weight", model.switch_loss_weight
        )
        SamplerBaseValidator.validate_non_negative_float(
            "zero_centred_loss_weight", model.zero_centred_loss_weight
        )
        SamplerBaseValidator.validate_non_negative_float(
            "mutual_information_loss_weight",
            model.mutual_information_loss_weight,
        )
        SamplerBaseValidator.validate_num_topk_samples(model)

    @staticmethod
    def validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, received {value!r}.")

    @staticmethod
    def validate_non_negative_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or value < 0:
            raise ValueError(
                f"{name} must be a non-negative integer, received {value!r}."
            )

    @staticmethod
    def validate_probability(name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0.0 and 1.0 inclusive, received {value!r}."
            )

    @staticmethod
    def validate_non_negative_float(name: str, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0.0, received {value!r}.")

    @staticmethod
    def validate_num_topk_samples(model: "SamplerBase") -> None:
        if model.num_topk_samples > model.top_k:
            raise ValueError(
                "num_topk_samples cannot exceed top_k, "
                f"received num_topk_samples={model.num_topk_samples}, "
                f"top_k={model.top_k}."
            )

    @staticmethod
    def validate_router_logit_scores(model: "SamplerBase", router_logit_scores) -> None:
        if not isinstance(router_logit_scores, Tensor):
            raise TypeError(
                "router_logit_scores must be a Tensor, "
                f"received {type(router_logit_scores).__name__}."
            )
        if router_logit_scores.dim() != 2:
            raise ValueError(
                "router_logit_scores must be a 2D tensor "
                "(batch_size, num_experts), received a "
                f"{router_logit_scores.dim()}D tensor with shape "
                f"{tuple(router_logit_scores.shape)}."
            )
        expected_dim = (
            model.num_experts * 2 if model.noisy_topk_flag else model.num_experts
        )
        if router_logit_scores.shape[-1] != expected_dim:
            raise ValueError(
                "router_logit_scores feature dimension is invalid, "
                f"expected {expected_dim}, received shape "
                f"{tuple(router_logit_scores.shape)}."
            )

    @staticmethod
    def validate_skip_mask(router_logit_scores: Tensor, skip_mask) -> None:
        if skip_mask is None:
            return
        if not isinstance(skip_mask, Tensor):
            raise TypeError(
                f"skip_mask must be a Tensor when provided, received {type(skip_mask).__name__}."
            )
        if skip_mask.shape[0] != router_logit_scores.shape[0]:
            raise ValueError(
                "skip_mask batch dimension must match router_logit_scores, "
                f"received skip_mask shape {tuple(skip_mask.shape)} and "
                f"router_logit_scores shape {tuple(router_logit_scores.shape)}."
            )


class SamplerSparseValidator(SamplerBaseValidator):
    @staticmethod
    def validate(model: "SamplerSparse") -> None:
        if model.top_k != 1:
            raise ValueError(
                f"top_k must be 1 when using SamplerSparse, received {model.top_k}."
            )
        if model.normalize_probabilities_flag is not False:
            raise ValueError(
                "normalize_probabilities_flag must be False when using SamplerSparse, "
                f"received {model.normalize_probabilities_flag!r}."
            )
        if model.num_topk_samples != 0:
            raise ValueError(
                "num_topk_samples must be 0 when using SamplerSparse, "
                f"received {model.num_topk_samples}."
            )
        if model.mutual_information_loss_weight != 0.0:
            raise ValueError(
                "mutual_information_loss_weight must be 0.0 when using SamplerSparse, "
                f"received {model.mutual_information_loss_weight}."
            )


class SamplerTopkValidator(SamplerBaseValidator):
    @staticmethod
    def validate(model: "SamplerTopk") -> None:
        if not (0 < model.top_k < model.num_experts):
            raise ValueError(
                "top_k must be greater than 0 and less than num_experts when using "
                f"SamplerTopk, received top_k={model.top_k}, "
                f"num_experts={model.num_experts}."
            )


class SamplerFullValidator(SamplerBaseValidator):
    @staticmethod
    def validate(model: "SamplerFull") -> None:
        if model.num_topk_samples != 0:
            raise ValueError(
                "num_topk_samples must be 0 when using SamplerFull, "
                f"received {model.num_topk_samples}."
            )
        if model.coefficient_of_variation_loss_weight != 0.0:
            raise ValueError(
                "coefficient_of_variation_loss_weight must be 0.0 when using "
                f"SamplerFull, received {model.coefficient_of_variation_loss_weight}."
            )
        if model.switch_loss_weight != 0.0:
            raise ValueError(
                "switch_loss_weight must be 0.0 when using SamplerFull, "
                f"received {model.switch_loss_weight}."
            )
        if model.zero_centred_loss_weight != 0.0:
            raise ValueError(
                "zero_centred_loss_weight must be 0.0 when using SamplerFull, "
                f"received {model.zero_centred_loss_weight}."
            )
        if model.mutual_information_loss_weight != 0.0:
            raise ValueError(
                "mutual_information_loss_weight must be 0.0 when using SamplerFull, "
                f"received {model.mutual_information_loss_weight}."
            )
        if model.top_k != model.num_experts:
            raise ValueError(
                "top_k must be equal to num_experts when using SamplerFull, "
                f"received top_k={model.top_k}, num_experts={model.num_experts}."
            )
