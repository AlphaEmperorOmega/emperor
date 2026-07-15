from typing import TYPE_CHECKING

from torch import Tensor

from emperor.base.validator import ValidatorBase

if TYPE_CHECKING:
    from emperor.sampler.core.base import SamplerBase
    from emperor.sampler.core.routers import RouterModel
    from emperor.sampler.core.variants import SamplerFull, SamplerSparse, SamplerTopk
    from emperor.sampler.model import SamplerModel


class SamplerModelValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"router_config"}

    @classmethod
    def validate(cls, model: "SamplerModel") -> None:
        cls.validate_required_fields(model.sampler_config)
        cls.validate_field_types(model.sampler_config)
        cls._validate_sampler_dimensions(model)
        cls._validate_router_config(model)

    @classmethod
    def _validate_sampler_dimensions(cls, model: "SamplerModel") -> None:
        cls._validate_positive_integer("top_k", model.sampler_config.top_k)
        cls._validate_positive_integer("num_experts", model.num_experts)
        if model.sampler_config.top_k > model.num_experts:
            raise ValueError(
                "top_k cannot exceed num_experts for SamplerModel, "
                f"received top_k={model.sampler_config.top_k}, "
                f"num_experts={model.num_experts}."
            )

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, received {value!r}.")

    @staticmethod
    def _validate_router_config(model: "SamplerModel") -> None:
        router_config = model.router_config
        if router_config is None:
            return
        from emperor.sampler.core.config import RouterConfig

        if not isinstance(router_config, RouterConfig):
            raise TypeError(
                "router_config must be a RouterConfig for SamplerModel, "
                f"got {type(router_config).__name__}."
            )
        if router_config.num_experts != model.sampler_config.num_experts:
            raise ValueError(
                "router_config.num_experts must match sampler_config.num_experts, "
                f"received router_config.num_experts={router_config.num_experts} and "
                f"sampler_config.num_experts={model.sampler_config.num_experts}."
            )
        if router_config.noisy_topk_flag != model.sampler_config.noisy_topk_flag:
            raise ValueError(
                "router_config.noisy_topk_flag must match "
                "sampler_config.noisy_topk_flag, received "
                f"router_config.noisy_topk_flag={router_config.noisy_topk_flag!r} "
                "and "
                f"sampler_config.noisy_topk_flag={model.sampler_config.noisy_topk_flag!r}."
            )

    @staticmethod
    def validate_forward_inputs(input_matrix) -> None:
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
    @classmethod
    def validate(cls, model: "RouterModel") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.input_dim,
            num_experts=model.num_experts,
        )
        cls._validate_positive_integer("input_dim", model.input_dim)
        cls._validate_positive_integer("num_experts", model.num_experts)
        cls._validate_model_config(model.cfg.model_config)

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, received {value!r}.")

    @staticmethod
    def _validate_model_config(model_config: object) -> None:
        from emperor.base.config import ConfigBase

        if not isinstance(model_config, ConfigBase):
            raise TypeError(
                "model_config must be a ConfigBase for RouterConfig, "
                f"got {type(model_config).__name__}."
            )

    @staticmethod
    def validate_forward_inputs(model: "RouterModel", input_batch) -> None:
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

    @classmethod
    def validate(cls, model: "SamplerBase") -> None:
        cfg = model.cfg
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_positive_integer("top_k", cfg.top_k)
        cls.validate_positive_integer("num_experts", cfg.num_experts)
        cls.validate_non_negative_integer(
            "num_topk_samples", cfg.num_topk_samples
        )
        cls.validate_probability("threshold", cfg.threshold)
        cls.validate_non_negative_float(
            "coefficient_of_variation_loss_weight",
            cfg.coefficient_of_variation_loss_weight,
        )
        cls.validate_non_negative_float(
            "switch_loss_weight", cfg.switch_loss_weight
        )
        cls.validate_non_negative_float(
            "zero_centred_loss_weight", cfg.zero_centred_loss_weight
        )
        cls.validate_non_negative_float(
            "mutual_information_loss_weight",
            cfg.mutual_information_loss_weight,
        )
        cls._validate_num_topk_samples(cfg.num_topk_samples, cfg.top_k)

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
    def _validate_num_topk_samples(num_topk_samples: int, top_k: int) -> None:
        if num_topk_samples > top_k:
            raise ValueError(
                "num_topk_samples cannot exceed top_k, "
                f"received num_topk_samples={num_topk_samples}, "
                f"top_k={top_k}."
            )

    @classmethod
    def validate_forward_inputs(
        cls,
        model: "SamplerBase",
        router_logit_scores,
        skip_mask,
    ) -> None:
        cls._validate_router_logit_scores(model, router_logit_scores)
        cls._validate_skip_mask(model, router_logit_scores, skip_mask)

    @staticmethod
    def _validate_router_logit_scores(
        model: "SamplerBase", router_logit_scores
    ) -> None:
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
    def _validate_skip_mask(
        model: "SamplerBase",
        router_logit_scores: Tensor,
        skip_mask,
    ) -> None:
        if skip_mask is None:
            return
        if not isinstance(skip_mask, Tensor):
            raise TypeError(
                "skip_mask must be a Tensor when provided, received "
                f"{type(skip_mask).__name__}."
            )
        if skip_mask.dim() != 2:
            raise ValueError(
                "skip_mask must be a 2D tensor with shape (batch_size, 1), "
                f"received a {skip_mask.dim()}D tensor with shape "
                f"{tuple(skip_mask.shape)}."
            )
        if skip_mask.shape[0] != router_logit_scores.shape[0]:
            raise ValueError(
                "skip_mask batch dimension must match router_logit_scores, "
                f"received skip_mask shape {tuple(skip_mask.shape)} and "
                f"router_logit_scores shape {tuple(router_logit_scores.shape)}."
            )
        if skip_mask.shape[-1] != 1:
            raise ValueError(
                "skip_mask feature dimension must be 1 so it broadcasts across "
                "experts, received skip_mask shape "
                f"{tuple(skip_mask.shape)} for num_experts={model.num_experts}."
            )


class SamplerSparseValidator(SamplerBaseValidator):
    @classmethod
    def validate(cls, model: "SamplerSparse") -> None:
        super().validate(model)
        cls._validate_sparse_configuration(model)

    @staticmethod
    def _validate_sparse_configuration(model: "SamplerSparse") -> None:
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
    @classmethod
    def validate(cls, model: "SamplerTopk") -> None:
        super().validate(model)
        cls._validate_topk_configuration(model)

    @staticmethod
    def _validate_topk_configuration(model: "SamplerTopk") -> None:
        if not (0 < model.top_k < model.num_experts):
            raise ValueError(
                "top_k must be greater than 0 and less than num_experts when using "
                f"SamplerTopk, received top_k={model.top_k}, "
                f"num_experts={model.num_experts}."
            )


class SamplerFullValidator(SamplerBaseValidator):
    @classmethod
    def validate(cls, model: "SamplerFull") -> None:
        super().validate(model)
        cls._validate_full_configuration(model)

    @staticmethod
    def _validate_full_configuration(model: "SamplerFull") -> None:
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
