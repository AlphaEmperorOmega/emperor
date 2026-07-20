import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.sampler._selection.base import SamplerBase
    from emperor.sampler._selection.full import SamplerFull
    from emperor.sampler._selection.sparse import SamplerSparse
    from emperor.sampler._selection.top_k import SamplerTopk


class SamplerBaseValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"router_config"}

    @classmethod
    def validate(cls, model: "SamplerBase") -> None:
        cfg = model.cfg
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_positive_integer("top_k", cfg.top_k)
        cls.validate_positive_integer("num_experts", cfg.num_experts)
        cls.validate_non_negative_integer("num_topk_samples", cfg.num_topk_samples)
        cls.validate_probability("threshold", cfg.threshold)
        cls.validate_non_negative_float(
            "coefficient_of_variation_loss_weight",
            cfg.coefficient_of_variation_loss_weight,
        )
        cls.validate_non_negative_float("switch_loss_weight", cfg.switch_loss_weight)
        cls.validate_non_negative_float(
            "zero_centred_loss_weight", cfg.zero_centred_loss_weight
        )
        cls.validate_non_negative_float(
            "mutual_information_loss_weight",
            cfg.mutual_information_loss_weight,
        )
        cls._validate_num_topk_samples(cfg.num_topk_samples, cfg.top_k)

    @classmethod
    def validate_config(cls, cfg) -> None:
        validation_target = SimpleNamespace(
            cfg=cfg,
            **{
                field_name: getattr(cfg, field_name)
                for field_name in cfg.__dataclass_fields__
            },
        )
        cls.validate(validation_target)

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
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} must be finite and >= 0.0, received {value!r}."
            )

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
        if router_logit_scores.shape[1] != expected_dim:
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
        if skip_mask.shape[1] != 1:
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
        if model.top_k >= model.num_experts:
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
