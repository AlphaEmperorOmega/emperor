import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.sampler._router import RouterModel
    from emperor.sampler._sampler import SamplerModel


class SamplerModelValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"router_config"}

    @classmethod
    def validate(cls, model: "SamplerModel") -> None:
        cls.validate_required_fields(model.sampler_config)
        cls.validate_field_types(model.sampler_config)
        cls._validate_sampler_dimensions(model)
        cls._validate_router_config(model)

    @classmethod
    def validate_config(
        cls,
        sampler_config,
        *,
        router_input_dim: int | None = None,
    ) -> None:
        validation_target = SimpleNamespace(
            sampler_config=sampler_config,
            num_experts=sampler_config.num_experts,
            router_config=sampler_config.router_config,
        )
        cls.validate(validation_target)

        from emperor.sampler._selection.full import SamplerFull
        from emperor.sampler._selection.sparse import SamplerSparse
        from emperor.sampler._selection.top_k import SamplerTopk

        if sampler_config.top_k == 1:
            selection_owner = SamplerSparse
        elif sampler_config.top_k == sampler_config.num_experts:
            selection_owner = SamplerFull
        else:
            selection_owner = SamplerTopk
        selection_owner.VALIDATOR.validate_config(sampler_config)

        if sampler_config.router_config is None:
            return
        resolved_router_config = copy.deepcopy(sampler_config.router_config)
        if router_input_dim is not None:
            resolved_router_config.input_dim = router_input_dim
        router_owner = resolved_router_config._registry_owner()
        router_owner.VALIDATOR.validate_config(resolved_router_config)

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
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, received {value!r}.")

    @staticmethod
    def _validate_router_config(model: "SamplerModel") -> None:
        router_config = model.router_config
        if router_config is None:
            return
        from emperor.sampler._config import RouterConfig

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
    OPTIONAL_FIELDS = {"input_dim"}

    @classmethod
    def validate(cls, model: "RouterModel") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls._validate_input_dimension(model.input_dim)
        cls._validate_positive_integer("input_dim", model.input_dim)
        cls._validate_positive_integer("num_experts", model.num_experts)
        cls._validate_model_config(model.cfg.model_config)

    @classmethod
    def validate_config(cls, router_config) -> None:
        cls.validate(
            SimpleNamespace(
                cfg=router_config,
                input_dim=router_config.input_dim,
                num_experts=router_config.num_experts,
            )
        )

    @staticmethod
    def _validate_input_dimension(value: object) -> None:
        if value is None:
            raise ValueError("input_dim is required for RouterConfig, received None")
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"input_dim must be int for RouterConfig, got {type(value).__name__}"
            )

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, received {value!r}.")

    @staticmethod
    def _validate_model_config(model_config: object) -> None:
        from emperor.config import ConfigBase

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
        if input_batch.shape[1] != model.input_dim:
            raise ValueError(
                "RouterModel input feature dimension must match input_dim, "
                f"received input_dim={model.input_dim} and input shape "
                f"{tuple(input_batch.shape)}."
            )
