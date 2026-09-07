from dataclasses import fields
from typing import TYPE_CHECKING

import torch
from torch.types import Tensor

from emperor._validation import ValidatorBase
from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveGeneratorValidatorBase,
)

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._weights.base import (
        DynamicWeightAbstract,
    )
    from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
        DepthMappingLayer,
        DepthMappingLayerConfig,
    )
    from emperor.layers import LayerStackConfig


class DynamicWeightValidator(AdaptiveGeneratorValidatorBase, ValidatorBase):
    OPTIONAL_FIELDS = {"bank_expansion_factor"}

    @staticmethod
    def validate_normalization_option(normalization_option) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            WeightNormalizationOptions,
        )

        if not isinstance(normalization_option, WeightNormalizationOptions):
            raise ValueError(
                f"Unsupported normalization_option value: {normalization_option!r}."
            )

    @staticmethod
    def validate_normalization_position_option(normalization_position_option) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            WeightNormalizationPositionOptions,
        )

        if not isinstance(
            normalization_position_option, WeightNormalizationPositionOptions
        ):
            raise ValueError(
                "Unsupported normalization_position_option value: "
                f"{normalization_position_option!r}."
            )

    @classmethod
    def validate_generator_stack(cls, config):
        from emperor.layers import Layer, LayerStack, LayerStackConfig
        from emperor.linears import LinearLayer

        if not isinstance(config, LayerStackConfig):
            raise TypeError("Active generator must use a LayerStackConfig.")
        if config.registry_owner() is not LayerStack:
            raise ValueError("Active generator must use an ordinary LayerStack.")
        layer = config.layer_config
        if layer is None or layer.registry_owner() is not Layer:
            raise ValueError("Active generator must use ordinary feed-forward Layers.")
        if (
            layer.layer_model_config is None
            or layer.layer_model_config.registry_owner() is not LinearLayer
        ):
            raise ValueError("Active generator must wrap ordinary LinearLayer configs.")
        for owner in (config, layer):
            for name in (
                "gate_config",
                "halting_config",
                "memory_config",
                "shared_gate_config",
                "shared_halting_config",
                "shared_memory_config",
            ):
                if getattr(owner, name, None) is not None:
                    raise ValueError(f"Active generator does not support {name}.")
        if config.hidden_dim is not None and (
            type(config.hidden_dim) is not int or config.hidden_dim <= 0
        ):
            raise ValueError("Generator hidden_dim must be a positive integer.")

    @classmethod
    def validate(cls, model: "DynamicWeightAbstract") -> None:
        cls.validate_initialization_fields(model)
        cls.validate_variant_config(model)
        cls.validate_model_config(model.cfg)

    @classmethod
    def validate_initialization_fields(cls, model: "DynamicWeightAbstract") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.cfg.input_dim,
            output_dim=model.cfg.output_dim,
        )
        cls.validate_decay_parameters(model.cfg)

    @classmethod
    def validate_variant_config(cls, model: "DynamicWeightAbstract") -> None:
        from emperor.augmentations.adaptive_parameters._weights.variants.layered_weighted_bank import (
            LayeredWeightedBankDynamicWeight,
        )
        from emperor.augmentations.adaptive_parameters._weights.variants.single_model import (
            SingleModelDynamicWeight,
        )
        from emperor.augmentations.adaptive_parameters._weights.variants.soft_weighted_bank import (
            SoftWeightedBankDynamicWeight,
        )

        if isinstance(model, SingleModelDynamicWeight):
            cls.validate_square_dimensions(model)
        if isinstance(
            model,
            (
                LayeredWeightedBankDynamicWeight,
                SoftWeightedBankDynamicWeight,
            ),
        ):
            cls.validate_bank_expansion_factor(model)

    @staticmethod
    def validate_bank_expansion_factor(model: "DynamicWeightAbstract") -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            BankExpansionFactorOptions,
        )

        factor = model.cfg.bank_expansion_factor
        if factor is None or not isinstance(factor, BankExpansionFactorOptions):
            raise ValueError(
                f"{type(model).__name__} requires bank_expansion_factor to be a "
                f"BankExpansionFactorOptions value, received {factor!r}."
            )
        if factor == BankExpansionFactorOptions.DISABLED:
            raise ValueError(
                f"{type(model).__name__} requires bank_expansion_factor > 0, "
                f"received {factor}. "
                f"Use FACTOR_OF_ONE, FACTOR_OF_TWO, FACTOR_OF_THREE, or "
                f"FACTOR_OF_FOUR."
            )

    @staticmethod
    def validate_square_dimensions(model: "DynamicWeightAbstract") -> None:
        if model.cfg.input_dim != model.cfg.output_dim:
            raise ValueError(
                f"{type(model).__name__} requires input_dim == output_dim, "
                f"received input_dim={model.cfg.input_dim}, "
                f"output_dim={model.cfg.output_dim}."
            )


class ModulatedLowRankValidator(DynamicWeightValidator):
    OPTIONAL_FIELDS = {
        "model_config",
        "input_factor_model_config",
        "output_factor_model_config",
    }

    @staticmethod
    def validate_supported_fields(cfg) -> None:
        if vars(cfg).keys() - {field.name for field in fields(cfg)} - {"_passed_args"}:
            raise ValueError("Modulated low-rank config contains unsupported fields.")

    @classmethod
    def validate(cls, model) -> None:
        cls.validate_supported_fields(model.cfg)
        cls.validate_initialization_fields(model)
        cls.validate_active_models(model.cfg)
        if model.cfg.generator_depth.value <= 0:
            raise ValueError("generator_depth must be greater than zero.")

    @classmethod
    def validate_active_models(cls, cfg, fallback=None) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            LowRankFactorSourceOptions,
        )

        default_model = cfg.model_config if cfg.model_config is not None else fallback
        for side in ("input", "output"):
            source = getattr(cfg, f"{side}_factor_source")
            if source is None:
                source = LowRankFactorSourceOptions.GENERATED
            if not isinstance(source, LowRankFactorSourceOptions):
                raise TypeError(
                    f"{side}_factor_source must be a LowRankFactorSourceOptions."
                )
            override = getattr(cfg, f"{side}_factor_model_config")
            if source is LowRankFactorSourceOptions.SHARED_PARAMETER:
                if override is not None:
                    raise ValueError(
                        f"{side}_factor_model_config is invalid for SHARED_PARAMETER."
                    )
            else:
                cls.validate_generator_stack(
                    override if override is not None else default_model
                )
        cls.validate_generator_stack(
            cfg.coefficient_model_config
            if cfg.coefficient_model_config is not None
            else default_model
        )


class DepthMappingValidator(ValidatorBase):
    OPTIONAL_FIELDS: set[str] = set()

    @classmethod
    def validate(cls, model: "DepthMappingLayer") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.cfg.input_dim,
            output_dim=model.cfg.output_dim,
        )
        cls._validate_generator_depth(model.cfg)

    @staticmethod
    def _validate_generator_depth(cfg: "DepthMappingLayerConfig") -> None:
        if cfg.generator_depth is None or cfg.generator_depth.value == 0:
            raise ValueError(
                f"generator_depth must be greater than 0 for DepthMappingLayer, "
                f"received {cfg.generator_depth}. "
                f"Use DEPTH_OF_ONE, DEPTH_OF_TWO, or DEPTH_OF_THREE."
            )

    @staticmethod
    def validate_input_is_2d(
        input_batch: Tensor,
        input_dim: int | None = None,
    ) -> None:
        if not isinstance(input_batch, Tensor):
            raise TypeError(
                "DepthMappingLayerStack input must be a Tensor, "
                f"received {type(input_batch).__name__}."
            )
        if not input_batch.dim() == 2:
            raise ValueError(
                "DepthMappingLayerStack expects a 2D input tensor "
                "(batch_size, features), "
                f"received a {input_batch.dim()}D tensor with shape "
                f"{tuple(input_batch.shape)}."
            )
        if input_dim is not None and input_batch.shape[-1] != input_dim:
            raise ValueError(
                "DepthMappingLayerStack input feature dimension must match "
                "input_dim, "
                f"received input_dim={input_dim} and input shape "
                f"{tuple(input_batch.shape)}."
            )

    @staticmethod
    def validate_inner_model_is_linear_layer_config(
        model_config: "LayerStackConfig",
    ) -> None:
        from emperor.linears import LinearLayerConfig

        inner_config = model_config.layer_config.layer_model_config
        if not isinstance(inner_config, LinearLayerConfig):
            raise TypeError(
                "model_config.layer_config.layer_model_config must be a "
                "LinearLayerConfig, "
                f"received {type(inner_config).__name__}."
            )

    @staticmethod
    def validate_layer_config_has_no_gate_or_halting(
        model_config: "LayerStackConfig",
    ) -> None:
        layer_config = model_config.layer_config
        if layer_config.gate_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support gate_config. "
                "Set gate_config to None."
            )
        if model_config.shared_gate_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support shared_gate_config. "
                "Set shared_gate_config to None."
            )
        if layer_config.halting_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support halting_config. "
                "Set halting_config to None."
            )
        if model_config.shared_halting_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support shared_halting_config. "
                "Set shared_halting_config to None."
            )
        if layer_config.memory_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support memory_config. "
                "Set memory_config to None."
            )
        if model_config.shared_memory_config is not None:
            raise ValueError(
                "DepthMappingLayerStack does not support shared_memory_config. "
                "Set shared_memory_config to None."
            )


class MatrixWeightsMixtureValidator(DynamicWeightValidator):
    OPTIONAL_FIELDS = {"model_config"}

    @staticmethod
    def validate_configuration(cfg):
        if vars(cfg).keys() - {field.name for field in fields(cfg)} - {"_passed_args"}:
            raise ValueError(
                "Complete parameter bank config contains unsupported fields."
            )
        for name in ("input_dim", "output_dim", "num_experts", "top_k"):
            value = getattr(cfg, name)
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"{name} must be a positive integer, received {value!r}."
                )
        if cfg.top_k > cfg.num_experts:
            raise ValueError("top_k cannot exceed num_experts.")

    @staticmethod
    def validate_route(model, probabilities, indices):
        if indices is None:
            if model.top_k != model.num_experts:
                raise ValueError(
                    "Missing indices require a full-bank route: top_k must equal num_experts."
                )
        else:
            if not isinstance(indices, Tensor) or indices.dtype != torch.long:
                raise TypeError("indices must be a torch.long Tensor.")
            if indices.device != model.parameter_bank.device:
                raise ValueError("indices must be on the parameter bank device.")
            if indices.ndim not in (1, 2) or (indices.ndim == 1 and model.top_k != 1):
                raise ValueError(
                    "indices must have shape [contexts, top_k], or [contexts] for top-1."
                )
            if indices.ndim == 2 and indices.shape[1] != model.top_k:
                raise ValueError("indices selection dimension must equal top_k.")
            if torch.any((indices < 0) | (indices >= model.num_experts)):
                raise ValueError("indices must lie within the parameter bank.")
        if probabilities is None:
            raise ValueError("Matrix mixture sampler must provide probabilities.")
        if (
            not isinstance(probabilities, Tensor)
            or not probabilities.is_floating_point()
        ):
            raise TypeError("probabilities must be a floating-point Tensor.")
        if probabilities.device != model.parameter_bank.device:
            raise ValueError("probabilities must be on the parameter bank device.")
        if probabilities.ndim not in (1, 2) or (
            probabilities.ndim == 1 and model.top_k != 1
        ):
            raise ValueError(
                "probabilities must have shape [contexts, top_k], or [contexts] for top-1."
            )
        if probabilities.ndim == 2 and probabilities.shape[1] != model.top_k:
            raise ValueError("probabilities selection dimension must equal top_k.")
        if indices is not None and probabilities.shape[0] != indices.shape[0]:
            raise ValueError(
                "probabilities and indices must have the same context count."
            )
        if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
            raise ValueError("probabilities must be finite and non-negative.")

    @classmethod
    def validate_initialization_config(cls, cfg) -> None:
        cls.validate_configuration(cfg)
        cls.validate_field_types(cfg)
        cls.validate_decay_parameters(cfg)
        cls.validate_sampler_config(cfg.sampler_config)

    @classmethod
    def validate_sampler_fields(cls, sampler, cfg) -> None:
        cls.validate_supported_fields(sampler)
        cls.validate_field_types(sampler)
        for name in ("num_experts", "top_k"):
            cls.validate_matching_value(
                f"sampler_config.{name}", getattr(sampler, name), getattr(cfg, name)
            )

    @staticmethod
    def validate_sampler_option(name, value, expected) -> None:
        if value is not None and value != expected:
            raise ValueError(
                f"matrix mixture sampler_config.{name} must be {expected!r}."
            )

    @staticmethod
    def validate_router_config(router_config) -> None:
        from emperor.sampler import RouterConfig

        if router_config is not None and not isinstance(router_config, RouterConfig):
            raise TypeError("sampler_config.router_config must be a RouterConfig.")

    @classmethod
    def validate_router_fields(cls, router, cfg) -> None:
        cls.validate_supported_fields(router)
        cls.validate_field_types(router)
        for name in ("input_dim", "num_experts"):
            cls.validate_matching_value(
                f"sampler_config.router_config.{name}",
                getattr(router, name),
                getattr(cfg, name),
            )
        if router.noisy_topk_flag not in (None, False):
            raise ValueError("matrix mixture router noisy_topk_flag must be False.")

    @staticmethod
    def validate_resolved_config(cfg) -> None:
        cfg.sampler_config.validate_for_router_input_dim(cfg.input_dim)

    @staticmethod
    def validate_matching_value(name, value, expected):
        if value is not None and (
            type(value) is not type(expected) or value != expected
        ):
            raise ValueError(f"{name} must match {expected!r}, received {value!r}.")

    @staticmethod
    def validate_supported_fields(cfg):
        if vars(cfg).keys() - {field.name for field in fields(cfg)} - {"_passed_args"}:
            raise ValueError(f"{type(cfg).__name__} contains unsupported fields.")

    @staticmethod
    def validate_sampler_config(sampler_config) -> None:
        from emperor.sampler import SamplerConfig

        if sampler_config is None:
            raise ValueError(
                "sampler_config is required for an adaptive matrix mixture."
            )
        if not isinstance(sampler_config, SamplerConfig):
            raise TypeError("sampler_config must be a SamplerConfig.")

    @staticmethod
    def validate_sampler_available(sampler) -> None:
        if sampler is None:
            raise ValueError("Adaptive matrix mixture requires an initialized sampler.")

    @staticmethod
    def validate_sampler_result(skip_mask: Tensor | None, loss: Tensor) -> None:
        if skip_mask is not None or loss.detach().ne(0).any():
            raise RuntimeError(
                "Matrix mixture sampler returned unsupported skip/loss state."
            )

    @classmethod
    def validate_forward_inputs(cls, model, weight_params, context):
        if (
            not isinstance(context, Tensor)
            or context.ndim != 2
            or context.shape[-1] != model.input_dim
        ):
            raise ValueError(
                "Matrix mixture context must have shape [contexts, input_dim]."
            )
        cls.validate_batched_weight_params(model, weight_params, context)
