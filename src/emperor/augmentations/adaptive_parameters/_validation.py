import math
from typing import TYPE_CHECKING

from torch.types import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._augmentation import (
        AdaptiveParameterAugmentation,
    )
    from emperor.augmentations.adaptive_parameters._config import (
        AdaptiveParameterAugmentationConfig,
    )
    from emperor.augmentations.adaptive_parameters._linear_adapter import (
        AdaptiveLinearLayer,
    )
    from emperor.layers import RowLayout


class AdaptiveLinearValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config"}

    @classmethod
    def validate(cls, model: "AdaptiveLinearLayer") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(
            input_dim=model.input_dim,
            output_dim=model.output_dim,
        )
        augmentation_validator = (
            model.cfg.adaptive_augmentation_config.registry_owner().VALIDATOR
        )
        augmentation_validator.validate_grouping_configuration(
            model.cfg.adaptive_augmentation_config
        )
        cls._validate_adaptive_bias_consistency(model)

    @staticmethod
    def _validate_adaptive_bias_consistency(model: "AdaptiveLinearLayer") -> None:
        adaptive_augmentation_config = model.cfg.adaptive_augmentation_config
        if model.bias_flag:
            return
        if adaptive_augmentation_config.bias_config is not None:
            raise ValueError(
                "bias_flag is False but adaptive_augmentation_config.bias_config "
                "is set; cannot apply a dynamic bias to a layer without bias."
            )

    @staticmethod
    def validate_input_is_2d(X: Tensor) -> None:
        if X.dim() != 2:
            raise ValueError(
                f"Input must be a 2D matrix (batch, input_dim), "
                f"got {X.dim()}D tensor with shape {X.shape}"
            )

    @staticmethod
    def validate_weight_context_count(
        input_batch: Tensor,
        weights: Tensor,
    ) -> None:
        if input_batch.shape[0] != weights.shape[0]:
            raise ValueError(
                "Dynamic weights context count must match affine input, "
                f"received {weights.shape[0]} and {input_batch.shape[0]}."
            )

    @staticmethod
    def validate_bias_context_count(
        input_batch: Tensor,
        bias: Tensor,
    ) -> None:
        if input_batch.shape[0] != bias.shape[0]:
            raise ValueError(
                "Dynamic bias context count must match affine input, "
                f"received {bias.shape[0]} and {input_batch.shape[0]}."
            )


class AdaptiveGeneratorValidatorBase:
    @staticmethod
    def validate_model_config(cfg) -> None:
        from emperor.layers import LayerStackConfig

        if not isinstance(cfg.model_config, LayerStackConfig):
            raise TypeError(
                "model_config must be a LayerStackConfig for "
                f"{type(cfg).__name__}, got {type(cfg.model_config).__name__}."
            )

    @classmethod
    def validate_generator_model(cls, generator_model) -> None:
        from torch.nn import Sequential

        from emperor.layers import Layer, LayerStack

        if isinstance(generator_model, Layer):
            cls._validate_generator_layer(generator_model)
            return
        if isinstance(generator_model, (Sequential, LayerStack)):
            cls._validate_generator_sequence(generator_model)
            return
        raise TypeError(
            "Expected model_config.build(...) to return a Layer, Sequential, or "
            f"LayerStack, received {type(generator_model).__name__}."
        )

    @classmethod
    def _validate_generator_sequence(cls, generator_sequence) -> None:
        from emperor.layers import Layer

        for generator_layer in generator_sequence:
            if not isinstance(generator_layer, Layer):
                raise TypeError(
                    "Expected each generator sequence item to be a Layer, "
                    f"received {type(generator_layer).__name__}."
                )
            cls._validate_generator_layer(generator_layer)

    @staticmethod
    def _validate_generator_layer(generator_layer) -> None:
        from emperor.linears import LinearLayer

        if not isinstance(generator_layer.model, LinearLayer):
            raise TypeError(
                "Expected each generator Layer to wrap a LinearLayer, "
                f"received {type(generator_layer.model).__name__}."
            )

    @staticmethod
    def validate_decay_parameters(cfg) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            WeightDecayScheduleOptions,
        )

        schedule = cfg.decay_schedule
        if schedule is None or schedule == WeightDecayScheduleOptions.DISABLED:
            return
        decay_rate = cfg.decay_rate
        if decay_rate is None:
            raise ValueError(
                f"decay_rate must be greater than 0.0 when decay_schedule is "
                f"{schedule.name}, received {decay_rate!r}."
            )
        if not math.isfinite(decay_rate):
            raise ValueError(
                f"decay_rate must be finite when decay_schedule is "
                f"{schedule.name}, received {decay_rate!r}."
            )
        if decay_rate <= 0.0:
            raise ValueError(
                f"decay_rate must be greater than 0.0 when decay_schedule is "
                f"{schedule.name}, received {decay_rate!r}."
            )
        bounded_schedules = {
            WeightDecayScheduleOptions.LINEAR,
            WeightDecayScheduleOptions.MULTIPLICATIVE,
        }
        if schedule in bounded_schedules and decay_rate >= 1.0:
            raise ValueError(
                f"decay_rate must be less than 1.0 for {schedule.name}, "
                f"received {decay_rate!r}."
            )
        decay_warmup_batches = cfg.decay_warmup_batches
        if decay_warmup_batches is not None and decay_warmup_batches < 0:
            raise ValueError(
                f"decay_warmup_batches must be >= 0, received {decay_warmup_batches!r}."
            )

    @staticmethod
    def validate_input_batch(model, input_batch) -> None:
        if not isinstance(input_batch, Tensor):
            raise TypeError(
                f"input must be a Tensor, received {type(input_batch).__name__}."
            )
        if input_batch.dim() != 2:
            raise ValueError(
                f"{type(model).__name__} expects a 2D input tensor "
                "(batch_size, input_dim), received a "
                f"{input_batch.dim()}D tensor with shape {tuple(input_batch.shape)}."
            )
        if input_batch.shape[-1] != model.input_dim:
            raise ValueError(
                f"{type(model).__name__} input feature dimension must match input_dim, "
                f"received input_dim={model.input_dim} and input shape "
                f"{tuple(input_batch.shape)}."
            )

    @staticmethod
    def validate_weight_params(model, weight_params) -> None:
        if not isinstance(weight_params, Tensor):
            raise TypeError(
                "weight_params must be a Tensor, "
                f"received {type(weight_params).__name__}."
            )
        if weight_params.dim() not in {2, 3}:
            raise ValueError(
                "weight_params must be a 2D tensor (input_dim, output_dim) "
                "or a 3D tensor (batch_size, input_dim, output_dim), "
                f"received a {weight_params.dim()}D tensor with shape "
                f"{tuple(weight_params.shape)}."
            )
        expected_shape = (model.input_dim, model.output_dim)
        if tuple(weight_params.shape[-2:]) != expected_shape:
            raise ValueError(
                "weight_params trailing dimensions must match "
                "(input_dim, output_dim), "
                f"expected {expected_shape}, received shape "
                f"{tuple(weight_params.shape)}."
            )

    @classmethod
    def validate_batched_weight_params(cls, model, weight_params, input_batch) -> None:
        cls.validate_weight_params(model, weight_params)
        if weight_params.dim() == 3 and weight_params.shape[0] != input_batch.shape[0]:
            raise ValueError(
                "weight_params batch dimension must match input batch dimension, "
                f"received weight_params shape {tuple(weight_params.shape)} and "
                f"input shape {tuple(input_batch.shape)}."
            )

    @staticmethod
    def validate_bias_params(model, bias_params, input_batch=None) -> None:
        if bias_params is None:
            return
        if not isinstance(bias_params, Tensor):
            raise TypeError(
                "bias_params must be a Tensor when provided, "
                f"received {type(bias_params).__name__}."
            )
        if bias_params.dim() not in {1, 2}:
            raise ValueError(
                "bias_params must be a 1D tensor (output_dim) or a 2D tensor "
                "(batch_size, output_dim), received a "
                f"{bias_params.dim()}D tensor with shape {tuple(bias_params.shape)}."
            )
        if bias_params.shape[-1] != model.output_dim:
            raise ValueError(
                "bias_params feature dimension must match output_dim, "
                f"received output_dim={model.output_dim} and bias_params shape "
                f"{tuple(bias_params.shape)}."
            )
        if (
            input_batch is not None
            and bias_params.dim() == 2
            and bias_params.shape[0] != input_batch.shape[0]
        ):
            raise ValueError(
                "bias_params batch dimension must match input batch dimension, "
                f"received bias_params shape {tuple(bias_params.shape)} and "
                f"input shape {tuple(input_batch.shape)}."
            )


class AdaptiveParameterAugmentationValidator(
    AdaptiveGeneratorValidatorBase, ValidatorBase
):
    OPTIONAL_FIELDS = {
        "diagonal_config",
        "weight_config",
        "bias_config",
        "mask_config",
        "model_config",
        "group_count",
    }

    @classmethod
    def validate(cls, model: "AdaptiveParameterAugmentation") -> None:
        cls.validate_grouping_configuration(model.cfg)
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls._validate_dimensions(model)
        cls._validate_model_config("model_config", model.model_config)
        cls._validate_sub_configs(model)

    @staticmethod
    def _validate_dimensions(model: "AdaptiveParameterAugmentation") -> None:
        if (
            model.input_dim is None
            or not isinstance(model.input_dim, int)
            or model.input_dim <= 0
        ):
            raise ValueError(
                f"input_dim must be a positive integer, received {model.input_dim!r}."
            )
        if (
            model.output_dim is None
            or not isinstance(model.output_dim, int)
            or model.output_dim <= 0
        ):
            raise ValueError(
                f"output_dim must be a positive integer, received {model.output_dim!r}."
            )

    @classmethod
    def _validate_sub_configs(cls, model: "AdaptiveParameterAugmentation") -> None:
        from emperor.augmentations.adaptive_parameters._biases.config import (
            DynamicBiasConfig,
        )
        from emperor.augmentations.adaptive_parameters._diagonals.config import (
            DynamicDiagonalConfig,
        )
        from emperor.augmentations.adaptive_parameters._masks.config import (
            AxisMaskConfig,
        )
        from emperor.augmentations.adaptive_parameters._weights.config import (
            DynamicWeightConfig,
        )

        sub_configs = [
            ("weight_config", model.weight_config, DynamicWeightConfig),
            ("diagonal_config", model.diagonal_config, DynamicDiagonalConfig),
            ("bias_config", model.bias_config, DynamicBiasConfig),
            ("mask_config", model.mask_config, AxisMaskConfig),
        ]
        for name, config, expected_type in sub_configs:
            if config is None:
                continue
            if not isinstance(config, expected_type):
                raise TypeError(
                    f"{name} must be a {expected_type.__name__} instance, "
                    f"got {type(config).__name__}."
                )
            cls._validate_model_config(f"{name}.model_config", config.model_config)
            if config.model_config is None and model.model_config is None:
                raise ValueError(
                    f"{type(config).__name__} requires a model_config but none "
                    "was provided on the sub-config or the parent "
                    "AdaptiveParameterAugmentationConfig."
                )

    @classmethod
    def validate_grouping_configuration(
        cls,
        config: "AdaptiveParameterAugmentationConfig",
    ) -> None:
        grouping_scope = config.grouping_scope
        group_count = config.group_count
        cls._validate_grouping_scope(grouping_scope)
        if not cls.grouping_is_enabled(config):
            if group_count is not None:
                cls._validate_group_count(group_count)
            return
        cls._validate_group_count(group_count)
        active_components = (
            config.diagonal_config,
            config.weight_config,
            config.bias_config,
            config.mask_config,
        )
        if not any(component is not None for component in active_components):
            raise ValueError(
                "enabled grouping requires at least one adaptive parameter component."
            )

    @classmethod
    def grouping_is_enabled(
        cls,
        config: "AdaptiveParameterAugmentationConfig",
    ) -> bool:
        from emperor.augmentations.adaptive_parameters._options import (
            AdaptiveParameterGroupingScopeOptions,
        )

        grouping_scope = config.grouping_scope
        return (
            grouping_scope is AdaptiveParameterGroupingScopeOptions.ROWS
            or grouping_scope is AdaptiveParameterGroupingScopeOptions.SEQUENCE
        )

    @staticmethod
    def _validate_grouping_scope(grouping_scope: object) -> None:
        from emperor.augmentations.adaptive_parameters._options import (
            AdaptiveParameterGroupingScopeOptions,
        )

        if grouping_scope is None:
            raise ValueError(
                "grouping_scope is required for a resolved "
                "AdaptiveParameterAugmentationConfig; use DISABLED, ROWS, or "
                "SEQUENCE."
            )
        if not isinstance(grouping_scope, AdaptiveParameterGroupingScopeOptions):
            raise TypeError(
                "grouping_scope must be an "
                "AdaptiveParameterGroupingScopeOptions value, received "
                f"{grouping_scope!r}."
            )

    @staticmethod
    def _validate_group_count(group_count: int | None) -> None:
        if (
            isinstance(group_count, bool)
            or not isinstance(group_count, int)
            or group_count <= 0
        ):
            raise ValueError(
                "group_count must be a positive integer when provided, "
                f"received {group_count!r}."
            )

    @staticmethod
    def _validate_model_config(name: str, model_config) -> None:
        if model_config is None:
            return
        from emperor.layers import LayerStackConfig

        if not isinstance(model_config, LayerStackConfig):
            raise TypeError(
                f"{name} must be a LayerStackConfig when provided, "
                f"got {type(model_config).__name__}."
            )

    @classmethod
    def validate_forward_inputs(
        cls,
        model: "AdaptiveParameterAugmentation",
        affine_transform_callback,
        weight_params,
        bias_params,
        input_batch,
    ) -> None:
        if not callable(affine_transform_callback):
            raise TypeError(
                "affine_transform_callback must be callable, "
                f"received {type(affine_transform_callback).__name__}."
            )
        cls.validate_input_batch(model, input_batch)
        cls.validate_batched_weight_params(model, weight_params, input_batch)
        cls.validate_bias_params(model, bias_params, input_batch)

    @staticmethod
    def validate_grouped_base_parameters(
        weight_params: Tensor,
        bias_params: Tensor | None,
    ) -> None:
        if weight_params.dim() != 2:
            raise ValueError(
                "Adaptive parameter grouping requires shared two-dimensional base "
                "weights; row-specific base weights are not supported."
            )
        if bias_params is not None and bias_params.dim() != 1:
            raise ValueError(
                "Adaptive parameter grouping requires a shared one-dimensional base "
                "bias; row-specific base bias is not supported."
            )

    @classmethod
    def validate_grouped_forward_inputs(
        cls,
        weight_params: Tensor,
        bias_params: Tensor | None,
        row_layout: "RowLayout | None",
    ) -> None:
        if row_layout is None:
            raise ValueError(
                "Enabled adaptive parameter grouping requires an explicit RowLayout."
            )
        cls.validate_grouped_base_parameters(weight_params, bias_params)

    @classmethod
    def validate_generated_parameters(
        cls,
        model: "AdaptiveParameterAugmentation",
        weights: Tensor,
        bias: Tensor | None,
        conditioning_input: Tensor,
    ) -> None:
        cls.validate_batched_weight_params(model, weights, conditioning_input)
        cls.validate_bias_params(model, bias, conditioning_input)
