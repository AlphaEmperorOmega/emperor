from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.patch._base import PatchBase


class PatchValidator(ValidatorBase):
    OPTIONAL_FIELDS = {"override_config", "class_token_flag"}

    @classmethod
    def validate(cls, model: "PatchBase") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        dimensions = {
            "embedding_dim": model.embedding_dim,
            "num_input_channels": model.num_input_channels,
            "patch_size": model.patch_size,
        }
        if hasattr(model.cfg, "stride"):
            dimensions["stride"] = model.cfg.stride
        cls.validate_dimensions(
            **dimensions,
        )
        if hasattr(model.cfg, "padding") and model.cfg.padding < 0:
            raise ValueError(
                "padding must be greater than or equal to 0, "
                f"received {model.cfg.padding}"
            )
        cls._validate_dropout_probability(model.dropout_probability)
        cls._validate_class_token_flag(model.cfg.class_token_flag)
        cls._validate_stack_config_types(model.cfg)
        cls._validate_convolutional_patch_geometry(model.cfg)

    @staticmethod
    def _validate_stack_config_types(config) -> None:
        from emperor.layers import LayerStackConfig

        for field_name in ("embedding_stack_config", "conv_stack_config"):
            if not hasattr(config, field_name):
                continue
            stack_config = getattr(config, field_name)
            if not isinstance(stack_config, LayerStackConfig):
                raise TypeError(
                    f"{field_name} must be an instance of LayerStackConfig for "
                    f"{type(config).__name__}, got {type(stack_config).__name__}"
                )

    @classmethod
    def _validate_convolutional_patch_geometry(cls, config) -> None:
        if not hasattr(config, "conv_stack_config"):
            return

        from emperor.convs import Conv2dLayerConfig

        stack_config = config.conv_stack_config
        layer_config = getattr(stack_config, "layer_config", None)
        layer_model_config = getattr(layer_config, "layer_model_config", None)
        if not isinstance(layer_model_config, Conv2dLayerConfig):
            raise TypeError(
                "conv_stack_config.layer_config.layer_model_config must be an "
                "instance of Conv2dLayerConfig for ConvPatchEmbeddingConfig, got "
                f"{type(layer_model_config).__name__}"
            )

        effective_patch_size = cls._effective_convolutional_patch_size(
            stack_config,
            layer_model_config,
        )
        if effective_patch_size is None or config.patch_size == effective_patch_size:
            return
        raise ValueError(
            "patch_size must match the effective convolutional receptive field "
            "for ConvPatchEmbeddingConfig, got "
            f"patch_size={config.patch_size} and "
            f"effective_patch_size={effective_patch_size}"
        )

    @staticmethod
    def _effective_convolutional_patch_size(
        stack_config,
        layer_model_config,
    ) -> int | None:
        from emperor.layers import MirroredLayerStackConfig

        num_layers = stack_config.num_layers
        kernel_size = layer_model_config.kernel_size
        stride = layer_model_config.stride
        if (
            type(num_layers) is not int
            or num_layers < 1
            or type(kernel_size) is not int
            or kernel_size < 1
            or type(stride) is not int
            or stride < 1
        ):
            return None

        physical_layer_count = (
            num_layers * 2
            if isinstance(stack_config, MirroredLayerStackConfig)
            else num_layers
        )
        effective_patch_size = 1
        effective_stride = 1
        for _ in range(physical_layer_count):
            effective_patch_size += (kernel_size - 1) * effective_stride
            effective_stride *= stride
        return effective_patch_size

    @staticmethod
    def _validate_dropout_probability(value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"dropout_probability must be in [0.0, 1.0], received {value}"
            )

    @staticmethod
    def _validate_class_token_flag(value: bool | None) -> None:
        if value is not None and not isinstance(value, bool):
            raise TypeError(
                "class_token_flag must be bool or None for PatchConfig, got "
                f"{type(value).__name__}"
            )

    @staticmethod
    def validate_forward_inputs(model: "PatchBase", X: Tensor) -> None:
        if not isinstance(X, Tensor):
            raise TypeError(
                f"Input Error: forward input must be a Tensor, "
                f"received {type(X).__name__}."
            )
        if X.dim() != 4:
            raise ValueError(
                f"Input Error: PatchBase expects a 4D input tensor "
                f"(batch, channels, height, width), received a "
                f"{X.dim()}D tensor with shape {tuple(X.shape)}."
            )
        if X.shape[1] != model.num_input_channels:
            raise ValueError(
                f"Input Error: input channel dimension must match "
                f"'num_input_channels', received "
                f"num_input_channels={model.num_input_channels} and input shape "
                f"{tuple(X.shape)}."
            )
