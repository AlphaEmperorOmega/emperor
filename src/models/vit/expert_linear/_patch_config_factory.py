from dataclasses import dataclass

import models.vit.expert_linear.config as config
from emperor.layers import LayerNormPositionOptions, NormalizationOptions
from emperor.patch import LinearPatchEmbeddingConfig
from models.vit.expert_linear import _config_defaults as config_defaults
from models.vit.expert_linear._linear_layer_config_factory import (
    LinearLayerConfigFactory,
)
from models.vit.expert_linear.runtime_options import (
    TransformerEncoderOptions,
    VitPatchOptions,
)


@dataclass(frozen=True)
class PatchConfigDependencies:
    hidden_dim: int
    patch_options: VitPatchOptions | None
    encoder_options: TransformerEncoderOptions | None
    linear_layer_config_factory: LinearLayerConfigFactory


class PatchConfigFactory:
    def __init__(self, dependencies: PatchConfigDependencies) -> None:
        self.hidden_dim = dependencies.hidden_dim
        self.patch_options = (
            config_defaults.vit_patch_options(config)
            if dependencies.patch_options is None
            else dependencies.patch_options
        )
        self.encoder_options = (
            config_defaults.vit_encoder_options(config)
            if dependencies.encoder_options is None
            else dependencies.encoder_options
        )
        self.linear_layer_config_factory = dependencies.linear_layer_config_factory

    @property
    def sequence_length(self) -> int:
        if self.patch_options.patch_size <= 0:
            raise ValueError(
                "image_patch_size must be positive, "
                f"received {self.patch_options.patch_size}."
            )
        if self.patch_options.image_height % self.patch_options.patch_size != 0:
            raise ValueError(
                "image_height must be divisible by image_patch_size, "
                f"received image_height={self.patch_options.image_height} and "
                f"image_patch_size={self.patch_options.patch_size}."
            )
        patches_per_axis = (
            self.patch_options.image_height // self.patch_options.patch_size
        )
        return patches_per_axis * patches_per_axis + 1

    def build_patch_config(self) -> LinearPatchEmbeddingConfig:
        options = self.patch_options
        embedding_stack_config = (
            self.linear_layer_config_factory.build_plain_linear_stack_config(
                input_dim=options.input_channels * options.patch_size**2,
                output_dim=self.hidden_dim,
                num_layers=1,
                bias_flag=options.bias_flag,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                normalization=NormalizationOptions.RMS_NORM,
                dropout_probability=self.encoder_options.dropout_probability,
                apply_output_postprocessing_flag=False,
            )
        )
        return LinearPatchEmbeddingConfig(
            embedding_dim=self.hidden_dim,
            num_input_channels=options.input_channels,
            patch_size=options.patch_size,
            stride=options.patch_size,
            padding=0,
            dropout_probability=options.dropout_probability,
            embedding_stack_config=embedding_stack_config,
        )
