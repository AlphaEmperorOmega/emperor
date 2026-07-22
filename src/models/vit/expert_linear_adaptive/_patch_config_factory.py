from dataclasses import dataclass

import models.vit.expert_linear_adaptive.config as config
from emperor.layers import LayerNormPositionOptions
from emperor.patch import LinearPatchEmbeddingConfig
from models.vit.expert_linear_adaptive._linear_layer_config_factory import (
    LinearLayerConfigFactory,
)
from models.vit.expert_linear_adaptive.runtime_options import (
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
        self.patch_options = self.__default_patch_options(dependencies.patch_options)
        self.encoder_options = self.__default_encoder_options(
            dependencies.encoder_options
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

    def __default_patch_options(
        self,
        patch_options: VitPatchOptions | None,
    ) -> VitPatchOptions:
        if patch_options is not None:
            return patch_options
        return VitPatchOptions(
            patch_size=config.IMAGE_PATCH_SIZE,
            input_channels=config.INPUT_CHANNELS,
            image_height=config.IMAGE_HEIGHT,
            dropout_probability=config.PATCH_DROPOUT_PROBABILITY,
            bias_flag=config.PATCH_BIAS_FLAG,
        )

    def __default_encoder_options(
        self,
        encoder_options: TransformerEncoderOptions | None,
    ) -> TransformerEncoderOptions:
        if encoder_options is not None:
            return encoder_options
        return TransformerEncoderOptions(
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.STACK_NUM_LAYERS,
            activation=config.STACK_ACTIVATION,
            dropout_probability=config.STACK_DROPOUT_PROBABILITY,
            layer_norm_position=config.LAYER_NORM_POSITION,
            causal_attention_mask_flag=False,
        )

    def build_patch_config(self) -> LinearPatchEmbeddingConfig:
        options = self.patch_options
        embedding_stack_config = (
            self.linear_layer_config_factory.build_plain_linear_stack_config(
                input_dim=options.input_channels * options.patch_size**2,
                output_dim=self.hidden_dim,
                num_layers=1,
                bias_flag=options.bias_flag,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                dropout_probability=self.encoder_options.dropout_probability,
                apply_output_pipeline_flag=False,
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
