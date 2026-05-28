from dataclasses import dataclass
from emperor.base.utils import ConfigBase, optional_field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.base.layer import LayerStackConfig


@dataclass
class PatchConfig(ConfigBase):
    embedding_dim: int | None = optional_field(
        "Output embedding dimension per patch token."
    )
    num_input_channels: int | None = optional_field(
        "Number of channels in the input image."
    )
    patch_size: int | None = optional_field(
        "Side length of each square patch in pixels."
    )
    dropout_probability: float | None = optional_field(
        "Dropout probability applied to the patch embeddings."
    )


@dataclass
class LinearPatchEmbeddingConfig(PatchConfig):
    stride: int | None = optional_field(
        "Stride of the unfold patch extractor."
    )
    padding: int | None = optional_field(
        "Padding of the unfold patch extractor."
    )
    embedding_stack_config: "LayerStackConfig | None" = optional_field(
        "LayerStack config used to project flattened patches to embedding_dim."
    )

    def _registry_owner(self) -> type:
        from emperor.patch.core.layers import PatchEmbeddingLinear

        return PatchEmbeddingLinear


@dataclass
class ConvPatchEmbeddingConfig(PatchConfig):
    conv_stack_config: "LayerStackConfig | None" = optional_field(
        "LayerStack config used to build the Conv2d stack. The final layer "
        "extracts patches; use last_layer_overrides on the stack to set the "
        "patch-extraction kernel_size/stride/padding."
    )

    def _registry_owner(self) -> type:
        from emperor.patch.core.layers import PatchEmbeddingConv

        return PatchEmbeddingConv
