from typing import TYPE_CHECKING

from torch import Tensor

from emperor.layers import Layer, LayerStackConfig
from emperor.patch._base import PatchBase

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.patch._config import ConvPatchEmbeddingConfig


class PatchEmbeddingConv(PatchBase):
    def __init__(
        self,
        cfg: "ConvPatchEmbeddingConfig | ModelConfig",
        overrides: "ConvPatchEmbeddingConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.cfg: ConvPatchEmbeddingConfig = self.cfg
        self.patch_model = self.__create_patch_extraction_model()

    def __create_patch_extraction_model(self):
        overrides = LayerStackConfig(
            input_dim=self.num_input_channels,
            output_dim=self.embedding_dim,
        )
        return self.cfg.conv_stack_config.build(overrides)

    def forward(self, X: Tensor):
        self.VALIDATOR.validate_forward_inputs(self, X)
        X = Layer.run_model_returning_hidden(self.patch_model, X)
        X = X.flatten(2)
        X = X.transpose(1, 2)
        X = self._concatenate_class_token(X)
        X = self.dropout(X)

        return X
