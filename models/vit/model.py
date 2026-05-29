from torch import Tensor
from emperor.base.layer import LayerStack
from emperor.experiments.classifier import ClassifierExperiment
from emperor.transformer import TransformerEncoderStack
from models.vit.config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.main_cfg: ExperimentConfig = self._resolve_main_config(self.cfg, cfg)

        self.patch_config = self.main_cfg.patch_config
        self.embedding_config = self.main_cfg.positional_embedding_config
        self.encoder_config = self.main_cfg.encoder_config
        self.output_config = self.main_cfg.output_config

        self.patch = self.patch_config.build()
        self.positional_embedding = self.embedding_config.build()
        self.transformer = TransformerEncoderStack(self.encoder_config)
        self.output = LayerStack(self.output_config).build_model()

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        X = self.patch(X)
        X = self.positional_embedding(X)
        X, loss = self.transformer(X)
        X = X[:, 0, :]
        X = self.output(X)
        return X
