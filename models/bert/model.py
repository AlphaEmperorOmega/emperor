import torch.nn as nn

from torch import Tensor
from emperor.base.utils import ConfigBase
from emperor.base.layer import LayerStack
from emperor.experiments.language_model import LanguageModelExperiment
from emperor.transformer.utils.stack import TransformerEncoderStack
from emperor.embedding.absolute.factory import AbsolutePositionalEmbeddingFactory
from models.bert.config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(LanguageModelExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.main_cfg: ExperimentConfig = self._resolve_main_config(self.cfg, cfg)

        self.embedding_config = self.main_cfg.positional_embedding_config
        self.encoder_config = self.main_cfg.encoder_config
        self.output_config = self.main_cfg.output_config

        self.token_embedding = nn.Embedding(cfg.input_dim, cfg.hidden_dim)
        self.positional_embedding = AbsolutePositionalEmbeddingFactory(
            self.embedding_config
        ).build()
        self.transformer = TransformerEncoderStack(self.encoder_config)
        self.output = LayerStack(self.output_config).build_model()

    def _resolve_main_config(
        self, sub_config: "ConfigBase", main_cfg: "ConfigBase"
    ) -> "ExperimentConfig":
        if sub_config.override_config is not None:
            return sub_config.override_config
        return main_cfg

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        token_emb = self.token_embedding(X)
        pos_emb = self.positional_embedding(X)
        X = token_emb + pos_emb
        X, loss = self.transformer(X)
        X = self.output(X)
        return X
