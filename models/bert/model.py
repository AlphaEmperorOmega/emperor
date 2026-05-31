import torch.nn as nn

from torch import Tensor
from emperor.base.layer import Layer
from emperor.experiments.language_model import LanguageModelExperiment
from emperor.transformer import TransformerEncoderStack
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
        if not isinstance(cfg.experiment_config, ExperimentConfig):
            raise TypeError("cfg.experiment_config must be a bert ExperimentConfig.")
        self.main_cfg: ExperimentConfig = cfg.experiment_config

        self.embedding_config = self.main_cfg.positional_embedding_config
        self.encoder_config = self.main_cfg.encoder_config
        self.output_config = self.main_cfg.output_config

        self.token_embedding = nn.Embedding(cfg.input_dim, cfg.hidden_dim)
        self.positional_embedding = self.embedding_config.build()
        self.transformer = TransformerEncoderStack(self.encoder_config)
        self.output = self.output_config.build()

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        token_emb = self.token_embedding(X)
        pos_emb = self.positional_embedding(X)
        X = token_emb + pos_emb
        X, loss = self.transformer(X)
        batch_size, sequence_length, hidden_dim = X.shape
        X = X.reshape(batch_size * sequence_length, hidden_dim)
        X = Layer.forward_with_state(self.output, X)
        X = X.reshape(batch_size, sequence_length, -1)
        return X
