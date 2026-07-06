from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from emperor.attention import AttentionLayerState
from emperor.experiments.bert_pretraining import BertPretrainingExperiment
from torch import Tensor

from models.bert._base_experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class BertPretrainingModel(BertPretrainingExperiment):
    def __init__(
        self,
        config: "ModelConfig",
    ):
        super().__init__(config)
        if not isinstance(config.experiment_config, ExperimentConfig):
            raise TypeError(
                "config.experiment_config must be a BERT ExperimentConfig."
            )
        if self.cfg.input_dim != self.cfg.output_dim:
            raise ValueError(
                "BERT pretraining ties the MLM decoder to token_embedding, "
                "so config.input_dim must equal config.output_dim."
            )
        self.experiment_config: ExperimentConfig = config.experiment_config
        self.token_embedding = nn.Embedding(self.cfg.input_dim, self.cfg.hidden_dim)
        self.token_type_embedding = nn.Embedding(2, self.cfg.hidden_dim)
        self.positional_embedding = (
            self.experiment_config.positional_embedding_config.build()
        )
        self.embedding_layer_norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.embedding_dropout = nn.Dropout(
            self.experiment_config.embedding_dropout_probability
        )
        self.transformer = self.experiment_config.encoder_config.build()
        self.encoder_layer_norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.mlm_dense = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)
        self.mlm_activation = nn.GELU()
        self.mlm_layer_norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.mlm_decoder = nn.Linear(
            self.cfg.hidden_dim, self.cfg.output_dim, bias=False
        )
        self.mlm_decoder.weight = self.token_embedding.weight
        self.mlm_decoder_bias = nn.Parameter(torch.zeros(self.cfg.output_dim))
        self.pooler = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)
        self.pooler_activation = nn.Tanh()
        self.nsp_head = nn.Linear(self.cfg.hidden_dim, 2)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(self.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = token_type_ids.to(self.device)

        token_embedding = self.token_embedding(input_ids)
        positional_embedding = self.positional_embedding(input_ids)
        segment_embedding = self.token_type_embedding(token_type_ids)
        hidden = token_embedding + positional_embedding + segment_embedding
        hidden = self.embedding_layer_norm(hidden)
        hidden = self.embedding_dropout(hidden)

        encoder_state = self.transformer(
            AttentionLayerState(
                hidden=hidden,
                key_padding_mask=attention_mask == 0,
            )
        )
        sequence_output = self.encoder_layer_norm(encoder_state.hidden)
        auxiliary_loss = (
            encoder_state.loss
            if encoder_state.loss is not None
            else sequence_output.new_zeros(())
        )

        mlm_hidden = self.mlm_dense(sequence_output)
        mlm_hidden = self.mlm_activation(mlm_hidden)
        mlm_hidden = self.mlm_layer_norm(mlm_hidden)
        mlm_logits = self.mlm_decoder(mlm_hidden) + self.mlm_decoder_bias

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        nsp_logits = self.nsp_head(pooled_output)
        return mlm_logits, nsp_logits, auxiliary_loss
