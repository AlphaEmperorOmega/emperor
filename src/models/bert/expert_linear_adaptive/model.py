from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from emperor.attention import AttentionLayerState
from emperor.experiments.bert_pretraining import BertPretrainingExperiment
from emperor.layers import LayerNormPositionOptions
from models.bert.expert_linear_adaptive.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(BertPretrainingExperiment):
    def __init__(
        self,
        config: "ModelConfig",
    ):
        experiment_config = self.__validate_experiment_config(config)
        super().__init__(config)
        self.__validate_pretraining_dimensions()
        self.experiment_config: ExperimentConfig = experiment_config
        self.token_embedding = self.__build_token_embedding()
        self.token_type_embedding = self.__build_token_type_embedding()
        self.positional_embedding = self.__build_positional_embedding()
        self.embedding_layer_norm = self.__build_embedding_layer_norm()
        self.embedding_dropout = self.__build_embedding_dropout()
        self.transformer = self.__build_encoder_model()
        self.encoder_layer_norm = self.__build_encoder_layer_norm()
        self.mlm_dense = self.__build_mlm_dense()
        self.mlm_activation = self.__build_mlm_activation()
        self.mlm_layer_norm = self.__build_mlm_layer_norm()
        self.mlm_decoder = self.__build_mlm_decoder()
        self.__tie_mlm_decoder_weights()
        self.mlm_decoder_bias = self.__build_mlm_decoder_bias()
        self.pooler = self.__build_pooler()
        self.pooler_activation = self.__build_pooler_activation()
        self.nsp_head = self.__build_nsp_head()

    @staticmethod
    def __validate_experiment_config(
        config: "ModelConfig",
    ) -> ExperimentConfig:
        if not isinstance(config.experiment_config, ExperimentConfig):
            raise TypeError(
                "config.experiment_config must be a BERT Expert Linear Adaptive "
                "ExperimentConfig."
            )
        return config.experiment_config

    def __validate_pretraining_dimensions(self) -> None:
        if self.cfg.input_dim != self.cfg.output_dim:
            raise ValueError(
                "BERT pretraining ties the MLM decoder to token_embedding, "
                "so config.input_dim must equal config.output_dim."
            )

    def __build_token_embedding(self) -> nn.Embedding:
        return nn.Embedding(self.cfg.input_dim, self.cfg.hidden_dim)

    def __build_token_type_embedding(self) -> nn.Embedding:
        return nn.Embedding(2, self.cfg.hidden_dim)

    def __build_positional_embedding(self) -> nn.Module:
        return self.experiment_config.positional_embedding_config.build()

    def __build_embedding_layer_norm(self) -> nn.LayerNorm:
        return nn.LayerNorm(self.cfg.hidden_dim)

    def __build_embedding_dropout(self) -> nn.Dropout:
        return nn.Dropout(self.experiment_config.embedding_dropout_probability)

    def __build_encoder_model(self) -> nn.Module:
        return self.experiment_config.encoder_config.build()

    def __build_encoder_layer_norm(self) -> nn.Module:
        encoder_config = self.experiment_config.encoder_config
        stack_config = getattr(encoder_config, "block_config", encoder_config)
        layer_config = stack_config.layer_config.layer_model_config
        if layer_config.layer_norm_position == LayerNormPositionOptions.BEFORE:
            return nn.LayerNorm(self.cfg.hidden_dim)
        return nn.Identity()

    def __build_mlm_dense(self) -> nn.Linear:
        return nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)

    def __build_mlm_activation(self) -> nn.GELU:
        return nn.GELU()

    def __build_mlm_layer_norm(self) -> nn.LayerNorm:
        return nn.LayerNorm(self.cfg.hidden_dim)

    def __build_mlm_decoder(self) -> nn.Linear:
        return nn.Linear(self.cfg.hidden_dim, self.cfg.output_dim, bias=False)

    def __tie_mlm_decoder_weights(self) -> None:
        self.mlm_decoder.weight = self.token_embedding.weight

    def __build_mlm_decoder_bias(self) -> nn.Parameter:
        return nn.Parameter(torch.zeros(self.cfg.output_dim))

    def __build_pooler(self) -> nn.Linear:
        return nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)

    def __build_pooler_activation(self) -> nn.Tanh:
        return nn.Tanh()

    def __build_nsp_head(self) -> nn.Linear:
        return nn.Linear(self.cfg.hidden_dim, 2)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        input_ids, attention_mask, token_type_ids = self.__prepare_inputs(
            input_ids,
            attention_mask,
            token_type_ids,
        )
        hidden = self.__build_input_embeddings(input_ids, token_type_ids)
        sequence_output, auxiliary_loss = self.__run_encoder(hidden, attention_mask)
        mlm_logits = self.__build_mlm_logits(sequence_output)
        nsp_logits = self.__build_nsp_logits(sequence_output)
        return mlm_logits, nsp_logits, auxiliary_loss

    def __prepare_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        token_type_ids: Tensor | None,
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
        return input_ids, attention_mask, token_type_ids

    def __build_input_embeddings(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
    ) -> Tensor:
        token_embedding = self.token_embedding(input_ids)
        positional_embedding = self.positional_embedding(input_ids)
        segment_embedding = self.token_type_embedding(token_type_ids)
        hidden = token_embedding + positional_embedding + segment_embedding
        hidden = self.embedding_layer_norm(hidden)
        return self.embedding_dropout(hidden)

    def __run_encoder(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
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
        return sequence_output, auxiliary_loss

    def __build_mlm_logits(self, sequence_output: Tensor) -> Tensor:
        mlm_hidden = self.mlm_dense(sequence_output)
        mlm_hidden = self.mlm_activation(mlm_hidden)
        mlm_hidden = self.mlm_layer_norm(mlm_hidden)
        return self.mlm_decoder(mlm_hidden) + self.mlm_decoder_bias

    def __build_nsp_logits(self, sequence_output: Tensor) -> Tensor:
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        return self.nsp_head(pooled_output)
