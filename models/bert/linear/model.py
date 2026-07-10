from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from emperor.attention import AttentionLayerState
from emperor.base.options import ActivationOptions
from emperor.experiments.bert_pretraining import BertPretrainingExperiment
from torch import Tensor

from models.bert.linear._boundary_config_factory import BertBoundaryConfig
from models.bert.linear.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


_ACTIVATION_MODULE_TYPES: dict[ActivationOptions, type[nn.Module]] = {
    ActivationOptions.DISABLED: nn.Identity,
    ActivationOptions.RELU: nn.ReLU,
    ActivationOptions.GELU: nn.GELU,
    ActivationOptions.SIGMOID: nn.Sigmoid,
    ActivationOptions.TANH: nn.Tanh,
    ActivationOptions.LEAKY_RELU: nn.LeakyReLU,
    ActivationOptions.ELU: nn.ELU,
    ActivationOptions.SELU: nn.SELU,
    ActivationOptions.SOFTPLUS: nn.Softplus,
    ActivationOptions.SOFTSIGN: nn.Softsign,
    ActivationOptions.SILU: nn.SiLU,
    ActivationOptions.MISH: nn.Mish,
}


def _build_activation_module(activation: ActivationOptions) -> nn.Module:
    try:
        module_type = _ACTIVATION_MODULE_TYPES[activation]
    except KeyError as error:
        raise ValueError(
            f"Unsupported BERT head activation: {activation!r}."
        ) from error
    return module_type()


class Model(BertPretrainingExperiment):
    def __init__(
        self,
        config: "ModelConfig",
    ):
        experiment_config = self.__validate_experiment_config(config)
        boundary_config = self.__validate_boundary_config(experiment_config)
        super().__init__(config)
        self.experiment_config: ExperimentConfig = experiment_config
        self.boundary_config: BertBoundaryConfig = boundary_config
        self.__validate_pretraining_dimensions()
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
                "config.experiment_config must be a BERT Linear ExperimentConfig."
            )
        return config.experiment_config

    @staticmethod
    def __validate_boundary_config(
        experiment_config: ExperimentConfig,
    ) -> BertBoundaryConfig:
        if not isinstance(experiment_config.boundary_config, BertBoundaryConfig):
            raise TypeError(
                "config.experiment_config.boundary_config must be a resolved "
                "BertBoundaryConfig."
            )
        return experiment_config.boundary_config

    def __validate_pretraining_dimensions(self) -> None:
        if not self.boundary_config.mlm_head_options.decoder_weight_tying_flag:
            return
        if self.cfg.input_dim != self.cfg.output_dim:
            raise ValueError(
                "BERT MLM decoder weight tying requires config.input_dim to equal "
                "config.output_dim."
            )

    def __build_token_embedding(self) -> nn.Embedding:
        return nn.Embedding(self.cfg.input_dim, self.cfg.hidden_dim)

    def __build_token_type_embedding(self) -> nn.Embedding:
        return nn.Embedding(
            self.boundary_config.embedding_options.token_type_vocab_size,
            self.cfg.hidden_dim,
        )

    def __build_positional_embedding(self) -> nn.Module:
        return self.experiment_config.positional_embedding_config.build()

    def __build_embedding_layer_norm(self) -> nn.Module:
        if not self.boundary_config.embedding_options.layer_norm_flag:
            return nn.Identity()
        return nn.LayerNorm(self.cfg.hidden_dim)

    def __build_embedding_dropout(self) -> nn.Dropout:
        return nn.Dropout(self.boundary_config.embedding_options.dropout_probability)

    def __build_encoder_model(self) -> nn.Module:
        return self.experiment_config.encoder_config.build()

    def __build_encoder_layer_norm(self) -> nn.LayerNorm:
        return nn.LayerNorm(self.cfg.hidden_dim)

    def __build_mlm_dense(self) -> nn.Linear:
        return nn.Linear(
            self.cfg.hidden_dim,
            self.cfg.hidden_dim,
            bias=self.boundary_config.mlm_head_options.dense_bias_flag,
        )

    def __build_mlm_activation(self) -> nn.Module:
        return _build_activation_module(
            self.boundary_config.mlm_head_options.activation
        )

    def __build_mlm_layer_norm(self) -> nn.Module:
        if not self.boundary_config.mlm_head_options.layer_norm_flag:
            return nn.Identity()
        return nn.LayerNorm(self.cfg.hidden_dim)

    def __build_mlm_decoder(self) -> nn.Linear:
        return nn.Linear(self.cfg.hidden_dim, self.cfg.output_dim, bias=False)

    def __tie_mlm_decoder_weights(self) -> None:
        if not self.boundary_config.mlm_head_options.decoder_weight_tying_flag:
            return
        self.mlm_decoder.weight = self.token_embedding.weight

    def __build_mlm_decoder_bias(self) -> nn.Parameter | None:
        if not self.boundary_config.mlm_head_options.decoder_bias_flag:
            return None
        return nn.Parameter(torch.zeros(self.cfg.output_dim))

    def __build_pooler(self) -> nn.Linear:
        return nn.Linear(
            self.cfg.hidden_dim,
            self.cfg.hidden_dim,
            bias=self.boundary_config.nsp_head_options.pooler_bias_flag,
        )

    def __build_pooler_activation(self) -> nn.Module:
        return _build_activation_module(
            self.boundary_config.nsp_head_options.pooler_activation
        )

    def __build_nsp_head(self) -> nn.Linear:
        return nn.Linear(
            self.cfg.hidden_dim,
            self.boundary_config.nsp_head_options.output_dim,
            bias=self.boundary_config.nsp_head_options.head_bias_flag,
        )

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
        mlm_logits = self.mlm_decoder(mlm_hidden)
        if self.mlm_decoder_bias is not None:
            mlm_logits = mlm_logits + self.mlm_decoder_bias
        return mlm_logits

    def __build_nsp_logits(self, sequence_output: Tensor) -> Tensor:
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        return self.nsp_head(pooled_output)
