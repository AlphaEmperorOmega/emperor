from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from emperor.experiments.language_model import LanguageModelExperiment
from emperor.transformer import TransformerDecoderLayerState
from models.gpt.expert_linear_adaptive._boundary_config_factory import GptBoundaryConfig
from models.gpt.expert_linear_adaptive.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class Model(LanguageModelExperiment):
    def __init__(
        self,
        config: "ModelConfig",
    ) -> None:
        experiment_config = self.__validate_experiment_config(config)
        boundary_config = self.__validate_boundary_config(experiment_config)
        self.__validate_tied_vocabulary_sizes(config, boundary_config)
        super().__init__(config)
        self.experiment_config: ExperimentConfig = experiment_config
        self.boundary_config: GptBoundaryConfig = boundary_config
        self.token_embedding = self.__build_token_embedding()
        self.positional_embedding = self.__build_positional_embedding()
        self.embedding_layer_norm = self.__build_embedding_layer_norm()
        self.embedding_dropout = self.__build_embedding_dropout()
        self.transformer = self.__build_decoder()
        self.decoder_layer_norm = self.__build_decoder_layer_norm()
        self.lm_head = self.__build_lm_head()
        self.__tie_lm_head_weights()

    @staticmethod
    def __validate_experiment_config(config: "ModelConfig") -> ExperimentConfig:
        if not isinstance(config.experiment_config, ExperimentConfig):
            raise TypeError(
                "config.experiment_config must be a GPT Expert Linear Adaptive "
                "ExperimentConfig."
            )
        return config.experiment_config

    @staticmethod
    def __validate_boundary_config(
        experiment_config: ExperimentConfig,
    ) -> GptBoundaryConfig:
        if not isinstance(experiment_config.boundary_config, GptBoundaryConfig):
            raise TypeError(
                "config.experiment_config.boundary_config must be a resolved "
                "GptBoundaryConfig."
            )
        return experiment_config.boundary_config

    @staticmethod
    def __validate_tied_vocabulary_sizes(
        config: "ModelConfig",
        boundary_config: GptBoundaryConfig,
    ) -> None:
        if (
            boundary_config.lm_head_options.weight_tying_flag
            and config.input_dim != config.output_dim
        ):
            raise ValueError(
                "GPT LM head weight tying requires input_dim to equal output_dim."
            )

    def __build_token_embedding(self) -> nn.Embedding:
        return nn.Embedding(self.cfg.input_dim, self.cfg.hidden_dim)

    def __build_positional_embedding(self) -> nn.Module:
        return self.experiment_config.positional_embedding_config.build()

    def __build_embedding_layer_norm(self) -> nn.Module:
        if not self.boundary_config.embedding_options.layer_norm_flag:
            return nn.Identity()
        return nn.LayerNorm(self.cfg.hidden_dim)

    def __build_embedding_dropout(self) -> nn.Dropout:
        return nn.Dropout(self.boundary_config.embedding_options.dropout_probability)

    def __build_decoder(self) -> nn.Module:
        return self.experiment_config.decoder_config.build()

    def __build_decoder_layer_norm(self) -> nn.LayerNorm:
        return nn.LayerNorm(self.cfg.hidden_dim)

    def __build_lm_head(self) -> nn.Linear:
        return nn.Linear(
            self.cfg.hidden_dim,
            self.cfg.output_dim,
            bias=self.boundary_config.lm_head_options.bias_flag,
        )

    def __tie_lm_head_weights(self) -> None:
        if not self.boundary_config.lm_head_options.weight_tying_flag:
            return
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        input_ids, attention_mask = self.__prepare_inputs(input_ids, attention_mask)
        hidden = self.__build_input_embeddings(input_ids)
        sequence_output, auxiliary_loss = self.__run_decoder(hidden, attention_mask)
        logits = self.lm_head(sequence_output)
        return logits, auxiliary_loss.reshape(())

    def __build_input_embeddings(self, input_ids: Tensor) -> Tensor:
        token_embedding = self.token_embedding(input_ids)
        positional_embedding = self.positional_embedding(input_ids)
        hidden = token_embedding + positional_embedding
        hidden = self.embedding_layer_norm(hidden)
        return self.embedding_dropout(hidden)

    def __run_decoder(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        decoder_state = self.transformer(
            TransformerDecoderLayerState(
                hidden=hidden,
                target_key_padding_mask=attention_mask == 0,
            )
        )
        sequence_output = self.decoder_layer_norm(decoder_state.hidden)
        auxiliary_loss = (
            decoder_state.loss
            if decoder_state.loss is not None
            else sequence_output.new_zeros(())
        )
        return sequence_output, auxiliary_loss

    def __prepare_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        input_ids = self.__prepare_input_ids(input_ids)
        attention_mask = self.__prepare_attention_mask(attention_mask, input_ids)
        return input_ids, attention_mask

    def __prepare_input_ids(self, input_ids: Tensor) -> Tensor:
        if not isinstance(input_ids, Tensor):
            raise TypeError("input_ids must be a torch.Tensor.")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 [batch, sequence] tensor.")
        if input_ids.size(0) == 0 or input_ids.size(1) == 0:
            raise ValueError("input_ids must contain a non-empty prompt per batch.")
        if input_ids.size(1) > self.cfg.sequence_length:
            raise ValueError(
                "input sequence length exceeds the configured context length "
                f"of {self.cfg.sequence_length}."
            )
        if input_ids.dtype not in _INTEGER_DTYPES:
            raise TypeError("input_ids must contain integer token IDs.")
        input_ids = input_ids.to(self.device, dtype=torch.long)
        if bool(torch.any(input_ids < 0).item()) or bool(
            torch.any(input_ids >= self.cfg.input_dim).item()
        ):
            raise ValueError(
                f"input_ids must be in the range [0, {self.cfg.input_dim})."
            )
        return input_ids

    def __prepare_attention_mask(
        self,
        attention_mask: Tensor | None,
        input_ids: Tensor,
    ) -> Tensor:
        if attention_mask is None:
            return torch.ones_like(input_ids)
        if not isinstance(attention_mask, Tensor):
            raise TypeError("attention_mask must be a torch.Tensor or None.")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids, "
                f"received {tuple(attention_mask.shape)} and "
                f"{tuple(input_ids.shape)}."
            )
        return attention_mask.to(self.device)

    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        self.__validate_max_new_tokens(max_new_tokens)
        input_ids, _ = self.__prepare_inputs(input_ids, attention_mask=None)
        self.__validate_generation_length(input_ids, max_new_tokens)
        return self.__generate_greedily(input_ids, max_new_tokens)

    @staticmethod
    def __validate_max_new_tokens(max_new_tokens: int) -> None:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")

    def __validate_generation_length(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
    ) -> None:
        if input_ids.size(1) + max_new_tokens > self.cfg.sequence_length:
            raise ValueError(
                "prompt plus max_new_tokens exceeds the configured context length "
                f"of {self.cfg.sequence_length}."
            )

    def __generate_greedily(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
    ) -> Tensor:
        was_training = self.training
        generated = input_ids.clone()
        try:
            self.eval()
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    logits, _auxiliary_loss = self(generated)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat((generated, next_token), dim=1)
        finally:
            self.train(was_training)
        return generated
