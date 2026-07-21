from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from emperor.experiments.language_model import LanguageModelExperiment
from emperor.transformer import TransformerDecoderLayerState
from models.gpt.expert_linear._boundary_config_factory import GptBoundaryConfig
from models.gpt.expert_linear.experiment_config import ExperimentConfig

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
        self.token_embedding = nn.Embedding(config.input_dim, config.hidden_dim)
        self.positional_embedding = (
            experiment_config.positional_embedding_config.build()
        )
        self.embedding_layer_norm = self.__build_embedding_layer_norm()
        self.embedding_dropout = nn.Dropout(
            boundary_config.embedding_options.dropout_probability
        )
        self.transformer = experiment_config.decoder_config.build()
        self.decoder_layer_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(
            config.hidden_dim,
            config.output_dim,
            bias=boundary_config.lm_head_options.bias_flag,
        )
        if boundary_config.lm_head_options.weight_tying_flag:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def __validate_experiment_config(config: "ModelConfig") -> ExperimentConfig:
        if not isinstance(config.experiment_config, ExperimentConfig):
            raise TypeError(
                "config.experiment_config must be a GPT Expert Linear ExperimentConfig."
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

    def __build_embedding_layer_norm(self) -> nn.Module:
        if not self.boundary_config.embedding_options.layer_norm_flag:
            return nn.Identity()
        return nn.LayerNorm(self.cfg.hidden_dim)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        input_ids, attention_mask = self.__prepare_inputs(input_ids, attention_mask)
        hidden = self.token_embedding(input_ids) + self.positional_embedding(input_ids)
        hidden = self.embedding_layer_norm(hidden)
        hidden = self.embedding_dropout(hidden)
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
        return self.lm_head(sequence_output), auxiliary_loss.reshape(())

    def __prepare_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
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
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            if not isinstance(attention_mask, Tensor):
                raise TypeError("attention_mask must be a torch.Tensor or None.")
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must have the same shape as input_ids, "
                    f"received {tuple(attention_mask.shape)} and "
                    f"{tuple(input_ids.shape)}."
                )
            attention_mask = attention_mask.to(self.device)
        return input_ids, attention_mask

    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        input_ids, _ = self.__prepare_inputs(input_ids, attention_mask=None)
        if input_ids.size(1) + max_new_tokens > self.cfg.sequence_length:
            raise ValueError(
                "prompt plus max_new_tokens exceeds the configured context length "
                f"of {self.cfg.sequence_length}."
            )

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
