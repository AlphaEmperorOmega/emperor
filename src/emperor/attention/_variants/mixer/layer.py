"""Private mixer-attention layer implementation."""

from copy import deepcopy
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention._base import MultiHeadAttentionAbstract
from emperor.attention._variants.mixer.validation import (
    MixerAttentionValidator,
)
from emperor.layers import Layer
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.attention._variants.mixer.config import MixerAttentionConfig
    from emperor.config import ModelConfig
    from emperor.experts import MixtureOfExpertsModelConfig
    from emperor.layers import LayerStackConfig, RecurrentLayerConfig


class MixerAttention(MultiHeadAttentionAbstract):
    """Apply one shared configured model across every token-axis vector."""

    VALIDATOR = MixerAttentionValidator

    def __init__(
        self,
        cfg: "MixerAttentionConfig | ModelConfig",
        overrides: "MixerAttentionConfig | None" = None,
    ) -> None:
        Module.__init__(self)
        config = getattr(cfg, "mixer_attention_config", cfg)
        self.cfg: MixerAttentionConfig = self._override_config(config, overrides)

        self.embedding_dim: int = self.cfg.embedding_dim
        self.sequence_length: int = self.cfg.sequence_length
        self.batch_first_flag: bool = self.cfg.batch_first_flag
        self.causal_attention_mask_flag: bool = self.cfg.causal_attention_mask_flag
        self.mixing_model_config: (
            LayerStackConfig | MixtureOfExpertsModelConfig | RecurrentLayerConfig
        ) = self.cfg.mixing_model_config

        self.VALIDATOR.validate(self)
        exact_mixing_config = self.__exact_mixing_model_config(self.mixing_model_config)
        self.mixing_model = exact_mixing_config.build()

    def __exact_mixing_model_config(self, config):
        from emperor.experts import MixtureOfExpertsModelConfig
        from emperor.layers import LayerStackConfig, RecurrentLayerConfig

        exact_config = deepcopy(config)
        exact_config.input_dim = self.sequence_length
        exact_config.output_dim = self.sequence_length
        if isinstance(exact_config, MixtureOfExpertsModelConfig):
            if exact_config.stack_config is None:
                raise ValueError(
                    "stack_config is required for a MixerAttention "
                    "MixtureOfExpertsModelConfig."
                )
            exact_config.stack_config = self.__exact_mixing_model_config(
                exact_config.stack_config
            )
        elif isinstance(exact_config, RecurrentLayerConfig):
            if exact_config.block_config is None:
                raise ValueError(
                    "block_config is required for a MixerAttention "
                    "RecurrentLayerConfig."
                )
            exact_config.block_config = self.__exact_mixing_model_config(
                exact_config.block_config
            )
        elif not isinstance(exact_config, LayerStackConfig):
            # RecurrentLayerConfig deliberately accepts any ConfigBase block with
            # an input/output dimension contract. The outer mixer configuration
            # has already been type-checked, so preserve that extensibility here.
            return exact_config
        return exact_config

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        k_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        static_k: Tensor | None = None,
        static_v: Tensor | None = None,
    ) -> tuple[Tensor, None, Tensor | None]:
        self.VALIDATOR.validate_forward_inputs(
            self,
            q,
            k,
            v,
            k_padding_mask,
            attention_mask,
            static_k,
            static_v,
        )

        sequence_axis = 1 if self.batch_first_flag else 0
        token_vectors = q.movedim(sequence_axis, -1)
        leading_shape = token_vectors.shape[:-1]
        flattened_input = token_vectors.reshape(-1, self.sequence_length)
        state = Layer.run_model_returning_state(
            self.mixing_model,
            flattened_input,
        )
        state = self.VALIDATOR.validate_mixing_state(state, flattened_input)

        mixed_vectors = state.hidden.reshape(*leading_shape, self.sequence_length)
        mixed = mixed_vectors.movedim(-1, sequence_axis)
        return mixed, None, state.loss
