from torch import Tensor
from emperor.base.utils import Module
from emperor.transformer.core._validator import TransformerValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.transformer.config import TransformerConfig


class Transformer(Module):
    def __init__(
        self,
        cfg: "TransformerConfig",
        overrides: "TransformerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "transformer_config", cfg)
        self.cfg: "TransformerConfig" = self._override_config(config, overrides)

        TransformerValidator.validate_transformer(self)

        self.encoder_model = self.__build_stack_if_configured(
            self.cfg.encoder_stack_config
        )
        self.decoder_model = self.__build_stack_if_configured(
            self.cfg.decoder_stack_config
        )

    def __build_stack_if_configured(self, stack_config):
        if stack_config is None:
            return None
        return stack_config.build()

    def forward(
        self,
        source_token_embeddings: Tensor | None = None,
        target_token_embeddings: Tensor | None = None,
        source_attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        source_key_padding_mask: Tensor | None = None,
        target_key_padding_mask: Tensor | None = None,
        encoder_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        TransformerValidator.validate_transformer_forward_inputs(
            self, source_token_embeddings, target_token_embeddings
        )
        encoder_output, encoder_loss = self.__run_encoder_if_present(
            source_token_embeddings=source_token_embeddings,
            source_key_padding_mask=source_key_padding_mask,
            attention_mask=source_attention_mask,
        )
        decoder_output, decoder_loss = self.__run_decoder_if_present(
            target_token_embeddings=target_token_embeddings,
            encoder_output=encoder_output,
            target_key_padding_mask=target_key_padding_mask,
            encoder_key_padding_mask=encoder_key_padding_mask,
            attention_mask=target_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

        final_output = self.__select_final_output(encoder_output, decoder_output)
        total_loss = self.__sum_present_losses(encoder_loss, decoder_loss)
        return final_output, total_loss

    def __select_final_output(
        self,
        encoder_output: Tensor | None,
        decoder_output: Tensor | None,
    ) -> Tensor | None:
        if decoder_output is not None:
            return decoder_output
        return encoder_output

    def __run_encoder_if_present(
        self,
        source_token_embeddings: Tensor | None,
        source_key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.encoder_model is None:
            return None, None
        return self.encoder_model(
            source_token_embeddings=source_token_embeddings,
            source_key_padding_mask=source_key_padding_mask,
            attention_mask=attention_mask,
        )

    def __run_decoder_if_present(
        self,
        target_token_embeddings: Tensor | None,
        encoder_output: Tensor | None,
        target_key_padding_mask: Tensor | None,
        encoder_key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
        encoder_attention_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.decoder_model is None:
            return None, None
        return self.decoder_model(
            target_token_embeddings=target_token_embeddings,
            encoder_output=encoder_output,
            target_key_padding_mask=target_key_padding_mask,
            encoder_key_padding_mask=encoder_key_padding_mask,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

    def __sum_present_losses(
        self,
        encoder_loss: Tensor | None,
        decoder_loss: Tensor | None,
    ) -> Tensor | None:
        present_losses = [
            loss for loss in (encoder_loss, decoder_loss) if loss is not None
        ]
        if not present_losses:
            return None
        return sum(present_losses)
