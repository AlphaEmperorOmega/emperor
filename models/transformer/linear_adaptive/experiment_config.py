from dataclasses import dataclass

from emperor.base.layer import LayerStackConfig, RecurrentLayerConfig
from emperor.base.config import ConfigBase, optional_field
from emperor.embedding.absolute.core.config import AbsolutePositionalEmbeddingConfig


@dataclass
class ExperimentConfig(ConfigBase):
    source_positional_embedding_config: AbsolutePositionalEmbeddingConfig | None = (
        optional_field("Source positions.")
    )
    target_positional_embedding_config: AbsolutePositionalEmbeddingConfig | None = (
        optional_field("Target positions.")
    )
    encoder_config: LayerStackConfig | RecurrentLayerConfig | None = optional_field(
        "Encoder stack."
    )
    decoder_config: LayerStackConfig | RecurrentLayerConfig | None = optional_field(
        "Decoder stack."
    )
    vocab_size: int | None = optional_field("Shared vocabulary size.")
    model_dim: int | None = optional_field("Transformer width.")
    source_sequence_length: int | None = optional_field("Maximum source length.")
    target_sequence_length: int | None = optional_field("Maximum target length.")
    dropout_probability: float | None = optional_field("Embedding dropout.")
    pad_token_id: int | None = optional_field("PAD ID.")
    bos_token_id: int | None = optional_field("BOS ID.")
    eos_token_id: int | None = optional_field("EOS ID.")
    label_smoothing: float | None = optional_field("Label smoothing.")
    warmup_steps: int | None = optional_field("Scheduler warmup.")
    generation_metrics_flag: bool | None = optional_field("Enable BLEU.")
