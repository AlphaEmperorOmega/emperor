from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field
from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.layers import LayerStackConfig, RecurrentLayerConfig


@dataclass
class ExperimentConfig(ConfigBase):
    source_positional_embedding_config: AbsolutePositionalEmbeddingConfig | None = (
        optional_field("Source positional embedding configuration.")
    )
    target_positional_embedding_config: AbsolutePositionalEmbeddingConfig | None = (
        optional_field("Target positional embedding configuration.")
    )
    encoder_config: LayerStackConfig | RecurrentLayerConfig | None = optional_field(
        "Stateful Emperor encoder block stack."
    )
    decoder_config: LayerStackConfig | RecurrentLayerConfig | None = optional_field(
        "Stateful Emperor decoder block stack."
    )
    vocab_size: int | None = optional_field("Shared source/target vocabulary size.")
    model_dim: int | None = optional_field("Transformer hidden dimension.")
    source_sequence_length: int | None = optional_field("Maximum source length.")
    target_sequence_length: int | None = optional_field("Maximum target length.")
    dropout_probability: float | None = optional_field("Embedding dropout.")
    pad_token_id: int | None = optional_field("Padding token ID.")
    bos_token_id: int | None = optional_field("Beginning-of-sequence token ID.")
    eos_token_id: int | None = optional_field("End-of-sequence token ID.")
    label_smoothing: float | None = optional_field("Cross-entropy label smoothing.")
    warmup_steps: int | None = optional_field("Inverse-square-root LR warmup.")
    generation_metrics_flag: bool | None = optional_field("Enable greedy BLEU metrics.")
