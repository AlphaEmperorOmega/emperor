from dataclasses import dataclass

from emperor.config._base import ConfigBase, optional_field


@dataclass
class ModelConfig(ConfigBase):
    batch_size: int | None = optional_field("Batch size for training and inference.")
    learning_rate: float | None = optional_field("Learning rate for training.")
    sequence_length: int | None = optional_field(
        "Number of tokens for each sequence in the input batch."
    )
    input_dim: int | None = optional_field("Dimension of the input features.")
    hidden_dim: int | None = optional_field("Dimension of the hidden features.")
    output_dim: int | None = optional_field("Dimension of the output features.")
    experiment_config: ConfigBase | None = optional_field(
        "Model-specific configuration tree."
    )
