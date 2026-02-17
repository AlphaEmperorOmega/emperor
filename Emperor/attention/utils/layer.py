from torch import Tensor
from dataclasses import dataclass, field
from Emperor.attention.utils.utils import Utils
from Emperor.base.utils import ConfigBase, Module
from Emperor.attention.utils.handlers.maks import MaskBuilder
from Emperor.attention.utils.enums import AttentionOptions
from Emperor.attention.utils.handlers.bias import KeyValueBias
from Emperor.attention.utils.handlers.reshaper import ReshaperBuilder
from Emperor.attention.utils.handlers.processor import ProcessorBuilder
from Emperor.attention.utils.handlers.projector import ProjectorBuilder
from Emperor.attention.utils.handlers.batch import BatchDimensionManager
from Emperor.attention.utils._validator import MultiHeadAttentionValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from torch.types import _dtype as DType
    from Emperor.linears.options import LinearLayerStackOptions
    from Emperor.adaptive.options import AdaptiveLayerStackOptions
    from Emperor.experts.utils.layers import MixtureOfExpertsConfig


@dataclass
class MultiHeadAttentionConfig(ConfigBase):
    model_type: "AdaptiveLayerStackOptions | LinearLayerStackOptions | None" = field(
        default=None,
        metadata={
            "help": "Type of model used to generate parameters query, key, and value projections"
        },
    )
    batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Number of samples (batch size) processed in parallel during training/inference."
        },
    )
    num_heads: int | None = field(
        default=None,
        metadata={"help": "Number of attention heads to use for multi-head attention."},
    )
    embedding_dim: int | None = field(
        default=None,
        metadata={
            "help": "Dimension of the input embedding vector (input feature size)."
        },
    )
    query_key_projection_dim: int | None = field(
        default=None,
        metadata={
            "help": "Dimension of the query and key projected vectors (defaults to embedding_dim if 0)."
        },
    )
    value_projection_dim: int | None = field(
        default=None,
        metadata={
            "help": "Dimension of the value projected vectors (defaults to embedding_dim if 0)."
        },
    )
    target_sequence_length: int | None = field(
        default=None,
        metadata={
            "help": "Length of the target sequence for decoding or output (number of target tokens)."
        },
    )
    source_sequence_length: int | None = field(
        default=None,
        metadata={
            "help": "Length of the source/input sequence (number of input tokens)."
        },
    )
    target_dtype: "DType | None" = field(
        default=None,
        metadata={
            "help": "Data type (dtype) for the attention outputs (e.g. torch.float32)."
        },
    )
    attention_option: AttentionOptions | None = field(
        default=None,
        metadata={"help": "Option for selecting the type of projector to use."},
    )
    dropout_probability: float | None = field(
        default=None,
        metadata={
            "help": "Dropout probability applied to attention weights (prevents overfitting)."
        },
    )
    key_value_bias_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, add learnable bias vectors to key and value projections."
        },
    )
    zero_attention_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, add zero vectors to attention keys/values to allow explicit non-attending positions."
        },
    )
    causal_attention_mask_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, use a causal mask to prevent attention to future positions (for decoding/generation)."
        },
    )
    add_key_value_bias_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    average_attention_weights_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    return_attention_weights_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    experts_config: "MixtureOfExpertsConfig | None" = field(
        default=None,
        metadata={
            "help": "Type of model used to generate parameters query, key, and value projections"
        },
    )
    use_kv_expert_models_flag: bool | None = field(
        default=None,
        metadata={
            "help": "Type of model used to generate parameters query, key, and value projections"
        },
    )


class MultiHeadAttention(Module):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig | ModelConfig",
        overrides: "MultiHeadAttentionConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "multi_head_attention_model_config", cfg)
        self.cfg: "MultiHeadAttentionConfig" = self._overwrite_config(config, overrides)

        self.main_cfg = cfg
        self.num_heads = self.cfg.num_heads
        self.batch_size = self.cfg.batch_size
        self.model_type = self.cfg.model_type
        self.target_dtype = self.cfg.target_dtype
        self.embedding_dim = self.cfg.embedding_dim
        self.dropout_probability = self.cfg.dropout_probability
        self.key_value_bias_flag = self.cfg.key_value_bias_flag
        self.zero_attention_flag = self.cfg.zero_attention_flag
        self.value_projection_dim = self.cfg.value_projection_dim
        self.query_key_projection_dim = self.cfg.query_key_projection_dim
        self.target_sequence_length = self.cfg.target_sequence_length
        self.source_sequence_length = self.cfg.source_sequence_length
        self.add_key_value_bias_flag = self.cfg.add_key_value_bias_flag
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag
        self.attention_option = self.cfg.attention_option
        self.average_attention_weights_flag = self.cfg.average_attention_weights_flag
        # self._validate_fields(self.cfg, MultiHeadAttentionConfig)
        # self.__initialize_utilities()

        self.validator = MultiHeadAttentionValidator(self.cfg)
        self.masks = MaskBuilder(self.cfg).build()
        self.projector = ProjectorBuilder(self.cfg).build()
        self.processor = ProcessorBuilder(self.cfg, self.projector).build()
        self.reshaper = ReshaperBuilder(self.cfg).build()
        self.bias = KeyValueBias(self.cfg)
        self.utils = Utils(self.cfg, self.validator)
        self.batch_utils = BatchDimensionManager(self.cfg)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        k_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        static_k: Tensor | None = None,
        static_v: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        q, k, v = self.batch_utils.enforce_batch_as_second_dim(q, k, v)
        self.validator.check_attention_input_shapes(
            q, k, v, k_padding_mask, attention_mask
        )
        q, k, v, k_padding_mask, attention_mask = (
            self.utils.add_batch_dimension_if_missing(
                q, k, v, k_padding_mask, attention_mask
            )
        )
        k_padding_mask, attention_mask = self.masks.process_attention_masks(
            k_padding_mask, attention_mask
        )
        q, k, v = self.projector.compute_qkv_projections(q, k, v)
        k, v, k_padding_mask, attention_mask = self.bias.add_kv_learnable_bias_vectors(
            k, v, k_padding_mask, attention_mask
        )
        q, k, v = self.reshaper.reshape_qkv_for_attention(q, k, v, static_k, static_v)
        k, v, attention_mask, k_padding_mask = self.utils.add_zero_attention(
            k, v, attention_mask, k_padding_mask
        )
        merged_masks = self.masks.merge_padding_and_attention_mask(
            k, k_padding_mask, attention_mask
        )
        attention_output, attention_weights = self.processor.compute_attention(
            q, k, v, merged_masks
        )
        attention_output = self.batch_utils.reverse_enforced_batch_as_second_dim(
            attention_output
        )
        return attention_output, attention_weights
