# import torch.nn as nn
# nn.TransformerEncoderLayer

from dataclasses import dataclass, field
from Emperor.layers.utils.enums import LayerTypes
from Emperor.attention.attention import MultiHeadAttention
from Emperor.base.utils import DataClassBase, Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class TransformerEncoderLayerConfig(DataClassBase):
    model_type: LayerTypes | None = field(
        default=None,
        metadata={"help": ""},
    )
    batch_size: int | None = field(
        default=None,
        metadata={"help": ""},
    )


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        cfg: "TransformerEncoderLayerConfig | ModelConfig",
        overrides: "TransformerEncoderLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "multi_head_attention_model_config", cfg)
        self.cfg: "TransformerEncoderLayerConfig" = self._overwrite_config(
            config, overrides
        )
        self.main_cfg = cfg
        self.num_heads = self.cfg.num_heads

        self.self_attention = MultiheadAttention(self.cfg)
