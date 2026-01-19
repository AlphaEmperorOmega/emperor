import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Parameter
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase, Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.linears.options import LinearLayerStackOptions
    from Emperor.transformer.utils.patch.selector import PatchOptions


@dataclass
class PatchConfig(ConfigBase):
    patch_option: "PatchOptions | None" = field(
        default=None,
        metadata={"help": ""},
    )
    embedding_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    num_input_channels: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    patch_size: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    stride: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    padding: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    dropout_probability: float | None = field(
        default=None,
        metadata={"help": ""},
    )


class PatchBase(Module):
    def __init__(
        self,
        cfg: "PatchConfig | ModelConfig",
        overrides: "PatchConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "PatchConfig" = self._overwrite_config(cfg, overrides)
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)
        self.embedding_dim = self.cfg.embedding_dim
        self.patch_size = self.cfg.patch_size
        self.stride = self.cfg.stride
        self.padding = self.cfg.padding
        self.dropout_probability = self.cfg.dropout_probability
        self.num_input_channels = self.cfg.num_input_channels

        self.class_token = self._create_class_token()
        self.dropout = nn.Dropout(self.dropout_probability)

    def _create_class_token(self, shape: tuple | None = None) -> Parameter:
        if shape is None:
            shape = (1, 1, self.embedding_dim)
        class_token_init = torch.randn(shape)
        return Parameter(class_token_init, requires_grad=True)

    def _concatenate_class_token(self, X: Tensor) -> Tensor:
        batch_size = X.size(0)
        class_token = self.class_token.expand(batch_size, -1, -1)
        return torch.cat([class_token, X], dim=1)
