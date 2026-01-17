import torch

from torch import Tensor
from torch.nn import Parameter
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase, Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.transformer.utils.patch.selector import PatchOptions
    from Emperor.config import ModelConfig


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
    dropout: float | None = field(
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
        self.embedding_dim = self.cfg.embedding_dim
        self.patch_size = self.cfg.patch_size
        self.stride = self.cfg.stride
        self.padding = self.cfg.padding
        self.dropout = self.cfg.dropout

        self.class_token = self._create_class_token()

    def _create_class_token(self, shape: tuple | None = None) -> Parameter:
        if shape is None:
            shape = (1, 1, self.embedding_dim)
        class_token_init = torch.randn(shape)
        return Parameter(class_token_init, requires_grad=True)

    def _concatenate_class_token(self, X: Tensor) -> Tensor:
        batch_size = X.size(0)
        class_token = self.class_token.expand(batch_size, -1, -1)
        return torch.cat([class_token, X], dim=1)
