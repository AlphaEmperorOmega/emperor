from Emperor.base.enums import BaseOptions
from Emperor.base.utils import Module
from Emperor.transformer.utils.patch.options.base import PatchConfig
from Emperor.transformer.utils.patch.options.patch_tokenizer import PatchTokenizer
from Emperor.transformer.utils.patch.options.patch_embedding import PatchEmbeddingConv


class PatchOptions(BaseOptions):
    DISABLED = 0
    TOKENIZER = 1
    CONV = 2


class PatchSelector(Module):
    def __init__(
        self,
        cfg: "PatchConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.patch_option = self.cfg.patch_option

    def build(self) -> Module:
        match self.patch_option:
            case PatchOptions.TOKENIZER:
                return PatchTokenizer(self.cfg)
            case PatchOptions.CONV:
                return PatchEmbeddingConv(self.cfg)
            case _:
                raise ValueError(
                    "If the `patch_option` is set to `DISABLED` or, this class should not be initialized"
                )
