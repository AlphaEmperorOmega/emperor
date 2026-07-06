from typing import TYPE_CHECKING

from models.bert._base_model import BertPretrainingModel

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(BertPretrainingModel):
    def __init__(
        self,
        config: "ModelConfig",
    ):
        super().__init__(config)
        self.experiment_config = config.experiment_config
