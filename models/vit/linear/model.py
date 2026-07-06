from typing import TYPE_CHECKING

from models.vit._base_model import VitClassifierModel

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(VitClassifierModel):
    def __init__(
        self,
        config: "ModelConfig",
    ):
        super().__init__(config)
        self.experiment_config = config.experiment_config
