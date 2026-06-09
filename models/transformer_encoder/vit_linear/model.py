import torch.nn as nn
from torch import Tensor

from emperor.base.layer import Layer
from emperor.experiments.classifier import ClassifierExperiment
from models.transformer_encoder.vit_linear.experiment_config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        if not isinstance(cfg.experiment_config, ExperimentConfig):
            raise TypeError(
                "cfg.experiment_config must be a vit_linear ExperimentConfig."
            )
        self.main_cfg: ExperimentConfig = cfg.experiment_config

        self.patch = self.main_cfg.patch_config.build()
        self.positional_embedding = self.main_cfg.positional_embedding_config.build()
        self.transformer = self.main_cfg.encoder_config.build()
        self.encoder_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.output = self.main_cfg.output_config.build()

    def forward(
        self,
        images: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:
        images = images.to(self.device)
        hidden = self.patch(images)
        hidden = self.positional_embedding(hidden)

        encoder_state = Layer.run_model_returning_state(self.transformer, hidden)
        hidden = self.encoder_layer_norm(encoder_state.hidden)

        cls_hidden = hidden[:, 0, :]
        logits = Layer.run_model_returning_hidden(self.output, cls_hidden)

        if encoder_state.loss is not None and encoder_state.loss.item() != 0.0:
            return logits, encoder_state.loss
        return logits
