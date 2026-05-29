import torch

from torch import Tensor
from torch.nn import Sequential
from emperor.base.layer import Layer
from emperor.base.layer.stack import LayerStack
from emperor.base.layer.config import LayerConfig, LayerStackConfig, RecurrentLayerConfig
from emperor.experiments.classifier import ClassifierExperiment
from models.linear_adaptive.experiment_config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.main_cfg = self.cfg
        self.cfg: ExperimentConfig = self.cfg.experiment_config
        self.input_model = self._build_input_model()
        self.model = self._build_model()
        self.output_model = self._build_output_model()

    def _build_input_model(self) -> Layer:
        return self._build_layer(
            self.cfg.input_model_config,
            input_dim=self.main_cfg.input_dim,
            output_dim=self.main_cfg.hidden_dim,
        )

    def _build_model(self) -> "torch.nn.Module":
        model_config = self.cfg.model_config
        if isinstance(model_config, RecurrentLayerConfig):
            return model_config.build(
                overrides=RecurrentLayerConfig(
                    input_dim=self.main_cfg.hidden_dim,
                    output_dim=self.main_cfg.hidden_dim,
                )
            )
        return self._build_layer_stack(
            model_config,
            input_dim=self.main_cfg.hidden_dim,
            output_dim=self.main_cfg.hidden_dim,
        )

    def _build_output_model(self) -> Layer:
        return self._build_layer(
            self.cfg.output_model_config,
            input_dim=self.main_cfg.hidden_dim,
            output_dim=self.main_cfg.output_dim,
        )

    def _build_layer(
        self,
        model_config: LayerConfig,
        input_dim: int,
        output_dim: int,
    ) -> Layer:
        override = LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        return Layer(model_config, override)

    def _build_layer_stack(
        self,
        model_config: LayerStackConfig,
        input_dim: int,
        output_dim: int,
    ) -> Layer | Sequential:
        override = LayerStackConfig(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        return LayerStack(model_config, override).build()

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        X = torch.flatten(X, start_dim=1)
        X = Layer.forward_with_state(self.input_model, X)
        X = Layer.forward_with_state(self.model, X)
        X = Layer.forward_with_state(self.output_model, X)
        return X
