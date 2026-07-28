import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._biases.config import (
    DynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters._biases.validation import (
    DynamicBiasValidator,
)
from emperor.augmentations.adaptive_parameters._decay import _DecayPolicy
from emperor.layers import Layer, LayerStack, LayerStackConfig
from emperor.nn import Module


class DynamicBiasAbstract(Module):
    VALIDATOR = DynamicBiasValidator

    def __init__(
        self,
        cfg: "DynamicBiasConfig",
        overrides: "DynamicBiasConfig | None" = None,
    ):
        super().__init__()
        self.cfg: DynamicBiasConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.model_config = self.cfg.model_config
        self.register_buffer("decay_step", torch.zeros(1))
        self.register_buffer("warmup_step", torch.zeros(1))
        self._decay_policy = _DecayPolicy(self.cfg)

    def _init_model(self, output_dim: int) -> "Layer | LayerStack":
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=output_dim,
        )
        generator_model = self.model_config.build(overrides)
        self.VALIDATOR.validate_generator_model(generator_model)
        return generator_model

    def _maybe_apply_bias_decay(self, bias_params: Tensor) -> Tensor:
        return self._decay_policy.apply(
            bias_params,
            decay_step=self.decay_step,
            warmup_step=self.warmup_step,
            training=self.training,
        )
