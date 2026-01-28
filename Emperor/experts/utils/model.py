import torch

from torch import Tensor
from torch.nn import Sequential
from Emperor.base.utils import Module
from Emperor.sampler.model import SamplerModel
from Emperor.base.layer import Layer, LayerStackConfig
from Emperor.experts.utils.enums import InitSamplerOptions
from Emperor.experts.utils.stack import MixtureOfExpertsStack
from Emperor.experts.utils.layers import MixtureOfExpertsConfig
from Emperor.sampler.utils.routers import RouterConfig, RouterModel
from Emperor.experts.utils._validator import MixtureOfExpertsModelValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class MixtureOfExpertsModel(Module):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ) -> None:
        super().__init__()
        config = getattr(cfg, "layer_stack_config", cfg)
        self.cfg = self._overwrite_config(config, overrides)
        self.main_cfg: "MixtureOfExpertsConfig" = self._resolve_main_config(
            self.cfg, cfg
        )

        self.top_k = self.main_cfg.top_k
        self.input_dim = self.main_cfg.input_dim
        self.init_sampler_option = self.main_cfg.init_sampler_option
        self.router_model_config = self.main_cfg.router_model_config
        self.sampler_model_config = self.main_cfg.sampler_model_config

        self.router, self.sampler = self.__maybe_create_router_and_sampler()
        self.expert_stack = self._create_expert_stack()
        self.validator = MixtureOfExpertsModelValidator(self)

    def get_top_k(self) -> int:
        return self.top_k

    def _create_expert_stack(self) -> Layer | Sequential:
        return MixtureOfExpertsStack(self.cfg).build_model()

    def __maybe_create_router_and_sampler(
        self,
    ) -> tuple[RouterModel | None, SamplerModel | None]:
        if self.init_sampler_option == InitSamplerOptions.SHARED:
            router_overrides = RouterConfig(input_dim=self.input_dim)
            router = RouterModel(self.router_model_config, router_overrides)
            sampler = SamplerModel(self.sampler_model_config)
            return router, sampler
        return None, None

    def forward(
        self,
        input: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        probabilities, indices, sampler_loss = self._maybe_compute_expert_indices(
            input, probabilities, indices
        )

        inputs = {
            "input_batch": input,
            "probabilities": probabilities,
            "indices": indices,
            "loss": None,
        }
        output, expert_loss = self.expert_stack(inputs)

        total_loss = sampler_loss + expert_loss
        return output, total_loss

    def _maybe_compute_expert_indices(
        self,
        inputs: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor]:
        if self.init_sampler_option != InitSamplerOptions.SHARED or (
            indices is not None or probabilities is not None
        ):
            return probabilities, indices, torch.tensor(0.0)

        logits = self.router.compute_logit_scores(inputs)

        # TODO: In the future see if `skip_mask` needs to be implemented
        skip_mask = None
        probabilities, indices, skip_mask, sampler_loss = (
            self.sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        return probabilities, indices, sampler_loss


class MixtureOfExpertsAttentionModel(Module):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ) -> None:
        super().__init__()
        config = getattr(cfg, "layer_stack_config", cfg)
        self.cfg = self._overwrite_config(config, overrides)
        self.main_cfg: "MixtureOfExpertsConfig" = self._resolve_main_config(
            self.cfg, cfg
        )

        self.top_k = self.main_cfg.top_k
        self.input_dim = self.main_cfg.input_dim
        self.init_sampler_option = self.main_cfg.init_sampler_option
        self.router_model_config = self.main_cfg.router_model_config
        self.sampler_model_config = self.main_cfg.sampler_model_config

        self.router, self.sampler = self.__maybe_create_router_and_sampler()
        self.input_expert_layer = self._create_expert_layer()
        self.output_expert_layer = self._create_expert_layer()

    def _create_expert_stack(self) -> Layer | Sequential:
        pass

    def __maybe_create_router_and_sampler(
        self,
    ) -> tuple[RouterModel | None, SamplerModel | None]:
        if self.init_sampler_option == InitSamplerOptions.SHARED:
            router_overrides = RouterConfig(input_dim=self.input_dim)
            router = RouterModel(self.router_model_config, router_overrides)
            sampler = SamplerModel(self.sampler_model_config)
            return router, sampler
        return None, None

    def get_top_k(self) -> int:
        return self.top_k

    def _create_expert_layer(self) -> Layer | Sequential:
        return MixtureOfExpertsStack(self.cfg).build_model()

    def compute_input_expert(
        self,
        input: Tensor,
        skip_mask: Tensor | None = None,
    ):
        probabilities, indices, sampler_loss = self._maybe_compute_expert_indices(
            input, probabilities, indices
        )
        output, expert_loss = self.input_expert_layer(
            input, probabilities=probabilities, indices=indices
        )
        return output, expert_loss

    def compute_output_expert(self):
        pass
