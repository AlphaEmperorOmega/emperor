from dataclasses import dataclass, field
import torch
from torch import Tensor
from Emperor.base.decorators import timer
from Emperor.base.utils import Module, DataClassBase

from Emperor.components.parameter_generators.utils.behaviours import (
    DynamicDiagonalParametersBehaviour,
)
from Emperor.components.parameter_generators.utils.samplers import SamplerModel
from Emperor.components.parameter_generators.utils.mixture import (
    GeneratorMixture,
    MatrixMixture,
    VectorMixture,
)
from Emperor.components.parameter_generators.utils.routers import (
    RouterModel,
    VectorRouterModel,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class ParameterLayerConfig(DataClassBase):
    bias_parameters_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` it will generate bias parameters for each input sample."
        },
    )
    time_tracker_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` it will generate bias parameters for each input sample."
        },
    )
    dynamic_diagonal_params_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` for weight parameters a set of `diagonal` and `anti_diagonal` parameters are added to the generated weight_parameters for each input sampele, for biases a set of parameters that scale the biases are generated and biases are added to the bias parameters that shift them for each sample."
        },
    )


class ParameterLayerBase(Module):
    def __init__(
        self,
        cfg: "ParameterLayerConfig | ModelConfig",
        overrides: "ParameterLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "parameter_generator_model_config", cfg)
        self.cfg: "ParameterLayerConfig" = self._overwrite_config(config, overrides)

        self.input_dim = cfg.mixture_model_config.input_dim
        self.output_dim = cfg.mixture_model_config.output_dim
        self.bias_parameters_flag = self.cfg.bias_parameters_flag
        self.time_tracker_flag = self.cfg.time_tracker_flag
        self.dynamic_diagonal_params_flag = self.cfg.dynamic_diagonal_params_flag
        self.dyagonal_params_model = self.__create_diagonal_params_model()

    def __create_diagonal_params_model(
        self,
    ) -> "DynamicDiagonalParametersBehaviour | None":
        if not self.dynamic_diagonal_params_flag:
            return None

        return DynamicDiagonalParametersBehaviour(
            self.input_dim,
            self.output_dim,
            anti_diagonal_flag=True,
            dynamic_bias_flag=True,
        )

    def forward(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        if self.time_tracker_flag:
            return self._track_layer_output_time(input_batch, skip_mask)
        return self._compute_layer_output(input_batch, skip_mask)

    @timer
    def _track_layer_output_time(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        return self.compute_layer_output(input_batch, skip_mask)

    def _compute_layer_output(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        generated_weights, generated_biases = self._generate_parameters(
            input_batch, skip_mask
        )
        generated_weights, generated_biases = self.__add_dynamic_diagonal_params(
            input_batch, generated_weights, generated_biases
        )

        output = self.__apply_generated_weights(input_batch, generated_weights)
        output = self.__apply_generated_biases(output, generated_biases)

        updated_skip_mask = self.__get_updated_skip_mask()
        total_layer_loss = self.__get_total_layer_loss()

        return output, updated_skip_mask, total_layer_loss

    def _generate_parameters(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        weight_probabilities, weight_indices = self._compute_probabilities_and_indices(
            input_batch, skip_mask=skip_mask
        )
        bias_probabilities, bias_indices = self._compute_bias_probabilities_and_indices(
            input_batch, skip_mask
        )
        weight_parameters, bias_parameters = self.mixture.compute_mixture(
            weight_probabilities, weight_indices, bias_probabilities, bias_indices
        )
        return weight_parameters, bias_parameters

    def __add_dynamic_diagonal_params(
        self,
        input_batch: Tensor,
        weight_params: Tensor,
        bias_params: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.dynamic_diagonal_params_flag:
            self.dyagonal_params_model.set_parameters(weight_params, bias_params)
            return self.dynamic_diagonal_params_model(input_batch)
        return weight_params, bias_params

    def __apply_generated_weights(
        self,
        input_batch: Tensor,
        generated_weights: Tensor,
    ) -> Tensor:
        return torch.einsum("bi,bij->bj", input_batch, generated_weights)

    def __apply_generated_biases(
        self,
        weighted_inputs: Tensor,
        generated_biases: Tensor | None = None,
    ) -> Tensor:
        if self.bias_parameters_flag:
            return weighted_inputs + generated_biases
        return weighted_inputs

    def _compute_bias_probabilities_and_indices(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.bias_parameters_flag:
            return self._compute_probabilities_and_indices(
                input_batch, self.bias_parameters_flag, skip_mask
            )
        return None, None

    def _compute_probabilities_and_indices(
        self,
        input_batch: Tensor,
        compute_bias_flag: bool = False,
        skip_mask: Tensor | None = None,
    ):
        if compute_bias_flag:
            logits = self._compute_logits(input_batch)
            return self._sample_probabilities_and_indices(
                logits, skip_mask, compute_bias_flag
            )
        logits = self._compute_logits(input_batch)
        return self._sample_probabilities_and_indices(
            logits, skip_mask, compute_bias_flag
        )

    def _compute_logits(
        self,
        input_batch: Tensor,
        compute_bias_flag: bool = False,
    ) -> Tensor:
        if compute_bias_flag:
            if self.bias_router is None or self.bias_sampler is None:
                raise RuntimeError("Bias router is not initialized.")
            return self.bias_router.compute_logit_scores(input_batch)
        return self.weight_router.compute_logit_scores(input_batch)

    def _sample_probabilities_and_indices(
        self,
        logits: Tensor,
        skip_mask: Tensor | None = None,
        compute_bias_flag: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if compute_bias_flag:
            if self.bias_sampler is None:
                raise RuntimeError("Bias sampler is not initialized.")
            probabilities, selected_indices, _, _ = (
                self.bias_sampler.sample_probabilities_and_indices(logits, skip_mask)
            )
            return probabilities, selected_indices

        probabilities, selected_indices, _, _ = (
            self.weight_sampler.sample_probabilities_and_indices(logits, skip_mask)
        )
        return probabilities, selected_indices

    def __get_updated_skip_mask(self):
        return self.weight_sampler.get_updated_skip_mask()

    def __get_total_layer_loss(self) -> Tensor:
        weight_loss = self.weight_sampler.get_auxiliary_loss()
        bias_loss = self.bias_sampler.get_auxiliary_loss()
        return weight_loss + bias_loss


class VectorParameterLayer(ParameterLayerBase):
    def __init__(
        self,
        cfg: "ParameterLayerConfig | ModelConfig",
        overrides: "ParameterLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.weight_router: VectorRouterModel = VectorRouterModel(cfg)
        self.bias_router = self._init_bias_router_model(cfg)
        self.weight_sampler: SamplerModel = SamplerModel(cfg)
        self.bias_sampler: SamplerModel = SamplerModel(cfg)
        self.mixture: VectorMixture = VectorMixture(cfg)

    def _init_bias_router_model(self, cfg: "ModelConfig") -> VectorRouterModel | None:
        if self.bias_parameters_flag:
            return VectorRouterModel(
                cfg,
                bias_parameters_flag=self.bias_parameters_flag,
            )
        return None

    def _compute_probabilities_and_indices(
        self,
        input_batch: Tensor,
        compute_bias_flag: bool = False,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        logits = self._compute_logits(input_batch, compute_bias_flag)
        input_dim, batch_size, depth_dim = logits.shape
        logits = logits.view(-1, depth_dim)

        probabilities, indices = self._sample_probabilities_and_indices(
            logits, skip_mask, compute_bias_flag
        )

        probabilities_shape = (input_dim, batch_size)
        indices_shape = (input_dim, batch_size)
        if self.mixture.top_k > 1:
            probabilities_shape = (input_dim, batch_size, -1)
            indices_shape = (input_dim, batch_size, -1)

        probabilities = probabilities.reshape(probabilities_shape)
        if indices is not None:
            indices = indices.reshape(indices_shape)

        return probabilities, indices


class MatrixParameterLayer(ParameterLayerBase):
    def __init__(
        self,
        cfg: "ParameterLayerConfig | ModelConfig",
        overrides: "ParameterLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.weight_router: RouterModel = RouterModel(cfg)
        self.bias_router = self._init_bias_router_model(cfg)
        self.weight_sampler: SamplerModel = SamplerModel(cfg)
        self.bias_sampler: SamplerModel = SamplerModel(cfg)
        self.mixture: MatrixMixture = MatrixMixture(cfg)

    def _init_bias_router_model(self, cfg: "ModelConfig") -> RouterModel | None:
        if self.bias_parameters_flag:
            return RouterModel(cfg)
        return None


class GeneratorParameterLayer(ParameterLayerBase):
    def __init__(
        self,
        cfg: "ParameterLayerConfig | ModelConfig",
        overrides: "ParameterLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.weight_router: RouterModel = RouterModel(cfg)
        self.bias_router = self._init_bias_router_model(cfg)
        self.weight_sampler: SamplerModel = SamplerModel(cfg)
        self.bias_sampler: SamplerModel = SamplerModel(cfg)
        self.mixture: GeneratorMixture = GeneratorMixture(cfg)

    def _init_bias_router_model(self, cfg: "ModelConfig") -> RouterModel | None:
        if self.bias_parameters_flag:
            return RouterModel(cfg)
        return None

    def _generate_parameters(
        self,
        input_batch: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        weight_probabilities, weight_indices = self._compute_probabilities_and_indices(
            input_batch,
        )
        bias_probabilities, bias_indices = self._compute_bias_probabilities_and_indices(
            input_batch
        )

        weight_parameters, bias_parameters = self.mixture.compute_mixture(
            weight_probabilities,
            weight_indices,
            bias_probabilities,
            bias_indices,
            input_batch,
        )

        return weight_parameters, bias_parameters
