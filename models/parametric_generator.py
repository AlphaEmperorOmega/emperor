import torch

from torch import Tensor
from models.parser import get_experiment_parser
from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.base.enums import BaseOptions
from emperor.datasets.image.mnist import Mnist
from emperor.base.layer import LayerStack, LayerStackConfig
from emperor.linears.utils.layers import LinearLayerConfig
from emperor.sampler.utils.routers import RouterConfig
from emperor.sampler.utils.samplers import SamplerConfig
from emperor.parametric.options import AdaptiveLayerOptions
from emperor.parametric.utils.layers import AdaptiveParameterLayerConfig, AdaptiveRouterOptions
from emperor.parametric.utils.mixtures.base import AdaptiveMixtureConfig
from emperor.parametric.utils.mixtures.options import AdaptiveBiasOptions, AdaptiveWeightOptions
from emperor.parametric.utils.mixtures.types.utils.enums import ClipParameterOptions
from emperor.behaviours.model import AdaptiveParameterBehaviourConfig
from emperor.experts.utils.layers import MixtureOfExpertsConfig
from emperor.experiments.classifier import ClassifierExperiment
from emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from emperor.experts.utils.enums import (
    ExpertWeightingPositionOptions,
    InitSamplerOptions,
)
from emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    create_search_space,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


@dataclass
class ExperimentConfig(ConfigBase):
    model_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.main_cfg: ExperimentConfig = self._resolve_main_config(self.cfg, cfg)
        self.model_config: LayerStackConfig = self.main_cfg.model_config
        self.model = LayerStack(self.model_config).build_model()

    def _resolve_main_config(
        self, sub_config: "ConfigBase", main_cfg: "ConfigBase"
    ) -> None:
        if sub_config.override_config is not None:
            return sub_config.override_config
        return main_cfg

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        X = torch.flatten(X, start_dim=1)
        X = self.model(X)
        return X


class ExperimentOptions(BaseOptions):
    DEFAULT = 0
    BASE = 1


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_option: ExperimentOptions | None = None,
    ) -> None:
        super().__init__(experiment_option)
        self.accelerator = "cpu"

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_enumeration(self) -> type[BaseOptions]:
        return ExperimentOptions


class ExperimentPresets(ExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.DEFAULT,
        dataset: type = Mnist,
        num_samples: int | None = None,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.DEFAULT:
                return self._default_config(dataset)
            case ExperimentOptions.BASE:
                return self.__base_grid_search_config(dataset, num_samples)
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `ExperimentOptions`."
                )

    def __base_grid_search_config(
        self,
        dataset: type = Mnist,
        num_random_search_samples: int | None = None,
    ) -> list["ModelConfig"]:
        base_config = self._dataset_config(dataset)

        return create_search_space(
            self._preset,
            base_config,
            self.__base_search_space(),
            num_random_search_samples,
        )

    def __base_search_space(self) -> dict:
        return {
            "learning_rate": [1e-4, 1e-3, 1e-2],
            "hidden_dim": [64, 128, 256],
            "stack_num_layers": [3, 6],
            "stack_dropout_probability": [0.0, 0.1],
            "stack_activation": [
                ActivationOptions.RELU,
                ActivationOptions.SILU,
                ActivationOptions.GELU,
                ActivationOptions.LEAKY_RELU,
            ],
            "adaptive_mixture_top_k": [1, 3],
            "adaptive_mixture_num_experts": [4, 6, 8],
            "adaptive_bias_option": [
                AdaptiveBiasOptions.DISABLED,
                AdaptiveBiasOptions.GENERATOR,
            ],
            "adaptive_mixture_clip_parameter_option": [
                ClipParameterOptions.NONE,
                ClipParameterOptions.BEFORE,
                ClipParameterOptions.AFTER,
            ],
            "adaptive_behaviour_generator_depth": [
                DynamicDepthOptions.DISABLED,
                DynamicDepthOptions.DEPTH_OF_ONE,
                DynamicDepthOptions.DEPTH_OF_TWO,
            ],
        }

    def _preset(
        self,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        input_dim: int = 28**2,
        hidden_dim: int = 256,
        output_dim: int = 10,
        stack_num_layers: int = 3,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
        adaptive_mixture_top_k: int = 3,
        adaptive_mixture_num_experts: int = 6,
        adaptive_mixture_weighted_parameters_flag: bool = False,
        adaptive_mixture_clip_parameter_option: ClipParameterOptions = ClipParameterOptions.BEFORE,
        adaptive_mixture_clip_range: float = 5.0,
        adaptive_bias_option: AdaptiveBiasOptions = AdaptiveBiasOptions.DISABLED,
        adaptive_behaviour_generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        adaptive_behaviour_diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        adaptive_behaviour_bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        adaptive_behaviour_memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        adaptive_behaviour_memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        adaptive_behaviour_memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        adaptive_generator_stack_num_layers: int = 2,
        adaptive_generator_stack_hidden_dim: int = 256,
        adaptive_generator_stack_activation: ActivationOptions = ActivationOptions.RELU,
        adaptive_generator_stack_residual_flag: bool = False,
        adaptive_generator_stack_dropout_probability: float = 0.0,
    ) -> "ModelConfig":
        from emperor.config import ModelConfig
        from emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions

        _hidden_dim = max(input_dim, output_dim)

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            override_config=ExperimentConfig(
                model_config=LayerStackConfig(
                    model_type=AdaptiveLayerOptions.BASE,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=stack_num_layers,
                    activation=stack_activation,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=stack_residual_flag,
                    adaptive_computation_flag=False,
                    dropout_probability=stack_dropout_probability,
                    override_config=AdaptiveParameterLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        adaptive_weight_option=AdaptiveWeightOptions.GENERATOR,
                        adaptive_bias_option=adaptive_bias_option,
                        init_sampler_model_option=AdaptiveRouterOptions.SHARED_ROUTER,
                        time_tracker_flag=False,
                        adaptive_behaviour_config=AdaptiveParameterBehaviourConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            generator_depth=adaptive_behaviour_generator_depth,
                            diagonal_option=adaptive_behaviour_diagonal_option,
                            bias_option=adaptive_behaviour_bias_option,
                            memory_option=adaptive_behaviour_memory_option,
                            memory_size_option=adaptive_behaviour_memory_size_option,
                            memory_position_option=adaptive_behaviour_memory_position_option,
                            override_config=LayerStackConfig(
                                model_type=LinearLayerOptions.BASE,
                                input_dim=input_dim,
                                hidden_dim=adaptive_generator_stack_hidden_dim,
                                output_dim=output_dim,
                                num_layers=adaptive_generator_stack_num_layers,
                                activation=adaptive_generator_stack_activation,
                                layer_norm_position=LayerNormPositionOptions.NONE,
                                residual_flag=adaptive_generator_stack_residual_flag,
                                adaptive_computation_flag=False,
                                dropout_probability=adaptive_generator_stack_dropout_probability,
                                override_config=LinearLayerConfig(
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    bias_flag=False,
                                    data_monitor=None,
                                    parameter_monitor=None,
                                    override_config=AdaptiveParameterBehaviourConfig(
                                        generator_depth=adaptive_behaviour_generator_depth,
                                    ),
                                ),
                            ),
                        ),
                        router_config=RouterConfig(
                            input_dim=input_dim,
                            layer_stack_option=LinearLayerStackOptions.BASE,
                            num_experts=adaptive_mixture_num_experts,
                            noisy_topk_flag=False,
                            override_config=LayerStackConfig(
                                model_type=LinearLayerOptions.BASE,
                                input_dim=input_dim,
                                hidden_dim=max(input_dim, adaptive_mixture_num_experts),
                                output_dim=adaptive_mixture_num_experts,
                                num_layers=2,
                                activation=stack_activation,
                                layer_norm_position=LayerNormPositionOptions.NONE,
                                residual_flag=False,
                                adaptive_computation_flag=False,
                                dropout_probability=0.0,
                                override_config=LinearLayerConfig(
                                    input_dim=input_dim,
                                    output_dim=adaptive_mixture_num_experts,
                                    bias_flag=False,
                                    data_monitor=None,
                                    parameter_monitor=None,
                                ),
                            ),
                        ),
                        sampler_config=SamplerConfig(
                            top_k=adaptive_mixture_top_k,
                            threshold=0.0,
                            filter_above_threshold=False,
                            num_topk_samples=0,
                            normalize_probabilities_flag=False,
                            noisy_topk_flag=False,
                            num_experts=adaptive_mixture_num_experts,
                            coefficient_of_variation_loss_weight=0.0,
                            switch_loss_weight=0.0,
                            zero_centred_loss_weight=0.0,
                            mutual_information_loss_weight=0.0,
                        ),
                        override_config=AdaptiveMixtureConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            top_k=adaptive_mixture_top_k,
                            num_experts=adaptive_mixture_num_experts,
                            weighted_parameters_flag=adaptive_mixture_weighted_parameters_flag,
                            clip_parameter_option=adaptive_mixture_clip_parameter_option,
                            clip_range=adaptive_mixture_clip_range,
                            override_config=MixtureOfExpertsConfig(
                                input_dim=input_dim,
                                output_dim=output_dim,
                                top_k=adaptive_mixture_top_k,
                                num_experts=adaptive_mixture_num_experts,
                                layer_stack_option=LinearLayerStackOptions.BASE,
                                compute_expert_mixture_flag=False,
                                weighted_parameters_flag=False,
                                weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
                                init_sampler_option=InitSamplerOptions.SHARED,
                                override_config=LayerStackConfig(
                                    model_type=LinearLayerOptions.BASE,
                                    input_dim=input_dim,
                                    hidden_dim=_hidden_dim,
                                    output_dim=output_dim,
                                    num_layers=adaptive_generator_stack_num_layers,
                                    activation=adaptive_generator_stack_activation,
                                    layer_norm_position=LayerNormPositionOptions.NONE,
                                    residual_flag=adaptive_generator_stack_residual_flag,
                                    adaptive_computation_flag=False,
                                    dropout_probability=adaptive_generator_stack_dropout_probability,
                                    override_config=LinearLayerConfig(
                                        input_dim=input_dim,
                                        output_dim=output_dim,
                                        bias_flag=False,
                                        data_monitor=None,
                                        parameter_monitor=None,
                                    ),
                                ),
                                router_model_config=RouterConfig(
                                    input_dim=input_dim,
                                    layer_stack_option=LinearLayerStackOptions.BASE,
                                    num_experts=adaptive_mixture_num_experts,
                                    noisy_topk_flag=False,
                                    override_config=LayerStackConfig(
                                        model_type=LinearLayerOptions.BASE,
                                        input_dim=input_dim,
                                        hidden_dim=max(input_dim, adaptive_mixture_num_experts),
                                        output_dim=adaptive_mixture_num_experts,
                                        num_layers=2,
                                        activation=stack_activation,
                                        layer_norm_position=LayerNormPositionOptions.NONE,
                                        residual_flag=False,
                                        adaptive_computation_flag=False,
                                        dropout_probability=0.0,
                                        override_config=LinearLayerConfig(
                                            input_dim=input_dim,
                                            output_dim=adaptive_mixture_num_experts,
                                            bias_flag=False,
                                            data_monitor=None,
                                            parameter_monitor=None,
                                        ),
                                    ),
                                ),
                                sampler_model_config=SamplerConfig(
                                    top_k=adaptive_mixture_top_k,
                                    threshold=0.0,
                                    filter_above_threshold=False,
                                    num_topk_samples=0,
                                    normalize_probabilities_flag=False,
                                    noisy_topk_flag=False,
                                    num_experts=adaptive_mixture_num_experts,
                                    coefficient_of_variation_loss_weight=0.0,
                                    switch_loss_weight=0.0,
                                    zero_centred_loss_weight=0.0,
                                    mutual_information_loss_weight=0.0,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )


if __name__ == "__main__":
    parser = get_experiment_parser(ExperimentOptions.names())
    args = parser.parse_args()
    config_option = ExperimentOptions.get_option(args.name)
    experiment = Experiment(config_option)
    experiment.train_model(num_samples=args.num_samples, log_folder=args.log_folder)
