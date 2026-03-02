from emperor.parametric.utils.mixtures.base import AdaptiveMixtureConfig
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
from emperor.parametric.utils.layers import (
    AdaptiveParameterLayerConfig,
    AdaptiveRouterOptions,
)
from emperor.parametric.utils.mixtures.options import (
    AdaptiveBiasOptions,
    AdaptiveWeightOptions,
)
from emperor.parametric.utils.mixtures.types.utils.enums import ClipParameterOptions
from emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from emperor.behaviours.model import AdaptiveParameterBehaviourConfig
from emperor.experiments.classifier import ClassifierExperiment
from emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    create_search_space,
)
from emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
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
        router_layer_stack_option: LinearLayerStackOptions = LinearLayerStackOptions.BASE,
        router_hidden_dim: int = 0,
        router_num_layers: int = 2,
        router_activation: ActivationOptions = ActivationOptions.RELU,
        router_layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.NONE,
        router_residual_flag: bool = False,
        router_dropout_probability: float = 0.0,
        router_bias_flag: bool = False,
        router_noisy_topk_flag: bool = False,
        router_generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        router_diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        router_bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        router_memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        router_memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        router_memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        router_adaptive_generator_stack_hidden_dim: int = 0,
        router_adaptive_generator_stack_num_layers: int = 2,
        router_adaptive_generator_stack_activation: ActivationOptions = ActivationOptions.RELU,
        router_adaptive_generator_stack_residual_flag: bool = False,
        router_adaptive_generator_stack_dropout_probability: float = 0.0,
        sampler_num_experts: int = 6,
        sampler_top_k: int = 3,
        sampler_threshold: float = 0.0,
        sampler_filter_above_threshold: bool = False,
        sampler_num_topk_samples: int = 0,
        sampler_normalize_probabilities_flag: bool = False,
        sampler_noisy_topk_flag: bool = False,
        sampler_coefficient_of_variation_loss_weight: float = 0.0,
        sampler_switch_loss_weight: float = 0.0,
        sampler_zero_centred_loss_weight: float = 0.0,
        sampler_mutual_information_loss_weight: float = 0.0,
    ) -> "ModelConfig":
        from emperor.config import ModelConfig

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
                        adaptive_weight_option=AdaptiveWeightOptions.VECTOR,
                        adaptive_bias_option=AdaptiveBiasOptions.DISABLED,
                        init_sampler_model_option=AdaptiveRouterOptions.INDEPENTENT_ROUTER,
                        time_tracker_flag=False,
                        adaptive_behaviour_config=AdaptiveParameterBehaviourConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            override_config=LayerStackConfig(
                                model_type=LinearLayerOptions.BASE,
                                input_dim=input_dim,
                                hidden_dim=_hidden_dim,
                                output_dim=output_dim,
                                num_layers=2,
                                activation=stack_activation,
                                layer_norm_position=LayerNormPositionOptions.NONE,
                                residual_flag=False,
                                adaptive_computation_flag=False,
                                dropout_probability=0.0,
                                override_config=LinearLayerConfig(
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    bias_flag=False,
                                    data_monitor=None,
                                    parameter_monitor=None,
                                ),
                            ),
                        ),
                        router_config=RouterConfig(
                            input_dim=input_dim,
                            layer_stack_option=router_layer_stack_option,
                            num_experts=adaptive_mixture_num_experts,
                            noisy_topk_flag=router_noisy_topk_flag,
                            override_config=self.__build_linear_layer_stack_config(
                                layer_stack_option=router_layer_stack_option,
                                input_dim=input_dim,
                                hidden_dim=(
                                    router_hidden_dim
                                    if router_hidden_dim > 0
                                    else max(input_dim, adaptive_mixture_num_experts)
                                ),
                                output_dim=adaptive_mixture_num_experts,
                                stack_num_layers=router_num_layers,
                                stack_activation=router_activation,
                                layer_norm_position=router_layer_norm_position,
                                stack_residual_flag=router_residual_flag,
                                stack_dropout_probability=router_dropout_probability,
                                bias_flag=router_bias_flag,
                                generator_depth=router_generator_depth,
                                diagonal_option=router_diagonal_option,
                                bias_option=router_bias_option,
                                memory_option=router_memory_option,
                                memory_size_option=router_memory_size_option,
                                memory_position_option=router_memory_position_option,
                                adaptive_generator_stack_hidden_dim=router_adaptive_generator_stack_hidden_dim,
                                adaptive_generator_stack_num_layers=router_adaptive_generator_stack_num_layers,
                                adaptive_generator_stack_activation=router_adaptive_generator_stack_activation,
                                adaptive_generator_stack_residual_flag=router_adaptive_generator_stack_residual_flag,
                                adaptive_generator_stack_dropout_probability=router_adaptive_generator_stack_dropout_probability,
                            ),
                        ),
                        sampler_config=SamplerConfig(
                            top_k=sampler_top_k,
                            threshold=sampler_threshold,
                            filter_above_threshold=sampler_filter_above_threshold,
                            num_topk_samples=sampler_num_topk_samples,
                            normalize_probabilities_flag=sampler_normalize_probabilities_flag,
                            noisy_topk_flag=sampler_noisy_topk_flag,
                            num_experts=sampler_num_experts,
                            coefficient_of_variation_loss_weight=sampler_coefficient_of_variation_loss_weight,
                            switch_loss_weight=sampler_switch_loss_weight,
                            zero_centred_loss_weight=sampler_zero_centred_loss_weight,
                            mutual_information_loss_weight=sampler_mutual_information_loss_weight,
                        ),
                        override_config=AdaptiveMixtureConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            top_k=adaptive_mixture_top_k,
                            num_experts=adaptive_mixture_num_experts,
                            weighted_parameters_flag=adaptive_mixture_weighted_parameters_flag,
                            clip_parameter_option=adaptive_mixture_clip_parameter_option,
                            clip_range=adaptive_mixture_clip_range,
                        ),
                    ),
                ),
            ),
        )

    def __build_linear_layer_stack_config(
        self,
        layer_stack_option: LinearLayerStackOptions,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        stack_num_layers: int,
        stack_activation: ActivationOptions,
        layer_norm_position: LayerNormPositionOptions,
        stack_residual_flag: bool,
        stack_dropout_probability: float,
        bias_flag: bool,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        adaptive_generator_stack_hidden_dim: int = 0,
        adaptive_generator_stack_num_layers: int = 2,
        adaptive_generator_stack_activation: ActivationOptions = ActivationOptions.RELU,
        adaptive_generator_stack_residual_flag: bool = False,
        adaptive_generator_stack_dropout_probability: float = 0.0,
    ) -> "LayerStackConfig":

        if layer_stack_option == LinearLayerStackOptions.ADAPTIVE:
            return LayerStackConfig(
                model_type=LinearLayerOptions.ADAPTIVE,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=stack_num_layers,
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_flag=stack_residual_flag,
                adaptive_computation_flag=False,
                dropout_probability=stack_dropout_probability,
                override_config=LinearLayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=bias_flag,
                    data_monitor=None,
                    parameter_monitor=None,
                    override_config=AdaptiveParameterBehaviourConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        generator_depth=generator_depth,
                        diagonal_option=diagonal_option,
                        bias_option=bias_option,
                        memory_option=memory_option,
                        memory_size_option=memory_size_option,
                        memory_position_option=memory_position_option,
                        override_config=LayerStackConfig(
                            model_type=LinearLayerOptions.BASE,
                            input_dim=input_dim,
                            hidden_dim=adaptive_generator_stack_hidden_dim,
                            output_dim=output_dim,
                            num_layers=adaptive_generator_stack_num_layers,
                            activation=adaptive_generator_stack_activation,
                            layer_norm_position=layer_norm_position,
                            residual_flag=adaptive_generator_stack_residual_flag,
                            adaptive_computation_flag=False,
                            dropout_probability=adaptive_generator_stack_dropout_probability,
                            override_config=LinearLayerConfig(
                                input_dim=input_dim,
                                output_dim=output_dim,
                                bias_flag=bias_flag,
                                data_monitor=None,
                                parameter_monitor=None,
                                override_config=AdaptiveParameterBehaviourConfig(
                                    generator_depth=generator_depth,
                                ),
                            ),
                        ),
                    ),
                ),
            )

        return LayerStackConfig(
            model_type=LinearLayerOptions.BASE,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            layer_norm_position=layer_norm_position,
            residual_flag=stack_residual_flag,
            adaptive_computation_flag=False,
            dropout_probability=stack_dropout_probability,
            override_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
            ),
        )


if __name__ == "__main__":
    parser = get_experiment_parser(ExperimentOptions.names())
    args = parser.parse_args()
    config_option = ExperimentOptions.get_option(args.name)
    experiment = Experiment(config_option)
    experiment.train_model(num_samples=args.num_samples, log_folder=args.log_folder)
