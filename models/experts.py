from torch import Tensor
from models.parser import get_experiment_parser
from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.base.enums import BaseOptions
from emperor.datasets.image.mnist import Mnist
from emperor.experts.utils.model import MixtureOfExpertsModel
from emperor.experts.utils.layers import MixtureOfExpertsConfig
from emperor.base.layer import LayerStack, LayerStackConfig
from emperor.linears.utils.layers import LinearLayerConfig
from emperor.sampler.utils.routers import RouterConfig
from emperor.sampler.utils.samplers import SamplerConfig
from emperor.experiments.classifier import ClassifierExperiment
from emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from emperor.experts.utils.enums import (
    ExpertWeightingPositionOptions,
    InitSamplerOptions,
)
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
    experts_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    output_config: "LayerStackConfig | None" = field(
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

        self.experts_config = self.main_cfg.experts_config
        self.output_config = self.main_cfg.output_config

        self.experts = MixtureOfExpertsModel(self.experts_config)
        self.output = LayerStack(self.output_config).build_model()

    def _resolve_main_config(
        self, sub_config: "ConfigBase", main_cfg: "ConfigBase"
    ) -> "ExperimentConfig":
        if sub_config.override_config is not None:
            return sub_config.override_config
        return main_cfg

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        X = self.experts(X)
        return self.output(X)


class ExperimentOptions(BaseOptions):
    DEFAULT = 0
    BASE = 1
    ADAPTIVE = 2


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_option: ExperimentOptions | None = None,
    ) -> None:
        super().__init__(experiment_option)

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_enumeration(self) -> type[BaseOptions]:
        return ExperimentOptions


class ExperimentPresets(ExperimentPresetsBase):
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
            case ExperimentOptions.ADAPTIVE:
                return self.__adaptive_grid_search_config(dataset, num_samples)
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `ExperimentOptions`."
                )

    def _default_config(
        self,
        dataset: type = Mnist,
    ) -> list["ModelConfig"]:
        return [self._preset(**self._dataset_config(dataset))]

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

    def __adaptive_grid_search_config(
        self,
        dataset: type = Mnist,
        num_random_search_samples: int | None = None,
    ) -> list["ModelConfig"]:
        base_config = self._dataset_config(dataset)

        search_space = {
            **self.__base_search_space(),
            "experts_model_generator_depth": [
                DynamicDepthOptions.DEPTH_OF_ONE,
                DynamicDepthOptions.DEPTH_OF_TWO,
                DynamicDepthOptions.DEPTH_OF_THREE,
            ],
            "experts_model_diagonal_option": [
                DynamicDiagonalOptions.DISABLED,
                DynamicDiagonalOptions.DIAGONAL,
                DynamicDiagonalOptions.ANTI_DIAGONAL,
                DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL,
            ],
            "experts_model_bias_option": [
                DynamicBiasOptions.DISABLED,
                DynamicBiasOptions.SCALE_AND_OFFSET,
                DynamicBiasOptions.ELEMENT_WISE_OFFSET,
                DynamicBiasOptions.DYNAMIC_PARAMETERS,
            ],
        }

        return create_search_space(
            self._preset, base_config, search_space, num_random_search_samples
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
            "experts_top_k": [1, 2, 3],
            "experts_num_experts": [4, 6, 8],
        }

    def _preset(
        self,
        batch_size: int = 64,
        input_dim: int = 28**2,
        hidden_dim: int = 256,
        output_dim: int = 10,
        learning_rate: float = 1e-3,
        bias_flag: bool = True,
        output_num_layers: int = 2,
        output_activation: ActivationOptions = ActivationOptions.SILU,
        output_dropout_probability: float = 0.1,
        router_noisy_topk_flag: bool = False,
        sampler_threshold: float = 0.0,
        sampler_filter_above_threshold: bool = False,
        sampler_num_topk_samples: int = 0,
        sampler_normalize_probabilities_flag: bool = False,
        sampler_coefficient_of_variation_loss_weight: float = 0.0,
        sampler_switch_loss_weight: float = 0.0,
        sampler_zero_centred_loss_weight: float = 0.0,
        sampler_mutual_information_loss_weight: float = 0.0,
        experts_top_k: int = 3,
        experts_num_experts: int = 6,
        experts_compute_expert_mixture_flag: bool = False,
        experts_weighted_parameters_flag: bool = False,
        experts_weighting_position_option: ExpertWeightingPositionOptions = ExpertWeightingPositionOptions.BEFORE_EXPERTS,
        experts_init_sampler_option: InitSamplerOptions = InitSamplerOptions.DISABLED,
        experts_model_generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        experts_model_diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        experts_model_bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        experts_model_memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        experts_model_memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        experts_model_memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        stack_num_layers: int = 3,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
    ) -> "ModelConfig":
        from emperor.config import ModelConfig
        from emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
        from emperor.behaviours.model import AdaptiveParameterBehaviourConfig

        experts_layer_stack_option = LinearLayerStackOptions.BASE
        if experts_model_generator_depth != DynamicDepthOptions.DISABLED:
            experts_layer_stack_option = LinearLayerStackOptions.ADAPTIVE

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            override_config=ExperimentConfig(
                experts_config=LayerStackConfig(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=stack_num_layers,
                    activation=stack_activation,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=stack_residual_flag,
                    adaptive_computation_flag=False,
                    dropout_probability=stack_dropout_probability,
                    override_config=MixtureOfExpertsConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        top_k=experts_top_k,
                        num_experts=experts_num_experts,
                        layer_stack_option=experts_layer_stack_option,
                        compute_expert_mixture_flag=experts_compute_expert_mixture_flag,
                        weighted_parameters_flag=experts_weighted_parameters_flag,
                        weighting_position_option=experts_weighting_position_option,
                        init_sampler_option=experts_init_sampler_option,
                        override_config=LayerStackConfig(
                            model_type=LinearLayerOptions.BASE,
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            num_layers=stack_num_layers,
                            activation=stack_activation,
                            layer_norm_position=LayerNormPositionOptions.NONE,
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
                                    generator_depth=experts_model_generator_depth,
                                    diagonal_option=experts_model_diagonal_option,
                                    bias_option=experts_model_bias_option,
                                    memory_option=experts_model_memory_option,
                                    memory_size_option=experts_model_memory_size_option,
                                    memory_position_option=experts_model_memory_position_option,
                                    override_config=LayerStackConfig(
                                        model_type=LinearLayerOptions.BASE,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=output_dim,
                                        num_layers=stack_num_layers,
                                        activation=stack_activation,
                                        layer_norm_position=LayerNormPositionOptions.NONE,
                                        residual_flag=stack_residual_flag,
                                        adaptive_computation_flag=False,
                                        dropout_probability=stack_dropout_probability,
                                        override_config=LinearLayerConfig(
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            bias_flag=bias_flag,
                                            data_monitor=None,
                                            parameter_monitor=None,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        router_model_config=RouterConfig(
                            input_dim=input_dim,
                            layer_stack_option=LinearLayerStackOptions.BASE,
                            num_experts=experts_num_experts,
                            noisy_topk_flag=router_noisy_topk_flag,
                            override_config=LayerStackConfig(
                                model_type=LinearLayerOptions.BASE,
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                output_dim=experts_num_experts,
                                num_layers=stack_num_layers,
                                activation=stack_activation,
                                layer_norm_position=LayerNormPositionOptions.NONE,
                                residual_flag=stack_residual_flag,
                                adaptive_computation_flag=False,
                                dropout_probability=stack_dropout_probability,
                                override_config=LinearLayerConfig(
                                    input_dim=input_dim,
                                    output_dim=experts_num_experts,
                                    bias_flag=bias_flag,
                                    data_monitor=None,
                                    parameter_monitor=None,
                                ),
                            ),
                        ),
                        sampler_model_config=SamplerConfig(
                            top_k=experts_top_k,
                            threshold=sampler_threshold,
                            filter_above_threshold=sampler_filter_above_threshold,
                            num_topk_samples=sampler_num_topk_samples,
                            normalize_probabilities_flag=sampler_normalize_probabilities_flag,
                            noisy_topk_flag=router_noisy_topk_flag,
                            num_experts=experts_num_experts,
                            coefficient_of_variation_loss_weight=sampler_coefficient_of_variation_loss_weight,
                            switch_loss_weight=sampler_switch_loss_weight,
                            zero_centred_loss_weight=sampler_zero_centred_loss_weight,
                            mutual_information_loss_weight=sampler_mutual_information_loss_weight,
                        ),
                    ),
                ),
                output_config=LayerStackConfig(
                    model_type=LinearLayerOptions.BASE,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=output_num_layers,
                    activation=output_activation,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=False,
                    adaptive_computation_flag=False,
                    dropout_probability=output_dropout_probability,
                    override_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                        data_monitor=None,
                        parameter_monitor=None,
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
