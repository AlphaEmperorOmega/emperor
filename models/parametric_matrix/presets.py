import models.parametric_matrix.config as config

from emperor.datasets.image.mnist import Mnist
from emperor.base.layer import LayerStackConfig
from emperor.linears.utils.config import LinearLayerConfig
from emperor.sampler.utils.routers import RouterConfig
from emperor.sampler.utils.samplers import SamplerConfig
from emperor.parametric.options import AdaptiveLayerOptions
from emperor.parametric.utils.mixtures.base import AdaptiveMixtureConfig
from emperor.base.enums import BaseOptions, ActivationOptions, LayerNormPositionOptions
from emperor.parametric.utils.mixtures.types.utils.enums import ClipParameterOptions
from emperor.parametric.utils.layers import (
    AdaptiveParameterLayerConfig,
    AdaptiveRouterOptions,
)
from emperor.parametric.utils.mixtures.options import (
    AdaptiveBiasOptions,
    AdaptiveWeightOptions,
)
from emperor.behaviours.model import AdaptiveParameterBehaviourConfig
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase
from models.parametric_matrix.config import ExperimentConfig
from models.parametric_matrix.model import Model
from emperor.experiments.base import SearchMode

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    PRESET = 0
    CONFIG = 1


class ExperimentPresets(ExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.PRESET,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.PRESET:
                return self._create_default_preset_configs(dataset)
            case ExperimentOptions.CONFIG:
                return self._create_default_search_space_configs(dataset, search_mode)
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `ExperimentOptions`."
                )

    def _preset(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_flag: bool = config.STACK_RESIDUAL_FLAG,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        adaptive_mixture_top_k: int = config.ADAPTIVE_MIXTURE_TOP_K,
        adaptive_mixture_num_experts: int = config.ADAPTIVE_MIXTURE_NUM_EXPERTS,
        adaptive_mixture_weighted_parameters_flag: bool = config.ADAPTIVE_MIXTURE_WEIGHTED_PARAMETERS_FLAG,
        adaptive_mixture_clip_parameter_option: ClipParameterOptions = config.ADAPTIVE_MIXTURE_CLIP_PARAMETER_OPTION,
        adaptive_mixture_clip_range: float = config.ADAPTIVE_MIXTURE_CLIP_RANGE,
        adaptive_bias_option: AdaptiveBiasOptions = config.ADAPTIVE_BIAS_OPTION,
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
                        adaptive_weight_option=AdaptiveWeightOptions.MATRIX,
                        adaptive_bias_option=adaptive_bias_option,
                        init_sampler_model_option=AdaptiveRouterOptions.SHARED_ROUTER,
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
                        ),
                    ),
                ),
            ),
        )


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
