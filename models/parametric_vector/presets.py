from typing import TYPE_CHECKING

import models.parametric_vector.config as config

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.options import (
    ActivationOptions,
    BaseOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.config import ModelConfig
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase, SearchMode
from emperor.linears.core.config import LinearLayerConfig
from emperor.parametric import (
    AdaptiveRouterOptions,
    ClipParameterOptions,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
    VectorWeightsMixtureConfig,
)
from emperor.sampler.core.config import RouterConfig, SamplerConfig
from models.parametric_vector.config import ExperimentConfig
from models.parametric_vector.model import Model

if TYPE_CHECKING:
    from emperor.config import ModelConfig as ModelConfigType


class ExperimentOptions(BaseOptions):
    PRESET = 0
    CONFIG = 1


class ExperimentPresets(ExperimentPresetsBase):
    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.PRESET,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfigType"]:
        match model_config_options:
            case ExperimentOptions.PRESET:
                return self._create_default_preset_configs(
                    dataset,
                    config_overrides=config_overrides,
                    search_overrides=search_overrides,
                )
            case ExperimentOptions.CONFIG:
                return self._create_preset_search_space_configs(
                    dataset,
                    search_mode,
                    search_keys=search_keys,
                    config_overrides=config_overrides,
                    search_overrides=search_overrides,
                )
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid "
                    "`ExperimentOptions`."
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
        sampler_threshold: float = config.SAMPLER_THRESHOLD,
        sampler_filter_above_threshold: bool = config.SAMPLER_FILTER_ABOVE_THRESHOLD,
        sampler_num_topk_samples: int = config.SAMPLER_NUM_TOPK_SAMPLES,
        sampler_normalize_probabilities_flag: bool = config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
        sampler_noisy_topk_flag: bool = config.SAMPLER_NOISY_TOPK_FLAG,
        sampler_coefficient_of_variation_loss_weight: float = config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
        sampler_switch_loss_weight: float = config.SAMPLER_SWITCH_LOSS_WEIGHT,
        sampler_zero_centred_loss_weight: float = config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
        sampler_mutual_information_loss_weight: float = config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT,
    ) -> "ModelConfigType":
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=self._linear_stack_config(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_layers=1,
                    activation=stack_activation,
                    residual_flag=False,
                    dropout_probability=0.0,
                    apply_output_pipeline_flag=True,
                ),
                model_config=self._parametric_stack_config(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_layers=stack_num_layers,
                    activation=stack_activation,
                    residual_flag=stack_residual_flag,
                    dropout_probability=stack_dropout_probability,
                    weight_mixture_config=VectorWeightsMixtureConfig(
                        input_dim=hidden_dim,
                        output_dim=hidden_dim,
                        top_k=adaptive_mixture_top_k,
                        num_experts=adaptive_mixture_num_experts,
                        weighted_parameters_flag=adaptive_mixture_weighted_parameters_flag,
                        clip_parameter_option=adaptive_mixture_clip_parameter_option,
                        clip_range=adaptive_mixture_clip_range,
                    ),
                    routing_initialization_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
                    router_config=self._router_config(
                        input_dim=hidden_dim,
                        num_experts=adaptive_mixture_num_experts,
                        activation=stack_activation,
                    ),
                    sampler_config=self._sampler_config(
                        top_k=adaptive_mixture_top_k,
                        num_experts=adaptive_mixture_num_experts,
                        threshold=sampler_threshold,
                        filter_above_threshold=sampler_filter_above_threshold,
                        num_topk_samples=sampler_num_topk_samples,
                        normalize_probabilities_flag=sampler_normalize_probabilities_flag,
                        noisy_topk_flag=sampler_noisy_topk_flag,
                        coefficient_of_variation_loss_weight=sampler_coefficient_of_variation_loss_weight,
                        switch_loss_weight=sampler_switch_loss_weight,
                        zero_centred_loss_weight=sampler_zero_centred_loss_weight,
                        mutual_information_loss_weight=sampler_mutual_information_loss_weight,
                    ),
                ),
                output_model_config=self._linear_stack_config(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=1,
                    activation=ActivationOptions.DISABLED,
                    residual_flag=False,
                    dropout_probability=0.0,
                    apply_output_pipeline_flag=False,
                ),
            ),
        )

    def _parametric_stack_config(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: ActivationOptions,
        residual_flag: bool,
        dropout_probability: float,
        weight_mixture_config: VectorWeightsMixtureConfig,
        routing_initialization_mode: AdaptiveRouterOptions,
        router_config: RouterConfig,
        sampler_config: SamplerConfig,
    ) -> LayerStackConfig:
        parametric_layer_config = ParametricLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            weight_mixture_config=weight_mixture_config,
            bias_mixture_config=None,
            routing_initialization_mode=routing_initialization_mode,
            router_config=router_config,
            sampler_config=sampler_config,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                weight_config=None,
                diagonal_config=None,
                bias_config=None,
                mask_config=None,
                model_config=None,
            ),
        )
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            layer_config=ParametricLayerHandlerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                activation=activation,
                residual_flag=residual_flag,
                dropout_probability=dropout_probability,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                shared_halting_flag=False,
                layer_model_config=parametric_layer_config,
            ),
        )

    def _linear_stack_config(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: ActivationOptions,
        residual_flag: bool,
        dropout_probability: float,
        apply_output_pipeline_flag: bool,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=LayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                activation=activation,
                residual_flag=residual_flag,
                dropout_probability=dropout_probability,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                shared_halting_flag=False,
                layer_model_config=LinearLayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=True,
                ),
            ),
        )

    def _router_config(
        self,
        input_dim: int,
        num_experts: int,
        activation: ActivationOptions,
    ) -> RouterConfig:
        return RouterConfig(
            input_dim=input_dim,
            num_experts=num_experts,
            noisy_topk_flag=False,
            model_config=self._linear_stack_config(
                input_dim=input_dim,
                hidden_dim=max(4, min(input_dim, 32)),
                output_dim=num_experts,
                num_layers=1,
                activation=activation,
                residual_flag=False,
                dropout_probability=0.0,
                apply_output_pipeline_flag=False,
            ),
        )

    def _sampler_config(
        self,
        top_k: int,
        num_experts: int,
        threshold: float,
        filter_above_threshold: bool,
        num_topk_samples: int,
        normalize_probabilities_flag: bool,
        noisy_topk_flag: bool,
        coefficient_of_variation_loss_weight: float,
        switch_loss_weight: float,
        zero_centred_loss_weight: float,
        mutual_information_loss_weight: float,
    ) -> SamplerConfig:
        return SamplerConfig(
            top_k=top_k,
            threshold=threshold,
            filter_above_threshold=filter_above_threshold,
            num_topk_samples=num_topk_samples,
            normalize_probabilities_flag=normalize_probabilities_flag,
            noisy_topk_flag=noisy_topk_flag,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=coefficient_of_variation_loss_weight,
            switch_loss_weight=switch_loss_weight,
            zero_centred_loss_weight=zero_centred_loss_weight,
            mutual_information_loss_weight=mutual_information_loss_weight,
            router_config=None,
        )


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_option: ExperimentOptions | None = None,
    ) -> None:
        super().__init__(experiment_option)

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return config.DATASET_OPTIONS

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_enumeration(self) -> type[BaseOptions]:
        return ExperimentOptions
