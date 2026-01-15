from Emperor.base.enums import ActivationOptions
from Emperor.experiments.utils.factories import Experiments
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.experts.utils.presets import MixtureOfExpertsPresets
from Emperor.experts.utils.enums import ExpertWeightingPositionOptions, LayerRoleOptions
from Emperor.experts.options import (
    MixtureOfExpertsOptions,
    MixtureOfExpertsStackOptions,
)
from Emperor.behaviours.utils.enums import (
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    DynamicBiasOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class MixtureOfExpertsExperiments(Experiments):
    def __init__(
        self,
        mini_datasetset_flag: bool = True,
    ) -> None:
        super().__init__(mini_datasetset_flag)

    def train_model(self, layer_type: MixtureOfExpertsOptions):
        preset = MixtureOfExpertsPreset(layer_type).get_config()
        self._set_model_config(preset)
        self._train_model(layer_type)

    def test_all_types(self):
        option_types = [MixtureOfExpertsStackOptions]

        for option_type in option_types:
            for option in option_type:
                print(option)
                self.train_model(option)


class MixtureOfExpertsPreset:
    def __init__(
        self,
        model_option: MixtureOfExpertsOptions | MixtureOfExpertsStackOptions,
    ) -> None:
        self.model_option = model_option

    def get_config(self) -> "ModelConfig":
        match self.model_option:
            case MixtureOfExpertsOptions.BASE:
                return MixtureOfExpertsPresets.experts_preset(
                    return_model_config_flag=True,
                    batch_size=64,
                    input_dim=784,
                    output_dim=10,
                    bias_flag=True,
                    top_k=3,
                    num_experts=8,
                    compute_expert_mixture_flag=True,
                    weighted_parameters_flag=True,
                    weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
                    init_sampler_model_flag=True,
                    layer_role_option=LayerRoleOptions.GENERAL,
                    generator_depth=DynamicDepthOptions.DISABLED,
                    diagonal_option=DynamicDiagonalOptions.DISABLED,
                    bias_option=DynamicBiasOptions.DISABLED,
                    memory_option=LinearMemoryOptions.DISABLED,
                    memory_size_option=LinearMemorySizeOptions.DISABLED,
                    memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
                    noisy_topk_flag=False,
                    threshold=0.0,
                    filter_above_threshold=False,
                    num_topk_samples=0,
                    normalize_probabilities_flag=False,
                    switch_loss_weight=0.1,
                    zero_centred_loss_weight=0.1,
                    mutual_information_loss_weight=0.0,
                    coefficient_of_variation_loss_weight=0.1,
                    layer_stack_option=LinearLayerStackOptions.BASE,
                    router_layer_stack_option=LinearLayerStackOptions.BASE,
                    stack_num_layers=2,
                    stack_hidden_dim=0,
                    stack_activation=ActivationOptions.RELU,
                    stack_residual_flag=False,
                    stack_dropout_probability=0.0,
                )
            case MixtureOfExpertsStackOptions.BASE:
                return MixtureOfExpertsPresets.experts_stack_preset(
                    return_model_config_flag=True,
                    batch_size=64,
                    input_dim=784,
                    output_dim=10,
                    bias_flag=True,
                    num_experts=8,
                    compute_expert_mixture_flag=True,
                    weighted_parameters_flag=True,
                    weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
                    init_sampler_model_flag=True,
                    layer_role_option=LayerRoleOptions.GENERAL,
                    generator_depth=DynamicDepthOptions.DISABLED,
                    diagonal_option=DynamicDiagonalOptions.DISABLED,
                    bias_option=DynamicBiasOptions.DISABLED,
                    memory_option=LinearMemoryOptions.DISABLED,
                    memory_size_option=LinearMemorySizeOptions.DISABLED,
                    memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
                    noisy_topk_flag=False,
                    threshold=0.0,
                    filter_above_threshold=False,
                    num_topk_samples=0,
                    normalize_probabilities_flag=False,
                    switch_loss_weight=0.1,
                    zero_centred_loss_weight=0.1,
                    mutual_information_loss_weight=0.0,
                    coefficient_of_variation_loss_weight=0.1,
                    layer_stack_option=LinearLayerStackOptions.BASE,
                    router_layer_stack_option=LinearLayerStackOptions.BASE,
                    stack_num_layers=2,
                    stack_hidden_dim=0,
                    stack_activation=ActivationOptions.RELU,
                    stack_residual_flag=False,
                    stack_dropout_probability=0.0,
                )
