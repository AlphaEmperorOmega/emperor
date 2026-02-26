import torch

from torch import Tensor
from Emperor.experts.utils.layers import MixtureOfExpertsConfig
from models.parser import get_parser
from Emperor.experts.utils.model import MixtureOfExpertsModel
from Emperor.config import ModelConfig
from dataclasses import dataclass, field
from Emperor.datasets.image.mnist import Mnist
from Emperor.base.utils import ConfigBase, Module
from Emperor.datasets.image.cifar_10 import Cifar10
from Emperor.datasets.image.cifar_100 import Cifar100
from Emperor.linears.utils.presets import LinearPresets
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.base.layer import LayerStack, LayerStackConfig
from Emperor.experiments.utils.base import Experiments
from Emperor.datasets.image.fashion_mnist import FashionMNIST
from Emperor.transformer.utils.layers import TransformerConfig
from Emperor.transformer.utils.presets import TransformerPresets
from Emperor.transformer.utils.patch.selector import PatchOptions
from Emperor.transformer.utils.patch.selector import PatchSelector
from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.transformer.utils.stack import TransformerEncoderStack
from Emperor.transformer.utils.patch.options.base import PatchConfig
from Emperor.transformer.utils.feed_forward import FeedForwardConfig
from Emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from Emperor.transformer.utils.embedding.selector import PositionalEmbeddingOptions
from Emperor.transformer.utils.embedding.selector import PositionalEmbeddingSelector
from Emperor.transformer.utils.embedding.options.base import PositionalEmbeddingConfig
from Emperor.base.enums import ActivationOptions, BaseOptions, LayerNormPositionOptions
from Emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
)

@dataclass
class ExpertsExperimentConfig(ConfigBase):
    experts_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    output_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )

class ExpertsModel(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.main_cfg: ExpertsExperimentConfig = self._resolve_main_config(self.cfg, cfg)

        self.experts_config = self.main_cfg.experts_config
        self.output_config = self.main_cfg.output_config

        self.experts = MixtureOfExpertsModel(self.experts_config)
        self.output = LayerStack(self.output_config).build_model()

    def forward(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        X = input_tensor.to(self.device)
        X = self.experts(X)
        return self.output(X)

class ExpertsExperimentOptions(BaseOptions):
    BASE = 0
    ADAPTIVE = 1

class ExpertsExperiment(Experiments):
    def __init__(
        self,
        model_config_option: ExpertsExperimentOptions | None = None,
        mini_datasetset_flag: bool = False,
    ) -> None:
        self.print_frequency = 50
        self.model_config_option = model_config_option
        super().__init__(mini_datasetset_flag, self.print_frequency)

    def _get_num_epochs(self) -> int:
        return 200

    def _get_learning_rates(self) -> list:
        return [1e-4, 1e-3, 1e-2]

    def _get_dataset_options(self) -> list:
        return [Mnist, FashionMNIST, Cifar10, Cifar100]

    def _get_model_config(self):
        if self.model_config_option is None:
            return None
        return ExpertsModelPresets().get_config(self.model_config_option)

    def _get_model_type(self) -> type:
        return ExpertsModel

    def train_model(self) -> None:
        if self.model_config_option is not None:
            super().train_model()
            return None

        for config_option in ExpertsExperimentOptions:
            self.model_config_option = config_option
            config = ExpertsModelPresets().get_config(config_option)
            self._set_model_config(config)
            super().train_model()


class ExpertsModelPresets:
    def __init__(self) -> None:
        self.batch_size = 64

    def get_config(
        self,
        model_config_options: ExpertsExperimentOptions = ExpertsExperimentOptions.BASE,
    ) -> "ModelConfig":
        match model_config_options:
            case ExpertsExperimentOptions.BASE:
                return self.__base_linear_transformer_config()
            case ExpertsExperimentOptions.ADAPTIVE:
                return self.__adaptive_linear_transformer_config()
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `VITExperimentOptions`."
                )

    def __base_linear_transformer_config(self) -> "ModelConfig":
        input_dim = 32
        embedding_dim = 32
        output_dim = 10
        dropout_probability = 0.1
        activation_function = ActivationOptions.SILU


        output_bias_flag = True
        output_num_layers = 2

        return ModelConfig(
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.embedding_dim,
            output_dim=self.output_dim,
            override_config=ExpertsExperimentConfig(
                experts_config=LayerStackConfig(
                    input_dim=input_dim,
                    hidden_dim=stack_hidden_dim,
                    output_dim=output_dim,
                    num_layers=experts_stack_num_layers,
                    activation=experts_stack_activation,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=experts_stack_residual_flag,
                    adaptive_computation_flag=False,
                    dropout_probability=experts_stack_dropout_probability,
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
                        override_config=expert_model_config,
                        router_model_config=RouterConfig(
                            input_dim=input_dim,
                            layer_stack_option=layer_stack_option,
                            num_experts=num_experts,
                            noisy_topk_flag=noisy_topk_flag,
                            override_config=LayerStackConfig(
                                model_type=LinearLayerOptions.BASE,
                                input_dim=input_dim,
                                hidden_dim=stack_hidden_dim,
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
                                    data_monitor=data_monitor,
                                    parameter_monitor=parameter_monitor,
                                )
                            )
                        ),
                        sampler_model_config=SamplerConfig(
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
                        )
                    )
                ),
                output_config=LayerStackConfig(
                    model_type=LinearLayerOptions.BASE,
                    input_dim=self.embedding_dim,
                    hidden_dim=self.embedding_dim,
                    output_dim=self.output_dim,
                    num_layers=self.output_num_layers,
                    activation=self.activation_function,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=False,
                    adaptive_computation_flag=False,
                    dropout_probability=self.dropout_probability,
                    override_config=LinearLayerConfig(
                        input_dim=self.embedding_dim,
                        output_dim=self.output_dim,
                        bias_flag=self.output_bias_flag,
                        data_monitor=None,
                        parameter_monitor=None,
                    ),
                ),
            )






if __name__ == "__main__":
    parser = get_parser(VITExperimentOptions.names())
    args = parser.parse_args()
    config_option = VITExperimentOptions.get_option(args.config_name)

    experiment = VITExperiment(config_option)
    experiment.train_model()
