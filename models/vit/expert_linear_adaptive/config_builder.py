from emperor.experts.config import MixtureOfExpertsModelConfig

import models.experts.linear_adaptive.config as adaptive_expert_defaults
import models.vit.expert_linear_adaptive.config as config
from models.experts._builder_adapter import linear_adaptive_builder_kwargs_from_flat
from models.experts._builder_options import ExpertsStackOptions
from models.experts.linear_adaptive._control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.vit.expert_linear.config_builder import VitExpertLinearConfigBuilder
from models.vit.expert_linear_adaptive.experiment_config import ExperimentConfig


class VitExpertLinearAdaptiveConfigBuilder(VitExpertLinearConfigBuilder):
    def __init__(
        self,
        *args,
        adaptive_generator_stack_options=None,
        hidden_adaptive_weight_options=None,
        hidden_adaptive_bias_options=None,
        hidden_adaptive_diagonal_options=None,
        hidden_adaptive_mask_options=None,
        router_layer_controller_options=None,
        router_dynamic_memory_options=None,
        router_recurrent_controller_options=None,
        router_adaptive_weight_options=None,
        router_adaptive_bias_options=None,
        router_adaptive_diagonal_options=None,
        router_adaptive_mask_options=None,
        submodule_stack_options=None,
        layer_controller_options=None,
        dynamic_memory_options=None,
        recurrent_controller_options=None,
        **kwargs,
    ) -> None:
        adaptive_defaults_kwargs = linear_adaptive_builder_kwargs_from_flat(
            {},
            adaptive_expert_defaults,
        )
        self.adaptive_generator_stack_options = (
            adaptive_generator_stack_options
            or adaptive_defaults_kwargs["adaptive_generator_stack_options"]
        )
        self.hidden_adaptive_weight_options = (
            hidden_adaptive_weight_options
            or adaptive_defaults_kwargs["hidden_adaptive_weight_options"]
        )
        self.hidden_adaptive_bias_options = (
            hidden_adaptive_bias_options
            or adaptive_defaults_kwargs["hidden_adaptive_bias_options"]
        )
        self.hidden_adaptive_diagonal_options = (
            hidden_adaptive_diagonal_options
            or adaptive_defaults_kwargs["hidden_adaptive_diagonal_options"]
        )
        self.hidden_adaptive_mask_options = (
            hidden_adaptive_mask_options
            or adaptive_defaults_kwargs["hidden_adaptive_mask_options"]
        )
        self.router_layer_controller_options = (
            router_layer_controller_options
            or adaptive_defaults_kwargs["router_layer_controller_options"]
        )
        self.router_dynamic_memory_options = (
            router_dynamic_memory_options
            or adaptive_defaults_kwargs["router_dynamic_memory_options"]
        )
        self.router_recurrent_controller_options = (
            router_recurrent_controller_options
            or adaptive_defaults_kwargs["router_recurrent_controller_options"]
        )
        self.router_adaptive_weight_options = (
            router_adaptive_weight_options
            or adaptive_defaults_kwargs["router_adaptive_weight_options"]
        )
        self.router_adaptive_bias_options = (
            router_adaptive_bias_options
            or adaptive_defaults_kwargs["router_adaptive_bias_options"]
        )
        self.router_adaptive_diagonal_options = (
            router_adaptive_diagonal_options
            or adaptive_defaults_kwargs["router_adaptive_diagonal_options"]
        )
        self.router_adaptive_mask_options = (
            router_adaptive_mask_options
            or adaptive_defaults_kwargs["router_adaptive_mask_options"]
        )
        self.submodule_stack_options = (
            submodule_stack_options
            or adaptive_defaults_kwargs["submodule_stack_options"]
        )
        self.layer_controller_options = (
            layer_controller_options
            or adaptive_defaults_kwargs["layer_controller_options"]
        )
        self.dynamic_memory_options = (
            dynamic_memory_options
            or adaptive_defaults_kwargs["dynamic_memory_options"]
        )
        self.recurrent_controller_options = (
            recurrent_controller_options
            or adaptive_defaults_kwargs["recurrent_controller_options"]
        )
        super().__init__(*args, **kwargs)
        self.experiment_config_type = ExperimentConfig

    def _build_expert_model_config(self) -> MixtureOfExpertsModelConfig:
        stack_options = ExpertsStackOptions(
            hidden_dim=self.hidden_dim,
            bias_flag=self.feed_forward_options.bias_flag,
            layer_norm_position=self.encoder_options.layer_norm_position,
            num_layers=self.feed_forward_options.num_layers,
            activation=self.encoder_options.activation,
            residual_connection_option=config.ResidualConnectionOptions.DISABLED,
            dropout_probability=self.encoder_options.dropout_probability,
            last_layer_bias_option=config.LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
        )
        factory = ControlConfigFactory(
            ControlConfigDependencies(
                stack_options=stack_options,
                submodule_stack_options=self.submodule_stack_options,
                mixture_options=self.mixture_options,
                expert_stack_options=self.expert_stack_options,
                sampler_options=self.sampler_options,
                router_options=self.router_options,
                router_stack_options=self.router_stack_options,
                router_layer_controller_options=self.router_layer_controller_options,
                router_dynamic_memory_options=self.router_dynamic_memory_options,
                router_recurrent_controller_options=(
                    self.router_recurrent_controller_options
                ),
                layer_controller_options=self.layer_controller_options,
                dynamic_memory_options=self.dynamic_memory_options,
                recurrent_controller_options=self.recurrent_controller_options,
                expert_layer_controller_options=self.expert_layer_controller_options,
                expert_dynamic_memory_options=self.expert_dynamic_memory_options,
                expert_recurrent_controller_options=(
                    self.expert_recurrent_controller_options
                ),
                adaptive_generator_stack_options=(
                    self.adaptive_generator_stack_options
                ),
                hidden_adaptive_weight_options=self.hidden_adaptive_weight_options,
                hidden_adaptive_bias_options=self.hidden_adaptive_bias_options,
                hidden_adaptive_diagonal_options=(
                    self.hidden_adaptive_diagonal_options
                ),
                hidden_adaptive_mask_options=self.hidden_adaptive_mask_options,
                router_adaptive_weight_options=self.router_adaptive_weight_options,
                router_adaptive_bias_options=self.router_adaptive_bias_options,
                router_adaptive_diagonal_options=(
                    self.router_adaptive_diagonal_options
                ),
                router_adaptive_mask_options=self.router_adaptive_mask_options,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
            )
        )
        model_config = factory.build()
        if isinstance(model_config, MixtureOfExpertsModelConfig):
            return model_config
        return model_config.block_config
