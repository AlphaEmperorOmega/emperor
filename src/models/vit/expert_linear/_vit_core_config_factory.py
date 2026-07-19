from dataclasses import dataclass
from typing import Protocol

import torch

from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.experts import MixtureOfExpertsModelConfig
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    RecurrentLayerConfig,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.transformer import (
    FeedForwardConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from models.vit.expert_linear._control_factory_dependencies import (
    VitControlFactoryDependencies,
)
from models.vit.expert_linear._encoder_control_config_factory import (
    VitGateConfigFactory,
    VitHaltingConfigFactory,
    VitMemoryConfigFactory,
    VitRecurrentConfigFactory,
)
from models.vit.expert_linear._linear_layer_config_factory import (
    LinearLayerConfigFactory as VitLinearLayerConfigFactory,
)
from models.vit.expert_linear.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
)


class _VitExpertConfigFactory(Protocol):
    def build_attention_config(
        self,
        *,
        batch_size: int,
        hidden_dim: int,
        sequence_length: int,
        projection_model_config: LayerStackConfig | RecurrentLayerConfig,
    ) -> MixtureOfAttentionHeadsConfig | None: ...

    def build_feed_forward_base_stack_config(
        self,
        feed_forward_stack_options: SubmoduleStackOptions,
    ) -> MixtureOfExpertsModelConfig: ...


@dataclass(frozen=True)
class CoreConfigDependencies:
    batch_size: int
    sequence_length: int
    encoder_options: TransformerEncoderOptions
    attention_options: TransformerAttentionOptions
    feed_forward_options: TransformerFeedForwardOptions
    attention_projection_stack_options: SubmoduleStackOptions
    attention_projection_layer_controller_options: LayerControllerOptions | None
    attention_projection_dynamic_memory_options: DynamicMemoryOptions | None
    attention_projection_recurrent_controller_options: RecurrentControllerOptions | None
    feed_forward_stack_options: SubmoduleStackOptions
    feed_forward_layer_controller_options: LayerControllerOptions | None
    feed_forward_dynamic_memory_options: DynamicMemoryOptions | None
    feed_forward_recurrent_controller_options: RecurrentControllerOptions | None
    stack_options: MainLayerStackOptions | None
    submodule_stack_options: SubmoduleStackOptions | None
    layer_controller_options: LayerControllerOptions | None
    dynamic_memory_options: DynamicMemoryOptions | None
    recurrent_controller_options: RecurrentControllerOptions | None
    linear_layer_config_factory: VitLinearLayerConfigFactory
    attention_projection_linear_layer_config_factory: (
        VitLinearLayerConfigFactory | None
    ) = None
    feed_forward_linear_layer_config_factory: VitLinearLayerConfigFactory | None = None
    expert_config_factory: _VitExpertConfigFactory | None = None


class VitCoreConfigFactory:
    def __init__(self, dependencies: CoreConfigDependencies) -> None:
        self.batch_size = dependencies.batch_size
        self.sequence_length = dependencies.sequence_length
        self.encoder_options = dependencies.encoder_options
        self.hidden_dim = dependencies.encoder_options.hidden_dim
        self.attention_options = dependencies.attention_options
        self.feed_forward_options = dependencies.feed_forward_options
        self.attention_projection_stack_options = (
            dependencies.attention_projection_stack_options
        )
        self.attention_projection_layer_controller_options = (
            dependencies.attention_projection_layer_controller_options
        )
        self.attention_projection_dynamic_memory_options = (
            dependencies.attention_projection_dynamic_memory_options
        )
        self.attention_projection_recurrent_controller_options = (
            dependencies.attention_projection_recurrent_controller_options
        )
        self.feed_forward_stack_options = dependencies.feed_forward_stack_options
        self.feed_forward_layer_controller_options = (
            dependencies.feed_forward_layer_controller_options
        )
        self.feed_forward_dynamic_memory_options = (
            dependencies.feed_forward_dynamic_memory_options
        )
        self.feed_forward_recurrent_controller_options = (
            dependencies.feed_forward_recurrent_controller_options
        )
        self.encoder_stack_options = dependencies.stack_options
        self.encoder_submodule_stack_options = dependencies.submodule_stack_options
        self.encoder_layer_controller_options = dependencies.layer_controller_options
        self.encoder_dynamic_memory_options = dependencies.dynamic_memory_options
        self.encoder_recurrent_controller_options = (
            dependencies.recurrent_controller_options
        )
        self.linear_layer_config_factory = dependencies.linear_layer_config_factory
        self.attention_projection_linear_layer_config_factory = (
            dependencies.attention_projection_linear_layer_config_factory
            or self.linear_layer_config_factory
        )
        self.feed_forward_linear_layer_config_factory = (
            dependencies.feed_forward_linear_layer_config_factory
            or self.linear_layer_config_factory
        )
        self.expert_config_factory = dependencies.expert_config_factory

    def build_encoder_config(self):
        return self._build_encoder_config()

    def _build_encoder_config(self):
        options = self.encoder_options
        layer_model_config = self._build_encoder_layer_config()
        dependencies = self._control_factory_dependencies()
        gate_factory = VitGateConfigFactory(dependencies).build_encoder_factory()
        halting_factory = VitHaltingConfigFactory(dependencies).build_encoder_factory()
        memory_factory = VitMemoryConfigFactory(dependencies).build_encoder_factory()
        halting_config = (
            halting_factory.build_halting_config()
            if halting_factory is not None
            else None
        )
        stack_residual_connection_option = self._stack_residual_connection_option()
        residual_config = (
            None
            if stack_residual_connection_option is None
            else ResidualConfig(option=stack_residual_connection_option)
        )
        layer_config = TransformerEncoderBlockLayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=residual_config,
            dropout_probability=0.0,
            gate_config=(
                gate_factory.build_gate_config() if gate_factory is not None else None
            ),
            halting_config=None,
            layer_model_config=layer_model_config,
        )
        stack_config = LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=self._stack_last_layer_bias_option(),
            apply_output_pipeline_flag=self._stack_apply_output_pipeline_flag(),
            shared_gate_config=self._shared_gate_config(),
            shared_halting_config=halting_config,
            shared_memory_config=(
                memory_factory.build_memory_config()
                if memory_factory is not None
                else None
            ),
            layer_config=layer_config,
        )
        recurrent_factory = VitRecurrentConfigFactory(
            dependencies
        ).build_encoder_factory()
        if recurrent_factory is None:
            return stack_config
        return recurrent_factory.build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    def _build_encoder_layer_config(self) -> TransformerEncoderLayerConfig:
        options = self.encoder_options
        attention_config = self._build_attention_config()
        feed_forward_config = self._build_feed_forward_config()
        return TransformerEncoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=options.layer_norm_position,
            dropout_probability=options.dropout_probability,
            residual_config=ResidualConfig(option=ResidualConnectionOptions.RESIDUAL),
            causal_attention_mask_flag=False,
            attention_config=attention_config,
            feed_forward_config=feed_forward_config,
        )

    def _build_attention_config(
        self,
    ) -> SelfAttentionConfig | MixtureOfAttentionHeadsConfig:
        encoder_options = self.encoder_options
        attention_options = self.attention_options
        projection_model_config = self._build_attention_projection_stack_config()
        if self.expert_config_factory is not None:
            attention_config = self.expert_config_factory.build_attention_config(
                batch_size=self.batch_size,
                hidden_dim=self.hidden_dim,
                sequence_length=self.sequence_length,
                projection_model_config=projection_model_config,
            )
            if attention_config is not None:
                return attention_config
        return SelfAttentionConfig(
            batch_size=self.batch_size,
            num_heads=attention_options.num_heads,
            embedding_dim=self.hidden_dim,
            query_key_projection_dim=self.hidden_dim,
            value_projection_dim=self.hidden_dim,
            target_sequence_length=self.sequence_length,
            source_sequence_length=self.sequence_length,
            target_dtype=torch.float32,
            dropout_probability=encoder_options.dropout_probability,
            zero_attention_flag=False,
            causal_attention_mask_flag=False,
            add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
            average_attention_weights_flag=False,
            return_attention_weights_flag=False,
            projection_model_config=projection_model_config,
            projection_strategy=SelfAttentionProjectionStrategy.SEPARATE,
        )

    def _build_feed_forward_config(self) -> FeedForwardConfig:
        stack_config = self._build_feed_forward_stack_config()
        return FeedForwardConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            stack_config=stack_config,
        )

    def _build_attention_projection_stack_config(self):
        stack_config = self._build_attention_projection_base_stack_config()
        self._apply_attention_projection_controls(stack_config)
        recurrent_factory = VitRecurrentConfigFactory(
            self._control_factory_dependencies()
        ).build_attention_projection_factory()
        if recurrent_factory is None:
            return stack_config
        return recurrent_factory.build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    def _build_attention_projection_base_stack_config(self) -> LayerStackConfig:
        options = self._effective_attention_projection_stack_options()
        factory = self.attention_projection_linear_layer_config_factory
        return factory.build_linear_stack_config(
            layer_model_config=(
                factory.build_backend_linear_layer_config(
                    bias_flag=options.bias_flag,
                )
            ),
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            activation=options.activation,
            residual_connection_option=options.residual_connection_option,
            layer_norm_position=options.layer_norm_position,
            dropout_probability=options.dropout_probability,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_pipeline_flag=options.apply_output_pipeline_flag,
        )

    def _apply_attention_projection_controls(
        self,
        stack_config: LayerStackConfig,
    ) -> None:
        dependencies = self._control_factory_dependencies()
        gate_factory = VitGateConfigFactory(
            dependencies
        ).build_attention_projection_factory()
        halting_factory = VitHaltingConfigFactory(
            dependencies
        ).build_attention_projection_factory()
        memory_factory = VitMemoryConfigFactory(
            dependencies
        ).build_attention_projection_factory()
        controller_options = self.attention_projection_layer_controller_options
        if (
            gate_factory is not None
            and controller_options is not None
            and controller_options.stack_gate_flag
        ):
            stack_config.layer_config.gate_config = gate_factory.build_gate_config()
        if (
            halting_factory is not None
            and controller_options is not None
            and controller_options.stack_halting_flag
        ):
            stack_config.layer_config.halting_config = (
                halting_factory.build_halting_config()
            )
        if (
            controller_options is not None
            and controller_options.shared_gate_config is not None
        ):
            stack_config.shared_gate_config = controller_options.shared_gate_config
        if (
            memory_factory is not None
            and self.attention_projection_dynamic_memory_options is not None
            and self.attention_projection_dynamic_memory_options.memory_flag
        ):
            stack_config.shared_memory_config = memory_factory.build_memory_config()

    def _build_feed_forward_stack_config(self):
        stack_config = self._build_feed_forward_base_stack_config()
        self._apply_feed_forward_controls(stack_config)
        recurrent_factory = VitRecurrentConfigFactory(
            self._control_factory_dependencies()
        ).build_feed_forward_factory()
        if recurrent_factory is None:
            return stack_config
        return recurrent_factory.build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    def _build_feed_forward_base_stack_config(self):
        options = self._effective_feed_forward_stack_options()
        if self.expert_config_factory is not None:
            return self.expert_config_factory.build_feed_forward_base_stack_config(
                options
            )
        factory = self.feed_forward_linear_layer_config_factory
        return factory.build_linear_stack_config(
            layer_model_config=(
                factory.build_backend_linear_layer_config(
                    bias_flag=options.bias_flag,
                )
            ),
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            activation=options.activation,
            residual_connection_option=options.residual_connection_option,
            layer_norm_position=options.layer_norm_position,
            dropout_probability=options.dropout_probability,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_pipeline_flag=options.apply_output_pipeline_flag,
        )

    def _apply_feed_forward_controls(self, stack_config) -> None:
        target_stack = self._feed_forward_control_target_stack(stack_config)
        dependencies = self._control_factory_dependencies()
        gate_factory = VitGateConfigFactory(dependencies).build_feed_forward_factory()
        halting_factory = VitHaltingConfigFactory(
            dependencies
        ).build_feed_forward_factory()
        memory_factory = VitMemoryConfigFactory(
            dependencies
        ).build_feed_forward_factory()
        controller_options = self.feed_forward_layer_controller_options
        if (
            gate_factory is not None
            and controller_options is not None
            and controller_options.stack_gate_flag
        ):
            target_stack.layer_config.gate_config = gate_factory.build_gate_config()
        if (
            halting_factory is not None
            and controller_options is not None
            and controller_options.stack_halting_flag
        ):
            target_stack.layer_config.halting_config = (
                halting_factory.build_halting_config()
            )
        if (
            controller_options is not None
            and controller_options.shared_gate_config is not None
        ):
            target_stack.shared_gate_config = controller_options.shared_gate_config
        if (
            memory_factory is not None
            and self.feed_forward_dynamic_memory_options is not None
            and self.feed_forward_dynamic_memory_options.memory_flag
        ):
            target_stack.shared_memory_config = memory_factory.build_memory_config()

    def _feed_forward_control_target_stack(self, stack_config) -> LayerStackConfig:
        if isinstance(stack_config, LayerStackConfig):
            return stack_config
        if isinstance(stack_config, MixtureOfExpertsModelConfig):
            return stack_config.stack_config
        raise TypeError(
            "Feed-forward controls require a LayerStackConfig or "
            f"MixtureOfExpertsModelConfig, got {type(stack_config).__name__}."
        )

    def _stack_residual_connection_option(self) -> ResidualConnectionOptions:
        if self.encoder_stack_options is None:
            return None
        return self.encoder_stack_options.residual_connection_option

    def _stack_last_layer_bias_option(self) -> LastLayerBiasOptions:
        if self.encoder_stack_options is None:
            return LastLayerBiasOptions.DEFAULT
        return self.encoder_stack_options.last_layer_bias_option

    def _stack_apply_output_pipeline_flag(self) -> bool:
        if self.encoder_stack_options is None:
            return True
        return self.encoder_stack_options.apply_output_pipeline_flag

    def _shared_gate_config(self):
        if self.encoder_layer_controller_options is None:
            return None
        return self.encoder_layer_controller_options.shared_gate_config

    def _effective_stack_options(self) -> MainLayerStackOptions:
        if self.encoder_stack_options is not None:
            return self.encoder_stack_options
        return MainLayerStackOptions(
            bias_flag=True,
            layer_norm_position=self.encoder_options.layer_norm_position,
            num_layers=self.encoder_options.num_layers,
            activation=self.encoder_options.activation,
            residual_connection_option=None,
            dropout_probability=self.encoder_options.dropout_probability,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
        )

    def _effective_attention_projection_stack_options(self) -> SubmoduleStackOptions:
        return self.attention_projection_stack_options

    def _effective_feed_forward_stack_options(self) -> SubmoduleStackOptions:
        return self.feed_forward_stack_options

    def _control_factory_dependencies(self) -> VitControlFactoryDependencies:
        return VitControlFactoryDependencies(
            hidden_dim=self.hidden_dim,
            encoder_stack_options=self._effective_stack_options(),
            encoder_submodule_stack_options=self.encoder_submodule_stack_options,
            attention_projection_stack_options=(
                self._effective_attention_projection_stack_options()
            ),
            feed_forward_stack_options=self._effective_feed_forward_stack_options(),
            encoder_layer_controller_options=self.encoder_layer_controller_options,
            attention_projection_layer_controller_options=(
                self.attention_projection_layer_controller_options
            ),
            feed_forward_layer_controller_options=(
                self.feed_forward_layer_controller_options
            ),
            encoder_dynamic_memory_options=self.encoder_dynamic_memory_options,
            attention_projection_dynamic_memory_options=(
                self.attention_projection_dynamic_memory_options
            ),
            feed_forward_dynamic_memory_options=(
                self.feed_forward_dynamic_memory_options
            ),
            encoder_recurrent_controller_options=(
                self.encoder_recurrent_controller_options
            ),
            attention_projection_recurrent_controller_options=(
                self.attention_projection_recurrent_controller_options
            ),
            feed_forward_recurrent_controller_options=(
                self.feed_forward_recurrent_controller_options
            ),
        )
