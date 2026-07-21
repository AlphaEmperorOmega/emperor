from dataclasses import dataclass

import torch

from emperor.attention import (
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.layers import (
    ActivationOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.transformer import (
    FeedForwardConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerConfig,
)
from models.gpt.linear_adaptive._control_factory_dependencies import (
    GptControlFactoryDependencies,
)
from models.gpt.linear_adaptive._decoder_control_config_factory import (
    GptGateConfigFactory,
    GptHaltingConfigFactory,
    GptMemoryConfigFactory,
    GptRecurrentConfigFactory,
)
from models.gpt.linear_adaptive._linear_layer_config_factory import (
    LinearLayerConfigFactory,
)
from models.gpt.linear_adaptive.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class CoreConfigDependencies:
    batch_size: int
    sequence_length: int
    decoder_options: TransformerDecoderOptions
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
    stack_options: MainLayerStackOptions
    submodule_stack_options: SubmoduleStackOptions | None
    layer_controller_options: LayerControllerOptions | None
    dynamic_memory_options: DynamicMemoryOptions | None
    recurrent_controller_options: RecurrentControllerOptions | None
    linear_layer_config_factory: LinearLayerConfigFactory
    attention_projection_linear_layer_config_factory: (
        LinearLayerConfigFactory | None
    ) = None
    feed_forward_linear_layer_config_factory: LinearLayerConfigFactory | None = None


class GptCoreConfigFactory:
    def __init__(self, dependencies: CoreConfigDependencies) -> None:
        self.batch_size = dependencies.batch_size
        self.sequence_length = dependencies.sequence_length
        self.decoder_options = dependencies.decoder_options
        self.hidden_dim = dependencies.decoder_options.hidden_dim
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
        self.decoder_stack_options = dependencies.stack_options
        self.decoder_submodule_stack_options = dependencies.submodule_stack_options
        self.decoder_layer_controller_options = dependencies.layer_controller_options
        self.decoder_dynamic_memory_options = dependencies.dynamic_memory_options
        self.decoder_recurrent_controller_options = (
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

    def build_decoder_config(self):
        dependencies = self._control_factory_dependencies()
        gate_factory = GptGateConfigFactory(dependencies).build_decoder_factory()
        halting_factory = GptHaltingConfigFactory(dependencies).build_decoder_factory()
        memory_factory = GptMemoryConfigFactory(dependencies).build_decoder_factory()
        halting_config = (
            halting_factory.build_halting_config()
            if halting_factory is not None
            else None
        )
        layer_config = TransformerDecoderBlockLayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None
            if (self.decoder_stack_options.residual_connection_option) is None
            else ResidualConfig(
                option=(self.decoder_stack_options.residual_connection_option)
            ),
            dropout_probability=0.0,
            gate_config=(
                gate_factory.build_gate_config() if gate_factory is not None else None
            ),
            halting_config=None,
            layer_model_config=self._build_decoder_layer_config(),
        )
        stack_config = LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self.decoder_options.num_layers,
            last_layer_bias_option=(self.decoder_stack_options.last_layer_bias_option),
            apply_output_pipeline_flag=(
                self.decoder_stack_options.apply_output_pipeline_flag
            ),
            shared_gate_config=(
                self.decoder_layer_controller_options.shared_gate_config
                if self.decoder_layer_controller_options is not None
                else None
            ),
            shared_halting_config=halting_config,
            shared_memory_config=(
                memory_factory.build_memory_config()
                if memory_factory is not None
                else None
            ),
            layer_config=layer_config,
        )
        recurrent_factory = GptRecurrentConfigFactory(
            dependencies
        ).build_decoder_factory()
        if recurrent_factory is None:
            return stack_config
        return recurrent_factory.build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    def _build_decoder_layer_config(self) -> TransformerDecoderLayerConfig:
        options = self.decoder_options
        return TransformerDecoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=options.layer_norm_position,
            dropout_probability=options.dropout_probability,
            residual_config=ResidualConfig(option=ResidualConnectionOptions.RESIDUAL),
            self_attention_config=self._build_attention_config(),
            cross_attention_config=None,
            feed_forward_config=self._build_feed_forward_config(),
        )

    def _build_attention_config(self) -> SelfAttentionConfig:
        decoder_options = self.decoder_options
        attention_options = self.attention_options
        return SelfAttentionConfig(
            batch_size=self.batch_size,
            num_heads=attention_options.num_heads,
            embedding_dim=self.hidden_dim,
            query_key_projection_dim=self.hidden_dim,
            value_projection_dim=self.hidden_dim,
            target_sequence_length=self.sequence_length,
            source_sequence_length=self.sequence_length,
            target_dtype=torch.float32,
            dropout_probability=decoder_options.dropout_probability,
            zero_attention_flag=False,
            causal_attention_mask_flag=True,
            add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
            average_attention_weights_flag=False,
            return_attention_weights_flag=False,
            batch_first_flag=True,
            projection_model_config=self._build_attention_projection_stack_config(),
            projection_strategy=SelfAttentionProjectionStrategy.SEPARATE,
        )

    def _build_feed_forward_config(self) -> FeedForwardConfig:
        return FeedForwardConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            stack_config=self._build_feed_forward_stack_config(),
        )

    def _build_attention_projection_stack_config(self):
        options = self.attention_projection_stack_options
        factory = self.attention_projection_linear_layer_config_factory
        stack_config = factory.build_linear_stack_config(
            layer_model_config=factory.build_backend_linear_layer_config(
                bias_flag=options.bias_flag
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
        self._apply_attention_projection_controls(stack_config)
        recurrent_factory = GptRecurrentConfigFactory(
            self._control_factory_dependencies()
        ).build_attention_projection_factory()
        if recurrent_factory is None:
            return stack_config
        return recurrent_factory.build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    def _apply_attention_projection_controls(
        self,
        stack_config: LayerStackConfig,
    ) -> None:
        dependencies = self._control_factory_dependencies()
        gate_factory = GptGateConfigFactory(
            dependencies
        ).build_attention_projection_factory()
        halting_factory = GptHaltingConfigFactory(
            dependencies
        ).build_attention_projection_factory()
        memory_factory = GptMemoryConfigFactory(
            dependencies
        ).build_attention_projection_factory()
        controller_options = self.attention_projection_layer_controller_options
        if gate_factory is not None:
            stack_config.layer_config.gate_config = gate_factory.build_gate_config()
        if halting_factory is not None:
            stack_config.layer_config.halting_config = (
                halting_factory.build_halting_config()
            )
        if controller_options is not None:
            stack_config.shared_gate_config = controller_options.shared_gate_config
        if memory_factory is not None:
            stack_config.shared_memory_config = memory_factory.build_memory_config()

    def _build_feed_forward_stack_config(self):
        options = self.feed_forward_stack_options
        factory = self.feed_forward_linear_layer_config_factory
        stack_config = factory.build_linear_stack_config(
            layer_model_config=factory.build_backend_linear_layer_config(
                bias_flag=options.bias_flag
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
        self._apply_feed_forward_controls(stack_config)
        recurrent_factory = GptRecurrentConfigFactory(
            self._control_factory_dependencies()
        ).build_feed_forward_factory()
        if recurrent_factory is None:
            return stack_config
        return recurrent_factory.build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    def _apply_feed_forward_controls(self, stack_config: LayerStackConfig) -> None:
        dependencies = self._control_factory_dependencies()
        gate_factory = GptGateConfigFactory(dependencies).build_feed_forward_factory()
        halting_factory = GptHaltingConfigFactory(
            dependencies
        ).build_feed_forward_factory()
        memory_factory = GptMemoryConfigFactory(
            dependencies
        ).build_feed_forward_factory()
        controller_options = self.feed_forward_layer_controller_options
        if gate_factory is not None:
            stack_config.layer_config.gate_config = gate_factory.build_gate_config()
        if halting_factory is not None:
            stack_config.layer_config.halting_config = (
                halting_factory.build_halting_config()
            )
        if controller_options is not None:
            stack_config.shared_gate_config = controller_options.shared_gate_config
        if memory_factory is not None:
            stack_config.shared_memory_config = memory_factory.build_memory_config()

    def _control_factory_dependencies(self) -> GptControlFactoryDependencies:
        return GptControlFactoryDependencies(
            hidden_dim=self.hidden_dim,
            decoder_stack_options=self.decoder_stack_options,
            decoder_submodule_stack_options=self.decoder_submodule_stack_options,
            attention_projection_stack_options=(
                self.attention_projection_stack_options
            ),
            feed_forward_stack_options=self.feed_forward_stack_options,
            decoder_layer_controller_options=self.decoder_layer_controller_options,
            attention_projection_layer_controller_options=(
                self.attention_projection_layer_controller_options
            ),
            feed_forward_layer_controller_options=(
                self.feed_forward_layer_controller_options
            ),
            decoder_dynamic_memory_options=self.decoder_dynamic_memory_options,
            attention_projection_dynamic_memory_options=(
                self.attention_projection_dynamic_memory_options
            ),
            feed_forward_dynamic_memory_options=(
                self.feed_forward_dynamic_memory_options
            ),
            decoder_recurrent_controller_options=(
                self.decoder_recurrent_controller_options
            ),
            attention_projection_recurrent_controller_options=(
                self.attention_projection_recurrent_controller_options
            ),
            feed_forward_recurrent_controller_options=(
                self.feed_forward_recurrent_controller_options
            ),
        )
