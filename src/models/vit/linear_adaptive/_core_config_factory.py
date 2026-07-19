from dataclasses import dataclass
from typing import Any

from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import LastLayerBiasOptions, LayerNormPositionOptions

import models.vit.linear_adaptive.config as config
from models.vit.linear_adaptive._linear_layer_config_factory import (
    LinearLayerConfigFactory,
)
from models.vit.linear_adaptive._vit_core_config_factory import (
    CoreConfigDependencies as _CoreDependencies,
)
from models.vit.linear_adaptive._vit_core_config_factory import VitCoreConfigFactory
from models.vit.linear_adaptive.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class CoreConfigDependencies:
    batch_size: int
    sequence_length: int
    encoder_options: TransformerEncoderOptions | None
    attention_options: TransformerAttentionOptions | None
    feed_forward_options: TransformerFeedForwardOptions | None
    attention_projection_stack_options: SubmoduleStackOptions | None
    attention_projection_layer_controller_options: LayerControllerOptions | None
    attention_projection_dynamic_memory_options: DynamicMemoryOptions | None
    attention_projection_recurrent_controller_options: RecurrentControllerOptions | None
    feed_forward_stack_options: SubmoduleStackOptions | None
    feed_forward_layer_controller_options: LayerControllerOptions | None
    feed_forward_dynamic_memory_options: DynamicMemoryOptions | None
    feed_forward_recurrent_controller_options: RecurrentControllerOptions | None
    stack_options: MainLayerStackOptions | None
    submodule_stack_options: SubmoduleStackOptions | None
    layer_controller_options: LayerControllerOptions | None
    dynamic_memory_options: DynamicMemoryOptions | None
    recurrent_controller_options: RecurrentControllerOptions | None
    linear_layer_config_factory: LinearLayerConfigFactory
    attention_projection_linear_layer_config_factory: (
        LinearLayerConfigFactory | None
    ) = None
    feed_forward_linear_layer_config_factory: LinearLayerConfigFactory | None = None
    expert_config_factory: Any | None = None


class CoreConfigFactory:
    def __init__(self, dependencies: CoreConfigDependencies) -> None:
        self.dependencies = dependencies
        self.encoder_options = self.__default_encoder_options(
            dependencies.encoder_options
        )
        self.attention_options = self.__default_attention_options(
            dependencies.attention_options
        )
        self.feed_forward_options = self.__default_feed_forward_options(
            dependencies.feed_forward_options
        )
        self.stack_options = self.__default_stack_options(dependencies.stack_options)
        self.attention_projection_stack_options = (
            self.__default_attention_projection_stack_options(
                dependencies.attention_projection_stack_options
            )
        )
        self.feed_forward_stack_options = self.__default_feed_forward_stack_options(
            dependencies.feed_forward_stack_options
        )

    def build_encoder_config(self):
        core_factory = VitCoreConfigFactory(self.__core_dependencies())
        return core_factory.build_encoder_config()

    def __default_encoder_options(
        self,
        encoder_options: TransformerEncoderOptions | None,
    ) -> TransformerEncoderOptions:
        if encoder_options is not None:
            return encoder_options
        return TransformerEncoderOptions(
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.STACK_NUM_LAYERS,
            activation=config.STACK_ACTIVATION,
            dropout_probability=config.STACK_DROPOUT_PROBABILITY,
            layer_norm_position=config.STACK_LAYER_NORM_POSITION,
            causal_attention_mask_flag=False,
        )

    def __default_attention_options(
        self,
        attention_options: TransformerAttentionOptions | None,
    ) -> TransformerAttentionOptions:
        if attention_options is not None:
            return attention_options
        return TransformerAttentionOptions(
            num_heads=config.ATTN_NUM_HEADS,
            num_layers=config.ATTN_NUM_LAYERS,
            bias_flag=config.ATTN_BIAS_FLAG,
            add_key_value_bias_flag=config.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
        )

    def __default_feed_forward_options(
        self,
        feed_forward_options: TransformerFeedForwardOptions | None,
    ) -> TransformerFeedForwardOptions:
        if feed_forward_options is not None:
            return feed_forward_options
        return TransformerFeedForwardOptions(
            num_layers=config.FF_NUM_LAYERS,
            bias_flag=config.FF_BIAS_FLAG,
        )

    def __default_stack_options(
        self,
        stack_options: MainLayerStackOptions | None,
    ) -> MainLayerStackOptions:
        if stack_options is not None:
            return stack_options
        return MainLayerStackOptions(
            bias_flag=config.STACK_BIAS_FLAG,
            layer_norm_position=config.STACK_LAYER_NORM_POSITION,
            num_layers=config.STACK_NUM_LAYERS,
            activation=config.STACK_ACTIVATION,
            residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
            dropout_probability=config.STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        )

    def __default_attention_projection_stack_options(
        self,
        stack_options: SubmoduleStackOptions | None,
    ) -> SubmoduleStackOptions:
        if stack_options is not None:
            return stack_options
        return SubmoduleStackOptions(
            hidden_dim=self.encoder_options.hidden_dim,
            num_layers=self.attention_options.num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            activation=self.encoder_options.activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            bias_flag=self.attention_options.bias_flag,
        )

    def __default_feed_forward_stack_options(
        self,
        stack_options: SubmoduleStackOptions | None,
    ) -> SubmoduleStackOptions:
        if stack_options is not None:
            return stack_options
        return SubmoduleStackOptions(
            hidden_dim=self.__scaled_feed_forward_hidden_dim(),
            num_layers=self.feed_forward_options.num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            activation=self.encoder_options.activation,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=self.encoder_options.dropout_probability,
            bias_flag=self.feed_forward_options.bias_flag,
        )

    def __scaled_feed_forward_hidden_dim(self) -> int:
        if (
            config.HIDDEN_DIM > 0
            and config.FF_STACK_HIDDEN_DIM % config.HIDDEN_DIM == 0
        ):
            return self.encoder_options.hidden_dim * (
                config.FF_STACK_HIDDEN_DIM // config.HIDDEN_DIM
            )
        return config.FF_STACK_HIDDEN_DIM

    def __core_dependencies(self) -> _CoreDependencies:
        dependencies = self.dependencies
        return _CoreDependencies(
            batch_size=dependencies.batch_size,
            sequence_length=dependencies.sequence_length,
            encoder_options=self.encoder_options,
            attention_options=self.attention_options,
            feed_forward_options=self.feed_forward_options,
            attention_projection_stack_options=(
                self.attention_projection_stack_options
            ),
            attention_projection_layer_controller_options=(
                dependencies.attention_projection_layer_controller_options
            ),
            attention_projection_dynamic_memory_options=(
                dependencies.attention_projection_dynamic_memory_options
            ),
            attention_projection_recurrent_controller_options=(
                dependencies.attention_projection_recurrent_controller_options
            ),
            feed_forward_stack_options=self.feed_forward_stack_options,
            feed_forward_layer_controller_options=(
                dependencies.feed_forward_layer_controller_options
            ),
            feed_forward_dynamic_memory_options=(
                dependencies.feed_forward_dynamic_memory_options
            ),
            feed_forward_recurrent_controller_options=(
                dependencies.feed_forward_recurrent_controller_options
            ),
            stack_options=self.stack_options,
            submodule_stack_options=dependencies.submodule_stack_options,
            layer_controller_options=dependencies.layer_controller_options,
            dynamic_memory_options=dependencies.dynamic_memory_options,
            recurrent_controller_options=dependencies.recurrent_controller_options,
            linear_layer_config_factory=dependencies.linear_layer_config_factory,
            attention_projection_linear_layer_config_factory=(
                dependencies.attention_projection_linear_layer_config_factory
            ),
            feed_forward_linear_layer_config_factory=(
                dependencies.feed_forward_linear_layer_config_factory
            ),
            expert_config_factory=dependencies.expert_config_factory,
        )
