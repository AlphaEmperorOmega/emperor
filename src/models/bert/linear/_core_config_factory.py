from dataclasses import dataclass, replace
from typing import Literal

import torch

import models.bert.linear.config as config
from emperor.attention import (
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
)
from emperor.transformer import (
    FeedForwardConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from models.bert.linear import _config_defaults as config_defaults
from models.bert.linear._encoder_control_config_factory import (
    GateConfigFactory,
    HaltingConfigFactory,
    MemoryConfigFactory,
    RecurrentConfigFactory,
)
from models.bert.linear._linear_layer_config_factory import LinearLayerConfigFactory
from models.bert.linear.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
)

from ._residual import build_residual_config


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


class CoreConfigFactory:
    def __init__(self, dependencies: CoreConfigDependencies) -> None:
        self.batch_size = dependencies.batch_size
        self.sequence_length = dependencies.sequence_length
        self.linear_layer_config_factory = dependencies.linear_layer_config_factory
        self.encoder_options = (
            dependencies.encoder_options or config_defaults.bert_encoder_options(config)
        )
        self.hidden_dim = self.encoder_options.hidden_dim
        self.attention_options = (
            dependencies.attention_options
            or config_defaults.bert_attention_options(config)
        )
        self.feed_forward_options = (
            dependencies.feed_forward_options
            or config_defaults.bert_feed_forward_options(config)
        )
        self.stack_options = (
            dependencies.stack_options
            or config_defaults.main_layer_stack_options(config)
        )
        self.submodule_stack_options = (
            dependencies.submodule_stack_options
            or config_defaults.submodule_stack_options(config, self.stack_options)
        )
        self.layer_controller_options = (
            dependencies.layer_controller_options
            or self.__layer_controller_options(
                role="main",
            )
        )
        self.dynamic_memory_options = (
            dependencies.dynamic_memory_options
            or self.__dynamic_memory_options(role="main")
        )
        self.recurrent_controller_options = (
            dependencies.recurrent_controller_options
            or self.__recurrent_controller_options(role="main")
        )
        self.attention_projection_stack_options = (
            dependencies.attention_projection_stack_options
            or config_defaults.attention_projection_stack_options(
                config,
                self.encoder_options,
                self.attention_options,
            )
        )
        self.attention_projection_layer_controller_options = (
            dependencies.attention_projection_layer_controller_options
            or self.__layer_controller_options(
                role="attention",
            )
        )
        self.attention_projection_dynamic_memory_options = (
            dependencies.attention_projection_dynamic_memory_options
            or self.__dynamic_memory_options(role="attention")
        )
        self.attention_projection_recurrent_controller_options = (
            dependencies.attention_projection_recurrent_controller_options
            or self.__recurrent_controller_options(role="attention")
        )
        self.feed_forward_stack_options = (
            dependencies.feed_forward_stack_options
            or config_defaults.feed_forward_stack_options(
                config,
                self.encoder_options,
                self.feed_forward_options,
            )
        )
        self.feed_forward_layer_controller_options = (
            dependencies.feed_forward_layer_controller_options
            or self.__layer_controller_options(
                role="feed_forward",
            )
        )
        self.feed_forward_dynamic_memory_options = (
            dependencies.feed_forward_dynamic_memory_options
            or self.__dynamic_memory_options(role="feed_forward")
        )
        self.feed_forward_recurrent_controller_options = (
            dependencies.feed_forward_recurrent_controller_options
            or self.__recurrent_controller_options(role="feed_forward")
        )

    def build_encoder_config(self):
        encoder_layer_config = self.__build_encoder_layer_config()
        gate_factory = self.__encoder_gate_factory()
        halting_factory = self.__encoder_halting_factory()
        memory_factory = self.__encoder_memory_factory()
        halting_config = halting_factory.build_halting_config()
        layer_config = TransformerEncoderBlockLayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            normalization=NormalizationOptions.RMS_NORM,
            residual_config=build_residual_config(
                self.stack_options.residual_connection_option,
                self.stack_options.residual_model_flag,
                self.stack_options.residual_stack_options,
            ),
            dropout_probability=0.0,
            gate_config=gate_factory.build_gate_config(),
            halting_config=None,
            layer_model_config=encoder_layer_config,
        )
        stack_config = LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self.encoder_options.num_layers,
            last_layer_bias_option=self.stack_options.last_layer_bias_option,
            apply_output_postprocessing_flag=(
                self.stack_options.apply_output_postprocessing_flag
            ),
            shared_gate_config=self.layer_controller_options.shared_gate_config,
            shared_halting_config=halting_config,
            shared_memory_config=memory_factory.build_memory_config(),
            layer_config=layer_config,
        )
        return self.__encoder_recurrent_factory().build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    def __build_encoder_layer_config(self) -> TransformerEncoderLayerConfig:
        options = self.encoder_options
        return TransformerEncoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            dropout_probability=options.dropout_probability,
            residual_config=AdditiveResidualConfig(),
            attention_config=self.__build_attention_config(),
            feed_forward_config=self.__build_feed_forward_config(),
        )

    def __build_attention_config(self) -> SelfAttentionConfig:
        encoder_options = self.encoder_options
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
            dropout_probability=encoder_options.dropout_probability,
            zero_attention_flag=False,
            causal_attention_mask_flag=encoder_options.causal_attention_mask_flag,
            add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
            average_attention_weights_flag=False,
            return_attention_weights_flag=False,
            batch_first_flag=True,
            projection_model_config=(self.__build_attention_projection_stack_config()),
            projection_strategy=SelfAttentionProjectionStrategy.SEPARATE,
        )

    def __build_feed_forward_config(self) -> FeedForwardConfig:
        return FeedForwardConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            stack_config=self.__build_feed_forward_stack_config(),
        )

    def __build_attention_projection_stack_config(self):
        options = self.attention_projection_stack_options
        stack_config = self.linear_layer_config_factory.build_linear_stack_config(
            layer_model_config=(
                self.linear_layer_config_factory.build_backend_linear_layer_config(
                    bias_flag=options.bias_flag,
                )
            ),
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            activation=options.activation,
            residual_connection_option=options.residual_connection_option,
            residual_model_flag=options.residual_model_flag,
            residual_stack_options=options.residual_stack_options,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            dropout_probability=options.dropout_probability,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
        )
        gate_factory = self.__attention_projection_gate_factory()
        halting_factory = self.__attention_projection_halting_factory()
        memory_factory = self.__attention_projection_memory_factory()
        stack_config.layer_config.gate_config = gate_factory.build_gate_config()
        stack_config.layer_config.halting_config = (
            halting_factory.build_halting_config()
        )
        stack_config.shared_gate_config = (
            self.attention_projection_layer_controller_options.shared_gate_config
        )
        stack_config.shared_memory_config = memory_factory.build_memory_config()
        return self.__attention_projection_recurrent_factory().build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    def __build_feed_forward_stack_config(self):
        options = self.feed_forward_stack_options
        stack_config = self.linear_layer_config_factory.build_linear_stack_config(
            layer_model_config=(
                self.linear_layer_config_factory.build_backend_linear_layer_config(
                    bias_flag=options.bias_flag,
                )
            ),
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            activation=options.activation,
            residual_connection_option=options.residual_connection_option,
            residual_model_flag=options.residual_model_flag,
            residual_stack_options=options.residual_stack_options,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            dropout_probability=options.dropout_probability,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
        )
        gate_factory = self.__feed_forward_gate_factory()
        halting_factory = self.__feed_forward_halting_factory()
        memory_factory = self.__feed_forward_memory_factory()
        stack_config.layer_config.gate_config = gate_factory.build_gate_config()
        stack_config.layer_config.halting_config = (
            halting_factory.build_halting_config()
        )
        stack_config.shared_gate_config = (
            self.feed_forward_layer_controller_options.shared_gate_config
        )
        stack_config.shared_memory_config = memory_factory.build_memory_config()
        return self.__feed_forward_recurrent_factory().build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    def __encoder_gate_factory(self) -> GateConfigFactory:
        return GateConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
        )

    def __encoder_halting_factory(self) -> HaltingConfigFactory:
        return HaltingConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
            output_dim=self.hidden_dim,
        )

    def __encoder_memory_factory(self) -> MemoryConfigFactory:
        return MemoryConfigFactory(
            hidden_dim=self.hidden_dim,
            stack_options=self.stack_options,
            dynamic_memory_options=self.dynamic_memory_options,
            submodule_stack_options=self.submodule_stack_options,
        )

    def __encoder_recurrent_factory(self) -> RecurrentConfigFactory:
        return RecurrentConfigFactory(
            recurrent_controller_options=self.recurrent_controller_options,
            gate_config_factory=self.__encoder_gate_factory(),
            halting_config_factory=self.__encoder_halting_factory(),
        )

    def __attention_projection_gate_factory(self) -> GateConfigFactory:
        return GateConfigFactory(
            layer_controller_options=(
                self.attention_projection_layer_controller_options
            ),
            recurrent_controller_options=(
                self.attention_projection_recurrent_controller_options
            ),
            submodule_stack_options=self.attention_projection_stack_options,
            recurrent_stack_inherits_gate_stack=False,
        )

    def __attention_projection_halting_factory(self) -> HaltingConfigFactory:
        options = self.attention_projection_stack_options
        return HaltingConfigFactory(
            layer_controller_options=(
                self.attention_projection_layer_controller_options
            ),
            recurrent_controller_options=(
                self.attention_projection_recurrent_controller_options
            ),
            submodule_stack_options=options,
            output_dim=self.hidden_dim,
            halting_stack_defaults=options,
            recurrent_stack_inherits_halting_stack=False,
        )

    def __attention_projection_memory_factory(self) -> MemoryConfigFactory:
        return MemoryConfigFactory(
            hidden_dim=self.hidden_dim,
            stack_options=self.stack_options,
            dynamic_memory_options=(self.attention_projection_dynamic_memory_options),
            submodule_stack_options=self.attention_projection_stack_options,
        )

    def __attention_projection_recurrent_factory(self) -> RecurrentConfigFactory:
        return RecurrentConfigFactory(
            recurrent_controller_options=(
                self.attention_projection_recurrent_controller_options
            ),
            gate_config_factory=self.__attention_projection_gate_factory(),
            halting_config_factory=self.__attention_projection_halting_factory(),
        )

    def __feed_forward_gate_factory(self) -> GateConfigFactory:
        return GateConfigFactory(
            layer_controller_options=self.feed_forward_layer_controller_options,
            recurrent_controller_options=(
                self.feed_forward_recurrent_controller_options
            ),
            submodule_stack_options=self.feed_forward_stack_options,
            recurrent_stack_inherits_gate_stack=False,
        )

    def __feed_forward_halting_factory(self) -> HaltingConfigFactory:
        options = self.feed_forward_stack_options
        return HaltingConfigFactory(
            layer_controller_options=self.feed_forward_layer_controller_options,
            recurrent_controller_options=(
                self.feed_forward_recurrent_controller_options
            ),
            submodule_stack_options=options,
            output_dim=self.hidden_dim,
            halting_stack_defaults=options,
            recurrent_stack_inherits_halting_stack=False,
        )

    def __feed_forward_memory_factory(self) -> MemoryConfigFactory:
        return MemoryConfigFactory(
            hidden_dim=self.hidden_dim,
            stack_options=self.stack_options,
            dynamic_memory_options=self.feed_forward_dynamic_memory_options,
            submodule_stack_options=self.feed_forward_stack_options,
        )

    def __feed_forward_recurrent_factory(self) -> RecurrentConfigFactory:
        return RecurrentConfigFactory(
            recurrent_controller_options=(
                self.feed_forward_recurrent_controller_options
            ),
            gate_config_factory=self.__feed_forward_gate_factory(),
            halting_config_factory=self.__feed_forward_halting_factory(),
        )

    def __layer_controller_options(
        self,
        *,
        role: Literal["main", "attention", "feed_forward"],
    ) -> LayerControllerOptions:
        defaults = config_defaults.linears_layer_controller_options(
            config,
            role,
        )
        return replace(defaults, halting_option=LayerControllerOptions.halting_option)

    def __dynamic_memory_options(
        self,
        *,
        role: Literal["main", "attention", "feed_forward"],
    ) -> DynamicMemoryOptions:
        return config_defaults.linears_dynamic_memory_options(
            config,
            role,
        )

    def __recurrent_controller_options(
        self,
        *,
        role: Literal["main", "attention", "feed_forward"],
    ) -> RecurrentControllerOptions:
        defaults = config_defaults.linears_recurrent_controller_options(
            config,
            role,
        )
        return replace(
            defaults,
            recurrent_halting_option=(
                RecurrentControllerOptions.recurrent_halting_option
            ),
        )
