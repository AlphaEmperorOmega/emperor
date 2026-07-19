from dataclasses import dataclass

import torch

import models.gpt.linear.config as config
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
from models.gpt.linear._decoder_control_config_factory import (
    GateConfigFactory,
    HaltingConfigFactory,
    MemoryConfigFactory,
    RecurrentConfigFactory,
)
from models.gpt.linear._linear_layer_config_factory import LinearLayerConfigFactory
from models.gpt.linear.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    SubmoduleStackSource,
    TransformerAttentionOptions,
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class CoreConfigDependencies:
    batch_size: int
    sequence_length: int
    decoder_options: TransformerDecoderOptions | None
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
        self.decoder_options = self.__default_decoder_options(
            dependencies.decoder_options
        )
        self.hidden_dim = self.decoder_options.hidden_dim
        self.attention_options = self.__default_attention_options(
            dependencies.attention_options
        )
        self.feed_forward_options = self.__default_feed_forward_options(
            dependencies.feed_forward_options
        )
        self.stack_options = self.__default_stack_options(dependencies.stack_options)
        self.submodule_stack_options = self.__default_submodule_stack_options(
            dependencies.submodule_stack_options
        )
        self.layer_controller_options = self.__default_layer_controller_options(
            dependencies.layer_controller_options,
            prefix="",
        )
        self.dynamic_memory_options = self.__default_dynamic_memory_options(
            dependencies.dynamic_memory_options,
            prefix="",
        )
        self.recurrent_controller_options = self.__default_recurrent_controller_options(
            dependencies.recurrent_controller_options,
            prefix="RECURRENT",
        )
        self.attention_projection_stack_options = (
            self.__default_attention_projection_stack_options(
                dependencies.attention_projection_stack_options
            )
        )
        self.attention_projection_layer_controller_options = (
            self.__default_layer_controller_options(
                dependencies.attention_projection_layer_controller_options,
                prefix="ATTN",
            )
        )
        self.attention_projection_dynamic_memory_options = (
            self.__default_dynamic_memory_options(
                dependencies.attention_projection_dynamic_memory_options,
                prefix="ATTN",
            )
        )
        self.attention_projection_recurrent_controller_options = (
            self.__default_recurrent_controller_options(
                dependencies.attention_projection_recurrent_controller_options,
                prefix="ATTN_RECURRENT",
            )
        )
        self.feed_forward_stack_options = self.__default_feed_forward_stack_options(
            dependencies.feed_forward_stack_options
        )
        self.feed_forward_layer_controller_options = (
            self.__default_layer_controller_options(
                dependencies.feed_forward_layer_controller_options,
                prefix="FF",
            )
        )
        self.feed_forward_dynamic_memory_options = (
            self.__default_dynamic_memory_options(
                dependencies.feed_forward_dynamic_memory_options,
                prefix="FF",
            )
        )
        self.feed_forward_recurrent_controller_options = (
            self.__default_recurrent_controller_options(
                dependencies.feed_forward_recurrent_controller_options,
                prefix="FF_RECURRENT",
            )
        )

    def build_decoder_config(self):
        decoder_layer_config = self.__build_decoder_layer_config()
        gate_factory = self.__decoder_gate_factory()
        halting_factory = self.__decoder_halting_factory()
        memory_factory = self.__decoder_memory_factory()
        halting_config = halting_factory.build_halting_config()
        layer_config = TransformerDecoderBlockLayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None
            if (self.stack_options.residual_connection_option) is None
            else ResidualConfig(option=(self.stack_options.residual_connection_option)),
            dropout_probability=0.0,
            gate_config=gate_factory.build_gate_config(),
            halting_config=None,
            layer_model_config=decoder_layer_config,
        )
        stack_config = LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self.decoder_options.num_layers,
            last_layer_bias_option=self.stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=(self.stack_options.apply_output_pipeline_flag),
            shared_gate_config=self.layer_controller_options.shared_gate_config,
            shared_halting_config=halting_config,
            shared_memory_config=memory_factory.build_memory_config(),
            layer_config=layer_config,
        )
        return self.__decoder_recurrent_factory().build_config(
            stack_config,
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    def __build_decoder_layer_config(self) -> TransformerDecoderLayerConfig:
        options = self.decoder_options
        return TransformerDecoderLayerConfig(
            embedding_dim=self.hidden_dim,
            layer_norm_position=options.layer_norm_position,
            dropout_probability=options.dropout_probability,
            residual_config=ResidualConfig(option=ResidualConnectionOptions.RESIDUAL),
            causal_attention_mask_flag=True,
            self_attention_config=self.__build_attention_config(),
            cross_attention_config=None,
            feed_forward_config=self.__build_feed_forward_config(),
        )

    def __build_attention_config(self) -> SelfAttentionConfig:
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
            layer_norm_position=options.layer_norm_position,
            dropout_probability=options.dropout_probability,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_pipeline_flag=options.apply_output_pipeline_flag,
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
            layer_norm_position=options.layer_norm_position,
            dropout_probability=options.dropout_probability,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_pipeline_flag=options.apply_output_pipeline_flag,
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

    def __decoder_gate_factory(self) -> GateConfigFactory:
        return GateConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
        )

    def __decoder_halting_factory(self) -> HaltingConfigFactory:
        return HaltingConfigFactory(
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            submodule_stack_options=self.submodule_stack_options,
            output_dim=self.hidden_dim,
        )

    def __decoder_memory_factory(self) -> MemoryConfigFactory:
        return MemoryConfigFactory(
            hidden_dim=self.hidden_dim,
            stack_options=self.stack_options,
            dynamic_memory_options=self.dynamic_memory_options,
            submodule_stack_options=self.submodule_stack_options,
        )

    def __decoder_recurrent_factory(self) -> RecurrentConfigFactory:
        return RecurrentConfigFactory(
            recurrent_controller_options=self.recurrent_controller_options,
            gate_config_factory=self.__decoder_gate_factory(),
            halting_config_factory=self.__decoder_halting_factory(),
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

    def __default_decoder_options(
        self,
        options: TransformerDecoderOptions | None,
    ) -> TransformerDecoderOptions:
        if options is not None:
            return options
        return TransformerDecoderOptions(
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.STACK_NUM_LAYERS,
            activation=config.STACK_ACTIVATION,
            dropout_probability=config.STACK_DROPOUT_PROBABILITY,
            layer_norm_position=config.STACK_LAYER_NORM_POSITION,
        )

    def __default_attention_options(
        self,
        options: TransformerAttentionOptions | None,
    ) -> TransformerAttentionOptions:
        if options is not None:
            return options
        return TransformerAttentionOptions(
            num_heads=config.ATTN_NUM_HEADS,
            num_layers=config.ATTN_NUM_LAYERS,
            bias_flag=config.ATTN_BIAS_FLAG,
            add_key_value_bias_flag=config.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
        )

    def __default_feed_forward_options(
        self,
        options: TransformerFeedForwardOptions | None,
    ) -> TransformerFeedForwardOptions:
        if options is not None:
            return options
        return TransformerFeedForwardOptions(
            num_layers=config.FF_NUM_LAYERS,
            bias_flag=config.FF_BIAS_FLAG,
        )

    def __default_stack_options(
        self,
        options: MainLayerStackOptions | None,
    ) -> MainLayerStackOptions:
        if options is not None:
            return options
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

    def __default_submodule_stack_options(
        self,
        options: SubmoduleStackOptions | None,
    ) -> SubmoduleStackOptions:
        if options is not None:
            return options
        return SubmoduleStackOptions(
            hidden_dim=config.SUBMODULE_STACK_HIDDEN_DIM,
            num_layers=config.SUBMODULE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.SUBMODULE_STACK_ACTIVATION,
            layer_norm_position=config.SUBMODULE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            dropout_probability=config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
            bias_flag=self.stack_options.bias_flag,
        )

    def __default_attention_projection_stack_options(
        self,
        options: SubmoduleStackOptions | None,
    ) -> SubmoduleStackOptions:
        if options is not None:
            return options
        return SubmoduleStackOptions(
            hidden_dim=self.hidden_dim,
            num_layers=self.attention_options.num_layers,
            last_layer_bias_option=config.ATTN_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(config.ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG),
            activation=self.decoder_options.activation,
            layer_norm_position=config.ATTN_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(config.ATTN_STACK_RESIDUAL_CONNECTION_OPTION),
            dropout_probability=config.ATTN_STACK_DROPOUT_PROBABILITY,
            bias_flag=self.attention_options.bias_flag,
        )

    def __default_feed_forward_stack_options(
        self,
        options: SubmoduleStackOptions | None,
    ) -> SubmoduleStackOptions:
        if options is not None:
            return options
        return SubmoduleStackOptions(
            hidden_dim=self.__scaled_feed_forward_hidden_dim(),
            num_layers=self.feed_forward_options.num_layers,
            last_layer_bias_option=config.FF_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            activation=self.decoder_options.activation,
            layer_norm_position=config.FF_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(config.FF_STACK_RESIDUAL_CONNECTION_OPTION),
            dropout_probability=self.decoder_options.dropout_probability,
            bias_flag=self.feed_forward_options.bias_flag,
        )

    def __scaled_feed_forward_hidden_dim(self) -> int:
        if (
            config.HIDDEN_DIM > 0
            and config.FF_STACK_HIDDEN_DIM % config.HIDDEN_DIM == 0
        ):
            return self.hidden_dim * (config.FF_STACK_HIDDEN_DIM // config.HIDDEN_DIM)
        return config.FF_STACK_HIDDEN_DIM

    def __default_layer_controller_options(
        self,
        options: LayerControllerOptions | None,
        *,
        prefix: str,
    ) -> LayerControllerOptions:
        if options is not None:
            return options
        config_prefix = f"{prefix}_" if prefix else ""
        return LayerControllerOptions(
            stack_gate_flag=getattr(config, f"{config_prefix}GATE_FLAG"),
            gate_option=getattr(config, f"{config_prefix}GATE_OPTION"),
            gate_activation=getattr(config, f"{config_prefix}GATE_ACTIVATION"),
            gate_stack_source=self.__controller_stack_source(
                f"{config_prefix}GATE_STACK"
            ),
            stack_halting_flag=getattr(config, f"{config_prefix}HALTING_FLAG"),
            halting_threshold=getattr(
                config,
                f"{config_prefix}HALTING_THRESHOLD",
            ),
            halting_dropout=getattr(config, f"{config_prefix}HALTING_DROPOUT"),
            halting_hidden_state_mode=getattr(
                config,
                f"{config_prefix}HALTING_HIDDEN_STATE_MODE",
            ),
            halting_stack_source=self.__controller_stack_source(
                f"{config_prefix}HALTING_STACK"
            ),
        )

    def __default_dynamic_memory_options(
        self,
        options: DynamicMemoryOptions | None,
        *,
        prefix: str,
    ) -> DynamicMemoryOptions:
        if options is not None:
            return options
        config_prefix = f"{prefix}_" if prefix else ""
        memory_prefix = f"{config_prefix}MEMORY"
        return DynamicMemoryOptions(
            memory_flag=getattr(config, f"{memory_prefix}_FLAG"),
            memory_option=getattr(config, f"{memory_prefix}_OPTION"),
            memory_position_option=getattr(
                config,
                f"{memory_prefix}_POSITION_OPTION",
            ),
            memory_test_time_training_learning_rate=getattr(
                config,
                f"{memory_prefix}_TEST_TIME_TRAINING_LEARNING_RATE",
            ),
            memory_test_time_training_num_inner_steps=getattr(
                config,
                f"{memory_prefix}_TEST_TIME_TRAINING_NUM_INNER_STEPS",
            ),
            memory_stack_source=self.__controller_stack_source(
                f"{memory_prefix}_STACK"
            ),
        )

    def __default_recurrent_controller_options(
        self,
        options: RecurrentControllerOptions | None,
        *,
        prefix: str,
    ) -> RecurrentControllerOptions:
        if options is not None:
            return options
        return RecurrentControllerOptions(
            recurrent_flag=getattr(config, f"{prefix}_FLAG"),
            recurrent_max_steps=getattr(config, f"{prefix}_MAX_STEPS"),
            recurrent_layer_norm_position=getattr(
                config,
                f"{prefix}_LAYER_NORM_POSITION",
            ),
            recurrent_gate_flag=getattr(config, f"{prefix}_GATE_FLAG"),
            recurrent_gate_option=getattr(config, f"{prefix}_GATE_OPTION"),
            recurrent_gate_activation=getattr(
                config,
                f"{prefix}_GATE_ACTIVATION",
            ),
            recurrent_gate_stack_source=self.__controller_stack_source(
                f"{prefix}_GATE_STACK"
            ),
            recurrent_halting_flag=getattr(config, f"{prefix}_HALTING_FLAG"),
            recurrent_halting_threshold=getattr(
                config,
                f"{prefix}_HALTING_THRESHOLD",
            ),
            recurrent_halting_dropout=getattr(
                config,
                f"{prefix}_HALTING_DROPOUT",
            ),
            recurrent_halting_hidden_state_mode=getattr(
                config,
                f"{prefix}_HALTING_HIDDEN_STATE_MODE",
            ),
            recurrent_halting_stack_source=self.__controller_stack_source(
                f"{prefix}_HALTING_STACK"
            ),
        )

    def __controller_stack_source(self, prefix: str) -> SubmoduleStackSource:
        return SubmoduleStackSource(
            independent_flag=getattr(config, f"{prefix}_INDEPENDENT_FLAG"),
            hidden_dim=getattr(config, f"{prefix}_HIDDEN_DIM"),
            num_layers=getattr(config, f"{prefix}_NUM_LAYERS"),
            last_layer_bias_option=getattr(
                config,
                f"{prefix}_LAST_LAYER_BIAS_OPTION",
            ),
            apply_output_pipeline_flag=getattr(
                config,
                f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG",
            ),
            activation=getattr(config, f"{prefix}_ACTIVATION"),
            layer_norm_position=getattr(
                config,
                f"{prefix}_LAYER_NORM_POSITION",
            ),
            residual_connection_option=getattr(
                config,
                f"{prefix}_RESIDUAL_CONNECTION_OPTION",
            ),
            dropout_probability=getattr(
                config,
                f"{prefix}_DROPOUT_PROBABILITY",
            ),
            bias_flag=getattr(config, f"{prefix}_BIAS_FLAG"),
        )
