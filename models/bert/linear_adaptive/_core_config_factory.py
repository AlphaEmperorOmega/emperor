from dataclasses import dataclass, replace

import models.bert.linear_adaptive.config as config
from models.bert.linear_adaptive import _config_defaults as config_defaults
from models.bert.linear_adaptive._bert_core_config_factory import (
    BertCoreConfigFactory,
)
from models.bert.linear_adaptive._bert_core_config_factory import (
    CoreConfigDependencies as _CoreDependencies,
)
from models.bert.linear_adaptive._linear_layer_config_factory import (
    LinearLayerConfigFactory,
)
from models.bert.linear_adaptive.runtime_options import (
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


class CoreConfigFactory:
    def __init__(self, dependencies: CoreConfigDependencies) -> None:
        self.dependencies = dependencies
        self.encoder_options = (
            dependencies.encoder_options or config_defaults.bert_encoder_options(config)
        )
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
            or config_defaults.linears_submodule_stack_options(
                config,
                "SUBMODULE_STACK",
            )
        )
        self.layer_controller_options = (
            dependencies.layer_controller_options
            or self.__layer_controller_options(
                gate_prefix="GATE",
                halting_prefix="HALTING",
            )
        )
        self.dynamic_memory_options = (
            dependencies.dynamic_memory_options
            or self.__dynamic_memory_options(memory_prefix="MEMORY")
        )
        self.recurrent_controller_options = (
            dependencies.recurrent_controller_options
            or self.__recurrent_controller_options(recurrent_prefix="RECURRENT")
        )
        self.attention_projection_stack_options = (
            dependencies.attention_projection_stack_options
            or self.__attention_projection_stack_options()
        )
        self.attention_projection_layer_controller_options = (
            dependencies.attention_projection_layer_controller_options
            or self.__layer_controller_options(
                gate_prefix="ATTN_GATE",
                halting_prefix="ATTN_HALTING",
            )
        )
        self.attention_projection_dynamic_memory_options = (
            dependencies.attention_projection_dynamic_memory_options
            or self.__dynamic_memory_options(memory_prefix="ATTN_MEMORY")
        )
        self.attention_projection_recurrent_controller_options = (
            dependencies.attention_projection_recurrent_controller_options
            or self.__recurrent_controller_options(recurrent_prefix="ATTN_RECURRENT")
        )
        self.feed_forward_stack_options = (
            dependencies.feed_forward_stack_options
            or self.__feed_forward_stack_options()
        )
        self.feed_forward_layer_controller_options = (
            dependencies.feed_forward_layer_controller_options
            or self.__layer_controller_options(
                gate_prefix="FF_GATE",
                halting_prefix="FF_HALTING",
            )
        )
        self.feed_forward_dynamic_memory_options = (
            dependencies.feed_forward_dynamic_memory_options
            or self.__dynamic_memory_options(memory_prefix="FF_MEMORY")
        )
        self.feed_forward_recurrent_controller_options = (
            dependencies.feed_forward_recurrent_controller_options
            or self.__recurrent_controller_options(recurrent_prefix="FF_RECURRENT")
        )

    def build_encoder_config(self):
        return BertCoreConfigFactory(self.__core_dependencies()).build_encoder_config()

    def __attention_projection_stack_options(self) -> SubmoduleStackOptions:
        defaults = config_defaults.linears_submodule_stack_options(
            config,
            "ATTN_STACK",
            num_layers_key="ATTN_NUM_LAYERS",
            bias_key="ATTN_BIAS_FLAG",
        )
        return replace(
            defaults,
            hidden_dim=self.encoder_options.hidden_dim,
            num_layers=self.attention_options.num_layers,
            activation=self.encoder_options.activation,
            bias_flag=self.attention_options.bias_flag,
        )

    def __feed_forward_stack_options(self) -> SubmoduleStackOptions:
        defaults = config_defaults.linears_submodule_stack_options(
            config,
            "FF_STACK",
            num_layers_key="FF_NUM_LAYERS",
            bias_key="FF_BIAS_FLAG",
        )
        return replace(
            defaults,
            hidden_dim=self.__scaled_feed_forward_hidden_dim(),
            num_layers=self.feed_forward_options.num_layers,
            activation=self.encoder_options.activation,
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

    def __layer_controller_options(
        self,
        *,
        gate_prefix: str,
        halting_prefix: str,
    ) -> LayerControllerOptions:
        return config_defaults.linears_layer_controller_options(
            config,
            gate_prefix=gate_prefix,
            gate_stack_prefix=f"{gate_prefix}_STACK",
            halting_prefix=halting_prefix,
            halting_stack_prefix=f"{halting_prefix}_STACK",
        )

    def __dynamic_memory_options(
        self,
        *,
        memory_prefix: str,
    ) -> DynamicMemoryOptions:
        return config_defaults.linears_dynamic_memory_options(
            config,
            memory_prefix=memory_prefix,
            memory_stack_prefix=f"{memory_prefix}_STACK",
        )

    def __recurrent_controller_options(
        self,
        *,
        recurrent_prefix: str,
    ) -> RecurrentControllerOptions:
        return config_defaults.linears_recurrent_controller_options(
            config,
            recurrent_prefix=recurrent_prefix,
            gate_stack_prefix=f"{recurrent_prefix}_GATE_STACK",
            halting_stack_prefix=f"{recurrent_prefix}_HALTING_STACK",
        )

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
                self.attention_projection_layer_controller_options
            ),
            attention_projection_dynamic_memory_options=(
                self.attention_projection_dynamic_memory_options
            ),
            attention_projection_recurrent_controller_options=(
                self.attention_projection_recurrent_controller_options
            ),
            feed_forward_stack_options=self.feed_forward_stack_options,
            feed_forward_layer_controller_options=(
                self.feed_forward_layer_controller_options
            ),
            feed_forward_dynamic_memory_options=(
                self.feed_forward_dynamic_memory_options
            ),
            feed_forward_recurrent_controller_options=(
                self.feed_forward_recurrent_controller_options
            ),
            stack_options=self.stack_options,
            submodule_stack_options=self.submodule_stack_options,
            layer_controller_options=self.layer_controller_options,
            dynamic_memory_options=self.dynamic_memory_options,
            recurrent_controller_options=self.recurrent_controller_options,
            linear_layer_config_factory=dependencies.linear_layer_config_factory,
            attention_projection_linear_layer_config_factory=(
                dependencies.attention_projection_linear_layer_config_factory
            ),
            feed_forward_linear_layer_config_factory=(
                dependencies.feed_forward_linear_layer_config_factory
            ),
        )
