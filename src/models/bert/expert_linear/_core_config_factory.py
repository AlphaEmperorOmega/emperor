from dataclasses import dataclass

import models.bert.expert_linear.config as config
from models.bert.expert_linear._bert_core_config_factory import BertCoreConfigFactory
from models.bert.expert_linear._bert_core_config_factory import (
    CoreConfigDependencies as _CoreDependencies,
)
from models.bert.expert_linear._bert_core_config_factory import (
    _BertExpertConfigFactory as _ExpertConfigFactory,
)
from models.bert.expert_linear._config_defaults import (
    attention_projection_stack_options,
    bert_attention_options,
    bert_encoder_options,
    bert_feed_forward_options,
    feed_forward_stack_options,
    main_layer_stack_options,
)
from models.bert.expert_linear._linear_layer_config_factory import (
    LinearLayerConfigFactory,
)
from models.bert.expert_linear.runtime_options import (
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
    expert_config_factory: _ExpertConfigFactory | None = None


class CoreConfigFactory:
    def __init__(self, dependencies: CoreConfigDependencies) -> None:
        self.dependencies = dependencies
        self.encoder_options = (
            bert_encoder_options(config)
            if dependencies.encoder_options is None
            else dependencies.encoder_options
        )
        self.attention_options = (
            bert_attention_options(config)
            if dependencies.attention_options is None
            else dependencies.attention_options
        )
        self.feed_forward_options = (
            bert_feed_forward_options(config)
            if dependencies.feed_forward_options is None
            else dependencies.feed_forward_options
        )
        self.stack_options = (
            main_layer_stack_options(config)
            if dependencies.stack_options is None
            else dependencies.stack_options
        )
        self.attention_projection_stack_options = (
            dependencies.attention_projection_stack_options
            or attention_projection_stack_options(
                config,
                self.encoder_options,
                self.attention_options,
            )
        )
        self.feed_forward_stack_options = (
            dependencies.feed_forward_stack_options
            or feed_forward_stack_options(
                config,
                self.encoder_options,
                self.feed_forward_options,
            )
        )

    def build_encoder_config(self):
        core_factory = BertCoreConfigFactory(self.__core_dependencies())
        return core_factory.build_encoder_config()

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
            expert_config_factory=dependencies.expert_config_factory,
        )
