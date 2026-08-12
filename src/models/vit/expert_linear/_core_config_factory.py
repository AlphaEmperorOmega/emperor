from dataclasses import dataclass, replace

import models.vit.expert_linear.config as config
from models.vit.expert_linear import _config_defaults as config_defaults
from models.vit.expert_linear._linear_layer_config_factory import (
    LinearLayerConfigFactory,
)
from models.vit.expert_linear._vit_core_config_factory import (
    CoreConfigDependencies as _CoreDependencies,
)
from models.vit.expert_linear._vit_core_config_factory import (
    VitCoreConfigFactory,
    _VitExpertConfigFactory,
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
    expert_config_factory: _VitExpertConfigFactory | None = None


class CoreConfigFactory:
    def __init__(self, dependencies: CoreConfigDependencies) -> None:
        self.dependencies = dependencies
        self.encoder_options = (
            config_defaults.vit_encoder_options(config)
            if dependencies.encoder_options is None
            else dependencies.encoder_options
        )
        self.attention_options = (
            config_defaults.vit_attention_options(config)
            if dependencies.attention_options is None
            else dependencies.attention_options
        )
        self.feed_forward_options = (
            config_defaults.vit_feed_forward_options(config)
            if dependencies.feed_forward_options is None
            else dependencies.feed_forward_options
        )
        self.stack_options = (
            config_defaults.main_layer_stack_options(config)
            if dependencies.stack_options is None
            else dependencies.stack_options
        )
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

    def __default_attention_projection_stack_options(
        self,
        stack_options: SubmoduleStackOptions | None,
    ) -> SubmoduleStackOptions:
        if stack_options is not None:
            return stack_options
        defaults = config_defaults.linears_submodule_stack_options(
            config, config_defaults.LinearRole.ATTENTION
        )
        return replace(
            defaults,
            hidden_dim=self.encoder_options.hidden_dim,
            num_layers=self.attention_options.num_layers,
            activation=self.encoder_options.activation,
            bias_flag=self.attention_options.bias_flag,
        )

    def __default_feed_forward_stack_options(
        self,
        stack_options: SubmoduleStackOptions | None,
    ) -> SubmoduleStackOptions:
        if stack_options is not None:
            return stack_options
        defaults = config_defaults.linears_submodule_stack_options(
            config, config_defaults.LinearRole.FEED_FORWARD
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
