from typing import TYPE_CHECKING

import models.gpt.expert_linear.config as config
from models.gpt.expert_linear._boundary_config_factory import (
    BoundaryConfigDependencies,
    BoundaryConfigFactory,
)
from models.gpt.expert_linear._core_config_factory import (
    CoreConfigDependencies,
    CoreConfigFactory,
)
from models.gpt.expert_linear._expert_config_factory import (
    ExpertConfigDependencies,
    ExpertConfigFactory,
)
from models.gpt.expert_linear._linear_layer_config_factory import (
    LinearLayerConfigDependencies,
    LinearLayerConfigFactory,
)
from models.gpt.expert_linear._positional_embedding_config_factory import (
    PositionalEmbeddingConfigDependencies,
    PositionalEmbeddingConfigFactory,
)
from models.gpt.expert_linear.experiment_config import ExperimentConfig
from models.gpt.expert_linear.runtime_defaults import DEFAULT_RUNTIME
from models.gpt.expert_linear.runtime_options import RuntimeOptions

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class _GptExpertLinearConfigBuilderImplementation:
    def __init__(self, runtime: RuntimeOptions) -> None:
        options = runtime._construction_options(config)
        self.batch_size = options.batch_size
        self.learning_rate = options.learning_rate
        self.input_dim = options.input_dim
        self.output_dim = options.output_dim
        self.sequence_length = options.sequence_length
        self.embedding_options = options.embedding_options
        self.decoder_options = options.decoder_options
        self.hidden_dim = self.__linear_layer_config_factory().hidden_dim
        self.positional_embedding_options = options.positional_embedding_options
        self.attention_options = options.attention_options
        self.feed_forward_options = options.feed_forward_options
        self.lm_head_options = options.lm_head_options
        self.attention_projection_stack_options = (
            options.attention_projection_stack_options
        )
        self.attention_projection_layer_controller_options = (
            options.attention_projection_layer_controller_options
        )
        self.attention_projection_dynamic_memory_options = (
            options.attention_projection_dynamic_memory_options
        )
        self.attention_projection_recurrent_controller_options = (
            options.attention_projection_recurrent_controller_options
        )
        self.feed_forward_stack_options = options.feed_forward_stack_options
        self.feed_forward_layer_controller_options = (
            options.feed_forward_layer_controller_options
        )
        self.feed_forward_dynamic_memory_options = (
            options.feed_forward_dynamic_memory_options
        )
        self.feed_forward_recurrent_controller_options = (
            options.feed_forward_recurrent_controller_options
        )
        self.decoder_stack_options = options.stack_options
        self.decoder_submodule_stack_options = options.submodule_stack_options
        self.decoder_layer_controller_options = options.layer_controller_options
        self.decoder_dynamic_memory_options = options.dynamic_memory_options
        self.decoder_recurrent_controller_options = options.recurrent_controller_options
        self.mixture_options = options.mixture_options
        self.expert_stack_options = options.expert_stack_options
        self.sampler_options = options.sampler_options
        self.router_options = options.router_options
        self.router_stack_options = options.router_stack_options
        self.expert_layer_controller_options = options.expert_layer_controller_options
        self.expert_dynamic_memory_options = options.expert_dynamic_memory_options
        self.expert_recurrent_controller_options = (
            options.expert_recurrent_controller_options
        )
        self.expert_attention_use_kv_expert_models_flag = (
            options.expert_attention_use_kv_expert_models_flag
        )

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            experiment_config=ExperimentConfig(
                positional_embedding_config=self.__positional_embedding_config(),
                boundary_config=self.__boundary_config(),
                decoder_config=self.__decoder_config(),
            ),
        )

    def __positional_embedding_config(self):
        factory = PositionalEmbeddingConfigFactory(
            PositionalEmbeddingConfigDependencies(
                hidden_dim=self.hidden_dim,
                sequence_length=self.sequence_length,
                positional_embedding_options=self.positional_embedding_options,
            )
        )
        return factory.build_positional_embedding_config()

    def __boundary_config(self):
        factory = BoundaryConfigFactory(
            BoundaryConfigDependencies(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                sequence_length=self.sequence_length,
                embedding_options=self.embedding_options,
                lm_head_options=self.lm_head_options,
            )
        )
        return factory.build_boundary_config()

    def __decoder_config(self):
        return CoreConfigFactory(
            self.__core_config_dependencies()
        ).build_decoder_config()

    def __core_config_dependencies(self) -> CoreConfigDependencies:
        return CoreConfigDependencies(
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            decoder_options=self.decoder_options,
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
            stack_options=self.decoder_stack_options,
            submodule_stack_options=self.decoder_submodule_stack_options,
            layer_controller_options=self.decoder_layer_controller_options,
            dynamic_memory_options=self.decoder_dynamic_memory_options,
            recurrent_controller_options=self.decoder_recurrent_controller_options,
            linear_layer_config_factory=self.__linear_layer_config_factory(),
            expert_config_factory=self.__expert_config_factory(),
        )

    def __linear_layer_config_factory(self) -> LinearLayerConfigFactory:
        return LinearLayerConfigFactory(
            LinearLayerConfigDependencies(decoder_options=self.decoder_options)
        )

    def __expert_config_factory(self) -> ExpertConfigFactory:
        return ExpertConfigFactory(
            ExpertConfigDependencies(
                hidden_dim=self.hidden_dim,
                decoder_options=self.decoder_options,
                attention_options=self.attention_options,
                feed_forward_options=self.feed_forward_options,
                mixture_options=self.mixture_options,
                expert_stack_options=self.expert_stack_options,
                sampler_options=self.sampler_options,
                router_options=self.router_options,
                router_stack_options=self.router_stack_options,
                expert_layer_controller_options=(self.expert_layer_controller_options),
                expert_dynamic_memory_options=self.expert_dynamic_memory_options,
                expert_recurrent_controller_options=(
                    self.expert_recurrent_controller_options
                ),
                expert_attention_use_kv_expert_models_flag=(
                    self.expert_attention_use_kv_expert_models_flag
                ),
            )
        )


class GptExpertLinearConfigBuilder(_GptExpertLinearConfigBuilderImplementation):
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.gpt.expert_linear GptExpertLinearConfigBuilder runtime must be RuntimeOptions"
            )
        self.runtime = runtime
        super().__init__(runtime)
