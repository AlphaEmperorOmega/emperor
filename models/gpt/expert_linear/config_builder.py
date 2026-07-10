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
from models.gpt.expert_linear.runtime_options import (
    DynamicMemoryOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsSubmoduleStackOptions,
    GptEmbeddingOptions,
    GptLmHeadOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class GptExpertLinearConfigBuilder:
    def __init__(
        self,
        *,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        output_dim: int = config.OUTPUT_DIM,
        sequence_length: int = config.SEQUENCE_LENGTH,
        embedding_options: GptEmbeddingOptions | None = None,
        decoder_options: TransformerDecoderOptions | None = None,
        positional_embedding_options: (
            TransformerPositionalEmbeddingOptions | None
        ) = None,
        attention_options: TransformerAttentionOptions | None = None,
        feed_forward_options: TransformerFeedForwardOptions | None = None,
        lm_head_options: GptLmHeadOptions | None = None,
        attention_projection_stack_options: SubmoduleStackOptions | None = None,
        attention_projection_layer_controller_options: (
            LayerControllerOptions | None
        ) = None,
        attention_projection_dynamic_memory_options: (
            DynamicMemoryOptions | None
        ) = None,
        attention_projection_recurrent_controller_options: (
            RecurrentControllerOptions | None
        ) = None,
        feed_forward_stack_options: SubmoduleStackOptions | None = None,
        feed_forward_layer_controller_options: LayerControllerOptions | None = None,
        feed_forward_dynamic_memory_options: DynamicMemoryOptions | None = None,
        feed_forward_recurrent_controller_options: (
            RecurrentControllerOptions | None
        ) = None,
        stack_options: MainLayerStackOptions | None = None,
        submodule_stack_options: SubmoduleStackOptions | None = None,
        layer_controller_options: LayerControllerOptions | None = None,
        dynamic_memory_options: DynamicMemoryOptions | None = None,
        recurrent_controller_options: RecurrentControllerOptions | None = None,
        mixture_options: ExpertsMixtureOptions | None = None,
        expert_stack_options: ExpertsSubmoduleStackOptions | None = None,
        sampler_options: ExpertsSamplerOptions | None = None,
        router_options: ExpertsRouterOptions | None = None,
        router_stack_options: ExpertsSubmoduleStackOptions | None = None,
        expert_layer_controller_options: ExpertsLayerControllerOptions | None = None,
        expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        expert_recurrent_controller_options: (
            ExpertsRecurrentControllerOptions | None
        ) = None,
        expert_attention_use_kv_expert_models_flag: bool = (
            config.EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG
        ),
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.embedding_options = embedding_options
        self.decoder_options = decoder_options
        self.hidden_dim = self.__linear_layer_config_factory().hidden_dim
        self.positional_embedding_options = positional_embedding_options
        self.attention_options = attention_options
        self.feed_forward_options = feed_forward_options
        self.lm_head_options = lm_head_options
        self.attention_projection_stack_options = attention_projection_stack_options
        self.attention_projection_layer_controller_options = (
            attention_projection_layer_controller_options
        )
        self.attention_projection_dynamic_memory_options = (
            attention_projection_dynamic_memory_options
        )
        self.attention_projection_recurrent_controller_options = (
            attention_projection_recurrent_controller_options
        )
        self.feed_forward_stack_options = feed_forward_stack_options
        self.feed_forward_layer_controller_options = (
            feed_forward_layer_controller_options
        )
        self.feed_forward_dynamic_memory_options = feed_forward_dynamic_memory_options
        self.feed_forward_recurrent_controller_options = (
            feed_forward_recurrent_controller_options
        )
        self.decoder_stack_options = stack_options
        self.decoder_submodule_stack_options = submodule_stack_options
        self.decoder_layer_controller_options = layer_controller_options
        self.decoder_dynamic_memory_options = dynamic_memory_options
        self.decoder_recurrent_controller_options = recurrent_controller_options
        self.mixture_options = mixture_options
        self.expert_stack_options = expert_stack_options
        self.sampler_options = sampler_options
        self.router_options = router_options
        self.router_stack_options = router_stack_options
        self.expert_layer_controller_options = expert_layer_controller_options
        self.expert_dynamic_memory_options = expert_dynamic_memory_options
        self.expert_recurrent_controller_options = expert_recurrent_controller_options
        self.expert_attention_use_kv_expert_models_flag = (
            expert_attention_use_kv_expert_models_flag
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
