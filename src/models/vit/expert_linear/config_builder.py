from typing import TYPE_CHECKING

import models.vit.expert_linear.config as config
from models.vit.expert_linear._boundary_config_factory import (
    BoundaryConfigDependencies,
    BoundaryConfigFactory,
)
from models.vit.expert_linear._core_config_factory import (
    CoreConfigDependencies,
    CoreConfigFactory,
)
from models.vit.expert_linear._expert_config_factory import (
    ExpertConfigDependencies,
    ExpertConfigFactory,
)
from models.vit.expert_linear._linear_layer_config_factory import (
    LinearLayerConfigDependencies,
    LinearLayerConfigFactory,
)
from models.vit.expert_linear._patch_config_factory import (
    PatchConfigDependencies,
    PatchConfigFactory,
)
from models.vit.expert_linear._positional_embedding_config_factory import (
    PositionalEmbeddingConfigDependencies,
    PositionalEmbeddingConfigFactory,
)
from models.vit.expert_linear.experiment_config import ExperimentConfig
from models.vit.expert_linear.runtime_options import (
    DynamicMemoryOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsSubmoduleStackOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
    VitOutputOptions,
    VitPatchOptions,
)

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class VitExpertLinearConfigBuilder:
    def __init__(
        self,
        *,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        output_dim: int = config.OUTPUT_DIM,
        patch_options: VitPatchOptions | None = None,
        encoder_options: TransformerEncoderOptions | None = None,
        positional_embedding_options: (
            TransformerPositionalEmbeddingOptions | None
        ) = None,
        attention_options: TransformerAttentionOptions | None = None,
        feed_forward_options: TransformerFeedForwardOptions | None = None,
        output_options: VitOutputOptions | None = None,
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
        feed_forward_layer_controller_options: LayerControllerOptions | None = (None),
        feed_forward_dynamic_memory_options: DynamicMemoryOptions | None = (None),
        feed_forward_recurrent_controller_options: (
            RecurrentControllerOptions | None
        ) = None,
        stack_options: MainLayerStackOptions | None = None,
        submodule_stack_options: SubmoduleStackOptions | None = None,
        layer_controller_options: LayerControllerOptions | None = (None),
        dynamic_memory_options: DynamicMemoryOptions | None = (None),
        recurrent_controller_options: RecurrentControllerOptions | None = (None),
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
        self.patch_options = patch_options
        self.encoder_options = encoder_options
        self.hidden_dim = self.__linear_layer_config_factory().hidden_dim
        self.positional_embedding_options = positional_embedding_options
        self.attention_options = attention_options
        self.feed_forward_options = feed_forward_options
        self.output_options = output_options
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
        self.encoder_stack_options = stack_options
        self.encoder_submodule_stack_options = submodule_stack_options
        self.encoder_layer_controller_options = layer_controller_options
        self.encoder_dynamic_memory_options = dynamic_memory_options
        self.encoder_recurrent_controller_options = recurrent_controller_options
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
        self.sequence_length = self.__patch_config_factory().sequence_length

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
                patch_config=self.__patch_config(),
                positional_embedding_config=self.__positional_embedding_config(),
                encoder_config=self.__encoder_config(),
                output_config=self.__output_config(),
            ),
        )

    def __patch_config(self):
        return self.__patch_config_factory().build_patch_config()

    def __patch_config_factory(self) -> PatchConfigFactory:
        return PatchConfigFactory(self.__patch_config_dependencies())

    def __patch_config_dependencies(self) -> PatchConfigDependencies:
        return PatchConfigDependencies(
            hidden_dim=self.hidden_dim,
            patch_options=self.patch_options,
            encoder_options=self.encoder_options,
            linear_layer_config_factory=self.__linear_layer_config_factory(),
        )

    def __positional_embedding_config(self):
        positional_embedding_config_dependencies = (
            self.__positional_embedding_config_dependencies()
        )
        positional_embedding_config_factory = PositionalEmbeddingConfigFactory(
            positional_embedding_config_dependencies
        )
        return positional_embedding_config_factory.build_positional_embedding_config()

    def __positional_embedding_config_dependencies(
        self,
    ) -> PositionalEmbeddingConfigDependencies:
        return PositionalEmbeddingConfigDependencies(
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length,
            positional_embedding_options=self.positional_embedding_options,
        )

    def __encoder_config(self):
        core_config_dependencies = self.__core_config_dependencies()
        core_config_factory = CoreConfigFactory(core_config_dependencies)
        return core_config_factory.build_encoder_config()

    def __core_config_dependencies(self) -> CoreConfigDependencies:
        return CoreConfigDependencies(
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            encoder_options=self.encoder_options,
            attention_options=self.attention_options,
            feed_forward_options=self.feed_forward_options,
            attention_projection_stack_options=self.attention_projection_stack_options,
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
            stack_options=self.encoder_stack_options,
            submodule_stack_options=self.encoder_submodule_stack_options,
            layer_controller_options=self.encoder_layer_controller_options,
            dynamic_memory_options=self.encoder_dynamic_memory_options,
            recurrent_controller_options=self.encoder_recurrent_controller_options,
            linear_layer_config_factory=self.__linear_layer_config_factory(),
            expert_config_factory=self.__expert_config_factory(),
        )

    def __output_config(self):
        boundary_config_dependencies = self.__boundary_config_dependencies()
        boundary_config_factory = BoundaryConfigFactory(boundary_config_dependencies)
        return boundary_config_factory.build_output_config()

    def __boundary_config_dependencies(self) -> BoundaryConfigDependencies:
        return BoundaryConfigDependencies(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            output_options=self.output_options,
        )

    def __linear_layer_config_factory(self) -> LinearLayerConfigFactory:
        linear_layer_config_dependencies = self.__linear_layer_config_dependencies()
        return LinearLayerConfigFactory(linear_layer_config_dependencies)

    def __linear_layer_config_dependencies(
        self,
    ) -> LinearLayerConfigDependencies:
        return LinearLayerConfigDependencies(
            encoder_options=self.encoder_options,
        )

    def __expert_config_factory(self) -> ExpertConfigFactory:
        expert_config_dependencies = self.__expert_config_dependencies()
        return ExpertConfigFactory(expert_config_dependencies)

    def __expert_config_dependencies(self) -> ExpertConfigDependencies:
        return ExpertConfigDependencies(
            hidden_dim=self.hidden_dim,
            encoder_options=self.encoder_options,
            attention_options=self.attention_options,
            feed_forward_options=self.feed_forward_options,
            mixture_options=self.mixture_options,
            expert_stack_options=self.expert_stack_options,
            sampler_options=self.sampler_options,
            router_options=self.router_options,
            router_stack_options=self.router_stack_options,
            expert_layer_controller_options=self.expert_layer_controller_options,
            expert_dynamic_memory_options=self.expert_dynamic_memory_options,
            expert_recurrent_controller_options=(
                self.expert_recurrent_controller_options
            ),
            expert_attention_use_kv_expert_models_flag=(
                self.expert_attention_use_kv_expert_models_flag
            ),
        )
