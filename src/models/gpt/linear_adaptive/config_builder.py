from typing import TYPE_CHECKING

import models.gpt.linear_adaptive.config as config
from models.gpt.linear_adaptive._boundary_config_factory import (
    BoundaryConfigDependencies,
    BoundaryConfigFactory,
)
from models.gpt.linear_adaptive._core_config_factory import (
    CoreConfigDependencies,
    CoreConfigFactory,
)
from models.gpt.linear_adaptive._linear_layer_config_factory import (
    AdaptiveAugmentationConfigFactory,
    AdaptiveAugmentationDependencies,
    LinearLayerConfigDependencies,
    LinearLayerConfigFactory,
)
from models.gpt.linear_adaptive._positional_embedding_config_factory import (
    PositionalEmbeddingConfigDependencies,
    PositionalEmbeddingConfigFactory,
)
from models.gpt.linear_adaptive.experiment_config import ExperimentConfig
from models.gpt.linear_adaptive.runtime_defaults import DEFAULT_RUNTIME
from models.gpt.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    DynamicMemoryOptions,
    GptEmbeddingOptions,
    GptLmHeadOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    RuntimeOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class _GptLinearAdaptiveConfigBuilderImplementation:
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
        adaptive_generator_stack_options: (AdaptiveGeneratorStackOptions | None) = None,
        hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None,
        hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None,
        hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None,
        hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None,
        attention_adaptive_generator_stack_options: (
            AdaptiveGeneratorStackOptions | None
        ) = None,
        attention_hidden_adaptive_weight_options: (
            HiddenAdaptiveWeightOptions | None
        ) = None,
        attention_hidden_adaptive_bias_options: (
            HiddenAdaptiveBiasOptions | None
        ) = None,
        attention_hidden_adaptive_diagonal_options: (
            HiddenAdaptiveDiagonalOptions | None
        ) = None,
        attention_hidden_adaptive_mask_options: (
            HiddenAdaptiveMaskOptions | None
        ) = None,
        feed_forward_adaptive_generator_stack_options: (
            AdaptiveGeneratorStackOptions | None
        ) = None,
        feed_forward_hidden_adaptive_weight_options: (
            HiddenAdaptiveWeightOptions | None
        ) = None,
        feed_forward_hidden_adaptive_bias_options: (
            HiddenAdaptiveBiasOptions | None
        ) = None,
        feed_forward_hidden_adaptive_diagonal_options: (
            HiddenAdaptiveDiagonalOptions | None
        ) = None,
        feed_forward_hidden_adaptive_mask_options: (
            HiddenAdaptiveMaskOptions | None
        ) = None,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.embedding_options = embedding_options
        self.decoder_options = decoder_options
        self.hidden_dim = self.__plain_linear_layer_config_factory().hidden_dim
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
        self.adaptive_generator_stack_options = adaptive_generator_stack_options
        self.hidden_adaptive_weight_options = hidden_adaptive_weight_options
        self.hidden_adaptive_bias_options = hidden_adaptive_bias_options
        self.hidden_adaptive_diagonal_options = hidden_adaptive_diagonal_options
        self.hidden_adaptive_mask_options = hidden_adaptive_mask_options
        self.attention_adaptive_generator_stack_options = (
            attention_adaptive_generator_stack_options
        )
        self.attention_hidden_adaptive_weight_options = (
            attention_hidden_adaptive_weight_options
        )
        self.attention_hidden_adaptive_bias_options = (
            attention_hidden_adaptive_bias_options
        )
        self.attention_hidden_adaptive_diagonal_options = (
            attention_hidden_adaptive_diagonal_options
        )
        self.attention_hidden_adaptive_mask_options = (
            attention_hidden_adaptive_mask_options
        )
        self.feed_forward_adaptive_generator_stack_options = (
            feed_forward_adaptive_generator_stack_options
        )
        self.feed_forward_hidden_adaptive_weight_options = (
            feed_forward_hidden_adaptive_weight_options
        )
        self.feed_forward_hidden_adaptive_bias_options = (
            feed_forward_hidden_adaptive_bias_options
        )
        self.feed_forward_hidden_adaptive_diagonal_options = (
            feed_forward_hidden_adaptive_diagonal_options
        )
        self.feed_forward_hidden_adaptive_mask_options = (
            feed_forward_hidden_adaptive_mask_options
        )
        self.adaptive_augmentation_config = self.__adaptive_augmentation_config()
        self.attention_adaptive_augmentation_config = (
            self.__attention_adaptive_augmentation_config()
        )
        self.feed_forward_adaptive_augmentation_config = (
            self.__feed_forward_adaptive_augmentation_config()
        )

    def __plain_linear_layer_config_factory(self) -> LinearLayerConfigFactory:
        linear_layer_config_dependencies = self.__linear_layer_config_dependencies()
        return LinearLayerConfigFactory(linear_layer_config_dependencies)

    def __adaptive_augmentation_config(self):
        adaptive_dependencies = self.__adaptive_augmentation_dependencies()
        adaptive_factory = AdaptiveAugmentationConfigFactory(adaptive_dependencies)
        return adaptive_factory.build_adaptive_augmentation_config()

    def __adaptive_augmentation_dependencies(self) -> AdaptiveAugmentationDependencies:
        return AdaptiveAugmentationDependencies(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            adaptive_generator_stack_options=self.adaptive_generator_stack_options,
            hidden_adaptive_weight_options=self.hidden_adaptive_weight_options,
            hidden_adaptive_bias_options=self.hidden_adaptive_bias_options,
            hidden_adaptive_diagonal_options=self.hidden_adaptive_diagonal_options,
            hidden_adaptive_mask_options=self.hidden_adaptive_mask_options,
        )

    def __attention_adaptive_augmentation_config(self):
        adaptive_dependencies = self.__attention_adaptive_augmentation_dependencies()
        adaptive_factory = AdaptiveAugmentationConfigFactory(adaptive_dependencies)
        return adaptive_factory.build_adaptive_augmentation_config()

    def __attention_adaptive_augmentation_dependencies(
        self,
    ) -> AdaptiveAugmentationDependencies:
        return AdaptiveAugmentationDependencies(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            adaptive_generator_stack_options=(
                self.attention_adaptive_generator_stack_options
                if self.attention_adaptive_generator_stack_options is not None
                else self.adaptive_generator_stack_options
            ),
            hidden_adaptive_weight_options=(
                self.attention_hidden_adaptive_weight_options
                if self.attention_hidden_adaptive_weight_options is not None
                else self.hidden_adaptive_weight_options
            ),
            hidden_adaptive_bias_options=(
                self.attention_hidden_adaptive_bias_options
                if self.attention_hidden_adaptive_bias_options is not None
                else self.hidden_adaptive_bias_options
            ),
            hidden_adaptive_diagonal_options=(
                self.attention_hidden_adaptive_diagonal_options
                if self.attention_hidden_adaptive_diagonal_options is not None
                else self.hidden_adaptive_diagonal_options
            ),
            hidden_adaptive_mask_options=(
                self.attention_hidden_adaptive_mask_options
                if self.attention_hidden_adaptive_mask_options is not None
                else self.hidden_adaptive_mask_options
            ),
        )

    def __feed_forward_adaptive_augmentation_config(self):
        adaptive_dependencies = self.__feed_forward_adaptive_augmentation_dependencies()
        adaptive_factory = AdaptiveAugmentationConfigFactory(adaptive_dependencies)
        return adaptive_factory.build_adaptive_augmentation_config()

    def __feed_forward_adaptive_augmentation_dependencies(
        self,
    ) -> AdaptiveAugmentationDependencies:
        return AdaptiveAugmentationDependencies(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            adaptive_generator_stack_options=(
                self.feed_forward_adaptive_generator_stack_options
                if self.feed_forward_adaptive_generator_stack_options is not None
                else self.adaptive_generator_stack_options
            ),
            hidden_adaptive_weight_options=(
                self.feed_forward_hidden_adaptive_weight_options
                if self.feed_forward_hidden_adaptive_weight_options is not None
                else self.hidden_adaptive_weight_options
            ),
            hidden_adaptive_bias_options=(
                self.feed_forward_hidden_adaptive_bias_options
                if self.feed_forward_hidden_adaptive_bias_options is not None
                else self.hidden_adaptive_bias_options
            ),
            hidden_adaptive_diagonal_options=(
                self.feed_forward_hidden_adaptive_diagonal_options
                if self.feed_forward_hidden_adaptive_diagonal_options is not None
                else self.hidden_adaptive_diagonal_options
            ),
            hidden_adaptive_mask_options=(
                self.feed_forward_hidden_adaptive_mask_options
                if self.feed_forward_hidden_adaptive_mask_options is not None
                else self.hidden_adaptive_mask_options
            ),
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

    def __boundary_config(self):
        boundary_config_dependencies = self.__boundary_config_dependencies()
        boundary_config_factory = BoundaryConfigFactory(boundary_config_dependencies)
        return boundary_config_factory.build_boundary_config()

    def __boundary_config_dependencies(self) -> BoundaryConfigDependencies:
        return BoundaryConfigDependencies(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            embedding_options=self.embedding_options,
            lm_head_options=self.lm_head_options,
        )

    def __decoder_config(self):
        core_config_dependencies = self.__core_config_dependencies()
        core_config_factory = CoreConfigFactory(core_config_dependencies)
        return core_config_factory.build_decoder_config()

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
            linear_layer_config_factory=self.__plain_linear_layer_config_factory(),
            attention_projection_linear_layer_config_factory=(
                self.__adaptive_linear_layer_config_factory(
                    self.attention_adaptive_augmentation_config
                )
            ),
            feed_forward_linear_layer_config_factory=(
                self.__adaptive_linear_layer_config_factory(
                    self.feed_forward_adaptive_augmentation_config
                )
            ),
        )

    def __adaptive_linear_layer_config_factory(
        self,
        adaptive_augmentation_config,
    ) -> LinearLayerConfigFactory:
        linear_layer_config_dependencies = self.__linear_layer_config_dependencies(
            adaptive_augmentation_config=adaptive_augmentation_config
        )
        return LinearLayerConfigFactory(linear_layer_config_dependencies)

    def __linear_layer_config_dependencies(
        self,
        *,
        adaptive_augmentation_config=None,
    ) -> LinearLayerConfigDependencies:
        return LinearLayerConfigDependencies(
            decoder_options=self.decoder_options,
            adaptive_augmentation_config=adaptive_augmentation_config,
        )


class GptLinearAdaptiveConfigBuilder(_GptLinearAdaptiveConfigBuilderImplementation):
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.gpt.linear_adaptive GptLinearAdaptiveConfigBuilder runtime must be RuntimeOptions"
            )
        self.runtime = runtime
        super().__init__(**runtime._as_construction_kwargs())
