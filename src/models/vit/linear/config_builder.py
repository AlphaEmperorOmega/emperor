from typing import TYPE_CHECKING

from models.vit.linear._boundary_config_factory import (
    BoundaryConfigDependencies,
    BoundaryConfigFactory,
)
from models.vit.linear._core_config_factory import (
    CoreConfigDependencies,
    CoreConfigFactory,
)
from models.vit.linear._linear_layer_config_factory import (
    LinearLayerConfigDependencies,
    LinearLayerConfigFactory,
)
from models.vit.linear._patch_config_factory import (
    PatchConfigDependencies,
    PatchConfigFactory,
)
from models.vit.linear._positional_embedding_config_factory import (
    PositionalEmbeddingConfigDependencies,
    PositionalEmbeddingConfigFactory,
)
from models.vit.linear.experiment_config import ExperimentConfig
from models.vit.linear.runtime_defaults import DEFAULT_RUNTIME
from models.vit.linear.runtime_options import RuntimeOptions

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class VitLinearConfigBuilder:
    def __init__(
        self,
        *,
        runtime: RuntimeOptions = DEFAULT_RUNTIME,
    ) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")
        self.runtime = runtime
        self.batch_size = self.runtime.batch_size
        self.learning_rate = self.runtime.learning_rate
        self.input_dim = self.runtime.input_dim
        self.output_dim = self.runtime.output_dim
        self.patch_options = self.runtime.patch_options
        self.encoder_options = self.runtime.encoder_options
        self.hidden_dim = self.__linear_layer_config_factory().hidden_dim
        self.positional_embedding_options = self.runtime.positional_embedding_options
        self.attention_options = self.runtime.attention_options
        self.feed_forward_options = self.runtime.feed_forward_options
        self.output_options = self.runtime.output_options
        self.attention_projection_stack_options = (
            self.runtime.attention_projection_stack_options
        )
        self.attention_projection_layer_controller_options = (
            self.runtime.attention_projection_layer_controller_options
        )
        self.attention_projection_dynamic_memory_options = (
            self.runtime.attention_projection_dynamic_memory_options
        )
        self.attention_projection_recurrent_controller_options = (
            self.runtime.attention_projection_recurrent_controller_options
        )
        self.feed_forward_stack_options = self.runtime.feed_forward_stack_options
        self.feed_forward_layer_controller_options = (
            self.runtime.feed_forward_layer_controller_options
        )
        self.feed_forward_dynamic_memory_options = (
            self.runtime.feed_forward_dynamic_memory_options
        )
        self.feed_forward_recurrent_controller_options = (
            self.runtime.feed_forward_recurrent_controller_options
        )
        self.encoder_stack_options = self.runtime.stack_options
        self.encoder_submodule_stack_options = self.runtime.submodule_stack_options
        self.encoder_layer_controller_options = self.runtime.layer_controller_options
        self.encoder_dynamic_memory_options = self.runtime.dynamic_memory_options
        self.encoder_recurrent_controller_options = (
            self.runtime.recurrent_controller_options
        )
        self.sequence_length = self.__sequence_length()

    def __sequence_length(self) -> int:
        patch_config_dependencies = self.__patch_config_dependencies()
        patch_config_factory = PatchConfigFactory(patch_config_dependencies)
        return patch_config_factory.sequence_length

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
        patch_config_dependencies = self.__patch_config_dependencies()
        patch_config_factory = PatchConfigFactory(patch_config_dependencies)
        return patch_config_factory.build_patch_config()

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
        )

    def __output_config(self):
        boundary_config_dependencies = self.__boundary_config_dependencies()
        boundary_config_factory = BoundaryConfigFactory(boundary_config_dependencies)
        return boundary_config_factory.build_output_config()

    def __boundary_config_dependencies(
        self,
    ) -> BoundaryConfigDependencies:
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
