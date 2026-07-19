from typing import TYPE_CHECKING, Any

from models.bert.linear._boundary_config_factory import (
    BoundaryConfigDependencies,
    BoundaryConfigFactory,
)
from models.bert.linear._core_config_factory import (
    CoreConfigDependencies,
    CoreConfigFactory,
)
from models.bert.linear._linear_layer_config_factory import (
    LinearLayerConfigDependencies,
    LinearLayerConfigFactory,
)
from models.bert.linear._positional_embedding_config_factory import (
    PositionalEmbeddingConfigDependencies,
    PositionalEmbeddingConfigFactory,
)
from models.bert.linear.experiment_config import ExperimentConfig
from models.bert.linear.runtime_defaults import (
    DEFAULT_RUNTIME,
    runtime_with_builder_options,
)
from models.bert.linear.runtime_options import RuntimeOptions

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class BertLinearConfigBuilder:
    def __init__(
        self,
        *,
        runtime: RuntimeOptions = DEFAULT_RUNTIME,
        **legacy_options: Any,
    ) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")
        self.runtime = runtime_with_builder_options(runtime, legacy_options)
        batch_size = self.runtime.batch_size
        learning_rate = self.runtime.learning_rate
        input_dim = self.runtime.input_dim
        output_dim = self.runtime.output_dim
        sequence_length = self.runtime.sequence_length
        embedding_options = self.runtime.embedding_options
        encoder_options = self.runtime.encoder_options
        positional_embedding_options = self.runtime.positional_embedding_options
        attention_options = self.runtime.attention_options
        feed_forward_options = self.runtime.feed_forward_options
        mlm_head_options = self.runtime.mlm_head_options
        nsp_head_options = self.runtime.nsp_head_options
        attention_projection_stack_options = (
            self.runtime.attention_projection_stack_options
        )
        attention_projection_layer_controller_options = (
            self.runtime.attention_projection_layer_controller_options
        )
        attention_projection_dynamic_memory_options = (
            self.runtime.attention_projection_dynamic_memory_options
        )
        attention_projection_recurrent_controller_options = (
            self.runtime.attention_projection_recurrent_controller_options
        )
        feed_forward_stack_options = self.runtime.feed_forward_stack_options
        feed_forward_layer_controller_options = (
            self.runtime.feed_forward_layer_controller_options
        )
        feed_forward_dynamic_memory_options = (
            self.runtime.feed_forward_dynamic_memory_options
        )
        feed_forward_recurrent_controller_options = (
            self.runtime.feed_forward_recurrent_controller_options
        )
        stack_options = self.runtime.stack_options
        submodule_stack_options = self.runtime.submodule_stack_options
        layer_controller_options = self.runtime.layer_controller_options
        dynamic_memory_options = self.runtime.dynamic_memory_options
        recurrent_controller_options = self.runtime.recurrent_controller_options
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.embedding_options = embedding_options
        self.encoder_options = encoder_options
        self.hidden_dim = self.__linear_layer_config_factory().hidden_dim
        self.positional_embedding_options = positional_embedding_options
        self.attention_options = attention_options
        self.feed_forward_options = feed_forward_options
        self.mlm_head_options = mlm_head_options
        self.nsp_head_options = nsp_head_options
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
                encoder_config=self.__encoder_config(),
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
            mlm_head_options=self.mlm_head_options,
            nsp_head_options=self.nsp_head_options,
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
            stack_options=self.encoder_stack_options,
            submodule_stack_options=self.encoder_submodule_stack_options,
            layer_controller_options=self.encoder_layer_controller_options,
            dynamic_memory_options=self.encoder_dynamic_memory_options,
            recurrent_controller_options=self.encoder_recurrent_controller_options,
            linear_layer_config_factory=self.__linear_layer_config_factory(),
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
