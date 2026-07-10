from dataclasses import dataclass

from models.gpt.linear_adaptive.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
)


@dataclass(frozen=True)
class GptControlFactoryDependencies:
    hidden_dim: int
    decoder_stack_options: MainLayerStackOptions
    decoder_submodule_stack_options: SubmoduleStackOptions | None
    attention_projection_stack_options: SubmoduleStackOptions
    feed_forward_stack_options: SubmoduleStackOptions
    decoder_layer_controller_options: LayerControllerOptions | None
    attention_projection_layer_controller_options: LayerControllerOptions | None
    feed_forward_layer_controller_options: LayerControllerOptions | None
    decoder_dynamic_memory_options: DynamicMemoryOptions | None
    attention_projection_dynamic_memory_options: DynamicMemoryOptions | None
    feed_forward_dynamic_memory_options: DynamicMemoryOptions | None
    decoder_recurrent_controller_options: RecurrentControllerOptions | None
    attention_projection_recurrent_controller_options: RecurrentControllerOptions | None
    feed_forward_recurrent_controller_options: RecurrentControllerOptions | None
