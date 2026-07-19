from dataclasses import dataclass

from models.vit.expert_linear.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
)


@dataclass(frozen=True)
class VitControlFactoryDependencies:
    hidden_dim: int
    encoder_stack_options: MainLayerStackOptions
    encoder_submodule_stack_options: SubmoduleStackOptions | None
    attention_projection_stack_options: SubmoduleStackOptions
    feed_forward_stack_options: SubmoduleStackOptions
    encoder_layer_controller_options: LayerControllerOptions | None
    attention_projection_layer_controller_options: LayerControllerOptions | None
    feed_forward_layer_controller_options: LayerControllerOptions | None
    encoder_dynamic_memory_options: DynamicMemoryOptions | None
    attention_projection_dynamic_memory_options: DynamicMemoryOptions | None
    feed_forward_dynamic_memory_options: DynamicMemoryOptions | None
    encoder_recurrent_controller_options: RecurrentControllerOptions | None
    attention_projection_recurrent_controller_options: RecurrentControllerOptions | None
    feed_forward_recurrent_controller_options: RecurrentControllerOptions | None
