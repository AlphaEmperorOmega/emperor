from models.vit.expert_linear._control_factory_dependencies import (
    VitControlFactoryDependencies,
)
from models.vit.expert_linear._gate_config_factory import GateConfigFactory
from models.vit.expert_linear._halting_config_factory import HaltingConfigFactory
from models.vit.expert_linear._memory_config_factory import MemoryConfigFactory
from models.vit.expert_linear._recurrent_config_factory import RecurrentConfigFactory


class VitGateConfigFactory:
    def __init__(self, dependencies: VitControlFactoryDependencies) -> None:
        self.dependencies = dependencies

    def build_encoder_factory(self) -> GateConfigFactory | None:
        dependencies = self.dependencies
        if (
            dependencies.encoder_layer_controller_options is None
            or dependencies.encoder_recurrent_controller_options is None
            or dependencies.encoder_submodule_stack_options is None
        ):
            return None
        return GateConfigFactory(
            layer_controller_options=dependencies.encoder_layer_controller_options,
            recurrent_controller_options=dependencies.encoder_recurrent_controller_options,
            submodule_stack_options=dependencies.encoder_submodule_stack_options,
        )

    def build_attention_projection_factory(self) -> GateConfigFactory | None:
        dependencies = self.dependencies
        if (
            dependencies.attention_projection_layer_controller_options is None
            or dependencies.attention_projection_recurrent_controller_options is None
        ):
            return None
        return GateConfigFactory(
            layer_controller_options=dependencies.attention_projection_layer_controller_options,
            recurrent_controller_options=dependencies.attention_projection_recurrent_controller_options,
            submodule_stack_options=dependencies.attention_projection_stack_options,
            recurrent_stack_inherits_gate_stack=False,
        )

    def build_feed_forward_factory(self) -> GateConfigFactory | None:
        dependencies = self.dependencies
        if (
            dependencies.feed_forward_layer_controller_options is None
            or dependencies.feed_forward_recurrent_controller_options is None
        ):
            return None
        return GateConfigFactory(
            layer_controller_options=dependencies.feed_forward_layer_controller_options,
            recurrent_controller_options=dependencies.feed_forward_recurrent_controller_options,
            submodule_stack_options=dependencies.feed_forward_stack_options,
            recurrent_stack_inherits_gate_stack=False,
        )


class VitHaltingConfigFactory:
    def __init__(self, dependencies: VitControlFactoryDependencies) -> None:
        self.dependencies = dependencies

    def build_encoder_factory(self) -> HaltingConfigFactory | None:
        dependencies = self.dependencies
        if (
            dependencies.encoder_layer_controller_options is None
            or dependencies.encoder_recurrent_controller_options is None
            or dependencies.encoder_submodule_stack_options is None
        ):
            return None
        return HaltingConfigFactory(
            layer_controller_options=dependencies.encoder_layer_controller_options,
            recurrent_controller_options=dependencies.encoder_recurrent_controller_options,
            submodule_stack_options=dependencies.encoder_submodule_stack_options,
            output_dim=dependencies.hidden_dim,
        )

    def build_attention_projection_factory(self) -> HaltingConfigFactory | None:
        dependencies = self.dependencies
        if (
            dependencies.attention_projection_layer_controller_options is None
            or dependencies.attention_projection_recurrent_controller_options is None
        ):
            return None
        stack_options = dependencies.attention_projection_stack_options
        return HaltingConfigFactory(
            layer_controller_options=dependencies.attention_projection_layer_controller_options,
            recurrent_controller_options=dependencies.attention_projection_recurrent_controller_options,
            submodule_stack_options=stack_options,
            output_dim=dependencies.hidden_dim,
            halting_stack_defaults=stack_options,
            recurrent_stack_inherits_halting_stack=False,
        )

    def build_feed_forward_factory(self) -> HaltingConfigFactory | None:
        dependencies = self.dependencies
        if (
            dependencies.feed_forward_layer_controller_options is None
            or dependencies.feed_forward_recurrent_controller_options is None
        ):
            return None
        stack_options = dependencies.feed_forward_stack_options
        return HaltingConfigFactory(
            layer_controller_options=dependencies.feed_forward_layer_controller_options,
            recurrent_controller_options=dependencies.feed_forward_recurrent_controller_options,
            submodule_stack_options=stack_options,
            output_dim=dependencies.hidden_dim,
            halting_stack_defaults=stack_options,
            recurrent_stack_inherits_halting_stack=False,
        )


class VitMemoryConfigFactory:
    def __init__(self, dependencies: VitControlFactoryDependencies) -> None:
        self.dependencies = dependencies

    def build_encoder_factory(self) -> MemoryConfigFactory | None:
        dependencies = self.dependencies
        if (
            dependencies.encoder_dynamic_memory_options is None
            or dependencies.encoder_submodule_stack_options is None
        ):
            return None
        return MemoryConfigFactory(
            hidden_dim=dependencies.hidden_dim,
            stack_options=dependencies.encoder_stack_options,
            dynamic_memory_options=dependencies.encoder_dynamic_memory_options,
            submodule_stack_options=dependencies.encoder_submodule_stack_options,
        )

    def build_attention_projection_factory(self) -> MemoryConfigFactory | None:
        dependencies = self.dependencies
        if dependencies.attention_projection_dynamic_memory_options is None:
            return None
        return MemoryConfigFactory(
            hidden_dim=dependencies.hidden_dim,
            stack_options=dependencies.encoder_stack_options,
            dynamic_memory_options=dependencies.attention_projection_dynamic_memory_options,
            submodule_stack_options=dependencies.attention_projection_stack_options,
        )

    def build_feed_forward_factory(self) -> MemoryConfigFactory | None:
        dependencies = self.dependencies
        if dependencies.feed_forward_dynamic_memory_options is None:
            return None
        return MemoryConfigFactory(
            hidden_dim=dependencies.hidden_dim,
            stack_options=dependencies.encoder_stack_options,
            dynamic_memory_options=dependencies.feed_forward_dynamic_memory_options,
            submodule_stack_options=dependencies.feed_forward_stack_options,
        )


class VitRecurrentConfigFactory:
    def __init__(self, dependencies: VitControlFactoryDependencies) -> None:
        self.dependencies = dependencies
        self.gate_factory = VitGateConfigFactory(dependencies)
        self.halting_factory = VitHaltingConfigFactory(dependencies)

    def build_encoder_factory(self) -> RecurrentConfigFactory | None:
        dependencies = self.dependencies
        if dependencies.encoder_recurrent_controller_options is None:
            return None
        gate_factory = self.gate_factory.build_encoder_factory()
        halting_factory = self.halting_factory.build_encoder_factory()
        if gate_factory is None or halting_factory is None:
            return None
        return RecurrentConfigFactory(
            recurrent_controller_options=dependencies.encoder_recurrent_controller_options,
            gate_config_factory=gate_factory,
            halting_config_factory=halting_factory,
        )

    def build_attention_projection_factory(self) -> RecurrentConfigFactory | None:
        dependencies = self.dependencies
        if dependencies.attention_projection_recurrent_controller_options is None:
            return None
        gate_factory = self.gate_factory.build_attention_projection_factory()
        halting_factory = self.halting_factory.build_attention_projection_factory()
        if gate_factory is None or halting_factory is None:
            return None
        return RecurrentConfigFactory(
            recurrent_controller_options=dependencies.attention_projection_recurrent_controller_options,
            gate_config_factory=gate_factory,
            halting_config_factory=halting_factory,
        )

    def build_feed_forward_factory(self) -> RecurrentConfigFactory | None:
        dependencies = self.dependencies
        if dependencies.feed_forward_recurrent_controller_options is None:
            return None
        gate_factory = self.gate_factory.build_feed_forward_factory()
        halting_factory = self.halting_factory.build_feed_forward_factory()
        if gate_factory is None or halting_factory is None:
            return None
        return RecurrentConfigFactory(
            recurrent_controller_options=dependencies.feed_forward_recurrent_controller_options,
            gate_config_factory=gate_factory,
            halting_config_factory=halting_factory,
        )
