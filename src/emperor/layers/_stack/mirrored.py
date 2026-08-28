from emperor.layers._stack.core import LayerStack


class MirroredLayerStack(LayerStack):
    _supports_rectangular_gate = True

    @property
    def depth_per_arm(self) -> int:
        return self.num_layers

    @property
    def expansion_layers(self):
        return self.layers[: self.depth_per_arm]

    @property
    def contraction_layers(self):
        return self.layers[self.depth_per_arm :]

    def _layer_dimensions(self) -> tuple[tuple[int, int], ...]:
        interior_arm_dimensions = ((self.hidden_dim, self.hidden_dim),) * (
            self.depth_per_arm - 1
        )
        expansion = (
            (self.input_dim, self.hidden_dim),
            *interior_arm_dimensions,
        )
        contraction = (
            *interior_arm_dimensions,
            (self.hidden_dim, self.output_dim),
        )
        return (*expansion, *contraction)
