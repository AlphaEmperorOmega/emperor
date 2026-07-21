from emperor.layers._stack import LayerStack


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

    def _layer_dimensions(self) -> list[tuple[int, int]]:
        expansion = [(self.input_dim, self.hidden_dim)]
        for _ in range(self.depth_per_arm - 1):
            expansion.append((self.hidden_dim, self.hidden_dim))

        contraction = []
        for _ in range(self.depth_per_arm - 1):
            contraction.append((self.hidden_dim, self.hidden_dim))
        contraction.append((self.hidden_dim, self.output_dim))
        mirrored_layer_dimensions = [*expansion, *contraction]
        return mirrored_layer_dimensions
