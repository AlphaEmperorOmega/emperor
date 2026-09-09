from emperor.layers._config import LayerStackConfig


class LayerStackTopology:
    """Resolve the ordered layer dimensions for an ordinary LayerStack."""

    def __init__(self, cfg: LayerStackConfig) -> None:
        self.cfg = cfg
        self.input_dim: int = self.cfg.input_dim
        self.hidden_dim: int = self.cfg.hidden_dim
        self.output_dim: int = self.cfg.output_dim
        self.num_layers: int = self.cfg.num_layers

    def resolve(self) -> tuple[tuple[int, int], ...]:
        if self.num_layers == 1:
            return ((self.input_dim, self.output_dim),)

        input_projection = (
            ((self.input_dim, self.hidden_dim),)
            if self.input_dim != self.hidden_dim
            else ()
        )
        hidden_layer_count = self.num_layers - len(input_projection) - 1
        hidden_layers = ((self.hidden_dim, self.hidden_dim),) * hidden_layer_count
        return (
            *input_projection,
            *hidden_layers,
            (self.hidden_dim, self.output_dim),
        )
