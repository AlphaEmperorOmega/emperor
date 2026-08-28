from typing import cast

from emperor.layers._config import LayerStackConfig


class LayerStackTopology:
    """Resolve the ordered layer dimensions for an ordinary LayerStack."""

    def __init__(self, stack_config: LayerStackConfig) -> None:
        self.__input_dim = cast(int, stack_config.input_dim)
        self.__hidden_dim = cast(int, stack_config.hidden_dim)
        self.__output_dim = cast(int, stack_config.output_dim)
        self.__num_layers = cast(int, stack_config.num_layers)

    def resolve(self) -> tuple[tuple[int, int], ...]:
        if self.__num_layers == 1:
            return ((self.__input_dim, self.__output_dim),)

        input_projection = (
            ((self.__input_dim, self.__hidden_dim),)
            if self.__input_dim != self.__hidden_dim
            else ()
        )
        hidden_layer_count = self.__num_layers - len(input_projection) - 1
        hidden_layers = ((self.__hidden_dim, self.__hidden_dim),) * hidden_layer_count
        return (
            *input_projection,
            *hidden_layers,
            (self.__hidden_dim, self.__output_dim),
        )
