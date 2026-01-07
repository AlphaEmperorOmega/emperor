from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.transformer.utils.feed_forward import FeedForward


class FeedForwardValidator:
    def __init__(self, model: "FeedForward"):
        self.model = model
        self.__ensure_valid_number_of_layers()

    def __ensure_valid_number_of_layers(self) -> None:
        if not (self.model.num_layers >= 2 and self.model.num_layers % 2 == 0):
            raise RuntimeError(
                "The Transformer FeedForward module requires at least 2 layers, and the number of layers is even."
            )
