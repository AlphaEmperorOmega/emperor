from torch import Tensor


class AuxiliaryLoss:
    """Resolve optional model loss into one graph-preserving scalar tensor."""

    def __init__(self, owner: str) -> None:
        self.__owner = owner

    def resolve(self, value: object | None, *, reference: Tensor) -> Tensor:
        if value is None:
            return reference.new_zeros(())
        self.__validate_scalar_tensor(value)
        return value.reshape(())

    def __validate_scalar_tensor(self, value: object) -> None:
        if not isinstance(value, Tensor) or value.numel() != 1:
            raise ValueError(
                f"{self.__owner} auxiliary loss must be a scalar tensor."
            )
