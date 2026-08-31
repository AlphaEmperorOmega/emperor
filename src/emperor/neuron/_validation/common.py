from torch import Tensor


class NeuronValidationMixin:
    @staticmethod
    def validate_integer(name: str, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be an integer, received {type(value).__name__}."
            )

    @classmethod
    def validate_positive_integer(cls, name: str, value: int) -> None:
        cls.validate_integer(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, received {value!r}.")

    @staticmethod
    def validate_tensor_rank(name: str, value, rank: int) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(
                f"{name} must be a Tensor, received {type(value).__name__}."
            )
        if value.dim() != rank:
            raise ValueError(
                f"{name} must be a {rank}D tensor, received a "
                f"{value.dim()}D tensor with shape {tuple(value.shape)}."
            )
