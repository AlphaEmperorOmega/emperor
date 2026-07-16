import torch
from torch.nn import Parameter

from emperor.nn._module import Module


class ParameterBank(Module):
    def __init__(
        self,
        shape: tuple,
        initializer: callable,
    ):
        super().__init__()
        self.shape = shape
        self.initializer = initializer
        self.parameter_bank = self.__create_bank()

    def __create_bank(self) -> Parameter:
        default_params = torch.randn(*self.shape)
        parameter_bank = Parameter(default_params)
        self.initializer(parameter_bank)
        return parameter_bank

    def get(self) -> Parameter:
        return self.parameter_bank
