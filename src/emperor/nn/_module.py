import copy
from dataclasses import fields
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from lightning import LightningModule
from torch.nn import Linear, Parameter, Sequential

from emperor.config import ConfigBase
from emperor.nn._visualization import ProgressBoard

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class Module(LightningModule):
    """The base class of models."""

    def __init__(
        self, plot_train_per_epoch=2, plot_valid_per_epoch=1, plotProgress=True
    ):
        super().__init__()
        self.plot_train_per_epoch = plot_train_per_epoch
        self.plot_valid_per_epoch = plot_valid_per_epoch
        self.plotProgress = plotProgress
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, "net"), "Neural network is defined"
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        if not isinstance(train, bool):
            raise TypeError(f"train must be bool, got {type(train).__name__}")
        if self.plotProgress:
            self.board.xlabel = "epoch"
            if train:
                x = self.trainer.train_batch_idx / self.trainer.num_train_batches
                n = self.trainer.num_train_batches / self.plot_train_per_epoch
            else:
                x = self.trainer.epoch + 1
                n = self.trainer.num_val_batches / self.plot_valid_per_epoch
            self.board.draw(
                x,
                value.detach().cpu().numpy(),
                ("train_" if train else "val_") + key,
                every_n=max(1, int(n)),
            )

    def training_step(self, batch):
        modelOutput, auxilary_loss = self(*batch[:-1])
        loss = self.loss(modelOutput, batch[-1])

        if auxilary_loss is not None:
            loss += auxilary_loss

        self.plot("loss", loss, train=True)
        return loss

    def validation_step(self, batch):
        modelOutput, auxilaryLoss = self(*batch[:-1])
        loss = self.loss(modelOutput, batch[-1])
        if auxilaryLoss is not None:
            loss += auxilaryLoss
        self.plot("loss", loss, train=False)

    def test_step(self, batch):
        modelOutput, auxilaryLoss = self(*batch[:-1])
        loss = self.loss(modelOutput, batch[-1])
        if auxilaryLoss is not None:
            loss += auxilaryLoss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)

    def _initialize_parameters(
        self, *parameters: Linear | Parameter | Sequential
    ) -> None:
        for parameter in parameters:
            if isinstance(parameter, Parameter):
                nn.init.xavier_uniform_(parameter)

            if isinstance(parameter, Linear):
                nn.init.xavier_uniform_(parameter.weight)
                if parameter.bias is not None:
                    nn.init.zeros_(parameter.bias)

            if isinstance(parameter, Sequential):
                for layer in parameter:
                    self._initialize_parameters(layer)

    def _override_config(
        self,
        cfg: "ConfigBase | ModelConfig",
        overrides: "ConfigBase | None" = None,
    ) -> "ConfigBase":
        if overrides is None:
            return cfg

        cfg = copy.deepcopy(cfg)
        for value in cfg.__dataclass_fields__:
            if (
                hasattr(overrides, "__dataclass_fields__")
                and value in overrides.__dataclass_fields__
            ):
                if getattr(overrides, value) is not None:
                    setattr(cfg, value, getattr(overrides, value))

        return cfg

    def _resolve_config_overrides(
        self,
        config: "ConfigBase",
        **kwargs,
    ) -> "ConfigBase":
        declared_fields = {field.name for field in fields(config)}
        override_kwargs = {
            name: value for name, value in kwargs.items() if name in declared_fields
        }
        return type(config)(**override_kwargs)

    def _build_from_config(
        self,
        config: "ConfigBase | None",
        **kwargs,
    ) -> "Module | None":
        if config is None:
            return None
        return config.build(overrides=self._resolve_config_overrides(config, **kwargs))

    def _resolve_main_config(
        self, sub_config: "ConfigBase", main_cfg: "ConfigBase"
    ) -> "ConfigBase":
        override = getattr(sub_config, "override_config", None)
        if override is not None:
            return override
        return main_cfg

    def _init_parameter_bank(
        self,
        parameter_shape: tuple,
        initializer: callable = None,
    ) -> Parameter:
        from emperor.nn._parameters import ParameterBank

        # TODO: Ensure you have the option to initialize the biases with
        # as a zero zensor.
        initializer = (
            initializer if initializer is not None else self._initialize_parameters
        )
        bank = ParameterBank(parameter_shape, initializer)
        return bank.get()

    def construct(
        self,
        class_type: type | None,
        cfg: "ConfigBase | None" = None,
    ) -> object | None:
        if class_type is None:
            return None
        if cfg is None:
            return class_type()
        return class_type(cfg)
