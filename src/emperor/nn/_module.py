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
                epoch_position = (
                    self.trainer.train_batch_idx / self.trainer.num_train_batches
                )
                batches_per_plot = (
                    self.trainer.num_train_batches / self.plot_train_per_epoch
                )
            else:
                epoch_position = self.trainer.epoch + 1
                batches_per_plot = (
                    self.trainer.num_val_batches / self.plot_valid_per_epoch
                )
            plotted_value = value.detach().cpu().numpy()
            series_label = ("train_" if train else "val_") + key
            plot_every_n_batches = max(1, int(batches_per_plot))
            self.board.draw(
                epoch_position,
                plotted_value,
                series_label,
                every_n=plot_every_n_batches,
            )

    def training_step(self, batch):
        predictions, auxiliary_loss = self(*batch[:-1])
        training_loss = self.loss(predictions, batch[-1])

        if auxiliary_loss is not None:
            training_loss += auxiliary_loss

        self.plot("loss", training_loss, train=True)
        return training_loss

    def validation_step(self, batch):
        predictions, auxiliary_loss = self(*batch[:-1])
        validation_loss = self.loss(predictions, batch[-1])
        if auxiliary_loss is not None:
            validation_loss += auxiliary_loss
        self.plot("loss", validation_loss, train=False)

    def test_step(self, batch):
        predictions, auxiliary_loss = self(*batch[:-1])
        test_loss = self.loss(predictions, batch[-1])
        if auxiliary_loss is not None:
            test_loss += auxiliary_loss

    def configure_optimizers(self):
        sgd_optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return sgd_optimizer

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)

    def _initialize_parameters(
        self, *parameters: Linear | Parameter | Sequential
    ) -> None:
        for initialization_target in parameters:
            if isinstance(initialization_target, Parameter):
                nn.init.xavier_uniform_(initialization_target)

            if isinstance(initialization_target, Linear):
                nn.init.xavier_uniform_(initialization_target.weight)
                if initialization_target.bias is not None:
                    nn.init.zeros_(initialization_target.bias)

            if isinstance(initialization_target, Sequential):
                for layer in initialization_target:
                    self._initialize_parameters(layer)

    def _override_config(
        self,
        cfg: "ConfigBase | ModelConfig",
        overrides: "ConfigBase | None" = None,
    ) -> "ConfigBase":
        if overrides is None:
            return cfg

        overridden_config = copy.deepcopy(cfg)
        for config_field_name in overridden_config.__dataclass_fields__:
            if (
                hasattr(overrides, "__dataclass_fields__")
                and config_field_name in overrides.__dataclass_fields__
            ):
                if getattr(overrides, config_field_name) is not None:
                    setattr(
                        overridden_config,
                        config_field_name,
                        getattr(overrides, config_field_name),
                    )

        return overridden_config

    def _resolve_config_overrides(
        self,
        config: "ConfigBase",
        **kwargs,
    ) -> "ConfigBase":
        declared_field_names = {config_field.name for config_field in fields(config)}
        applicable_override_kwargs = {
            config_field_name: override_value
            for config_field_name, override_value in kwargs.items()
            if config_field_name in declared_field_names
        }
        config_overrides = type(config)(**applicable_override_kwargs)
        return config_overrides

    def _build_from_config(
        self,
        config: "ConfigBase | None",
        **kwargs,
    ) -> "Module | None":
        if config is None:
            return None
        config_overrides = self._resolve_config_overrides(config, **kwargs)
        built_module = config.build(overrides=config_overrides)
        return built_module

    def _resolve_main_config(
        self, sub_config: "ConfigBase", main_cfg: "ConfigBase"
    ) -> "ConfigBase":
        sub_config_override = getattr(sub_config, "override_config", None)
        if sub_config_override is not None:
            return sub_config_override
        return main_cfg

    def _init_parameter_bank(
        self,
        parameter_shape: tuple,
        initializer: callable = None,
    ) -> Parameter:
        from emperor.nn._parameters import ParameterBank

        # TODO: Ensure you have the option to initialize the biases with
        # as a zero zensor.
        resolved_initializer = (
            initializer if initializer is not None else self._initialize_parameters
        )
        parameter_bank = ParameterBank(parameter_shape, resolved_initializer)
        initialized_parameter = parameter_bank.get()
        return initialized_parameter

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
