from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from lightning.pytorch.callbacks import Callback

from emperor.layers._monitoring.callbacks._hooks import (
    _extract_hidden_tensor,
    _install_method_replacement,
    _MethodReplacement,
    _remove_hooks,
    _restore_method_replacements,
)
from emperor.layers._monitoring.diagnostics import (
    _LayerActivationTrackingContext,
    _LayerDropoutTrackingContext,
    _LayerGateTrackingContext,
    _LayerNormTrackingContext,
    _LayerResidualTrackingContext,
)

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.hooks import RemovableHandle


class LayerControllerMonitorCallback(Callback):
    """Log activation, gate, dropout, normalization, and residual diagnostics."""

    def __init__(self, log_every_n_steps: int = 100) -> None:
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self._hooks: list[RemovableHandle] = []
        self._wrapped_methods: list[_MethodReplacement] = []
        self._hooked_gate_model_ids: set[int] = set()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from emperor.layers._layer import Layer

        self.__cleanup()
        for module_name, layer in pl_module.named_modules():
            if not isinstance(layer, Layer):
                continue
            self.__attach_gate_hook(module_name, layer, pl_module)
            self.__attach_dropout_hook(module_name, layer, pl_module)
            self.__attach_layer_norm_hook(module_name, layer, pl_module)
            if self.__should_track_activation(layer):
                self.__wrap_activation(module_name, layer, pl_module)
            self.__wrap_residual(module_name, layer, pl_module)

    @staticmethod
    def __should_track_activation(layer: Module) -> bool:
        from emperor.layers._options import ActivationOptions

        return layer.activation_function != ActivationOptions.DISABLED

    def __attach_gate_hook(
        self,
        module_name: str,
        layer: Module,
        pl_module: LightningModule,
    ) -> None:
        gate = getattr(layer, "gate_model", None)
        gate_model = getattr(gate, "model", None)
        if gate_model is None or id(gate_model) in self._hooked_gate_model_ids:
            return
        self._hooked_gate_model_ids.add(id(gate_model))
        self._hooks.append(
            gate_model.register_forward_hook(
                self.__make_gate_hook(module_name, layer, pl_module)
            )
        )

    def __attach_dropout_hook(
        self,
        module_name: str,
        layer: Module,
        pl_module: LightningModule,
    ) -> None:
        dropout_module = getattr(layer, "dropout_module", None)
        if dropout_module is not None:
            self._hooks.append(
                dropout_module.register_forward_hook(
                    self.__make_dropout_hook(module_name, pl_module)
                )
            )

    def __attach_layer_norm_hook(
        self,
        module_name: str,
        layer: Module,
        pl_module: LightningModule,
    ) -> None:
        layer_norm_module = getattr(layer, "layer_norm_module", None)
        if layer_norm_module is not None:
            self._hooks.append(
                layer_norm_module.register_forward_hook(
                    self.__make_layer_norm_hook(module_name, pl_module)
                )
            )

    def __make_gate_hook(
        self,
        module_name: str,
        layer: Module,
        pl_module: LightningModule,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def log_gate_output(
            _gate_model: Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> None:
            if not self.__should_sample(pl_module):
                return
            raw_gate_values = _extract_hidden_tensor(output)
            if raw_gate_values is None:
                return
            detached_values = raw_gate_values.detach().float()
            effective_values = self.__effective_layer_gate_values(
                layer,
                detached_values,
            )
            context = _LayerGateTrackingContext(
                pl_module=pl_module,
                module_name=module_name,
                raw_values=detached_values,
                effective_values=effective_values,
            )
            self.__track_gate_diagnostics(context)

        return log_gate_output

    def __track_gate_diagnostics(
        self,
        context: _LayerGateTrackingContext,
    ) -> None:
        self.__track_raw_gate_mean(context)
        self.__track_raw_gate_variance(context)
        self.__track_raw_gate_positive_fraction(context)
        self.__track_raw_gate_saturation_fraction(context)
        self.__track_effective_gate_mean(context)
        self.__track_effective_gate_variance(context)
        self.__track_effective_gate_positive_fraction(context)
        self.__track_effective_gate_saturation_fraction(context)

    @staticmethod
    def __track_raw_gate_mean(context: _LayerGateTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/gate/output_mean",
            context.raw_values.mean(),
        )

    @staticmethod
    def __track_raw_gate_variance(context: _LayerGateTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/gate/output_var",
            context.raw_values.var(unbiased=False),
        )

    @staticmethod
    def __track_raw_gate_positive_fraction(
        context: _LayerGateTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/gate/positive_fraction",
            (context.raw_values > 0).float().mean(),
        )

    @staticmethod
    def __track_raw_gate_saturation_fraction(
        context: _LayerGateTrackingContext,
    ) -> None:
        raw_values = context.raw_values
        context.pl_module.log(
            f"{context.module_name}/gate/saturation_fraction",
            ((raw_values < -0.99) | (raw_values > 0.99)).float().mean(),
        )

    @staticmethod
    def __effective_layer_gate_values(
        layer: Module,
        raw_gate_values: Tensor,
    ) -> Tensor | None:
        gate = getattr(layer, "gate_model", None)
        if gate is None or not hasattr(gate, "effective_values"):
            return raw_gate_values
        return gate.effective_values(raw_gate_values)

    @staticmethod
    def __track_effective_gate_mean(context: _LayerGateTrackingContext) -> None:
        if context.effective_values is None:
            return
        context.pl_module.log(
            f"{context.module_name}/gate/effective_mean",
            context.effective_values.mean(),
        )

    @staticmethod
    def __track_effective_gate_variance(context: _LayerGateTrackingContext) -> None:
        if context.effective_values is None:
            return
        context.pl_module.log(
            f"{context.module_name}/gate/effective_var",
            context.effective_values.var(unbiased=False),
        )

    @staticmethod
    def __track_effective_gate_positive_fraction(
        context: _LayerGateTrackingContext,
    ) -> None:
        if context.effective_values is None:
            return
        context.pl_module.log(
            f"{context.module_name}/gate/effective_positive_fraction",
            (context.effective_values > 0).float().mean(),
        )

    @staticmethod
    def __track_effective_gate_saturation_fraction(
        context: _LayerGateTrackingContext,
    ) -> None:
        if context.effective_values is None:
            return
        effective_values = context.effective_values
        saturation_mask = (effective_values < 0.01) | (effective_values > 0.99)
        context.pl_module.log(
            f"{context.module_name}/gate/effective_saturation_fraction",
            saturation_mask.float().mean(),
        )

    def __make_dropout_hook(
        self,
        module_name: str,
        pl_module: LightningModule,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def log_dropout_output(
            _dropout_module: Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            if not self.__should_sample(pl_module):
                return
            if (
                not inputs
                or not torch.is_tensor(inputs[0])
                or not torch.is_tensor(output)
            ):
                return
            context = _LayerDropoutTrackingContext(
                pl_module=pl_module,
                module_name=module_name,
                input_values=inputs[0].detach().float(),
                output_values=output.detach().float(),
            )
            self.__track_dropout_diagnostics(context)

        return log_dropout_output

    def __track_dropout_diagnostics(
        self,
        context: _LayerDropoutTrackingContext,
    ) -> None:
        self.__track_dropout_zero_fraction(context)
        self.__track_dropped_nonzero_fraction(context)

    @staticmethod
    def __track_dropout_zero_fraction(
        context: _LayerDropoutTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/dropout/zero_fraction",
            (context.output_values == 0.0).float().mean(),
        )

    @staticmethod
    def __track_dropped_nonzero_fraction(
        context: _LayerDropoutTrackingContext,
    ) -> None:
        nonzero_input = context.input_values != 0.0
        if nonzero_input.any():
            context.pl_module.log(
                f"{context.module_name}/dropout/dropped_nonzero_fraction",
                ((context.output_values == 0.0) & nonzero_input).float().sum()
                / nonzero_input.float().sum().clamp_min(1.0),
            )

    def __make_layer_norm_hook(
        self,
        module_name: str,
        pl_module: LightningModule,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def log_layer_norm_output(
            _layer_norm: Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            if not self.__should_sample(pl_module):
                return
            if (
                not inputs
                or not torch.is_tensor(inputs[0])
                or not torch.is_tensor(output)
            ):
                return
            context = _LayerNormTrackingContext(
                pl_module=pl_module,
                module_name=module_name,
                input_values=inputs[0].detach().float(),
                output_values=output.detach().float(),
            )
            self.__track_layer_norm_diagnostics(context)

        return log_layer_norm_output

    def __track_layer_norm_diagnostics(
        self,
        context: _LayerNormTrackingContext,
    ) -> None:
        self.__track_layer_norm_output_mean(context)
        self.__track_layer_norm_output_variance(context)
        self.__track_layer_norm_relative_delta_norm(context)

    @staticmethod
    def __track_layer_norm_output_mean(context: _LayerNormTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/layer_norm/output_mean",
            context.output_values.mean(),
        )

    @staticmethod
    def __track_layer_norm_output_variance(
        context: _LayerNormTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/layer_norm/output_var",
            context.output_values.var(unbiased=False),
        )

    @staticmethod
    def __track_layer_norm_relative_delta_norm(
        context: _LayerNormTrackingContext,
    ) -> None:
        if context.input_values.shape != context.output_values.shape:
            return
        output_delta = context.output_values - context.input_values
        context.pl_module.log(
            f"{context.module_name}/layer_norm/relative_delta_norm",
            output_delta.norm() / context.input_values.norm().clamp_min(1e-6),
        )

    def __wrap_activation(
        self,
        module_name: str,
        layer: Module,
        pl_module: LightningModule,
    ) -> None:
        method_name = "_Layer__maybe_apply_activation"
        original_activation = getattr(layer, method_name)

        def monitored_activation(*args: object, **kwargs: object) -> object:
            output = original_activation(*args, **kwargs)
            if self.__should_sample(pl_module) and torch.is_tensor(output):
                context = _LayerActivationTrackingContext(
                    pl_module=pl_module,
                    module_name=module_name,
                    activation_values=output.detach().float(),
                )
                self.__track_activation_diagnostics(context)
            return output

        _install_method_replacement(
            self._wrapped_methods,
            layer,
            method_name,
            original_activation,
            monitored_activation,
        )

    def __track_activation_diagnostics(
        self,
        context: _LayerActivationTrackingContext,
    ) -> None:
        self.__track_activation_zero_fraction(context)
        self.__track_activation_saturation_fraction(context)

    @staticmethod
    def __track_activation_zero_fraction(
        context: _LayerActivationTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/activation/zero_fraction",
            (context.activation_values == 0.0).float().mean(),
        )

    @staticmethod
    def __track_activation_saturation_fraction(
        context: _LayerActivationTrackingContext,
    ) -> None:
        activation_values = context.activation_values
        context.pl_module.log(
            f"{context.module_name}/activation/saturation_fraction",
            ((activation_values < -0.99) | (activation_values > 0.99)).float().mean(),
        )

    def __wrap_residual(
        self,
        module_name: str,
        layer: Module,
        pl_module: LightningModule,
    ) -> None:
        method_name = "_Layer__maybe_apply_residual_connection"
        if getattr(layer, "residual_connection", None) is None:
            return
        original_residual = getattr(layer, method_name)

        def monitored_residual(*args: object, **kwargs: object) -> object:
            output = original_residual(*args, **kwargs)
            input_values = args[0] if args else kwargs.get("input")
            previous_values = args[1] if len(args) > 1 else kwargs.get("prev_input")
            if self.__can_log_residual(
                pl_module,
                output,
                input_values,
                previous_values,
            ):
                context = _LayerResidualTrackingContext(
                    pl_module=pl_module,
                    module_name=module_name,
                    output_values=output.detach().float(),
                    input_values=input_values.detach().float(),
                    previous_values=previous_values.detach().float(),
                )
                self.__track_residual_diagnostics(context)
            return output

        _install_method_replacement(
            self._wrapped_methods,
            layer,
            method_name,
            original_residual,
            monitored_residual,
        )

    def __can_log_residual(
        self,
        pl_module: LightningModule,
        output: object,
        input_values: object,
        previous_values: object,
    ) -> bool:
        return (
            self.__should_sample(pl_module)
            and torch.is_tensor(output)
            and torch.is_tensor(input_values)
            and torch.is_tensor(previous_values)
            and output.shape == input_values.shape
        )

    def __track_residual_diagnostics(
        self,
        context: _LayerResidualTrackingContext,
    ) -> None:
        self.__track_residual_contribution_ratio(context)
        self.__track_residual_input_ratio(context)

    @staticmethod
    def __track_residual_contribution_ratio(
        context: _LayerResidualTrackingContext,
    ) -> None:
        residual_contribution = context.output_values - context.input_values
        context.pl_module.log(
            f"{context.module_name}/residual/contribution_ratio",
            residual_contribution.norm() / context.output_values.norm().clamp_min(1e-6),
        )

    @staticmethod
    def __track_residual_input_ratio(
        context: _LayerResidualTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/residual/input_ratio",
            context.previous_values.norm()
            / context.input_values.norm().clamp_min(1e-6),
        )

    def __should_sample(self, pl_module: LightningModule) -> bool:
        global_step = getattr(pl_module, "global_step", 0)
        return global_step % self.log_every_n_steps == 0

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__cleanup()

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        self.__cleanup()

    def __cleanup(self) -> None:
        _remove_hooks(self._hooks)
        _restore_method_replacements(self._wrapped_methods)
        self._hooked_gate_model_ids.clear()
