import torch

from lightning.pytorch.callbacks import Callback
from emperor.linears.core.layers import LinearAbstract

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from lightning import Trainer, LightningModule


class LinearMonitorCallback(Callback):
    DEAD_FEATURE_RELATIVE_FLOOR = 1e-3

    def __init__(
        self,
        log_every_n_steps: int = 100,
        log_weight_conditioning: bool = True,
    ):
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self.log_weight_conditioning = log_weight_conditioning
        self._hooks = []
        self._linear_modules = []
        self._parameter_snapshots = {}

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.__remove_hooks()
        self._linear_modules.clear()
        self._parameter_snapshots.clear()
        for name, module in pl_module.named_modules():
            if isinstance(module, LinearAbstract):
                self._linear_modules.append((name, module))
                hook = module.register_forward_hook(
                    self.__make_forward_stats_hook(name, pl_module)
                )
                self._hooks.append(hook)

    def __make_forward_stats_hook(self, name: str, module: "LightningModule"):
        log_every_n_steps = self.log_every_n_steps

        def hook(layer, input, output):
            step = module.global_step
            if step % log_every_n_steps != 0:
                return
            inp = input[0].detach().float()
            out = output.detach().float()
            module.log(f"{name}/input/mean", inp.mean())
            module.log(f"{name}/input/var", inp.var(unbiased=False))
            module.log(f"{name}/output/mean", out.mean())
            module.log(f"{name}/output/var", out.var(unbiased=False))

        return hook

    def on_fit_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.__remove_hooks()
        self._linear_modules.clear()
        self._parameter_snapshots.clear()

    def __remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        self.__log_weight_bias_parameter_stats(pl_module)
        self.__log_weight_bias_gradient_stats(pl_module)
        self.__log_weight_matrix_health(pl_module)

    def __log_weight_bias_parameter_stats(self, module: "LightningModule") -> None:
        for name, layer in self._linear_modules:
            w = layer.weight_params.detach().float()
            module.log(f"{name}/weights/mean", w.mean())
            module.log(f"{name}/weights/var", w.var(unbiased=False))
            module.log(f"{name}/weights/l2_norm", w.norm())
            self.__log_parameter_delta(
                module,
                name=name,
                channel="weights",
                current=w,
            )

            if layer.bias_params is not None:
                b = layer.bias_params.detach().float()
                module.log(f"{name}/bias/mean", b.mean())
                module.log(f"{name}/bias/var", b.var(unbiased=False))
                module.log(f"{name}/bias/l2_norm", b.norm())
                self.__log_parameter_delta(
                    module,
                    name=name,
                    channel="bias",
                    current=b,
                )
            else:
                self._parameter_snapshots.pop((name, "bias"), None)

    def __log_parameter_delta(
        self,
        module: "LightningModule",
        *,
        name: str,
        channel: str,
        current,
    ) -> None:
        snapshot_key = (name, channel)
        previous = self._parameter_snapshots.get(snapshot_key)
        if previous is not None and previous.shape == current.shape:
            delta_norm = (current - previous).norm()
            relative_delta_norm = delta_norm / previous.norm().clamp_min(1e-12)
            module.log(f"{name}/{channel}/delta_norm", delta_norm)
            module.log(f"{name}/{channel}/relative_delta_norm", relative_delta_norm)
        self._parameter_snapshots[snapshot_key] = current.detach().clone()

    def __log_weight_bias_gradient_stats(self, module: "LightningModule") -> None:
        for name, layer in self._linear_modules:
            w = layer.weight_params
            if w.grad is not None:
                g = w.grad.detach().float()
                module.log(f"{name}/weights/grad_mean", g.mean())
                module.log(f"{name}/weights/grad_var", g.var(unbiased=False))
                module.log(f"{name}/weights/grad_norm", g.norm())
                module.log(
                    f"{name}/weights/update_ratio",
                    g.norm() / w.detach().float().norm().clamp_min(1e-6),
                )

            if layer.bias_params is not None and layer.bias_params.grad is not None:
                bg = layer.bias_params.grad.detach().float()
                module.log(f"{name}/bias/grad_mean", bg.mean())
                module.log(f"{name}/bias/grad_var", bg.var(unbiased=False))
                module.log(f"{name}/bias/grad_norm", bg.norm())

    def __log_weight_matrix_health(self, module: "LightningModule") -> None:
        for name, layer in self._linear_modules:
            weight = layer.weight_params.detach().float()
            self.__log_dead_feature_fractions(module, name, weight)
            if self.log_weight_conditioning:
                self.__log_weight_conditioning(module, name, weight)

    def __log_dead_feature_fractions(
        self,
        module: "LightningModule",
        name: str,
        weight: "Tensor",
    ) -> None:
        input_feature_norms = weight.norm(dim=1)
        output_feature_norms = weight.norm(dim=0)
        module.log(
            f"{name}/weights/dead_input_fraction",
            self.__dead_feature_fraction(input_feature_norms),
        )
        module.log(
            f"{name}/weights/dead_output_fraction",
            self.__dead_feature_fraction(output_feature_norms),
        )

    def __dead_feature_fraction(self, feature_norms: "Tensor") -> "Tensor":
        dead_threshold = self.DEAD_FEATURE_RELATIVE_FLOOR * feature_norms.mean()
        return (feature_norms <= dead_threshold).float().mean()

    def __log_weight_conditioning(
        self,
        module: "LightningModule",
        name: str,
        weight: "Tensor",
    ) -> None:
        singular_values = torch.linalg.svdvals(weight)
        spectral_norm = singular_values.max()
        condition_number = spectral_norm / singular_values.min().clamp_min(1e-12)
        normalized_spectrum = singular_values / singular_values.sum().clamp_min(1e-12)
        spectral_entropy = -(
            normalized_spectrum.clamp_min(1e-12).log() * normalized_spectrum
        ).sum()
        module.log(f"{name}/weights/spectral_norm", spectral_norm)
        module.log(f"{name}/weights/condition_number", condition_number)
        module.log(f"{name}/weights/effective_rank", spectral_entropy.exp())
