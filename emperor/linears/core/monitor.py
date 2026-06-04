from lightning.pytorch.callbacks import Callback
from emperor.linears.core.layers import LinearAbstract

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning import Trainer, LightningModule


class LinearMonitorCallback(Callback):
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self._hooks = []
        self._linear_modules = []

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
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
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._linear_modules.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        self.__log_weight_bias_parameter_stats(pl_module)
        self.__log_weight_bias_gradient_stats(pl_module)

    def __log_weight_bias_parameter_stats(self, module: "LightningModule") -> None:
        for name, layer in self._linear_modules:
            w = layer.weight_params.detach().float()
            module.log(f"{name}/weights/mean", w.mean())
            module.log(f"{name}/weights/var", w.var(unbiased=False))
            module.log(f"{name}/weights/l2_norm", w.norm())

            if layer.bias_params is not None:
                b = layer.bias_params.detach().float()
                module.log(f"{name}/bias/mean", b.mean())
                module.log(f"{name}/bias/var", b.var(unbiased=False))
                module.log(f"{name}/bias/l2_norm", b.norm())

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
