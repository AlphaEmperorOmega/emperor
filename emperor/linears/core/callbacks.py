from lightning.pytorch.callbacks import Callback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning import Trainer, LightningModule


class LinearMonitorCallback(Callback):
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._hooks = []

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        from emperor.linears.core.layers import LinearBase

        for name, module in pl_module.named_modules():
            if isinstance(module, LinearBase):
                hook = module.register_forward_hook(
                    self._make_data_hook(name, pl_module)
                )
                self._hooks.append(hook)

    def on_fit_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        from emperor.linears.core.layers import LinearBase

        if batch_idx % self.log_every_n_steps != 0:
            return

        for name, module in pl_module.named_modules():
            if not isinstance(module, LinearBase):
                continue

            w = module.weight_params.detach()
            pl_module.log(f"{name}/weights/mean", w.mean())
            pl_module.log(f"{name}/weights/var", w.var())
            pl_module.log(f"{name}/weights/l2_norm", w.norm())

            if module.bias_params is not None:
                b = module.bias_params.detach()
                pl_module.log(f"{name}/bias/mean", b.mean())
                pl_module.log(f"{name}/bias/l2_norm", b.norm())

    def _make_data_hook(self, name: str, pl_module: "LightningModule"):
        log_every_n_steps = self.log_every_n_steps

        def hook(module, input, output):
            step = pl_module.global_step
            if step % log_every_n_steps != 0:
                return
            inp = input[0].detach()
            out = output.detach()
            pl_module.log(f"{name}/input/mean", inp.mean())
            pl_module.log(f"{name}/input/var", inp.var())
            pl_module.log(f"{name}/output/mean", out.mean())
            pl_module.log(f"{name}/output/var", out.var())

        return hook
