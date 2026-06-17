from typing import TYPE_CHECKING

import torch
from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor
    from torch.nn import Module


class MemoryMonitorCallback(Callback):
    GATE_SUBMODULE_NAMES = ("memory_gate_model", "memory_weight_model")

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        if not isinstance(log_every_n_steps, int) or isinstance(
            log_every_n_steps, bool
        ):
            raise TypeError(
                "log_every_n_steps must be a positive integer, "
                f"received {type(log_every_n_steps).__name__}."
            )
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self._hooks = []
        self._memory_modules = []
        self._latest_gate_logits = {}

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        from emperor.memory.core.base import DynamicMemoryAbstract

        self.__cleanup_hooks()
        for name, module in pl_module.named_modules():
            if not isinstance(module, DynamicMemoryAbstract):
                continue
            self._memory_modules.append((name, module))
            self._hooks.append(
                module.register_forward_hook(
                    self.__make_memory_forward_hook(name, trainer, pl_module)
                )
            )
            gate_submodule = self.__find_gate_submodule(module)
            if gate_submodule is not None:
                self._hooks.append(
                    gate_submodule.register_forward_hook(
                        self.__make_gate_capture_hook(name)
                    )
                )

    def __find_gate_submodule(self, module: "Module") -> "Module | None":
        for submodule_name in self.GATE_SUBMODULE_NAMES:
            submodule = getattr(module, submodule_name, None)
            if submodule is not None:
                return submodule
        return None

    def __make_gate_capture_hook(self, name: str):
        def hook(submodule: "Module", inputs: tuple, output: object) -> None:
            gate_logits = self.__extract_hidden_tensor(output)
            if gate_logits is not None:
                self._latest_gate_logits[name] = gate_logits.detach()

        return hook

    def __extract_hidden_tensor(self, output: object) -> "Tensor | None":
        if torch.is_tensor(output):
            return output
        hidden = getattr(output, "hidden", None)
        if torch.is_tensor(hidden):
            return hidden
        return None

    def __make_memory_forward_hook(
        self,
        name: str,
        trainer: "Trainer",
        pl_module: "LightningModule",
    ):
        log_every_n_steps = self.log_every_n_steps

        def hook(memory_module: "Module", inputs: tuple, output: object) -> None:
            step = getattr(trainer, "global_step", getattr(pl_module, "global_step", 0))
            if step % log_every_n_steps != 0:
                return
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            if not torch.is_tensor(output):
                return
            logits = inputs[0].detach().float()
            output_tensor = output.detach().float()
            prefix = f"{name}/memory"
            self.__log_output_stats(pl_module, prefix, output_tensor)
            self.__log_memory_contribution(pl_module, prefix, logits, output_tensor)
            gate_logits = self._latest_gate_logits.get(name)
            if gate_logits is not None:
                self.__log_gate_stats(pl_module, prefix, memory_module, gate_logits)

        return hook

    def __log_output_stats(
        self,
        module: "LightningModule",
        prefix: str,
        output_tensor: "Tensor",
    ) -> None:
        module.log(f"{prefix}/output_mean", output_tensor.mean())
        module.log(f"{prefix}/output_var", output_tensor.var(unbiased=False))
        module.log(f"{prefix}/output_l2_norm", output_tensor.norm())

    def __log_memory_contribution(
        self,
        module: "LightningModule",
        prefix: str,
        logits: "Tensor",
        output_tensor: "Tensor",
    ) -> None:
        delta = output_tensor - logits
        module.log(f"{prefix}/contribution/delta_mean", delta.mean())
        module.log(f"{prefix}/contribution/delta_var", delta.var(unbiased=False))
        module.log(f"{prefix}/contribution/delta_norm", delta.norm())
        module.log(
            f"{prefix}/contribution/relative_delta_norm",
            delta.norm() / logits.norm().clamp_min(1e-6),
        )

    def __log_gate_stats(
        self,
        module: "LightningModule",
        prefix: str,
        memory_module: "Module",
        gate_logits: "Tensor",
    ) -> None:
        gate_logits = gate_logits.float()
        if type(memory_module).__name__ == "WeightedDynamicMemory":
            memory_share = torch.softmax(gate_logits, dim=-1)[..., -1]
            module.log(f"{prefix}/gate/open_mean", memory_share.mean())
            module.log(
                f"{prefix}/gate/open_fraction",
                (memory_share > 0.5).float().mean(),
            )
            return
        gate = torch.sigmoid(gate_logits)
        module.log(f"{prefix}/gate/open_mean", gate.mean())
        module.log(f"{prefix}/gate/open_fraction", (gate > 0.5).float().mean())
        module.log(
            f"{prefix}/gate/saturation_fraction",
            ((gate < 0.01) | (gate > 0.99)).float().mean(),
        )

    def __cleanup_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._memory_modules.clear()
        self._latest_gate_logits.clear()

    def on_fit_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.__cleanup_hooks()
