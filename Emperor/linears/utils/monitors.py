import os
from typing import Optional

import torch
from torch.nn import Parameter
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class MonitorBase:
    LOG_DIR = "logs"

    def __init__(self, log_subdir: str):
        log_dir = os.path.join(self.LOG_DIR, log_subdir)
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def increase_step(self) -> int:
        self.step += 1
        return self.step

    def get_step(self) -> int:
        return self.step

    def reset_step(self) -> None:
        self.step = 0

    def add_histogram(self, key: str, value: Tensor) -> None:
        self.writer.add_histogram(key, value.detach().cpu(), self.step)

    def add_scalar(self, key: str, value: float | int) -> None:
        self.writer.add_scalar(key, float(value), self.step)

    def close(self) -> None:
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass


class StatisticsMonitor:
    _instance_counter = 0

    def __init__(self, hist_freq: int = 1000, log_subdir: Optional[str] = None):
        subdir = log_subdir or os.path.join("parameter_logs", f"run_{StatisticsMonitor._instance_counter}")
        StatisticsMonitor._instance_counter += 1
        self.monitor = MonitorBase(subdir)
        self.hist_freq = max(1, int(hist_freq))

    def update(self, weights: Tensor, biases: Tensor | None = None) -> None:
        step = self.monitor.increase_step()

        w = weights.detach()
        self.monitor.add_scalar("Parameters/Weights/mean", torch.mean(w).item())
        self.monitor.add_scalar("Parameters/Weights/var", torch.var(w).item())
        self.monitor.add_scalar("Parameters/Weights/l2_norm", torch.norm(w).item())

        if biases is not None:
            b = biases.detach()
            self.monitor.add_scalar("Parameters/Biases/mean", torch.mean(b).item())
            self.monitor.add_scalar("Parameters/Biases/var", torch.var(b).item())
            self.monitor.add_scalar("Parameters/Biases/l2_norm", torch.norm(b).item())

        if step % self.hist_freq == 0:
            try:
                self.monitor.add_histogram("Parameters/Weights/hist", w)
            except Exception:
                pass
            if biases is not None:
                try:
                    self.monitor.add_histogram("Parameters/Biases/hist", b)
                except Exception:
                    pass

    def close(self) -> None:
        self.monitor.close()


class TensorMonitor:
    _instance_counter = 0

    def __init__(self, hist_freq: int = 200, log_subdir: Optional[str] = None):
        subdir = log_subdir or os.path.join("data_logs", f"run_{TensorMonitor._instance_counter}")
        TensorMonitor._instance_counter += 1
        self.monitor = MonitorBase(subdir)
        self.hist_freq = max(1, int(hist_freq))

    def update(self, inputs: Tensor | Parameter, outputs: Tensor | Parameter | None) -> None:
        step = self.monitor.increase_step()

        inp = inputs.detach()
        self.monitor.add_scalar("Input/Mean", torch.mean(inp).item())
        self.monitor.add_scalar("Input/Var", torch.var(inp).item())

        if outputs is not None:
            out = outputs.detach()
            self.monitor.add_scalar("Output/Mean", torch.mean(out).item())
            self.monitor.add_scalar("Output/Var", torch.var(out).item())

        if step % self.hist_freq == 0:
            try:
                self.monitor.add_histogram("Input/Hist", inp)
            except Exception:
                pass
            if outputs is not None:
                try:
                    self.monitor.add_histogram("Output/Hist", out)
                except Exception:
                    pass

    def close(self) -> None:
        self.monitor.close()
