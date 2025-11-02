import os
import random

import torch
from torch.types import Tensor
from torch.utils.tensorboard import SummaryWriter


class MonitorBase:
    LOG_DIR = "logs"

    def __init__(self, log_dir: str):
        log_dir = os.path.join(self.LOG_DIR, log_dir)
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def increase_step(self):
        self.step += 1

    def get_step(self):
        return self.step

    def reset_step(self):
        self.step = 0

    def add_historgram(self, key: str, value: Tensor):
        self.writer.add_histogram(key, value, self.step)

    def add_scalar(self, key: str, value: Tensor):
        self.writer.add_scalar(key, value, self.step)


class ParameterMonitor:
    def __init__(self):
        log_dir = "parameter_logs"

        log_dir = os.path.join("parameter_logs", str(DataMonitor.test))
        self.monitor = MonitorBase(log_dir)

    def update(self, weights: Tensor, biases: Tensor | None = None) -> None:
        self.monitor.increase_step()
        if self.monitor.get_step() % 935 == 0:
            self.monitor.add_historgram("Parameters/Weights", weights)
            if biases is not None:
                self.monitor.add_historgram("Parameters/Biases", biases)


class DataMonitor:
    test = 1

    def __init__(self):
        log_dir = os.path.join("data_logs", str(DataMonitor.test))
        self.monitor = MonitorBase(log_dir)
        self.input_tensors = []
        self.output_tensors = []
        self.update_frequency = 20
        DataMonitor.test += 1

    def update(self, inputs: Tensor, outputs: Tensor) -> None:
        self.monitor.increase_step()

        input_mean = torch.mean(inputs)
        input_variance = torch.var(inputs)
        self.monitor.add_scalar("Input/Mean", input_mean)
        self.monitor.add_scalar("Input/Variance", input_variance)

        output_mean = torch.mean(outputs)
        output_variance = torch.var(outputs)
        self.monitor.add_scalar("Output/Mean", output_mean)
        self.monitor.add_scalar("Output/Variance", output_variance)
