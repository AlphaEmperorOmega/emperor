import os

from torch.types import Tensor
from torch.utils.tensorboard import SummaryWriter


class ScalarMonitor:
    LOG_DIR = "logs"
    TRAIN_KEY = "train/"
    TEST_KEY = "test/"
    LOSS_KEY = "Loss/"
    ACCURACY_KEY = "Accuracy/"

    def __init__(self, log_dir: str):
        log_dir = os.path.join(self.LOG_DIR, log_dir)
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def __increase_step(self):
        self.step += 1

    def reset_step(self):
        self.step = 0

    def __add_scalar(self, key: str, value: Tensor):
        self.writer.add_scalar(key, value, self.step)

    def __compose_key(self, metric_type: str, training_flag: bool = True):
        training_key = self.TRAIN_KEY if training_flag else self.TEST_KEY
        return metric_type + training_key

    def log_loss_metrics(self, loss: Tensor, training_flag: bool = True):
        self.__increase_step()
        loss_key = self.__compose_key(self.LOSS_KEY, training_flag)
        self.__add_scalar(loss_key, loss)

    def log_accuracy_metrics(self, accuracy: Tensor, training_flag: bool = True):
        self.__increase_step()
        accuracy_key = self.__compose_key(self.ACCURACY_KEY, training_flag)
        self.__add_scalar(accuracy_key, accuracy)
