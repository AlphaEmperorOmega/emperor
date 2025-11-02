from tensorboard.summary.writer.event_file_writer import os
import torch.nn.functional as F

from .utils import Module
from torch.types import Tensor
from Emperor.monitor.monitors import ScalarMonitor
from .utils import reshape, astype, argmax, float32, reduce_mean, randn, device


class Classifier(Module):
    def initialize_monitor(self, log_dir: str):
        self.monitor = ClassifierMonitor(log_dir)

    def log_train_metrics(self, loss: Tensor, accuracy: Tensor):
        if hasattr(self, "monitor"):
            self.monitor.log_training_metrics(loss, accuracy, self.training)

    def log_validation_metrics(self, loss: Tensor, accuracy: Tensor):
        if hasattr(self, "monitor"):
            self.monitor.log_validation_metrics(loss, accuracy, self.training)

    def training_step(self, batch):
        output, auxilary_loss = self(*batch[:-1])
        loss = self.loss(output, batch[-1])
        accuracy = self.accuracy(output, batch[-1])

        self.log_train_metrics(loss, accuracy)

        if auxilary_loss is not None:
            loss += auxilary_loss

        return loss, auxilary_loss

    def validation_step(self, batch):
        Y_hat, auxilaryLoss = self(*batch[:-1])
        loss = self.loss(Y_hat, batch[-1])
        accuracy = self.accuracy(Y_hat, batch[-1])

        self.log_validation_metrics(loss, accuracy)

        if auxilaryLoss is not None:
            loss += auxilaryLoss

        return loss, auxilaryLoss

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions."""
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1])).to("cpu")
        preds = astype(argmax(Y_hat, axis=1), Y.dtype)
        compare = astype(preds == reshape(Y, -1), float32)
        return reduce_mean(compare) if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = reshape(Y, (-1,)).to(device)
        loss = F.cross_entropy(Y_hat, Y, reduction="mean" if averaged else "none")
        return loss

    def layer_summary(self, X_shape):
        X = randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)


class ClassifierMonitor:
    def __init__(self, log_dir: str):
        self.training_monitor = ScalarMonitor(log_dir)
        self.validation_monitor = ScalarMonitor(log_dir)

    def log_training_metrics(self, loss: Tensor, accuracy: Tensor, training_flag: bool):
        self.training_monitor.log_loss_metrics(loss, training_flag)
        self.training_monitor.log_accuracy_metrics(accuracy, training_flag)

    def log_validation_metrics(
        self, loss: Tensor, accuracy: Tensor, training_flag: bool
    ):
        self.validation_monitor.log_loss_metrics(loss, training_flag)
        self.validation_monitor.log_accuracy_metrics(accuracy, training_flag)
