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

        if auxilary_loss is not None:
            loss += auxilary_loss

        self.log_train_metrics(loss, accuracy)

        return loss, auxilary_loss

    def validation_step(self, batch):
        Y_hat, auxilary_loss = self(*batch[:-1])
        loss = self.loss(Y_hat, batch[-1])
        accuracy = self.accuracy(Y_hat, batch[-1])

        if auxilary_loss is not None:
            loss += auxilary_loss

        self.log_validation_metrics(loss, accuracy)
        return loss, auxilary_loss

    def accuracy(self, Y_hat, Y, averaged=True):
        num_classes = Y_hat.size(-1)
        Y_hat = Y_hat.detach().view(-1, num_classes)
        Y = Y.detach().view(-1)
        if Y.device != Y_hat.device:
            Y = Y.to(Y_hat.device)
        preds = Y_hat.argmax(dim=1)
        correct = (preds == Y).float()
        return correct.mean() if averaged else correct

    def loss(self, Y_hat, Y, averaged=True):
        num_classes = Y_hat.size(-1)
        Y_hat = Y_hat.view(-1, num_classes)
        Y = Y.view(-1)
        if Y.device != Y_hat.device:
            Y = Y.to(Y_hat.device)
        reduction = "mean" if averaged else "none"
        return F.cross_entropy(Y_hat, Y, reduction=reduction)

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
