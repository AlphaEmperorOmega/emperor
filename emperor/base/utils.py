import inspect

import torch

from emperor.base.config import ConfigBase
from emperor.base.config import optional_field as optional_field
from emperor.base.data import DataModule
from emperor.base.module import Module, ParameterBank
from emperor.base.visualization import (
    HyperParameters,
    ProgressBoard,
    show_images,
)


class Trainer(HyperParameters):
    """Historical training loop retained for import compatibility."""

    def __init__(self, max_epochs, num_gpu=0, gradient_clip_val=0, view_progress=True):
        self.save_hyperparameters()
        self.print_loss = False
        self.gpus = [
            torch.device(f"cuda:{i}")
            for i in range(min(num_gpu, torch.cuda.device_count()))
        ]

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )

    def fit(
        self,
        model,
        data,
        print_loss_flag: bool = False,
        print_loss_frequency: int = 50,
    ):
        self.print_loss_flag = print_loss_flag
        self.print_loss_frequency = print_loss_frequency
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.fit_epoch()

    def fit_epoch(self):
        self.model.train()
        for batch_idx, batch in enumerate(self.train_dataloader):
            loss, auxiliary_loss = self.model.training_step(self.prepare_batch(batch))
            self.__print_batch_messages(loss, auxiliary_loss, batch_idx=batch_idx)
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch_idx, batch in enumerate(self.val_dataloader):
            with torch.no_grad():
                loss, auxiliary_loss = self.model.validation_step(
                    self.prepare_batch(batch)
                )
                self.__print_batch_messages(loss, auxiliary_loss, batch_idx=batch_idx)

    def __print_batch_messages(
        self,
        loss: torch.Tensor,
        auxiliary_loss: torch.Tensor,
        batch_idx: int = 0,
    ) -> None:
        if self.print_loss_flag and batch_idx % self.print_loss_frequency == 0:
            message = [
                f"Epoch: {self.epoch}",
                f"Batch: {batch_idx}",
                f"Total loss: {round(loss.item(), 4)}",
                f"Model loss: {round(loss.item() - auxiliary_loss.item(), 4)}",
                f"Auxiliary loss: {round(auxiliary_loss.item(), 4)}",
            ]
            print(", ".join(message))

    def prepare_batch(self, batch):
        if self.gpus:
            batch = [a.to(self.gpus[0]) for a in batch]
        return batch

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm


class ConfigUtils:
    @staticmethod
    def get_method_arguments():
        frame = inspect.currentframe()
        parent_frame = frame.f_back
        parent_arguments = inspect.getargvalues(parent_frame)
        parent_inputs = {
            key: parent_arguments.locals[key]
            for key in parent_arguments.locals
            if key in parent_arguments.args
        }

        del frame
        del parent_frame
        return parent_inputs


__all__ = [
    "ConfigBase",
    "ConfigUtils",
    "DataModule",
    "HyperParameters",
    "Module",
    "ParameterBank",
    "ProgressBoard",
    "Trainer",
    "optional_field",
    "show_images",
]
