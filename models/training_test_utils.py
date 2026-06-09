import torch

from lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, TensorDataset


class RandomImageClassificationDataModule(LightningDataModule):
    def __init__(self, dataset: type, batch_size: int = 8, num_batches: int = 2):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = batch_size * num_batches

    def _build_random_loader(self) -> DataLoader:
        images = torch.randn(
            self.num_samples,
            self.dataset.num_channels,
            self.dataset.default_height,
            self.dataset.default_width,
        )
        labels = torch.randint(0, self.dataset.num_classes, (self.num_samples,))
        return DataLoader(TensorDataset(images, labels), batch_size=self.batch_size)

    def train_dataloader(self) -> DataLoader:
        return self._build_random_loader()

    def val_dataloader(self) -> DataLoader:
        return self._build_random_loader()

    def test_dataloader(self) -> DataLoader:
        return self._build_random_loader()


class RandomBertPretrainingDataModule(LightningDataModule):
    def __init__(self, cfg, batch_size: int = 2, num_batches: int = 2):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_samples = batch_size * num_batches

    def _build_random_loader(self) -> DataLoader:
        input_ids = torch.randint(
            5,
            self.cfg.input_dim,
            (self.num_samples, self.cfg.sequence_length),
        )
        input_ids[:, 0] = 2
        input_ids[:, -1] = 0
        attention_mask = (input_ids != 0).long()
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[:, self.cfg.sequence_length // 2 : -1] = 1
        mlm_labels = torch.full_like(input_ids, -100)
        mlm_labels[:, 1] = input_ids[:, 1]
        next_sentence_labels = torch.randint(0, 2, (self.num_samples,))
        return DataLoader(
            TensorDataset(
                input_ids,
                mlm_labels,
                attention_mask,
                token_type_ids,
                next_sentence_labels,
            ),
            batch_size=self.batch_size,
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_random_loader()

    def val_dataloader(self) -> DataLoader:
        return self._build_random_loader()

    def test_dataloader(self) -> DataLoader:
        return self._build_random_loader()


def tiny_cpu_trainer() -> Trainer:
    return Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        limit_train_batches=2,
        limit_val_batches=2,
        num_sanity_val_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
