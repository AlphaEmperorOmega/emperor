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


class RandomLanguageModelDataModule(LightningDataModule):
    def __init__(self, cfg, batch_size: int = 2, num_batches: int = 2):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_samples = batch_size * num_batches

    def _build_random_loader(self) -> DataLoader:
        streams = torch.randint(
            0,
            self.cfg.input_dim,
            (self.num_samples, self.cfg.sequence_length + 1),
        )
        return DataLoader(
            TensorDataset(streams[:, :-1], streams[:, 1:]),
            batch_size=self.batch_size,
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_random_loader()

    def val_dataloader(self) -> DataLoader:
        return self._build_random_loader()

    def test_dataloader(self) -> DataLoader:
        return self._build_random_loader()


class RandomTranslationDataModule(LightningDataModule):
    def __init__(
        self,
        cfg,
        batch_size: int = 2,
        num_batches: int = 2,
        seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_samples = batch_size * num_batches
        self.seed = seed
        experiment_config = cfg.experiment_config
        self.source_sequence_length = experiment_config.source_sequence_length
        self.target_sequence_length = experiment_config.target_sequence_length
        self.vocab_size = cfg.input_dim
        self.pad_token_id = experiment_config.pad_token_id
        self.bos_token_id = experiment_config.bos_token_id
        self.eos_token_id = experiment_config.eos_token_id

    def _random_ids(self, length: int, generator: torch.Generator) -> torch.Tensor:
        if self.vocab_size > 4:
            token_ids = torch.randint(
                4,
                self.vocab_size,
                (self.num_samples, length),
                generator=generator,
            )
        else:
            token_ids = torch.ones((self.num_samples, length), dtype=torch.long)
        token_ids[:, 0] = self.bos_token_id
        token_ids[:, -1] = self.eos_token_id
        if length > 3:
            token_ids[::2, -2] = self.eos_token_id
            token_ids[::2, -1] = self.pad_token_id
        return token_ids

    def _build_random_loader(self, seed_offset: int) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed + seed_offset)
        source_ids = self._random_ids(self.source_sequence_length, generator)
        target_ids = self._random_ids(self.target_sequence_length, generator)
        return DataLoader(
            TensorDataset(source_ids, target_ids),
            batch_size=self.batch_size,
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_random_loader(0)

    def val_dataloader(self) -> DataLoader:
        return self._build_random_loader(1)

    def test_dataloader(self) -> DataLoader:
        return self._build_random_loader(2)

    def decode_ids(self, token_ids) -> str:
        decoded: list[str] = []
        for token_id in token_ids:
            value = int(token_id)
            if value == self.eos_token_id:
                break
            if value not in (self.pad_token_id, self.bos_token_id):
                decoded.append(str(value))
        return " ".join(decoded)

    def decode_batch(self, batch_token_ids) -> list[str]:
        return [self.decode_ids(token_ids) for token_ids in batch_token_ids]


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
