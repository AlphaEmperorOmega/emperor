import torch
from lightning import LightningDataModule

from emperor.datasets._metadata import _ResolvedDatasetMetadata
from emperor.datasets._visualization import show_images


class DataModule(LightningDataModule):
    """The base class of data."""

    _SETUP_HOOKS = {
        "fit": "_setup_fit",
        "validate": "_setup_validate",
        "test": "_setup_test",
    }
    _SUPPORTED_STAGES: frozenset[str] | None = None

    def __init__(
        self,
        root="data",
        num_workers=4,
    ):
        super().__init__()
        self.root = root
        self.num_workers = num_workers
        self._resolved_metadata = _ResolvedDatasetMetadata()

    @property
    def resolved_metadata(self) -> _ResolvedDatasetMetadata:
        return self._resolved_metadata

    def _resolve_metadata(
        self,
        *,
        vocab_size: int | None = None,
        num_classes: int | None = None,
        flattened_input_dim: int | None = None,
    ) -> None:
        self._resolved_metadata = self._resolved_metadata.resolve(
            vocab_size=vocab_size,
            num_classes=num_classes,
            flattened_input_dim=flattened_input_dim,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage is None:
            self._setup_all_supported_stages()
            return
        if stage not in self._SETUP_HOOKS:
            raise ValueError(f"Unsupported dataset setup stage: {stage!r}")
        self._setup_stage(stage)

    def _setup_all_supported_stages(self) -> None:
        supported_stages = self._supported_stages()
        if "fit" in supported_stages:
            self._setup_stage("fit")
        elif "validate" in supported_stages:
            self._setup_stage("validate")
        if "test" in supported_stages:
            self._setup_stage("test")
        if not supported_stages:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement a dataset setup stage"
            )

    def _setup_stage(self, stage: str) -> None:
        if stage not in self._supported_stages():
            raise NotImplementedError(
                f"{type(self).__name__} does not support the {stage!r} stage"
            )
        getattr(self, self._SETUP_HOOKS[stage])()

    def _supported_stages(self) -> frozenset[str]:
        declared_stages = type(self)._SUPPORTED_STAGES
        if declared_stages is not None:
            return declared_stages
        return frozenset(
            stage
            for stage, hook_name in self._SETUP_HOOKS.items()
            if getattr(type(self), hook_name) is not getattr(DataModule, hook_name)
        )

    def _setup_fit(self) -> None:
        raise NotImplementedError(
            "The method '_setup_fit' must be implemented in the subclass."
        )

    def _setup_validate(self) -> None:
        raise NotImplementedError(
            "The method '_setup_validate' must be implemented in the subclass."
        )

    def _setup_test(self) -> None:
        raise NotImplementedError(
            "The method '_setup_test' must be implemented in the subclass."
        )

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        self._require_stage_support("fit")
        self._require_dataset("train", "call setup('fit')")
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        if not self._supported_stages().intersection({"fit", "validate"}):
            raise NotImplementedError(
                f"{type(self).__name__} does not support validation data"
            )
        self._require_dataset(
            "val",
            "call setup('fit') or setup('validate')",
        )
        return self.get_dataloader(train=False)

    def test_dataloader(self):
        self._require_stage_support("test")
        self._require_dataset("test", "call setup('test')")
        return self._get_test_dataloader()

    def _get_test_dataloader(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement '_get_test_dataloader'"
        )

    def _require_stage_support(self, stage: str) -> None:
        if stage not in self._supported_stages():
            raise NotImplementedError(
                f"{type(self).__name__} does not support the {stage!r} stage"
            )

    def _require_dataset(self, attribute: str, setup_hint: str) -> None:
        if not hasattr(self, attribute) or getattr(self, attribute) is None:
            raise RuntimeError(
                f"{type(self).__name__} {attribute} data is not ready; "
                f"{setup_hint} before requesting its loader"
            )

    def get_tensorloader(  # noqa: B008 - preserves the historical public default
        self,
        tensors,
        train,
        indices=slice(0, None),  # noqa: B008
    ):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

    def visualize(self, batch, nrows=1, ncols=8, labels=None):
        X, y = batch
        if not labels:
            labels = self._text_labels(y)
        show_images(X.squeeze(1).permute(0, 2, 3, 1), nrows, ncols, titles=labels)

    def _text_labels(self, indices) -> list:
        raise NotImplementedError(
            "The 'test_labels' method must be implemented in the subclass."
        )
