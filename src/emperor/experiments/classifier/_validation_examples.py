from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from emperor.monitoring import MonitorEmissionPolicy


@dataclass(frozen=True)
class _ValidationExample:
    confidence: float
    image: Tensor
    prediction: int
    target: int


class _ClassifierValidationExamples:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._examples: list[_ValidationExample] = []
        self._emission_policy = MonitorEmissionPolicy()

    def reset(self) -> None:
        self._examples = []

    def update(
        self,
        examples: Tensor | None,
        logits: Tensor,
        labels: Tensor,
    ) -> None:
        if examples is None or examples.ndim != 4:
            return
        if examples.size(1) not in (1, 3):
            return

        probabilities = logits.detach().softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        targets = labels.detach().to(predictions.device)
        wrong_indices = (predictions != targets).nonzero(as_tuple=False).flatten()

        for index in wrong_indices.tolist():
            image = examples[index].detach().cpu()
            self._examples.append(
                _ValidationExample(
                    confidence=float(confidence[index].detach().cpu().item()),
                    image=image,
                    prediction=int(predictions[index].detach().cpu().item()),
                    target=int(targets[index].detach().cpu().item()),
                )
            )
        self._examples.sort(
            key=lambda item: item.confidence,
            reverse=True,
        )
        del self._examples[self._limit :]

    def emit(self, logger, epoch: int) -> None:
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not self._examples:
            return

        image_grid = cast(Tensor, self._grid())
        self._emission_policy.emit_image(
            experiment,
            "validation/examples/most_confident_wrong",
            image_grid,
            global_step=epoch,
            module_key="validation",
        )

        add_text = getattr(experiment, "add_text", None)
        if callable(add_text):
            lines = [
                (
                    f"{index}. true={example.target} "
                    f"predicted={example.prediction} "
                    f"confidence={example.confidence:.4f}"
                )
                for index, example in enumerate(self._examples, start=1)
            ]
            add_text(
                "validation/examples/most_confident_wrong_labels",
                "\n".join(lines),
                global_step=epoch,
            )

    def _grid(self) -> Tensor | None:
        images = [example.image for example in self._examples]
        if not images:
            return None

        image_batch = torch.stack(images).float()
        image_min = image_batch.amin(dim=(1, 2, 3), keepdim=True)
        image_max = image_batch.amax(dim=(1, 2, 3), keepdim=True)
        image_range = image_max - image_min
        image_batch = torch.where(
            image_range > 0,
            (image_batch - image_min) / image_range.clamp_min(1e-12),
            image_batch.clamp(0, 1),
        )
        if image_batch.size(1) == 1:
            image_batch = image_batch.repeat(1, 3, 1, 1)

        image_count, channels, height, width = image_batch.shape
        columns = min(4, image_count)
        rows = (image_count + columns - 1) // columns
        grid = image_batch.new_zeros(channels, rows * height, columns * width)
        for index, image in enumerate(image_batch):
            row = index // columns
            column = index % columns
            grid[
                :,
                row * height : (row + 1) * height,
                column * width : (column + 1) * width,
            ] = image
        return grid
