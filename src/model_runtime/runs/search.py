from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, Literal

from model_runtime.runs.errors import InvalidRunRequest
from model_runtime.runs.records import RandomSource


def _combination_count(axes: Sequence[Sequence[Any]]) -> int:
    return prod(len(axis) for axis in axes)


def _combination_at_index(
    axes: Sequence[Sequence[Any]],
    index: int,
) -> tuple[Any, ...]:
    combination_count = _combination_count(axes)
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("Combination index must be an integer.")
    if index < 0 or index >= combination_count:
        raise IndexError(
            f"Combination index {index} is outside [0, {combination_count})."
        )

    values: list[Any] = []
    remaining = index
    for axis in reversed(axes):
        remaining, offset = divmod(remaining, len(axis))
        values.append(axis[offset])
    return tuple(reversed(values))


def _sample_unique_combination_indices(
    total_combinations: int,
    num_samples: int,
    random_source: RandomSource,
) -> list[int]:
    sample_count = min(num_samples, total_combinations)
    if sample_count < 0:
        raise ValueError("Sample larger than population or is negative")
    if sample_count == 0:
        return []

    try:
        return random_source.sample(range(total_combinations), sample_count)
    except OverflowError:
        pass

    selected_indices: list[int] = []
    selected_index_set: set[int] = set()
    start = total_combinations - sample_count
    for offset in range(sample_count):
        upper_bound = start + offset
        candidate = random_source.randrange(upper_bound + 1)
        if candidate in selected_index_set:
            candidate = upper_bound
        selected_index_set.add(candidate)
        selected_indices.append(candidate)
    return selected_indices


@dataclass(frozen=True, slots=True)
class PreparedSearch:
    axes: tuple[Sequence[Any], ...]
    mode: Literal["grid", "random"] = "grid"
    random_samples: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        if self.mode not in {"grid", "random"}:
            raise InvalidRunRequest("Training search mode must be 'grid' or 'random'.")
        if any(len(axis) == 0 for axis in self.axes):
            raise InvalidRunRequest(
                "Training search axes require at least one selected value."
            )
        if self.mode == "grid" and self.random_samples is not None:
            raise InvalidRunRequest(
                "Grid search does not accept a random sample count."
            )
        if self.mode == "random":
            if isinstance(self.random_samples, bool) or not isinstance(
                self.random_samples,
                int,
            ):
                raise InvalidRunRequest(
                    "Random search sample count must be an integer."
                )
            if self.random_samples < 1:
                raise InvalidRunRequest(
                    "Random search sample count must be at least 1."
                )

    @property
    def combination_count(self) -> int:
        return _combination_count(self.axes)

    @property
    def selected_count(self) -> int:
        if self.mode == "grid":
            return self.combination_count
        assert self.random_samples is not None
        return min(self.random_samples, self.combination_count)

    def combinations(
        self,
        random_source: RandomSource | None = None,
    ) -> Iterator[tuple[Any, ...]]:
        if self.mode == "grid":
            indices: Sequence[int] = range(self.combination_count)
        else:
            if random_source is None:
                raise InvalidRunRequest(
                    "Random search requires an explicit random source."
                )
            assert self.random_samples is not None
            indices = _sample_unique_combination_indices(
                self.combination_count,
                self.random_samples,
                random_source,
            )
        for index in indices:
            yield _combination_at_index(self.axes, index)


__all__ = ["PreparedSearch"]
