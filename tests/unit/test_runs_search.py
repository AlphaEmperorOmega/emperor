from __future__ import annotations

import random
import unittest
from collections.abc import Sequence
from typing import Any

from emperor.runs.errors import InvalidRunRequest
from emperor.runs.search import (
    PreparedSearch,
    _sample_unique_combination_indices,
)


class _ScriptedRandom:
    def __init__(self, values: list[int]) -> None:
        self.values = iter(values)
        self.stops: list[int] = []

    def sample(self, population: Sequence[int], k: int) -> list[int]:
        raise OverflowError

    def randrange(self, stop: int) -> int:
        self.stops.append(stop)
        return next(self.values)


class _FailingSampleRandom:
    def sample(self, population: Sequence[int], k: int) -> list[int]:
        raise RuntimeError("sample failure")

    def randrange(self, stop: int) -> int:
        raise AssertionError("Non-overflow sample failures must propagate.")


class _VirtualAxis(Sequence[int]):
    def __init__(self, size: int) -> None:
        self.size = size
        self.reads: list[int] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            raise AssertionError("The virtual search axis must not be sliced.")
        if index < 0 or index >= self.size:
            raise IndexError(index)
        self.reads.append(index)
        return index

    def __iter__(self):
        raise AssertionError("The virtual search axis must not be materialized.")


class RunsSearchTests(unittest.TestCase):
    def test_grid_does_not_consume_supplied_random_state(self) -> None:
        random_source = random.Random(31)
        expected = random.Random(31)

        self.assertEqual(
            list(
                PreparedSearch(axes=((1, 2), (3, 4))).combinations(
                    random_source
                )
            ),
            [(1, 3), (1, 4), (2, 3), (2, 4)],
        )
        self.assertEqual(random_source.random(), expected.random())

    def test_random_sample_count_truncates_to_population(self) -> None:
        prepared = PreparedSearch(
            axes=(("a", "b"),),
            mode="random",
            random_samples=10,
        )

        self.assertEqual(prepared.selected_count, 2)
        self.assertEqual(
            set(prepared.combinations(random.Random(5))),
            {("a",), ("b",)},
        )

    def test_non_overflow_sample_failure_propagates(self) -> None:
        prepared = PreparedSearch(
            axes=((1, 2),),
            mode="random",
            random_samples=1,
        )

        with self.assertRaisesRegex(RuntimeError, "sample failure"):
            list(prepared.combinations(_FailingSampleRandom()))

    def test_grid_uses_mixed_radix_order_with_rightmost_axis_fastest(self) -> None:
        prepared = PreparedSearch(
            axes=(
                ("a0", "a1"),
                ("b0", "b1", "b2"),
                ("c0", "c1"),
            )
        )

        self.assertEqual(prepared.combination_count, 12)
        self.assertEqual(
            list(prepared.combinations()),
            [
                ("a0", "b0", "c0"),
                ("a0", "b0", "c1"),
                ("a0", "b1", "c0"),
                ("a0", "b1", "c1"),
                ("a0", "b2", "c0"),
                ("a0", "b2", "c1"),
                ("a1", "b0", "c0"),
                ("a1", "b0", "c1"),
                ("a1", "b1", "c0"),
                ("a1", "b1", "c1"),
                ("a1", "b2", "c0"),
                ("a1", "b2", "c1"),
            ],
        )

    def test_normal_seeded_random_selection_preserves_python_sample_order(
        self,
    ) -> None:
        random_source = random.Random(23)
        prepared = PreparedSearch(
            axes=(
                ("a0", "a1"),
                ("b0", "b1", "b2"),
                ("c0", "c1"),
            ),
            mode="random",
            random_samples=5,
        )

        self.assertEqual(
            list(prepared.combinations(random_source)),
            [
                ("a0", "b2", "c0"),
                ("a0", "b0", "c1"),
                ("a0", "b0", "c0"),
                ("a1", "b2", "c1"),
                ("a1", "b0", "c0"),
            ],
        )
        self.assertEqual(random_source.random(), 0.37918500997191673)

    def test_approved_overflow_selector_is_bounded_and_seeded(self) -> None:
        random_source = random.Random(23)

        indices = _sample_unique_combination_indices(
            2**63,
            5,
            random_source,
        )

        self.assertEqual(
            indices,
            [
                7188536481533917196,
                2674009078779859983,
                7652102777077138150,
                157503859921753048,
                5658426562401212583,
            ],
        )
        self.assertEqual(random_source.random(), 0.4237474082349614)

    def test_overflow_collision_substitutes_without_redrawing(self) -> None:
        total = 2**63
        random_source = _ScriptedRandom([5, 5, 5])

        indices = _sample_unique_combination_indices(total, 3, random_source)

        self.assertEqual(indices, [5, total - 2, total - 1])
        self.assertEqual(random_source.stops, [total - 2, total - 1, total])

    def test_huge_search_decodes_only_selected_values(self) -> None:
        axes = tuple(_VirtualAxis(10**9) for _ in range(3))
        random_source = random.Random(11)
        prepared = PreparedSearch(
            axes=axes,
            mode="random",
            random_samples=2,
        )

        self.assertEqual(
            list(prepared.combinations(random_source)),
            [
                (692964648, 458948364, 968338797),
                (967101122, 921085372, 330631083),
            ],
        )
        self.assertEqual(sum(len(axis.reads) for axis in axes), 6)
        self.assertEqual(random_source.random(), 0.4656500700997733)

    def test_random_search_requires_an_explicit_random_source(self) -> None:
        prepared = PreparedSearch(
            axes=((1, 2),),
            mode="random",
            random_samples=1,
        )

        with self.assertRaisesRegex(
            InvalidRunRequest,
            "requires an explicit random source",
        ):
            list(prepared.combinations())

    def test_random_search_rejects_nonpositive_sample_count(self) -> None:
        with self.assertRaisesRegex(
            InvalidRunRequest,
            "must be at least 1",
        ):
            PreparedSearch(
                axes=((1, 2),),
                mode="random",
                random_samples=0,
            )


if __name__ == "__main__":
    unittest.main()
