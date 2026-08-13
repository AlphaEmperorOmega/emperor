from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any, cast

from model_runtime.cli import preset_locks_to_wire
from model_runtime.inspection import configuration_schema
from model_runtime.packages import ModelPackage, PresetLock, RuntimeDefaultsError
from models.catalog import model_package


class _StatefulLock:
    def __init__(
        self,
        name: str,
        value: object,
        reason: object,
        events: list[str],
        *,
        falsey: bool = False,
    ) -> None:
        self._name = name
        self._value = value
        self._reason = reason
        self._events = events
        self._falsey = falsey
        self.value_reads = 0
        self.reason_reads = 0
        self.truth_reads = 0

    @property
    def value(self) -> object:
        self.value_reads += 1
        self._events.append(f"{self._name}.value")
        return self._value

    @property
    def reason(self) -> object:
        self.reason_reads += 1
        self._events.append(f"{self._name}.reason")
        return self._reason

    def __bool__(self) -> bool:
        self.truth_reads += 1
        return not self._falsey


class _RawLockProvider:
    def __init__(self, locks: Mapping[str, object]) -> None:
        self.locks = locks
        self.calls = 0

    def locked_fields(self, _preset: object) -> Mapping[str, object]:
        self.calls += 1
        return self.locks


class _RawLockAdapter:
    def __init__(self, delegate: ModelPackage, provider: _RawLockProvider) -> None:
        self._delegate = delegate
        self._provider = provider

    def load_metadata(self) -> Any:
        return self._delegate.metadata

    def load_preset_type(self) -> type[Any]:
        return self._delegate.preset_type

    def load_presets(self) -> _RawLockProvider:
        return self._provider


def _package_with_raw_locks(
    locks: Mapping[str, object],
) -> tuple[ModelPackage, _RawLockProvider]:
    delegate = model_package("linears/linear")
    assert delegate is not None
    provider = _RawLockProvider(locks)
    package = ModelPackage(
        delegate.identity,
        cast(Any, _RawLockAdapter(delegate, provider)),
        delegate.inspection_construction_limits,
    )
    return package, provider


class RuntimeDefaultsPresetLockTests(unittest.TestCase):
    def test_raw_interface_stays_structural_and_runtime_defaults_snapshot_once(
        self,
    ) -> None:
        events: list[str] = []
        opaque_value = object()
        falsey_lock = _StatefulLock(
            "hidden",
            23,
            "Falsey lock reason.",
            events,
            falsey=True,
        )
        opaque_lock = _StatefulLock(
            "opaque",
            opaque_value,
            "Opaque lock reason.",
            events,
        )
        raw_locks = {
            "HIDDEN_DIM": falsey_lock,
            "STACK_NUM_LAYERS": None,
            "OPAQUE_FIELD": opaque_lock,
        }
        package, provider = _package_with_raw_locks(raw_locks)
        preset = package.resolve_preset("baseline")

        public_locks = package.preset_locks(preset)

        self.assertIsNot(public_locks, raw_locks)
        self.assertIs(public_locks["HIDDEN_DIM"], falsey_lock)
        self.assertIsNone(public_locks["STACK_NUM_LAYERS"])
        self.assertEqual(events, [])
        self.assertEqual(provider.calls, 1)

        normalized = package.runtime_defaults_spec.locks_for_preset(preset)

        self.assertEqual(
            tuple(normalized),
            ("hidden_dim", "stack_num_layers", "opaque_field"),
        )
        self.assertTrue(
            all(isinstance(lock, PresetLock) for lock in normalized.values())
        )
        self.assertEqual(
            normalized["hidden_dim"], PresetLock(23, "Falsey lock reason.")
        )
        self.assertEqual(normalized["stack_num_layers"], PresetLock(None, ""))
        self.assertIs(normalized["opaque_field"].value, opaque_value)
        self.assertEqual(
            tuple(events),
            (
                "hidden.value",
                "hidden.reason",
                "opaque.value",
                "opaque.reason",
            ),
        )
        self.assertEqual(falsey_lock.truth_reads, 0)
        self.assertEqual(provider.calls, 2)

        wire = preset_locks_to_wire(normalized)

        self.assertEqual(
            wire["hidden_dim"],
            {"value": 23, "reason": "Falsey lock reason."},
        )
        self.assertEqual(
            wire["stack_num_layers"],
            {"value": None, "reason": ""},
        )
        self.assertEqual(falsey_lock.value_reads, 1)
        self.assertEqual(falsey_lock.reason_reads, 1)
        self.assertEqual(opaque_lock.value_reads, 1)
        self.assertEqual(opaque_lock.reason_reads, 1)

        fields = {
            field.key: field
            for field in configuration_schema(package, preset="baseline").fields
        }

        self.assertTrue(fields["HIDDEN_DIM"].locked)
        self.assertEqual(fields["HIDDEN_DIM"].locked_value, 23)
        self.assertEqual(fields["HIDDEN_DIM"].locked_reason, "Falsey lock reason.")
        self.assertTrue(fields["STACK_NUM_LAYERS"].locked)
        self.assertIsNone(fields["STACK_NUM_LAYERS"].locked_value)
        self.assertEqual(fields["STACK_NUM_LAYERS"].locked_reason, "")
        self.assertEqual(falsey_lock.value_reads, 2)
        self.assertEqual(falsey_lock.reason_reads, 2)
        self.assertEqual(falsey_lock.truth_reads, 0)
        self.assertEqual(opaque_lock.value_reads, 2)
        self.assertEqual(opaque_lock.reason_reads, 2)
        self.assertEqual(provider.calls, 3)

    def test_equal_aliases_keep_first_lock_and_read_each_source_once(self) -> None:
        events: list[str] = []
        first = _StatefulLock("first", 64, "First reason.", events)
        duplicate = _StatefulLock("duplicate", 64, "Second reason.", events)
        trailing = _StatefulLock("trailing", 2, "Trailing reason.", events)
        package, _provider = _package_with_raw_locks(
            {
                "HIDDEN_DIM": first,
                "hidden_dim": duplicate,
                "STACK_NUM_LAYERS": trailing,
            }
        )
        preset = package.resolve_preset("baseline")

        locks = package.runtime_defaults_spec.locks_for_preset(
            preset,
            label="baseline",
        )

        self.assertEqual(tuple(locks), ("hidden_dim", "stack_num_layers"))
        self.assertEqual(locks["hidden_dim"], PresetLock(64, "First reason."))
        self.assertEqual(
            tuple(events),
            (
                "first.value",
                "first.reason",
                "duplicate.value",
                "duplicate.reason",
                "trailing.value",
                "trailing.reason",
            ),
        )
        for lock in (first, duplicate, trailing):
            self.assertEqual(lock.value_reads, 1)
            self.assertEqual(lock.reason_reads, 1)

    def test_conflicting_aliases_stop_after_single_snapshot_per_source(self) -> None:
        events: list[str] = []
        first = _StatefulLock("first", 64, "First reason.", events)
        conflict = _StatefulLock("conflict", 128, "Conflict reason.", events)
        unread = _StatefulLock("unread", 2, "Unread reason.", events)
        package, _provider = _package_with_raw_locks(
            {
                "HIDDEN_DIM": first,
                "hidden_dim": conflict,
                "STACK_NUM_LAYERS": unread,
            }
        )
        preset = package.resolve_preset("baseline")

        with self.assertRaises(RuntimeDefaultsError) as raised:
            package.runtime_defaults_spec.locks_for_preset(
                preset,
                label="baseline",
            )

        self.assertEqual(
            str(raised.exception),
            "Preset 'baseline' for model 'linears/linear' defines conflicting "
            "locks for Runtime Defaults parameter 'hidden_dim' through "
            "'HIDDEN_DIM' and 'hidden_dim'.",
        )
        self.assertEqual(
            tuple(events),
            (
                "first.value",
                "first.reason",
                "conflict.value",
                "conflict.reason",
            ),
        )
        self.assertEqual(first.value_reads, 1)
        self.assertEqual(first.reason_reads, 1)
        self.assertEqual(conflict.value_reads, 1)
        self.assertEqual(conflict.reason_reads, 1)
        self.assertEqual(unread.value_reads, 0)
        self.assertEqual(unread.reason_reads, 0)


if __name__ == "__main__":
    unittest.main()
