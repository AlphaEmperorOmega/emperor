from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import emperor_workbench.tensorboard as tensorboard_reader
except ModuleNotFoundError as exc:
    if exc.name != "tensorboard":
        raise
    tensorboard_reader = None

import emperor_workbench.tensorboard as tensorboard_interface
from emperor_workbench.run_history._query import LogRunQueryService
from emperor_workbench.tensorboard import TENSORBOARD_TAG_SIZE_GUIDANCE
from tests.unit.run_history._support import log_run_scanner
from tests.unit.tensorboard._support import patch_event_accumulator_loader


class FakeTagsAccumulator:
    def Tags(self) -> dict[str, list[str]]:
        return {
            "scalars": [],
            "histograms": [],
            "images": [],
            "tensors": [],
        }


@unittest.skipIf(tensorboard_reader is None, "tensorboard is not installed")
class LogRunQueryTensorBoardObservationTests(unittest.TestCase):
    def test_tag_read_reuses_one_shared_event_file_observation(self) -> None:
        assert tensorboard_reader is not None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath("events.out.tfevents.test").write_bytes(b"events")
            query = LogRunQueryService(scanner=log_run_scanner())
            with (
                patch.object(
                    tensorboard_interface,
                    "event_file_index",
                    wraps=tensorboard_interface.event_file_index,
                ) as observe,
                patch_event_accumulator_loader(
                    return_value=FakeTagsAccumulator(),
                ) as load,
            ):
                query.read_tags(root)

        self.assertEqual(observe.call_count, 1)
        load.assert_called_once()
        args, kwargs = load.call_args
        self.assertEqual(args, (root,))
        self.assertEqual(
            kwargs["size_guidance"],
            TENSORBOARD_TAG_SIZE_GUIDANCE,
        )
        self.assertEqual(
            kwargs["event_files"],
            (root / "events.out.tfevents.test",),
        )
        self.assertEqual(kwargs["trusted_root"], root)


if __name__ == "__main__":
    unittest.main()
