from __future__ import annotations

import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from collections import OrderedDict
from dataclasses import asdict, fields
from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from model_runtime.runs import (
    DEFAULT_CHECKPOINT_ADMISSION_POLICY,
    CheckpointAdmissionPolicy,
    execute_runs,
)
from model_runtime.runs import _checkpoint_isolation as checkpoint_isolation
from model_runtime.runs import _checkpoint_snapshot as checkpoint_snapshot
from model_runtime.runs._checkpoint_payload import validate_checkpoint_payload
from model_runtime.runs._checkpoint_receipt import CheckpointPayloadReceipt
from model_runtime.runs._checkpoint_worker_protocol import (
    decode_worker_response,
    encode_worker_receipt,
)
from model_runtime.runs.checkpoints import (
    CheckpointContinuation,
    CheckpointContinuationLifecycle,
)
from model_runtime.runs.errors import InvalidCheckpointContinuation


def _payload(**extra: object) -> dict[str, object]:
    return {
        "pytorch-lightning_version": "2.6.5",
        "state_dict": {"weight": torch.ones(1)},
        "epoch": 0,
        "global_step": 1,
        "optimizer_states": [{}],
        **extra,
    }


def _process_is_running(process_id: int) -> bool:
    try:
        state = Path(f"/proc/{process_id}/stat").read_text(encoding="utf-8").split()[2]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state not in {"X", "Z"}


class CheckpointAdmissionPolicyTests(unittest.TestCase):
    def test_public_continuation_record_contract_is_unchanged(self) -> None:
        continuation = CheckpointContinuation("relative/last.ckpt")
        same = CheckpointContinuation(Path("relative/last.ckpt"))

        self.assertEqual([field.name for field in fields(continuation)], ["checkpoint_path"])
        self.assertEqual(
            list(signature(CheckpointContinuation).parameters),
            ["checkpoint_path"],
        )
        self.assertEqual(
            signature(CheckpointContinuation).parameters["checkpoint_path"].kind,
            Parameter.POSITIONAL_OR_KEYWORD,
        )
        self.assertEqual(continuation, same)
        self.assertEqual(
            repr(continuation),
            f"CheckpointContinuation(checkpoint_path={Path('relative/last.ckpt')!r})",
        )
        self.assertEqual(CheckpointContinuation.__match_args__, ("checkpoint_path",))
        self.assertEqual(pickle.loads(pickle.dumps(continuation)), continuation)

    def test_public_execution_rejects_non_policy_admission_before_plan_use(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "Checkpoint admission must be a CheckpointAdmissionPolicy",
        ):
            execute_runs(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                artifacts=object(),  # type: ignore[arg-type]
                checkpoint_admission=object(),  # type: ignore[arg-type]
            )

    def test_public_policy_export_is_lazy(self) -> None:
        script = """
import sys
from model_runtime.runs import CheckpointAdmissionPolicy, DEFAULT_CHECKPOINT_ADMISSION_POLICY
assert isinstance(DEFAULT_CHECKPOINT_ADMISSION_POLICY, CheckpointAdmissionPolicy)
for prefix in ('torch', 'lightning'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
        subprocess.run(  # noqa: S603 - fixed interpreter and test script
            [sys.executable, "-P", "-c", script],
            check=True,
            capture_output=True,
            timeout=10,
        )

        self.assertIsInstance(
            DEFAULT_CHECKPOINT_ADMISSION_POLICY,
            CheckpointAdmissionPolicy,
        )

    def test_policy_rejects_protocol_unsafe_numeric_values(self) -> None:
        cases = (
            {"max_file_bytes": True},
            {"max_payload_nodes": 2**63},
            {"worker_cpu_seconds": 86_401},
            {"worker_wall_timeout_seconds": 86_401.0},
        )

        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                CheckpointAdmissionPolicy(**values)  # type: ignore[arg-type]

    def test_policy_requires_a_bounded_frozenset_of_dtype_names(self) -> None:
        cases = (
            {"allowed_dtypes": ["torch.float32"]},
            {"allowed_dtypes": frozenset({"float32"})},
            {"allowed_dtypes": frozenset({"torch." + "x" * 65})},
        )

        for values in cases:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                CheckpointAdmissionPolicy(**values)  # type: ignore[arg-type]


class CheckpointPayloadAccountingTests(unittest.TestCase):
    def test_schema_counters_are_protocol_bounded_on_every_decode_path(self) -> None:
        payload = _payload(epoch=2**63, global_step=2**63)

        with self.assertRaisesRegex(
            InvalidCheckpointContinuation,
            "bounded nonnegative epoch",
        ):
            validate_checkpoint_payload(
                payload,
                Path("counter.ckpt"),
                CheckpointAdmissionPolicy(),
            )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "counter.ckpt"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(
                InvalidCheckpointContinuation,
                "bounded nonnegative epoch",
            ):
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(checkpoint),
                    SimpleNamespace(runs=(object(),)),
                )

    def test_raw_storage_and_unknown_leaves_fail_closed(self) -> None:
        cases = (
            (torch.UntypedStorage(8), "raw tensor storage"),
            (object(), "unsupported payload value"),
        )

        for leaf, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(InvalidCheckpointContinuation, message),
            ):
                validate_checkpoint_payload(
                    _payload(callback_state=leaf),
                    Path("source.ckpt"),
                    CheckpointAdmissionPolicy(),
                )

    def test_safe_torch_value_singletons_are_admitted(self) -> None:
        validated = validate_checkpoint_payload(
            _payload(
                callback_state={
                    "device": torch.device("cpu"),
                    "dtype": torch.float32,
                    "layout": torch.strided,
                }
            ),
            Path("source.ckpt"),
            CheckpointAdmissionPolicy(),
        )

        self.assertGreater(validated.receipt.payload_nodes, 0)

    def test_tensor_rank_and_aggregate_dimensions_are_bounded(self) -> None:
        high_rank = torch.empty((0,) * 5)
        with self.assertRaisesRegex(
            InvalidCheckpointContinuation,
            "5 dimensions.*limit of 4",
        ):
            validate_checkpoint_payload(
                _payload(state_dict={"weight": high_rank}),
                Path("rank.ckpt"),
                CheckpointAdmissionPolicy(max_tensor_dimensions=4),
            )

        with self.assertRaisesRegex(
            InvalidCheckpointContinuation,
            "aggregate tensor dimensions",
        ):
            validate_checkpoint_payload(
                _payload(
                    state_dict={
                        "first": torch.ones((1, 1)),
                        "second": torch.ones((1, 1)),
                    }
                ),
                Path("dimensions.ckpt"),
                CheckpointAdmissionPolicy(
                    max_tensor_dimensions=2,
                    max_aggregate_tensor_dimensions=3,
                ),
            )

    def test_aliases_count_references_but_share_storage(self) -> None:
        storage_owner = torch.ones(8)
        validated = validate_checkpoint_payload(
            _payload(
                state_dict={
                    "first": storage_owner[:4],
                    "second": storage_owner[4:],
                }
            ),
            Path("aliases.ckpt"),
            CheckpointAdmissionPolicy(),
        )

        self.assertEqual(validated.receipt.tensor_count, 2)
        self.assertEqual(
            validated.receipt.unique_storage_bytes,
            storage_owner.untyped_storage().nbytes(),
        )

    def test_tensor_attributes_are_included_in_resource_accounting(self) -> None:
        carrier = torch.ones(1)
        carrier.hidden = {"tensor": torch.ones(100)}  # type: ignore[attr-defined]

        with self.assertRaisesRegex(
            InvalidCheckpointContinuation,
            "2 tensors.*limit of 1",
        ):
            validate_checkpoint_payload(
                _payload(state_dict={"weight": carrier}),
                Path("tensor-attributes.ckpt"),
                CheckpointAdmissionPolicy(
                    max_tensor_count=1,
                    max_tensor_elements=100,
                    max_tensor_bytes=400,
                    max_aggregate_tensor_elements=100,
                    max_aggregate_tensor_bytes=400,
                ),
            )

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux isolation")
    def test_container_attributes_are_accounted_in_process_and_isolated(self) -> None:
        state = OrderedDict(weight=torch.ones(1))
        state.hidden = {"tensor": torch.ones(100)}  # type: ignore[attr-defined]
        payload = _payload(state_dict=state)
        policy = CheckpointAdmissionPolicy(
            max_tensor_count=1,
            max_tensor_elements=100,
            max_tensor_bytes=400,
            max_aggregate_tensor_elements=100,
            max_aggregate_tensor_bytes=400,
            require_isolated_decode=True,
        )

        with self.assertRaisesRegex(
            InvalidCheckpointContinuation,
            "2 tensors.*limit of 1",
        ):
            validate_checkpoint_payload(payload, Path("attributes.ckpt"), policy)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "attributes.ckpt"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(
                InvalidCheckpointContinuation,
                "2 tensors.*limit of 1",
            ):
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(checkpoint),
                    SimpleNamespace(runs=(object(),)),
                    admission_policy=policy,
                )

    def test_nested_wide_payload_stops_at_the_node_budget(self) -> None:
        nested = {"outer": [{str(index): index for index in range(50)}]}

        with self.assertRaisesRegex(
            InvalidCheckpointContinuation,
            "more than 20 payload values",
        ):
            validate_checkpoint_payload(
                _payload(callback_state=nested),
                Path("wide.ckpt"),
                CheckpointAdmissionPolicy(max_payload_nodes=20),
            )

    def test_cycles_terminate_and_container_depth_is_bounded(self) -> None:
        cycle: list[object] = []
        cycle.append(cycle)
        validated = validate_checkpoint_payload(
            _payload(callback_state=cycle),
            Path("cycle.ckpt"),
            CheckpointAdmissionPolicy(),
        )
        self.assertLess(validated.receipt.payload_nodes, 30)

        deep: object = "leaf"
        for _ in range(5):
            deep = [deep]
        with self.assertRaisesRegex(
            InvalidCheckpointContinuation,
            "containers deeper than the limit of 2",
        ):
            validate_checkpoint_payload(
                _payload(callback_state=deep),
                Path("deep.ckpt"),
                CheckpointAdmissionPolicy(max_container_depth=2),
            )

    def test_tensor_logical_and_storage_limits_cover_aliasing_views(self) -> None:
        cases = (
            (
                {"state_dict": {"weight": torch.ones(2)}},
                {"max_tensor_elements": 1},
                "2 elements.*per-tensor limit of 1",
            ),
            (
                {"state_dict": {"weight": torch.ones(2)}},
                {"max_tensor_bytes": 4},
                "8 logical bytes.*per-tensor limit of 4",
            ),
            (
                {
                    "state_dict": {
                        "first": torch.ones(2),
                        "second": torch.ones(2),
                    }
                },
                {
                    "max_tensor_elements": 2,
                    "max_aggregate_tensor_elements": 3,
                },
                "4 aggregate tensor elements.*limit of 3",
            ),
            (
                {
                    "state_dict": {
                        "first": torch.ones(2),
                        "second": torch.ones(2),
                    }
                },
                {
                    "max_tensor_bytes": 8,
                    "max_aggregate_tensor_bytes": 12,
                },
                "16 aggregate logical tensor bytes.*limit of 12",
            ),
        )
        for payload_values, policy_values, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(InvalidCheckpointContinuation, message),
            ):
                validate_checkpoint_payload(
                    _payload(**payload_values),
                    Path("tensor-limit.ckpt"),
                    CheckpointAdmissionPolicy(**policy_values),  # type: ignore[arg-type]
                )

        backing = torch.ones(100)
        with self.assertRaisesRegex(
            InvalidCheckpointContinuation,
            "400 unique tensor storage bytes.*limit of 4",
        ):
            validate_checkpoint_payload(
                _payload(state_dict={"tiny-view": backing[:1]}),
                Path("storage.ckpt"),
                CheckpointAdmissionPolicy(max_unique_storage_bytes=4),
            )

    def test_tensor_dtype_layout_and_device_policy_fails_closed(self) -> None:
        cases = (
            (
                torch.ones(1, dtype=torch.int64),
                CheckpointAdmissionPolicy(
                    allowed_dtypes=frozenset({"torch.float32"})
                ),
                "unsupported dtype torch.int64",
            ),
            (
                torch.sparse_coo_tensor([[0]], [1.0], (1,)),
                CheckpointAdmissionPolicy(),
                "unsupported layout",
            ),
            (
                torch.empty(1, device="meta"),
                CheckpointAdmissionPolicy(),
                "unsupported device meta",
            ),
        )
        for tensor, policy, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(InvalidCheckpointContinuation, message),
            ):
                validate_checkpoint_payload(
                    _payload(state_dict={"weight": tensor}),
                    Path("tensor-policy.ckpt"),
                    policy,
                )


class CheckpointWorkerProtocolTests(unittest.TestCase):
    def test_response_rejects_boolean_counters_and_invalid_hashes(self) -> None:
        receipt = CheckpointPayloadReceipt(
            lightning_version_sha256="0" * 64,
            epoch=0,
            completed_epochs=1,
            global_step=1,
            state_dict_keys=1,
            optimizer_states=1,
            archive_records=1,
            archive_uncompressed_bytes=100,
            payload_nodes=10,
            maximum_container_depth=2,
            tensor_count=1,
            tensor_dimensions=1,
            tensor_elements=1,
            tensor_bytes=4,
            unique_storage_bytes=4,
            scalar_bytes=16,
        )
        controls = {
            "addressSpaceBytes": 1024,
            "coreBytes": 0,
            "cpuSeconds": 1,
            "fileSizeBytes": 4096,
        }
        valid = json.loads(
            encode_worker_receipt("a" * 64, 100, receipt, controls)
        )
        corruptions = (
            {**valid, "sizeBytes": True},
            {**valid, "sha256": "A" * 64},
            {
                **valid,
                "receipt": {**asdict(receipt), "tensor_count": True},
            },
            {
                **valid,
                "receipt": {**asdict(receipt), "extra": 1},
            },
        )

        for payload in corruptions:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                decode_worker_response(json.dumps(payload).encode("utf-8"))

    def test_pre_limit_worker_import_graph_is_torch_free(self) -> None:
        script = """
import sys
import model_runtime.runs._checkpoint_worker_protocol
import model_runtime.runs._checkpoint_worker
for prefix in ('torch', 'lightning', 'models', 'emperor', 'emperor_workbench'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
        subprocess.run(  # noqa: S603 - fixed interpreter and test script
            [sys.executable, "-P", "-c", script],
            check=True,
            capture_output=True,
            timeout=10,
        )

    def test_worker_applies_hard_controls_before_importing_torch(self) -> None:
        script = r"""
import io
import sys
import tempfile
from pathlib import Path
from model_runtime.runs.checkpoint_admission import CheckpointAdmissionPolicy
from model_runtime.runs._checkpoint_worker_protocol import CheckpointWorkerRequest, encode_worker_request
from model_runtime.runs import _checkpoint_worker as worker

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    marker = root / 'marker'
    request = encode_worker_request(CheckpointWorkerRequest(
        snapshot_path=root / 'missing.ckpt',
        receipt_path=root / 'receipt.json',
        source_name='source.ckpt',
        sha256='0' * 64,
        size_bytes=0,
        policy=CheckpointAdmissionPolicy(),
    ))
    def controls(*_args):
        forbidden = ('torch', 'lightning', 'models', 'emperor', 'emperor_workbench')
        dirty = any(
            name == prefix or name.startswith(prefix + '.')
            for prefix in forbidden
            for name in sys.modules
        )
        marker.write_text('dirty' if dirty else 'clean', encoding='utf-8')
        return {'addressSpaceBytes': 1, 'coreBytes': 0, 'cpuSeconds': 1, 'fileSizeBytes': 4096}
    worker._apply_hard_limits = controls
    sys.stdin = io.TextIOWrapper(io.BytesIO(request), encoding='utf-8')
    assert worker.main() == 0
    assert marker.read_text(encoding='utf-8') == 'clean'
"""
        subprocess.run(  # noqa: S603 - fixed interpreter and test script
            [sys.executable, "-P", "-c", script],
            check=True,
            capture_output=True,
            timeout=20,
        )

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux controls")
    def test_worker_clamps_to_an_existing_stronger_hard_limit(self) -> None:
        script = r"""
import resource
from model_runtime.runs._checkpoint_worker import _effective_limit
resource.setrlimit(resource.RLIMIT_FSIZE, (2048, 2048))
assert _effective_limit(resource.RLIMIT_FSIZE, 4096) == 2048
assert resource.getrlimit(resource.RLIMIT_FSIZE) == (2048, 2048)
"""
        subprocess.run(  # noqa: S603 - fixed interpreter and test script
            [sys.executable, "-P", "-c", script],
            check=True,
            capture_output=True,
            timeout=10,
        )

    def test_worker_launch_stays_in_the_callers_process_group(self) -> None:
        with patch.object(
            checkpoint_isolation.subprocess,
            "Popen",
            return_value=object(),
        ) as popen:
            checkpoint_isolation._start_worker()

        launch_options = popen.call_args.kwargs
        self.assertNotIn("start_new_session", launch_options)
        self.assertNotIn("process_group", launch_options)

    def test_interrupted_worker_communication_kills_and_reaps_the_worker(self) -> None:
        process = SimpleNamespace(
            communicate=Mock(side_effect=KeyboardInterrupt),
            kill=Mock(),
            wait=Mock(return_value=0),
            stdin=SimpleNamespace(close=Mock()),
            returncode=None,
        )

        with self.assertRaises(KeyboardInterrupt):
            checkpoint_isolation._communicate_with_worker(
                process,  # type: ignore[arg-type]
                b"request",
                CheckpointAdmissionPolicy(),
            )

        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5.0)
        process.stdin.close.assert_called_once_with()

    def test_abnormal_worker_return_is_rejected(self) -> None:
        process = SimpleNamespace(
            communicate=Mock(return_value=(b"", b"")),
            stdin=SimpleNamespace(close=Mock()),
            returncode=-signal.SIGKILL,
        )
        with (
            patch.object(
                checkpoint_isolation,
                "_start_worker",
                return_value=process,
            ),
            self.assertRaisesRegex(
                InvalidCheckpointContinuation,
                "terminated abnormally",
            ),
        ):
            checkpoint_isolation.decode_checkpoint_isolated(
                Path("snapshot.ckpt"),
                Path("source.ckpt"),
                "0" * 64,
                0,
                CheckpointAdmissionPolicy(),
            )

        process.communicate.assert_called_once()
        process.stdin.close.assert_called_once_with()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process groups")
    def test_outer_process_group_cancellation_reaches_the_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decoder_pid_path = root / "decoder.pid"
            worker_script = (
                "import os,time; from pathlib import Path; "
                "Path(os.environ['DECODER_PID_PATH']).write_text("
                "str(os.getpid()), encoding='utf-8'); time.sleep(60)"
            )
            outer_script = f"""
import sys
from pathlib import Path
from model_runtime.runs import _checkpoint_isolation as isolation
from model_runtime.runs.checkpoint_admission import CheckpointAdmissionPolicy
isolation._WORKER_COMMAND = (sys.executable, '-P', '-c', {worker_script!r})
isolation.decode_checkpoint_isolated(
    Path({str(root / 'snapshot.ckpt')!r}),
    Path('source.ckpt'),
    '0' * 64,
    0,
    CheckpointAdmissionPolicy(worker_wall_timeout_seconds=60),
)
"""
            environment = dict(os.environ)
            environment["DECODER_PID_PATH"] = str(decoder_pid_path)
            outer = subprocess.Popen(  # noqa: S603 - fixed interpreter/test script
                [sys.executable, "-P", "-c", outer_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=environment,
            )
            decoder_pid: int | None = None
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and not decoder_pid_path.exists():
                    time.sleep(0.02)
                self.assertTrue(decoder_pid_path.exists())
                decoder_pid = int(decoder_pid_path.read_text(encoding="utf-8"))
                self.assertEqual(os.getpgid(decoder_pid), outer.pid)

                os.killpg(outer.pid, signal.SIGTERM)
                outer.wait(timeout=5)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and _process_is_running(decoder_pid):
                    time.sleep(0.02)
                self.assertFalse(_process_is_running(decoder_pid))
            finally:
                if outer.poll() is None:
                    os.killpg(outer.pid, signal.SIGKILL)
                    outer.wait(timeout=5)
                if decoder_pid is not None and _process_is_running(decoder_pid):
                    os.kill(decoder_pid, signal.SIGKILL)

    def test_parent_rejects_malformed_receipts_and_control_attestations(self) -> None:
        policy = CheckpointAdmissionPolicy()
        receipt = CheckpointPayloadReceipt(
            lightning_version_sha256="0" * 64,
            epoch=0,
            completed_epochs=1,
            global_step=1,
            state_dict_keys=1,
            optimizer_states=1,
            archive_records=1,
            archive_uncompressed_bytes=100,
            payload_nodes=10,
            maximum_container_depth=2,
            tensor_count=1,
            tensor_dimensions=1,
            tensor_elements=1,
            tensor_bytes=4,
            unique_storage_bytes=4,
            scalar_bytes=16,
        )
        with tempfile.TemporaryDirectory() as tmp:
            response_path = Path(tmp) / "receipt.json"
            cases = (
                b"{}",
                encode_worker_receipt(
                    "a" * 64,
                    100,
                    receipt,
                    {
                        "addressSpaceBytes": policy.worker_memory_bytes + 1,
                        "coreBytes": 0,
                        "cpuSeconds": 1,
                        "fileSizeBytes": 4096,
                    },
                ),
            )
            for raw in cases:
                with (
                    self.subTest(raw=raw),
                    self.assertRaisesRegex(
                        InvalidCheckpointContinuation,
                        "invalid receipt",
                    ),
                ):
                    response_path.write_bytes(raw)
                    checkpoint_isolation._validated_worker_receipt(
                        response_path,
                        Path("source.ckpt"),
                        "a" * 64,
                        100,
                        policy,
                    )


class CheckpointSnapshotSecurityTests(unittest.TestCase):
    def test_invalid_source_path_uses_the_domain_error_interface(self) -> None:
        with self.assertRaisesRegex(
            InvalidCheckpointContinuation,
            "readable regular file",
        ):
            CheckpointContinuationLifecycle.admit(
                CheckpointContinuation(Path("invalid\0checkpoint.ckpt")),
                SimpleNamespace(runs=(object(),)),
            )

    def test_temporary_root_creation_and_mode_failures_use_domain_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.ckpt"
            torch.save(_payload(), source)
            with (
                patch.object(
                    checkpoint_snapshot.tempfile,
                    "mkdtemp",
                    side_effect=OSError("creation failed"),
                ),
                self.assertRaisesRegex(
                    InvalidCheckpointContinuation,
                    "could not create a private snapshot",
                ),
            ):
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(source),
                    SimpleNamespace(runs=(object(),)),
                )

            leaked_root = Path(tmp) / "would-leak"
            original_chmod = Path.chmod

            def failing_chmod(path: Path, mode: int) -> None:
                if path == leaked_root:
                    raise OSError("mode failed")
                original_chmod(path, mode)

            with (
                patch.object(
                    checkpoint_snapshot.tempfile,
                    "mkdtemp",
                    return_value=str(leaked_root),
                ),
                patch.object(Path, "chmod", autospec=True, side_effect=failing_chmod),
                self.assertRaisesRegex(
                    InvalidCheckpointContinuation,
                    "could not create a private snapshot",
                ),
            ):
                leaked_root.mkdir()
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(source),
                    SimpleNamespace(runs=(object(),)),
                )

            self.assertFalse(leaked_root.exists())

            cancelled_root = Path(tmp) / "cancelled-root"

            def cancelled_chmod(path: Path, mode: int) -> None:
                if path == cancelled_root:
                    raise KeyboardInterrupt
                original_chmod(path, mode)

            with (
                patch.object(
                    checkpoint_snapshot.tempfile,
                    "mkdtemp",
                    return_value=str(cancelled_root),
                ),
                patch.object(
                    Path,
                    "chmod",
                    autospec=True,
                    side_effect=cancelled_chmod,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                cancelled_root.mkdir()
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(source),
                    SimpleNamespace(runs=(object(),)),
                )

            self.assertFalse(cancelled_root.exists())

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux controls")
    def test_worker_memory_control_fails_closed_and_cleans_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.ckpt"
            torch.save(_payload(), source)
            created_roots: list[Path] = []
            original_mkdtemp = checkpoint_snapshot.tempfile.mkdtemp

            def tracked_mkdtemp(*args: object, **kwargs: object) -> str:
                root = original_mkdtemp(*args, **kwargs)  # type: ignore[arg-type]
                created_roots.append(Path(root))
                return root

            with (
                patch.object(
                    checkpoint_snapshot.tempfile,
                    "mkdtemp",
                    side_effect=tracked_mkdtemp,
                ),
                self.assertRaisesRegex(
                    InvalidCheckpointContinuation,
                    "terminated abnormally",
                ),
            ):
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(source),
                    SimpleNamespace(runs=(object(),)),
                    admission_policy=CheckpointAdmissionPolicy(
                        require_isolated_decode=True,
                        worker_memory_bytes=64 * 1024**2,
                    ),
                )

            self.assertTrue(created_roots)
            self.assertTrue(all(not root.exists() for root in created_roots))

    def test_snapshot_cleanup_is_retryable_after_a_transient_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.ckpt"
            torch.save(_payload(), source)
            lifecycle = CheckpointContinuationLifecycle.admit(
                CheckpointContinuation(source),
                SimpleNamespace(runs=(object(),)),
            )
            execution = lifecycle.bind_training_runs([SimpleNamespace(num_epochs=2)])
            snapshot = execution.checkpoint_path
            assert snapshot is not None
            original_rmtree = checkpoint_snapshot.shutil.rmtree
            attempts = 0

            def fail_once(path: object) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError("transient cleanup failure")
                original_rmtree(path)  # type: ignore[arg-type]

            with patch.object(
                checkpoint_snapshot.shutil,
                "rmtree",
                side_effect=fail_once,
            ):
                with self.assertRaisesRegex(
                    InvalidCheckpointContinuation,
                    "snapshot could not be removed",
                ):
                    lifecycle.close()
                self.assertTrue(snapshot.exists())
                lifecycle.close()

            self.assertEqual(attempts, 2)
            self.assertFalse(snapshot.exists())

    def test_cancellation_during_snapshot_copy_is_preserved_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.ckpt"
            torch.save(_payload(), source)
            created_roots: list[Path] = []
            original_mkdtemp = checkpoint_snapshot.tempfile.mkdtemp

            def tracked_mkdtemp(*args: object, **kwargs: object) -> str:
                root = original_mkdtemp(*args, **kwargs)  # type: ignore[arg-type]
                created_roots.append(Path(root))
                return root

            with (
                patch.object(
                    checkpoint_snapshot.tempfile,
                    "mkdtemp",
                    side_effect=tracked_mkdtemp,
                ),
                patch.object(
                    checkpoint_snapshot,
                    "_copy_checkpoint_bytes",
                    side_effect=KeyboardInterrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(source),
                    SimpleNamespace(runs=(object(),)),
                )

            self.assertTrue(created_roots)
            self.assertTrue(all(not root.exists() for root in created_roots))

    def test_cancellation_after_snapshot_is_preserved_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.ckpt"
            torch.save(_payload(), source)
            created_roots: list[Path] = []
            original_mkdtemp = checkpoint_snapshot.tempfile.mkdtemp

            def tracked_mkdtemp(*args: object, **kwargs: object) -> str:
                root = original_mkdtemp(*args, **kwargs)  # type: ignore[arg-type]
                created_roots.append(Path(root))
                return root

            with (
                patch.object(
                    checkpoint_snapshot.tempfile,
                    "mkdtemp",
                    side_effect=tracked_mkdtemp,
                ),
                patch(
                    "model_runtime.runs.checkpoints._isolated_payload_receipt",
                    side_effect=KeyboardInterrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(source),
                    SimpleNamespace(runs=(object(),)),
                )

            self.assertTrue(created_roots)
            self.assertTrue(all(not root.exists() for root in created_roots))

    def test_worker_failure_never_falls_back_to_parent_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.ckpt"
            torch.save(_payload(), source)
            created_roots: list[Path] = []
            original_mkdtemp = checkpoint_snapshot.tempfile.mkdtemp

            def tracked_mkdtemp(*args: object, **kwargs: object) -> str:
                root = original_mkdtemp(*args, **kwargs)  # type: ignore[arg-type]
                created_roots.append(Path(root))
                return root

            for message in ("terminated abnormally", "memory limit exceeded"):
                with (
                    self.subTest(message=message),
                    patch.object(
                        checkpoint_snapshot.tempfile,
                        "mkdtemp",
                        side_effect=tracked_mkdtemp,
                    ),
                    patch(
                        "model_runtime.runs.checkpoints.isolated_decode_available",
                        return_value=True,
                    ),
                    patch(
                        "model_runtime.runs.checkpoints.decode_checkpoint_isolated",
                        side_effect=InvalidCheckpointContinuation(message),
                    ),
                    patch("model_runtime.runs.checkpoints.torch.load") as parent_load,
                    self.assertRaisesRegex(InvalidCheckpointContinuation, message),
                ):
                    CheckpointContinuationLifecycle.admit(
                        CheckpointContinuation(source),
                        SimpleNamespace(runs=(object(),)),
                    )
                parent_load.assert_not_called()

            self.assertTrue(created_roots)
            self.assertTrue(all(not root.exists() for root in created_roots))

    def test_required_isolation_fails_closed_when_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.ckpt"
            torch.save(_payload(), source)
            with (
                patch(
                    "model_runtime.runs.checkpoints.isolated_decode_available",
                    return_value=False,
                ),
                patch("model_runtime.runs.checkpoints.torch.load") as parent_load,
                self.assertRaisesRegex(
                    InvalidCheckpointContinuation,
                    "hard decoder controls are unavailable",
                ),
            ):
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(source),
                    SimpleNamespace(runs=(object(),)),
                    admission_policy=CheckpointAdmissionPolicy(
                        require_isolated_decode=True
                    ),
                )
            parent_load.assert_not_called()

    def test_compressed_archive_is_rejected_before_load_without_temp_path_leak(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "compressed.ckpt"
            with zipfile.ZipFile(
                source,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                archive.writestr("large-record", b"x" * 10_000)

            with self.assertRaisesRegex(
                InvalidCheckpointContinuation,
                "archive record expands.*limit of 100 bytes",
            ) as raised:
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(source),
                    SimpleNamespace(runs=(object(),)),
                    admission_policy=CheckpointAdmissionPolicy(
                        max_archive_record_bytes=100,
                        max_archive_uncompressed_bytes=1_000,
                    ),
                )

            message = str(raised.exception)
            self.assertIn(str(source), message)
            self.assertNotIn("emperor-checkpoint", message)
            self.assertNotIn("admitted.ckpt", message)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO requires POSIX")
    def test_fifo_is_rejected_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "checkpoint.fifo"
            os.mkfifo(source)

            with self.assertRaisesRegex(
                InvalidCheckpointContinuation,
                "readable regular file",
            ):
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(source),
                    SimpleNamespace(runs=(object(),)),
                )

    def test_symlink_to_regular_file_preserves_requested_provenance_and_modes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.ckpt"
            link = root / "requested.ckpt"
            torch.save(_payload(), target)
            link.symlink_to(target)

            with CheckpointContinuationLifecycle.admit(
                CheckpointContinuation(link),
                SimpleNamespace(runs=(object(),)),
            ) as lifecycle:
                execution = lifecycle.bind_training_runs(
                    [SimpleNamespace(num_epochs=2)]
                )
                snapshot = execution.checkpoint_path
                assert snapshot is not None
                self.assertEqual(snapshot.stat().st_mode & 0o777, 0o400)
                self.assertEqual(snapshot.parent.stat().st_mode & 0o777, 0o700)
                self.assertEqual(execution.provenance["checkpoint"], link.name)
                self.assertEqual(
                    execution.provenance["sha256"],
                    hashlib.sha256(target.read_bytes()).hexdigest(),
                )

            self.assertFalse(snapshot.exists())


if __name__ == "__main__":
    unittest.main()
