from __future__ import annotations

import random
import subprocess
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from queue import Empty, Queue
from typing import Any, TypeVar

from model_runtime.cli import (
    configuration_schema_from_wire,
    inspection_result_from_wire,
    run_plan_from_wire,
    search_space_from_wire,
    to_wire,
)
from model_runtime.inspection import (
    ConfigurationSchema,
    InspectionRequest,
    InspectionResult,
    ParsedOverrides,
    SearchSpace,
)
from model_runtime.packages import split_model_id
from model_runtime.runs import PlanningBudget, RunPlan, RunRequest, SubmittedRun

from emperor_workbench.failures import FailureKind
from emperor_workbench.project_adapter._contracts import ModelPackageReference
from emperor_workbench.project_adapter._errors import ProjectAdapterFailure
from emperor_workbench.project_adapter._wire import (
    MAX_PROJECT_ADAPTER_RESPONSE_BYTES,
    ProjectAdapterProtocolFailure,
    decode_response,
    default_project_adapter_command,
    encode_request,
    require_field,
    require_list,
    require_mapping,
    require_string,
    tuple_tree,
)

DecodedT = TypeVar("DecodedT")


class ProjectAdapterClient:
    """Explicit owner of one project Adapter process and its metadata cache."""

    def __init__(
        self,
        command: tuple[str, ...] | list[str] | None = None,
        *,
        timeout_seconds: float | None = 120.0,
        persistent: bool = True,
    ) -> None:
        self.command = tuple(command or default_project_adapter_command())
        self.timeout_seconds = timeout_seconds
        self.persistent = persistent
        self._process: subprocess.Popen[bytes] | None = None
        self._process_lock = threading.Lock()
        self._metadata_lock = threading.Lock()
        self._metadata_cache: dict[str, dict[str, Any]] = {}
        self._closed = False

    def close(self) -> None:
        with self._metadata_lock:
            with self._process_lock:
                if self._closed:
                    return
                self._closed = True
                self._metadata_cache.clear()
                process = self._process
                self._process = None
                if process is not None:
                    _shutdown_process(process)

    def __enter__(self) -> ProjectAdapterClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def call(self, operation: str, payload: dict[str, Any] | None = None) -> Any:
        encoded = encode_request(
            operation,
            payload,
            line_delimited=self.persistent,
        )
        started_at = time.monotonic()
        if self.persistent:
            return self._persistent_call(encoded, started_at=started_at)
        self._acquire_process_lock(started_at)
        try:
            self._require_open()
            return self._one_shot_call(encoded, started_at=started_at)
        finally:
            self._process_lock.release()

    def _one_shot_call(self, encoded: bytes, *, started_at: float) -> Any:
        try:
            process = subprocess.Popen(  # noqa: S603 - configured Adapter command
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise ProjectAdapterFailure(
                "The model project Adapter is unavailable.",
                kind=FailureKind.UNAVAILABLE,
            ) from exc
        try:
            try:
                raw_response = self._one_shot_exchange(
                    process,
                    encoded,
                    timeout=self._remaining_timeout(started_at),
                )
            except OSError as exc:
                raise ProjectAdapterFailure(
                    "The model project Adapter exited unexpectedly.",
                    kind=FailureKind.UNAVAILABLE,
                ) from exc
            if len(raw_response) > MAX_PROJECT_ADAPTER_RESPONSE_BYTES:
                raise ProjectAdapterFailure(
                    "The model project Adapter response exceeded its size limit.",
                    kind=FailureKind.TOO_LARGE,
                )
            try:
                process.wait(timeout=self._remaining_timeout(started_at))
            except subprocess.TimeoutExpired as exc:
                raise self._timeout_failure() from exc
            decoded_response = decode_response(raw_response)
            if process.returncode != 0:
                raise ProjectAdapterFailure(
                    "The model project Adapter exited unexpectedly.",
                    kind=FailureKind.UNAVAILABLE,
                )
            return decoded_response
        finally:
            _shutdown_process(process)

    def _one_shot_exchange(
        self,
        process: subprocess.Popen[bytes],
        encoded: bytes,
        *,
        timeout: float | None,
    ) -> bytes:
        if process.stdin is None or process.stdout is None:
            raise ProjectAdapterFailure(
                "The model project Adapter has no protocol streams.",
                kind=FailureKind.UNAVAILABLE,
            )

        def exchange() -> bytes:
            process.stdin.write(encoded)
            process.stdin.close()
            return process.stdout.read(MAX_PROJECT_ADAPTER_RESPONSE_BYTES + 1)

        return self._bounded_exchange(
            process,
            exchange,
            timeout=timeout,
            thread_name="project-adapter-one-shot",
        )

    def _persistent_call(self, encoded: bytes, *, started_at: float) -> Any:
        self._acquire_process_lock(started_at)
        try:
            self._require_open()
            process = self._process
            if process is None or process.poll() is not None:
                if process is not None:
                    _shutdown_process(process)
                process = self._start_process()
                self._process = process
            if process.stdin is None or process.stdout is None:
                raise ProjectAdapterFailure(
                    "The model project Adapter has no protocol streams.",
                    kind=FailureKind.UNAVAILABLE,
                )
            try:
                raw_response = self._persistent_exchange(
                    process,
                    encoded,
                    timeout=self._remaining_timeout(started_at),
                )
            except OSError as exc:
                self._discard_process(process)
                raise ProjectAdapterFailure(
                    "The model project Adapter exited unexpectedly.",
                    kind=FailureKind.UNAVAILABLE,
                ) from exc
            if (
                not raw_response
                or len(raw_response) > MAX_PROJECT_ADAPTER_RESPONSE_BYTES
            ):
                self._discard_process(process)
                raise ProjectAdapterFailure(
                    "The model project Adapter response is missing or too large.",
                    kind=(
                        FailureKind.TOO_LARGE
                        if raw_response
                        else FailureKind.UNAVAILABLE
                    ),
                )
            try:
                return decode_response(raw_response)
            except ProjectAdapterProtocolFailure:
                self._discard_process(process)
                raise
        finally:
            self._process_lock.release()

    def _persistent_exchange(
        self,
        process: subprocess.Popen[bytes],
        encoded: bytes,
        *,
        timeout: float | None,
    ) -> bytes:
        if process.stdin is None or process.stdout is None:
            raise ProjectAdapterFailure(
                "The model project Adapter has no protocol streams.",
                kind=FailureKind.UNAVAILABLE,
            )

        def exchange() -> bytes:
            process.stdin.write(encoded + b"\n")
            process.stdin.flush()
            return process.stdout.readline(MAX_PROJECT_ADAPTER_RESPONSE_BYTES + 1)

        if timeout is None:
            return exchange()

        return self._bounded_exchange(
            process,
            exchange,
            timeout=timeout,
            thread_name="project-adapter-exchange",
            discard_persistent=True,
        )

    def _bounded_exchange(
        self,
        process: subprocess.Popen[bytes],
        exchange: Callable[[], bytes],
        *,
        timeout: float | None,
        thread_name: str,
        discard_persistent: bool = False,
    ) -> bytes:
        if timeout is None:
            return exchange()

        response_queue: Queue[bytes | BaseException] = Queue(maxsize=1)

        def run_exchange() -> None:
            try:
                response: bytes | BaseException = exchange()
            except BaseException as exc:  # pragma: no cover - OS pipe failure
                response = exc
            response_queue.put(response)

        worker = threading.Thread(
            target=run_exchange,
            name=thread_name,
            daemon=True,
        )
        worker.start()
        try:
            response = response_queue.get(timeout=timeout)
        except Empty as exc:
            if discard_persistent:
                self._discard_process(process)
            else:
                _shutdown_process(process)
            worker.join(timeout=1.0)
            raise self._timeout_failure() from exc
        worker.join()
        if isinstance(response, BaseException):
            raise response
        return response

    def _acquire_process_lock(self, started_at: float) -> None:
        timeout = self._remaining_timeout(started_at)
        if timeout is None:
            self._process_lock.acquire()
            return
        if not self._process_lock.acquire(timeout=timeout):
            raise self._timeout_failure()

    def _remaining_timeout(self, started_at: float) -> float | None:
        if self.timeout_seconds is None:
            return None
        remaining = self.timeout_seconds - (time.monotonic() - started_at)
        if remaining <= 0:
            raise self._timeout_failure()
        return remaining

    @staticmethod
    def _timeout_failure() -> ProjectAdapterFailure:
        return ProjectAdapterFailure(
            "The model project Adapter request exceeded its time limit.",
            kind=FailureKind.TIMEOUT,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise ProjectAdapterFailure(
                "The model project Adapter client is closed.",
                kind=FailureKind.UNAVAILABLE,
            )

    def _discard_process(self, process: subprocess.Popen[bytes]) -> None:
        if self._process is process:
            self._process = None
        _shutdown_process(process)

    def _start_process(self) -> subprocess.Popen[bytes]:
        try:
            return subprocess.Popen(  # noqa: S603 - configured Adapter command
                [*self.command, "--serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        except OSError as exc:
            raise ProjectAdapterFailure(
                "The model project Adapter is unavailable.",
                kind=FailureKind.UNAVAILABLE,
            ) from exc

    def _package_metadata(self, model_id: str) -> dict[str, Any]:
        with self._metadata_lock:
            cached_metadata = self._metadata_cache.get(model_id)
            if cached_metadata is None:
                cached_metadata = require_mapping(
                    self.call(
                        "package_metadata",
                        {"model_id": model_id},
                    )
                )
                self._metadata_cache[model_id] = cached_metadata
            return deepcopy(cached_metadata)

    def catalog(self) -> list[ModelPackageReference]:
        catalog_payload = require_list(self.call("catalog"))
        references: list[ModelPackageReference] = []
        for raw_package in catalog_payload:
            package_payload = require_mapping(raw_package)
            references.append(
                ModelPackageReference(
                    model_type=require_string(
                        require_field(package_payload, "modelType")
                    ),
                    model=require_string(require_field(package_payload, "model")),
                    client=self,
                )
            )
        return references

    def package(self, model_id: str) -> ModelPackageReference:
        identity = split_model_id(model_id)
        if identity is None:
            raise ProjectAdapterFailure(f"Unknown model: {model_id}")
        reference = ModelPackageReference(
            identity.model_type,
            identity.model,
            self,
        )
        reference.metadata_payload()
        return reference

    def configuration(self, model_id: str, preset: str | None) -> ConfigurationSchema:
        configuration_payload = self.call(
            "configuration",
            {"model_id": model_id, "preset": preset},
        )
        return _decode_wire_result(
            configuration_schema_from_wire,
            configuration_payload,
            name="configuration",
        )

    def search_space(
        self,
        model_id: str,
        preset: str | None,
        presets: tuple[str, ...] | list[str] | None = None,
    ) -> SearchSpace:
        search_space_payload = self.call(
            "search_space",
            {
                "model_id": model_id,
                "preset": preset,
                "presets": list(presets) if presets is not None else None,
            },
        )
        return _decode_wire_result(
            search_space_from_wire,
            search_space_payload,
            name="search space",
        )

    def inspect(self, model_id: str, request: InspectionRequest) -> InspectionResult:
        overrides = (
            request.overrides.values
            if isinstance(request.overrides, ParsedOverrides)
            else request.overrides
        )
        inspection_payload = self.call(
            "inspect",
            {
                "model_id": model_id,
                "preset": request.preset,
                "overrides": to_wire(overrides),
                "dataset": request.dataset,
                "experiment_task": request.experiment_task,
            },
        )
        return _decode_wire_result(
            inspection_result_from_wire,
            inspection_payload,
            name="inspection",
        )

    def execute_run_plan(
        self,
        model_id: str,
        plan: RunPlan,
        *,
        logs_root: str,
        log_folder: str | None,
        progress_path: str,
        progress_step_interval: int,
        monitors: tuple[str, ...] | list[str],
    ) -> Any:
        return self.call(
            "execute_run_plan",
            {
                "model_id": model_id,
                "plan": to_wire(plan),
                "logs_root": logs_root,
                "log_folder": log_folder,
                "progress_path": progress_path,
                "progress_step_interval": progress_step_interval,
                "monitors": list(monitors),
            },
        )

    def plan_runs(
        self,
        model_id: str,
        request: RunRequest,
        *,
        budget: PlanningBudget,
        random_source: random.Random | None = None,
    ) -> RunPlan:
        plan_response = self.call(
            "plan_runs",
            {
                "model_id": model_id,
                "request": to_wire(request),
                "budget": to_wire(budget),
                "random_state": (
                    to_wire(random_source.getstate())
                    if random_source is not None
                    else None
                ),
            },
        )
        plan_payload = require_mapping(plan_response)
        if random_source is not None and plan_payload.get("random_state") is not None:
            try:
                random_source.setstate(tuple_tree(plan_payload["random_state"]))
            except (TypeError, ValueError) as exc:
                raise ProjectAdapterProtocolFailure(
                    "The project Adapter random state is invalid."
                ) from exc
        return _decode_wire_result(
            run_plan_from_wire,
            require_field(plan_payload, "plan"),
            name="Run Plan",
        )

    def accept_run_plan(
        self,
        model_id: str,
        request: RunRequest,
        runs: tuple[SubmittedRun, ...] | list[SubmittedRun],
        *,
        budget: PlanningBudget,
    ) -> RunPlan:
        accepted_plan_payload = self.call(
            "accept_run_plan",
            {
                "model_id": model_id,
                "request": to_wire(request),
                "runs": to_wire(runs),
                "budget": to_wire(budget),
            },
        )
        return _decode_wire_result(
            run_plan_from_wire,
            accepted_plan_payload,
            name="Run Plan",
        )


def _decode_wire_result(
    decoder: Callable[[dict[str, Any]], DecodedT],
    value: object,
    *,
    name: str,
) -> DecodedT:
    try:
        return decoder(require_mapping(value))
    except ProjectAdapterProtocolFailure:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectAdapterProtocolFailure(
            f"The project Adapter {name} result is invalid."
        ) from exc


def _shutdown_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
    if process.stdin is not None:
        try:
            process.stdin.close()
        except (OSError, ValueError):
            pass
    if process.stdout is not None:
        try:
            process.stdout.close()
        except (OSError, ValueError):
            pass
