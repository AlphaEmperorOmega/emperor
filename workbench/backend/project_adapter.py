from __future__ import annotations

import atexit
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar

from model_runtime.cli import (
    PROTOCOL_VERSION,
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
from model_runtime.packages import ModelIdentity
from model_runtime.runs import (
    PlanningBudget,
    RunPlan,
    RunRequest,
    SubmittedRun,
)
from workbench.backend.failures import DomainFailure, FailureKind

PROJECT_ADAPTER_COMMAND_ENV = "EMPEROR_PROJECT_ADAPTER_COMMAND"
MAX_PROJECT_ADAPTER_RESPONSE_BYTES = 64 * 1024 * 1024


class ProjectAdapterFailure(DomainFailure):
    """The configured model project Adapter rejected or failed a request."""

    def __init__(
        self,
        detail: str,
        *,
        kind: FailureKind = FailureKind.INVALID,
        remote_type: str | None = None,
        remote_cause_detail: str | None = None,
    ) -> None:
        super().__init__(detail, kind=kind)
        self.remote_type = remote_type
        self.remote_cause_detail = remote_cause_detail


def _default_command() -> tuple[str, ...]:
    configured = os.environ.get(PROJECT_ADAPTER_COMMAND_ENV)
    if configured:
        command = tuple(shlex.split(configured))
        if command:
            return command
    installed = shutil.which("emperor-project-adapter")
    if installed:
        return (installed,)
    return (sys.executable, "-m", "models.adapter_cli")


class ProjectAdapterClient:
    def __init__(
        self,
        command: tuple[str, ...] | list[str] | None = None,
        *,
        timeout_seconds: float | None = 120.0,
        persistent: bool = True,
    ) -> None:
        self.command = tuple(command or _default_command())
        self.timeout_seconds = timeout_seconds
        self.persistent = persistent
        self._process: subprocess.Popen[bytes] | None = None
        self._process_lock = threading.Lock()

    def close(self) -> None:
        with self._process_lock:
            process = self._process
            self._process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        if process.stdout is not None:
            process.stdout.close()

    def __enter__(self) -> ProjectAdapterClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (OSError, subprocess.SubprocessError):
            pass

    def call(self, operation: str, payload: dict[str, Any] | None = None) -> Any:
        request = {
            "version": PROTOCOL_VERSION,
            "operation": operation,
            "payload": payload or {},
        }
        encoded = json.dumps(
            request,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if self.persistent:
            return self._persistent_call(encoded)
        try:
            completed = subprocess.run(  # noqa: S603 - configured Adapter command
                self.command,
                input=encoded,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProjectAdapterFailure(
                "The model project Adapter is unavailable.",
                kind=FailureKind.UNAVAILABLE,
            ) from exc
        if len(completed.stdout) > MAX_PROJECT_ADAPTER_RESPONSE_BYTES:
            raise ProjectAdapterFailure(
                "The model project Adapter response exceeded its size limit.",
                kind=FailureKind.TOO_LARGE,
            )
        try:
            envelope = json.loads(completed.stdout)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProjectAdapterFailure(
                "The model project Adapter returned an invalid response.",
                kind=FailureKind.UNAVAILABLE,
            ) from exc
        if (
            not isinstance(envelope, dict)
            or envelope.get("version") != PROTOCOL_VERSION
        ):
            raise ProjectAdapterFailure(
                "The model project Adapter returned an incompatible response.",
                kind=FailureKind.UNAVAILABLE,
            )
        if envelope.get("ok") is not True:
            error = envelope.get("error")
            if not isinstance(error, dict) or not isinstance(error.get("message"), str):
                raise ProjectAdapterFailure(
                    "The model project Adapter returned an invalid failure.",
                    kind=FailureKind.UNAVAILABLE,
                )
            raw_kind = error.get("kind")
            try:
                kind = FailureKind(str(raw_kind))
            except ValueError:
                kind = FailureKind.UNAVAILABLE
            cause = error.get("cause")
            raise ProjectAdapterFailure(
                error["message"],
                kind=kind,
                remote_type=(
                    str(error["type"]) if error.get("type") is not None else None
                ),
                remote_cause_detail=(
                    str(cause["message"])
                    if isinstance(cause, dict) and cause.get("message") is not None
                    else None
                ),
            )
        if completed.returncode != 0:
            raise ProjectAdapterFailure(
                "The model project Adapter exited unexpectedly.",
                kind=FailureKind.UNAVAILABLE,
            )
        return envelope.get("result")

    def _persistent_call(self, encoded: bytes) -> Any:
        with self._process_lock:
            process = self._process
            if process is None or process.poll() is not None:
                try:
                    process = subprocess.Popen(  # noqa: S603 - configured Adapter
                        [*self.command, "--serve"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                    )
                except OSError as exc:
                    raise ProjectAdapterFailure(
                        "The model project Adapter is unavailable.",
                        kind=FailureKind.UNAVAILABLE,
                    ) from exc
                self._process = process
            if process.stdin is None or process.stdout is None:
                raise ProjectAdapterFailure(
                    "The model project Adapter has no protocol streams.",
                    kind=FailureKind.UNAVAILABLE,
                )
            try:
                process.stdin.write(encoded + b"\n")
                process.stdin.flush()
                raw_response = process.stdout.readline(
                    MAX_PROJECT_ADAPTER_RESPONSE_BYTES + 1
                )
            except (BrokenPipeError, OSError) as exc:
                self._process = None
                raise ProjectAdapterFailure(
                    "The model project Adapter exited unexpectedly.",
                    kind=FailureKind.UNAVAILABLE,
                ) from exc
        if not raw_response or len(raw_response) > MAX_PROJECT_ADAPTER_RESPONSE_BYTES:
            raise ProjectAdapterFailure(
                "The model project Adapter response is missing or too large.",
                kind=FailureKind.UNAVAILABLE,
            )
        try:
            envelope = json.loads(raw_response)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProjectAdapterFailure(
                "The model project Adapter returned an invalid response.",
                kind=FailureKind.UNAVAILABLE,
            ) from exc
        return self._result_from_envelope(envelope)

    def _result_from_envelope(self, envelope: object) -> Any:
        if (
            not isinstance(envelope, dict)
            or envelope.get("version") != PROTOCOL_VERSION
        ):
            raise ProjectAdapterFailure(
                "The model project Adapter returned an incompatible response.",
                kind=FailureKind.UNAVAILABLE,
            )
        if envelope.get("ok") is not True:
            error = envelope.get("error")
            if not isinstance(error, dict) or not isinstance(error.get("message"), str):
                raise ProjectAdapterFailure(
                    "The model project Adapter returned an invalid failure.",
                    kind=FailureKind.UNAVAILABLE,
                )
            try:
                kind = FailureKind(str(error.get("kind")))
            except ValueError:
                kind = FailureKind.UNAVAILABLE
            cause = error.get("cause")
            raise ProjectAdapterFailure(
                error["message"],
                kind=kind,
                remote_type=(
                    str(error["type"]) if error.get("type") is not None else None
                ),
                remote_cause_detail=(
                    str(cause["message"])
                    if isinstance(cause, dict) and cause.get("message") is not None
                    else None
                ),
            )
        return envelope.get("result")

    def catalog(self) -> list[ModelPackageReference]:
        result = self.call("catalog")
        if not isinstance(result, list):
            raise ProjectAdapterFailure("The project catalog response is invalid.")
        return [
            ModelPackageReference(
                model_type=str(item["modelType"]),
                model=str(item["model"]),
                client=self,
            )
            for item in result
            if isinstance(item, dict)
        ]

    def package(self, model_id: str) -> ModelPackageReference:
        if not isinstance(model_id, str) or len(model_id.split("/")) != 2:
            raise ProjectAdapterFailure(f"Unknown model: {model_id}")
        model_type, model = model_id.split("/", 1)
        reference = ModelPackageReference(model_type, model, self)
        reference.metadata_payload()
        return reference

    def configuration(self, model_id: str, preset: str | None) -> ConfigurationSchema:
        result = self.call(
            "configuration",
            {"model_id": model_id, "preset": preset},
        )
        return configuration_schema_from_wire(_dict(result))

    def search_space(
        self,
        model_id: str,
        preset: str | None,
        presets: tuple[str, ...] | list[str] | None = None,
    ) -> SearchSpace:
        result = self.call(
            "search_space",
            {
                "model_id": model_id,
                "preset": preset,
                "presets": list(presets) if presets is not None else None,
            },
        )
        return search_space_from_wire(_dict(result))

    def inspect(self, model_id: str, request: InspectionRequest) -> InspectionResult:
        overrides = (
            request.overrides.values
            if isinstance(request.overrides, ParsedOverrides)
            else request.overrides
        )
        result = self.call(
            "inspect",
            {
                "model_id": model_id,
                "preset": request.preset,
                "overrides": to_wire(overrides),
                "dataset": request.dataset,
                "experiment_task": request.experiment_task,
            },
        )
        return inspection_result_from_wire(_dict(result))

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
        result = self.call(
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
        result_payload = _dict(result)
        if random_source is not None and result_payload.get("random_state") is not None:
            random_source.setstate(_tuple_tree(result_payload["random_state"]))
        return run_plan_from_wire(_dict(result_payload["plan"]))

    def accept_run_plan(
        self,
        model_id: str,
        request: RunRequest,
        runs: tuple[SubmittedRun, ...] | list[SubmittedRun],
        *,
        budget: PlanningBudget,
    ) -> RunPlan:
        result = self.call(
            "accept_run_plan",
            {
                "model_id": model_id,
                "request": to_wire(request),
                "runs": to_wire(runs),
                "budget": to_wire(budget),
            },
        )
        return run_plan_from_wire(_dict(result))


@dataclass(frozen=True)
class PresetReference:
    name: str
    public_name: str


@dataclass(frozen=True)
class DatasetReference:
    __name__: str


@dataclass(frozen=True)
class MonitorReference:
    name: str


@dataclass(frozen=True)
class ModelPackageReference:
    model_type: str
    model: str
    client: ProjectAdapterClient
    _metadata_cache: ClassVar[dict[tuple[tuple[str, ...], str], dict[str, Any]]] = {}

    @property
    def catalog_key(self) -> str:
        return f"{self.model_type}/{self.model}"

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(self.model_type, self.model)

    def metadata_payload(self) -> dict[str, Any]:
        key = (self.client.command, self.catalog_key)
        cached = self._metadata_cache.get(key)
        if cached is None:
            cached = _dict(
                self.client.call(
                    "package_metadata",
                    {"model_id": self.catalog_key},
                )
            )
            self._metadata_cache[key] = cached
        return cached

    @property
    def runtime_defaults(self) -> SimpleNamespace:
        return SimpleNamespace(**dict(self.metadata_payload()["runtime_defaults"]))

    def resolve_experiment_task(self, value: str | None) -> str:
        result = self._resolve(experiment_task=value)
        return str(result["experiment_task"])

    def task_name(self, task: object) -> str:
        return str(task)

    def resolve_datasets(
        self,
        values: list[str],
        task: object,
    ) -> list[DatasetReference]:
        result = self._resolve(experiment_task=str(task), datasets=values)
        return [DatasetReference(str(name)) for name in result["datasets"]]

    def resolve_preset(self, value: str) -> PresetReference:
        result = self._resolve(presets=[value])
        preset = _dict(result["presets"][0])
        return PresetReference(name=str(preset["key"]), public_name=str(preset["name"]))

    def preset_name(self, preset: PresetReference) -> str:
        return preset.public_name

    def resolve_monitors(self, values: list[str] | None) -> list[MonitorReference]:
        result = self._resolve(monitors=list(values or ()))
        return [MonitorReference(str(name)) for name in result["monitors"]]

    def checkpoint_config_overrides(
        self,
        tensor_shapes: Mapping[str, tuple[int, ...]],
    ) -> dict[str, Any]:
        return _dict(
            self.client.call(
                "checkpoint_config_overrides",
                {
                    "model_id": self.catalog_key,
                    "tensor_shapes": to_wire(tensor_shapes),
                },
            )
        )

    def _resolve(self, **values: Any) -> dict[str, Any]:
        return _dict(
            self.client.call(
                "resolve",
                {"model_id": self.catalog_key, **values},
            )
        )


_DEFAULT_CLIENT: ProjectAdapterClient | None = None


def project_adapter() -> ProjectAdapterClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = ProjectAdapterClient()
    return _DEFAULT_CLIENT


def _close_default_client() -> None:
    if _DEFAULT_CLIENT is not None:
        _DEFAULT_CLIENT.close()


atexit.register(_close_default_client)


def _dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProjectAdapterFailure("The project Adapter result is invalid.")
    return value


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


__all__ = [
    "DatasetReference",
    "ModelPackageReference",
    "MonitorReference",
    "PROJECT_ADAPTER_COMMAND_ENV",
    "PresetReference",
    "ProjectAdapterClient",
    "ProjectAdapterFailure",
    "project_adapter",
]
