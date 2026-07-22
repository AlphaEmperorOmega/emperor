#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VENV_ROOT = REPOSITORY_ROOT / "torchenv"
WORKBENCH_ROOT = REPOSITORY_ROOT / "apps" / "workbench"
API_ROOT = WORKBENCH_ROOT / "api"
FRONTEND_ROOT = WORKBENCH_ROOT / "web"
DEFAULT_RUNTIME_ROOT = REPOSITORY_ROOT / ".runtime" / "workbench"
RUNTIME_ROOT = (
    Path(os.environ.get("WORKBENCH_RUNTIME_ROOT", str(DEFAULT_RUNTIME_ROOT)))
    .expanduser()
    .resolve()
)
SETUP_MARKER = VENV_ROOT / ".emperor-setup.json"
PINNED_PIP_VERSION = "26.1.2"
SUPPORTED_PYTHON = (3, 13)
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYTORCH_CUDA_130_INDEX = "https://download.pytorch.org/whl/cu130"
PYTORCH_CUDA_126_INDEX = "https://download.pytorch.org/whl/cu126"
SETUP_INPUTS = (
    REPOSITORY_ROOT / "pyproject.toml",
    API_ROOT / "pyproject.toml",
    FRONTEND_ROOT / "package.json",
    FRONTEND_ROOT / "package-lock.json",
)


@dataclass(frozen=True, slots=True)
class PlatformSpec:
    os_name: str
    architecture: str
    is_wsl: bool = False

    @property
    def key(self) -> str:
        return f"{self.os_name}-{self.architecture}"


@dataclass(frozen=True, slots=True)
class TorchWheelSource:
    index: str
    build_suffix: str


@dataclass(frozen=True, slots=True)
class ProfilePolicy:
    name: str
    supported_platforms: tuple[str, ...]
    lockfiles: tuple[tuple[str, str], ...]
    torch_sources: tuple[tuple[str, TorchWheelSource], ...]
    expected_torch_version: str
    expected_torchvision_version: str
    expected_cuda_version: str | None
    required_architectures: tuple[str, ...] = ()

    def lockfile(self, platform_key: str) -> str | None:
        return dict(self.lockfiles).get(platform_key)

    def torch_source(self, platform_key: str) -> TorchWheelSource | None:
        return dict(self.torch_sources).get(platform_key)


@dataclass(frozen=True, slots=True)
class TorchBuild:
    version: str
    cuda_version: str | None
    architectures: tuple[str, ...]


PROFILE_POLICIES = {
    "cpu": ProfilePolicy(
        name="cpu",
        supported_platforms=(
            "linux-x86_64",
            "windows-x86_64",
            "macos-arm64",
        ),
        lockfiles=(
            ("linux-x86_64", "python-3.13-linux-x86_64-cpu.txt"),
            ("windows-x86_64", "python-3.13-windows-x86_64-cpu.txt"),
            ("macos-arm64", "python-3.13-macos-arm64-cpu.txt"),
        ),
        torch_sources=(
            (
                "linux-x86_64",
                TorchWheelSource(PYTORCH_CPU_INDEX, "cpu"),
            ),
            (
                "windows-x86_64",
                TorchWheelSource(PYTORCH_CPU_INDEX, "cpu"),
            ),
        ),
        expected_torch_version="2.12.0",
        expected_torchvision_version="0.27.0",
        expected_cuda_version=None,
    ),
    "cuda": ProfilePolicy(
        name="cuda",
        supported_platforms=("linux-x86_64",),
        lockfiles=(("linux-x86_64", "python-3.13-linux-x86_64-cuda.txt"),),
        torch_sources=(
            (
                "linux-x86_64",
                TorchWheelSource(PYTORCH_CUDA_130_INDEX, "cu130"),
            ),
        ),
        expected_torch_version="2.12.0",
        expected_torchvision_version="0.27.0",
        expected_cuda_version="13.0",
    ),
    "cuda-legacy": ProfilePolicy(
        name="cuda-legacy",
        supported_platforms=("linux-x86_64",),
        lockfiles=(
            (
                "linux-x86_64",
                "python-3.13-linux-x86_64-cuda-legacy.txt",
            ),
        ),
        torch_sources=(
            (
                "linux-x86_64",
                TorchWheelSource(PYTORCH_CUDA_126_INDEX, "cu126"),
            ),
        ),
        expected_torch_version="2.12.0",
        expected_torchvision_version="0.27.0",
        expected_cuda_version="12.6",
        required_architectures=("sm_61",),
    ),
}
PROFILE_CHOICES = tuple(PROFILE_POLICIES)


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    name: str
    port: int
    command: tuple[str, ...]
    command_identity: str
    cwd: Path
    environment: dict[str, str]
    ready_url: str

    @property
    def metadata_path(self) -> Path:
        return RUNTIME_ROOT / f"{self.name}.json"

    @property
    def log_path(self) -> Path:
        return RUNTIME_ROOT / f"{self.name}.log"


_RUNTIME_METADATA_REQUIRED_KEYS = frozenset(
    {
        "argv",
        "commandIdentity",
        "createTime",
        "pid",
        "port",
    }
)
_RUNTIME_METADATA_OPTIONAL_KEYS = frozenset({"jobName"})


@dataclass(frozen=True, slots=True)
class _RuntimeProcessMetadata:
    argv: tuple[str, ...]
    command_identity: str
    create_time: float
    pid: int
    port: int
    job_name: str | None


@dataclass(frozen=True, slots=True)
class _RuntimeProcessObservation:
    process: Any
    metadata: _RuntimeProcessMetadata


def _normalize_architecture(machine: str) -> str:
    value = machine.lower().replace("-", "_")
    if value in {"x86_64", "amd64"}:
        return "x86_64"
    if value in {"arm64", "aarch64"}:
        return "arm64"
    return value


def detect_platform() -> PlatformSpec:
    system = platform.system().lower()
    os_name = {"darwin": "macos"}.get(system, system)
    architecture = _normalize_architecture(platform.machine())
    is_wsl = False
    if os_name == "linux":
        try:
            release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
        except OSError:
            release = platform.release()
        is_wsl = "microsoft" in release.lower()
    supported = {
        ("linux", "x86_64"),
        ("windows", "x86_64"),
        ("macos", "arm64"),
    }
    if (os_name, architecture) not in supported:
        raise SystemExit(
            "Unsupported platform: "
            f"{platform.system()} {platform.machine()}. Supported targets are "
            "Linux/WSL x86_64, Windows x86_64, and macOS arm64. Intel macOS "
            "has no PyTorch 2.12 wheel for the required Python 3.13 runtime."
        )
    return PlatformSpec(os_name, architecture, is_wsl)


def _profile_policy(profile: str) -> ProfilePolicy:
    try:
        return PROFILE_POLICIES[profile]
    except KeyError as exc:
        choices = ", ".join(repr(choice) for choice in PROFILE_CHOICES)
        raise SystemExit(f"Setup profile must be one of: {choices}.") from exc


def constraint_path(profile: str, spec: PlatformSpec | None = None) -> Path:
    target = spec or detect_platform()
    policy = _profile_policy(profile)
    lockfile = policy.lockfile(target.key)
    if lockfile is None:
        supported_labels = {
            "linux-x86_64": "Linux x86_64",
            "windows-x86_64": "Windows x86_64",
            "macos-arm64": "macOS arm64",
        }
        supported = ", ".join(
            supported_labels.get(platform_key, platform_key)
            for platform_key in policy.supported_platforms
        )
        raise SystemExit(
            f"The {profile} profile is only supported on {supported}; {target.key} "
            "uses the guaranteed CPU profile. Run: mise run setup --profile cpu"
        )
    path = REPOSITORY_ROOT / "constraints" / lockfile
    if not path.is_file():
        raise SystemExit(f"Missing verified constraint file: {path}")
    return path


def venv_python(root: Path = VENV_ROOT, *, os_name: str | None = None) -> Path:
    windows = (os_name == "windows") if os_name else os.name == "nt"
    return root / ("Scripts/python.exe" if windows else "bin/python")


def venv_scripts(root: Path = VENV_ROOT, *, os_name: str | None = None) -> Path:
    windows = (os_name == "windows") if os_name else os.name == "nt"
    return root / ("Scripts" if windows else "bin")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _protect_runtime_path(path: Path) -> None:
    _reject_runtime_link(path)
    if os.name == "posix":
        path.chmod(0o700 if path.is_dir() else 0o600)
        return
    if os.name == "nt":
        from emperor_workbench.filesystem import (
            apply_owner_only_permissions,
        )

        apply_owner_only_permissions(path)


def _reject_runtime_link(path: Path) -> None:
    from emperor_workbench.filesystem import reject_link_like

    reject_link_like(path, "Workbench runtime path")


def _constraint_inputs(lock: Path) -> tuple[Path, ...]:
    """Return one constraint file and all recursively included constraints."""

    pending = [lock]
    ordered: list[Path] = []
    seen: set[Path] = set()
    constraints_root = (REPOSITORY_ROOT / "constraints").resolve()
    while pending:
        path = pending.pop(0).resolve()
        if path in seen:
            continue
        try:
            path.relative_to(constraints_root)
        except ValueError as exc:
            raise SystemExit(
                f"Constraint includes must stay under {constraints_root}: {path}"
            ) from exc
        if not path.is_file():
            raise SystemExit(f"Missing included constraint file: {path}")
        seen.add(path)
        ordered.append(path)
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            included = None
            if line.startswith("-c "):
                included = line[3:].strip()
            elif line.startswith("--constraint "):
                included = line.removeprefix("--constraint ").strip()
            elif line.startswith("--constraint="):
                included = line.removeprefix("--constraint=").strip()
            if included:
                pending.append(path.parent / included)
    return tuple(ordered)


def _setup_signature(
    policy: ProfilePolicy | str,
    spec: PlatformSpec,
    lock: Path,
) -> str:
    resolved_policy = _profile_policy(policy) if isinstance(policy, str) else policy
    digest = hashlib.sha256()
    digest.update(
        (f"python=3.13\nplatform={spec.key}\nprofile={resolved_policy.name}\n").encode()
    )
    digest.update(
        json.dumps(
            asdict(resolved_policy),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(b"\0")
    for path in (*SETUP_INPUTS, *_constraint_inputs(lock)):
        digest.update(str(path.relative_to(REPOSITORY_ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _venv_version(python: Path) -> tuple[int, int] | None:
    if not python.is_file():
        return None
    completed = subprocess.run(
        [
            str(python),
            "-c",
            "import sys; print(sys.version_info.major, sys.version_info.minor)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        major, minor = completed.stdout.split()
        return int(major), int(minor)
    except ValueError:
        return None


def _venv_recreation_reason(
    *,
    marker: dict[str, Any] | None,
    profile: str,
    platform_key: str,
    signature: str,
    version: tuple[int, int] | None,
) -> str | None:
    if version is not None and version != SUPPORTED_PYTHON:
        return f"Python {version[0]}.{version[1]} is not 3.13"
    if marker is None:
        return None
    if marker.get("profile") != profile or marker.get("platform") != platform_key:
        return "the selected platform/profile changed"
    if marker.get("signature") != signature:
        return "the verified setup inputs changed"
    return None


def _remove_owned_venv(reason: str) -> None:
    if not VENV_ROOT.exists():
        return
    if VENV_ROOT.is_symlink():
        raise SystemExit(
            f"Refusing to replace symlink virtual environment: {VENV_ROOT}"
        )
    if not (VENV_ROOT / "pyvenv.cfg").is_file():
        raise SystemExit(
            f"Refusing to replace {VENV_ROOT}: it is not a Python virtual environment."
        )
    print(f"Recreating {VENV_ROOT} ({reason})...")
    shutil.rmtree(VENV_ROOT)


def _run_checked(command: list[str], *, cwd: Path = REPOSITORY_ROOT) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _constraint_version(lock: Path, package: str) -> str | None:
    prefix = f"{package.lower()}=="
    for constraint in _constraint_inputs(lock):
        for line in constraint.read_text(encoding="utf-8").splitlines():
            normalized = line.strip().lower()
            if normalized.startswith(prefix):
                return line.strip().split("==", 1)[1].split(";", 1)[0].strip()
    return None


def _pytorch_install_command(
    python: Path,
    profile: str,
    spec: PlatformSpec,
) -> list[str] | None:
    policy = _profile_policy(profile)
    source = policy.torch_source(spec.key)
    if source is None:
        return None
    return [
        str(python),
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        "--force-reinstall",
        "--no-deps",
        "--index-url",
        source.index,
        f"torch=={policy.expected_torch_version}+{source.build_suffix}",
        (f"torchvision=={policy.expected_torchvision_version}+{source.build_suffix}"),
    ]


def _torch_build_matches_profile(
    profile: str,
    spec: PlatformSpec,
    *,
    torch_version: str,
    cuda_version: str | None,
    architectures: tuple[str, ...] = (),
) -> bool:
    policy = _profile_policy(profile)
    if spec.key not in policy.supported_platforms:
        return False
    if torch_version.split("+", 1)[0] != policy.expected_torch_version:
        return False
    if policy.expected_cuda_version is None:
        if cuda_version is not None:
            return False
        source = policy.torch_source(spec.key)
        if source is None:
            return True
        local_version = torch_version.partition("+")[2].casefold()
        return local_version == source.build_suffix.casefold()
    if cuda_version != policy.expected_cuda_version:
        return False
    return _compiled_architectures_support(
        policy.required_architectures,
        architectures,
    )


def _compute_capability(architecture: str) -> tuple[int, int] | None:
    prefix, separator, digits = architecture.casefold().partition("_")
    if prefix != "sm" or separator != "_" or not digits.isdecimal():
        return None
    if len(digits) < 2:
        return None
    return int(digits[:-1]), int(digits[-1])


def _compiled_architectures_support(
    required: tuple[str, ...],
    compiled: tuple[str, ...],
) -> bool:
    compiled_capabilities = tuple(
        capability
        for architecture in compiled
        if (capability := _compute_capability(architecture)) is not None
    )
    for architecture in required:
        required_capability = _compute_capability(architecture)
        if required_capability is None:
            return False
        required_major, required_minor = required_capability
        if not any(
            compiled_major == required_major and compiled_minor <= required_minor
            for compiled_major, compiled_minor in compiled_capabilities
        ):
            return False
    return True


def _installed_torch_build(python: Path) -> TorchBuild | None:
    source = (
        "import json,torch; "
        "architectures=torch.cuda.get_arch_list(); "
        "architectures=architectures or "
        "((torch._C._cuda_getArchFlags() or '').split() "
        "if torch.version.cuda else []); "
        "print(json.dumps({'version':str(torch.__version__),"
        "'cuda':torch.version.cuda,"
        "'architectures':architectures}))"
    )
    completed = subprocess.run(
        [str(python), "-c", source],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
        version = payload["version"]
        cuda_version = payload["cuda"]
        architectures = payload["architectures"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(version, str) or not (
        cuda_version is None or isinstance(cuda_version, str)
    ):
        return None
    if not isinstance(architectures, list) or not all(
        isinstance(architecture, str) for architecture in architectures
    ):
        return None
    return TorchBuild(version, cuda_version, tuple(architectures))


def _torch_install_matches_profile(
    python: Path,
    profile: str,
    spec: PlatformSpec,
    lock: Path,
) -> bool:
    policy = _profile_policy(profile)
    build = _installed_torch_build(python)
    expected_torch = _constraint_version(lock, "torch")
    expected_torchvision = _constraint_version(lock, "torchvision")
    if build is None or expected_torch is None or expected_torchvision is None:
        return False
    if expected_torch.split("+", 1)[0] != policy.expected_torch_version:
        return False
    if expected_torchvision.split("+", 1)[0] != policy.expected_torchvision_version:
        return False
    return _torch_build_matches_profile(
        profile,
        spec,
        torch_version=build.version,
        cuda_version=build.cuda_version,
        architectures=build.architectures,
    )


def _dependencies_available(python: Path) -> bool:
    modules = (
        "emperor",
        "emperor_workbench",
        "fastapi",
        "filelock",
        "httpx",
        "lightning.pytorch",
        "model_runtime",
        "models",
        "psutil",
        "pydantic_settings",
        "ruff",
        "torch",
        "torchvision",
        "uvicorn",
    )
    source = (
        "import importlib.util,sys; "
        "missing=[name for name in sys.argv[1:] "
        "if importlib.util.find_spec(name) is None]; "
        "print(','.join(missing)); raise SystemExit(bool(missing))"
    )
    return (
        subprocess.run(
            [str(python), "-c", source, *modules],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        == 0
    )


def setup(profile: str = "cpu") -> int:
    spec = detect_platform()
    policy = _profile_policy(profile)
    lock = constraint_path(profile, spec)
    if sys.version_info[:2] != SUPPORTED_PYTHON:
        raise SystemExit(
            "Setup requires mise Python 3.13. Run it through "
            "'mise run setup --profile cpu'."
        )
    signature = _setup_signature(policy, spec, lock)
    python = venv_python(os_name=spec.os_name)
    marker = _read_json(SETUP_MARKER)
    version = _venv_version(python)
    recreation_reason = _venv_recreation_reason(
        marker=marker,
        profile=profile,
        platform_key=spec.key,
        signature=signature,
        version=version,
    )
    if recreation_reason is not None:
        _remove_owned_venv(recreation_reason)
        marker = None
    if not python.is_file():
        print(f"Creating Python 3.13 virtual environment at {VENV_ROOT}...")
        _run_checked([sys.executable, "-m", "venv", str(VENV_ROOT)])

    dependencies_current = (
        marker is not None
        and marker.get("signature") == signature
        and _dependencies_available(python)
        and _torch_install_matches_profile(python, profile, spec, lock)
    )
    if not dependencies_current:
        print("Installing verified binary Python dependencies...")
        _run_checked(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                f"pip=={PINNED_PIP_VERSION}",
            ]
        )
        pytorch_command = _pytorch_install_command(python, profile, spec)
        if pytorch_command is not None:
            _run_checked(pytorch_command)
        _run_checked(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--constraint",
                str(lock),
                "--build-constraint",
                str(lock),
                "--editable",
                ".[dev]",
                "--editable",
                "./apps/workbench/api[dev]",
            ]
        )

    frontend_current = (
        marker is not None
        and marker.get("signature") == signature
        and (FRONTEND_ROOT / "node_modules").is_dir()
    )
    if not frontend_current:
        npm = resolve_executable("npm", windows_name="npm.cmd")
        print("Installing locked Workbench frontend dependencies...")
        _run_checked([npm, "ci"], cwd=FRONTEND_ROOT)
    _run_checked([str(python), "-m", "pip", "check"])
    if not _torch_install_matches_profile(python, profile, spec, lock):
        build = _installed_torch_build(python)
        observed = (
            "unavailable"
            if build is None
            else (
                f"{build.version} (CUDA {build.cuda_version}; "
                f"architectures {', '.join(build.architectures) or 'none'})"
            )
        )
        raise SystemExit(
            f"Installed PyTorch build {observed} does not match the {profile} "
            f"profile for {spec.key}."
        )
    _write_json_atomic(
        SETUP_MARKER,
        {
            "architecture": spec.architecture,
            "constraint": str(lock.relative_to(REPOSITORY_ROOT)),
            "platform": spec.key,
            "profile": profile,
            "python": "3.13",
            "signature": signature,
        },
    )
    label = "WSL2/Linux" if spec.is_wsl else spec.key
    print(f"Setup ready: {label}, {profile}, {python}", flush=True)
    return 0


def resolve_executable(name: str, *, windows_name: str | None = None) -> str:
    candidates = [windows_name, name] if os.name == "nt" else [name]
    for candidate in candidates:
        if candidate and (resolved := shutil.which(candidate)):
            return resolved
    expected = windows_name if os.name == "nt" and windows_name else name
    raise SystemExit(f"Required executable not found on PATH: {expected}")


def _require_venv_python() -> Path:
    python = venv_python()
    if not python.is_file():
        raise SystemExit("Emperor is not set up. Run: mise run setup --profile cpu")
    return python


def _run_venv(arguments: list[str]) -> int:
    environment = {
        **os.environ,
        "MPLCONFIGDIR": os.environ.get(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "emperor-matplotlib"),
        ),
    }
    return subprocess.run(
        [str(_require_venv_python()), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    ).returncode


def _port(value: str | None, default: int, name: str) -> int:
    try:
        result = int(value) if value is not None else default
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer port.") from exc
    if not 1 <= result <= 65535:
        raise SystemExit(f"{name} must be between 1 and 65535.")
    return result


def service_specs() -> tuple[ServiceSpec, ServiceSpec]:
    python = _require_venv_python()
    node = resolve_executable("node", windows_name="node.exe")
    backend_port = _port(
        os.environ.get("WORKBENCH_BACKEND_PORT"), 9999, "WORKBENCH_BACKEND_PORT"
    )
    frontend_port = _port(
        os.environ.get("WORKBENCH_FRONTEND_PORT"), 9000, "WORKBENCH_FRONTEND_PORT"
    )
    api_url = os.environ.get(
        "NEXT_PUBLIC_WORKBENCH_API_URL", f"http://127.0.0.1:{backend_port}"
    )
    backend_environment = {
        **os.environ,
        "MPLCONFIGDIR": os.environ.get(
            "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib")
        ),
        "WORKBENCH_API_ALLOW_LOG_IMPORTS": os.environ.get(
            "WORKBENCH_API_ALLOW_LOG_IMPORTS", "true"
        ),
        "WORKBENCH_API_ALLOW_UNSAFE_LOCAL_MUTATIONS": os.environ.get(
            "WORKBENCH_API_ALLOW_UNSAFE_LOCAL_MUTATIONS", "true"
        ),
        "WORKBENCH_API_SNAPSHOTS_ROOT": os.environ.get(
            "WORKBENCH_API_SNAPSHOTS_ROOT", str(RUNTIME_ROOT / "snapshots")
        ),
        "WORKBENCH_API_STATE_ROOT": os.environ.get(
            "WORKBENCH_API_STATE_ROOT", str(RUNTIME_ROOT / "state")
        ),
    }
    backend_command = [
        str(python),
        "-m",
        "emperor_workbench",
        "--host",
        "127.0.0.1",
        "--port",
        str(backend_port),
    ]
    if os.environ.get("WORKBENCH_BACKEND_RELOAD", "true").lower() not in {
        "0",
        "false",
        "no",
    }:
        backend_command.append("--reload")
    frontend_environment = {
        **os.environ,
        "NEXT_PUBLIC_WORKBENCH_API_URL": api_url,
        "PORT": str(frontend_port),
    }
    return (
        ServiceSpec(
            "backend",
            backend_port,
            tuple(backend_command),
            "emperor_workbench",
            REPOSITORY_ROOT,
            backend_environment,
            f"http://127.0.0.1:{backend_port}/health",
        ),
        ServiceSpec(
            "frontend",
            frontend_port,
            (node, str(FRONTEND_ROOT / "scripts" / "start-next.mjs"), "dev"),
            "scripts/start-next.mjs",
            FRONTEND_ROOT,
            frontend_environment,
            f"http://127.0.0.1:{frontend_port}/",
        ),
    )


def _http_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=0.6) as response:  # noqa: S310 - loopback URL
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def _same_executable(observed: str, expected: str) -> bool:
    try:
        return os.path.samefile(observed, expected)
    except OSError:
        return os.path.normcase(os.path.abspath(observed)) == os.path.normcase(
            os.path.abspath(expected)
        )


def _same_argv(
    observed: list[str] | tuple[str, ...], expected: tuple[str, ...]
) -> bool:
    return (
        bool(observed)
        and bool(expected)
        and _same_executable(observed[0], expected[0])
        and tuple(observed[1:]) == expected[1:]
    )


def _launcher_options_match(arguments: tuple[str, ...], *, port: int) -> bool:
    observed_port: str | None = None
    observed_host = False
    observed_reload = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--reload":
            if observed_reload:
                return False
            observed_reload = True
            index += 1
            continue
        if argument == "--host":
            if observed_host or index + 1 >= len(arguments):
                return False
            observed_host = bool(arguments[index + 1])
            if not observed_host:
                return False
            index += 2
            continue
        if argument.startswith("--host="):
            if observed_host or not argument.removeprefix("--host="):
                return False
            observed_host = True
            index += 1
            continue
        if argument == "--port":
            if observed_port is not None or index + 1 >= len(arguments):
                return False
            observed_port = arguments[index + 1]
            index += 2
            continue
        if argument.startswith("--port="):
            if observed_port is not None:
                return False
            observed_port = argument.removeprefix("--port=")
            index += 1
            continue
        return False
    return observed_port == str(port)


def _backend_module_command_matches(
    spec: ServiceSpec,
    argv: tuple[str, ...],
    *,
    module: str,
) -> bool:
    return (
        len(argv) >= 3
        and _same_executable(argv[0], spec.command[0])
        and argv[1:3] == ("-m", module)
        and _launcher_options_match(argv[3:], port=spec.port)
    )


def _metadata_matches_current_command(
    spec: ServiceSpec,
    *,
    command_identity: str,
    argv: tuple[str, ...],
) -> bool:
    if spec.name == "backend":
        return (
            command_identity == spec.command_identity
            and _backend_module_command_matches(
                spec,
                argv,
                module=spec.command_identity,
            )
        )
    if command_identity != spec.command_identity or not argv:
        return False
    if spec.name == "frontend":
        if len(argv) != 3 or argv[2] != "dev":
            return False
        if not _same_executable(argv[0], spec.command[0]):
            return False
        if not _same_executable(argv[1], spec.command[1]):
            return False
        return True
    if not _same_executable(argv[0], spec.command[0]):
        return False
    if command_identity.casefold() not in " ".join(argv[1:]).casefold():
        return False
    return True


def _runtime_metadata(
    spec: ServiceSpec,
    payload: dict[str, Any] | None,
) -> _RuntimeProcessMetadata | None:
    if payload is None:
        return None
    keys = frozenset(payload)
    if not _RUNTIME_METADATA_REQUIRED_KEYS.issubset(keys) or not keys.issubset(
        _RUNTIME_METADATA_REQUIRED_KEYS | _RUNTIME_METADATA_OPTIONAL_KEYS
    ):
        return None
    raw_argv = payload.get("argv")
    command_identity = payload.get("commandIdentity")
    raw_create_time = payload.get("createTime")
    raw_pid = payload.get("pid")
    raw_port = payload.get("port")
    raw_job_name = payload.get("jobName")
    if (
        not isinstance(raw_argv, list)
        or not raw_argv
        or any(not isinstance(argument, str) or not argument for argument in raw_argv)
        or not isinstance(command_identity, str)
        or not command_identity
        or isinstance(raw_create_time, bool)
        or not isinstance(raw_create_time, (int, float))
        or not math.isfinite(float(raw_create_time))
        or float(raw_create_time) <= 0
        or isinstance(raw_pid, bool)
        or not isinstance(raw_pid, int)
        or raw_pid <= 0
        or isinstance(raw_port, bool)
        or not isinstance(raw_port, int)
        or raw_port != spec.port
        or (
            raw_job_name is not None
            and (not isinstance(raw_job_name, str) or not raw_job_name)
        )
    ):
        return None
    argv = tuple(raw_argv)
    if not _metadata_matches_current_command(
        spec,
        command_identity=command_identity,
        argv=argv,
    ):
        return None
    return _RuntimeProcessMetadata(
        argv=argv,
        command_identity=command_identity,
        create_time=float(raw_create_time),
        pid=raw_pid,
        port=raw_port,
        job_name=raw_job_name,
    )


def _validated_runtime_process(
    spec: ServiceSpec,
    *,
    require_current_command: bool,
) -> _RuntimeProcessObservation | None:
    import psutil

    _reject_runtime_link(spec.metadata_path)
    metadata = _runtime_metadata(spec, _read_json(spec.metadata_path))
    if metadata is None:
        return None
    if require_current_command and not _same_argv(metadata.argv, spec.command):
        return None
    try:
        process = psutil.Process(metadata.pid)
        if abs(process.create_time() - metadata.create_time) > 0.01:
            return None
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        if Path(process.cwd()).resolve() != spec.cwd.resolve():
            return None
        observed_argv = process.cmdline()
        if not isinstance(observed_argv, list) or not _same_argv(
            observed_argv,
            metadata.argv,
        ):
            return None
        return _RuntimeProcessObservation(process=process, metadata=metadata)
    except (OSError, psutil.Error, ValueError):
        return None


def _terminate_process_tree(process) -> None:
    import psutil

    descendants = process.children(recursive=True)
    targets = [*reversed(descendants), process]
    for target in targets:
        try:
            target.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(targets, timeout=5.0)
    for target in alive:
        try:
            target.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5.0)


def _terminate_runtime_process(
    spec: ServiceSpec,
    observation: _RuntimeProcessObservation,
) -> None:
    import psutil

    process = observation.process
    descendants = process.children(recursive=True)
    targets = [*reversed(descendants), process]
    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except (AttributeError, psutil.Error, OSError):
            pass
        _, alive = psutil.wait_procs(targets, timeout=5.0)
        if alive:
            _terminate_windows_service_job(observation.metadata.job_name, spec)
            _, alive = psutil.wait_procs(alive, timeout=5.0)
        for target in alive:
            try:
                target.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=5.0)
        return
    _terminate_process_tree(process)


def _drain_runtime_metadata(spec: ServiceSpec) -> bool:
    observation = _validated_runtime_process(
        spec,
        require_current_command=False,
    )
    if observation is None:
        spec.metadata_path.unlink(missing_ok=True)
        return False
    print(
        f"Replacing previous Workbench {spec.name} process "
        f"(PID {observation.process.pid}, port {spec.port})..."
    )
    _terminate_runtime_process(spec, observation)
    spec.metadata_path.unlink(missing_ok=True)
    _wait_for_stopped_port(spec)
    return True


def _wait_ready(spec: ServiceSpec, process, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process.is_running() or process.status() == "zombie":
            raise SystemExit(
                f"Workbench {spec.name} exited during startup. See {spec.log_path}"
            )
        if _http_ready(spec.ready_url):
            return
        time.sleep(0.1)
    raise SystemExit(
        f"Workbench {spec.name} did not become HTTP-ready at {spec.ready_url}. "
        f"See {spec.log_path}"
    )


def _start_service(spec: ServiceSpec) -> None:
    import psutil

    _reject_runtime_link(spec.metadata_path)
    _reject_runtime_link(spec.log_path)
    existing = _validated_runtime_process(
        spec,
        require_current_command=True,
    )
    if existing is not None:
        if _http_ready(spec.ready_url):
            print(
                "Workbench "
                f"{spec.name}: running (PID {existing.process.pid}, port {spec.port})"
            )
            return
        _wait_ready(spec, existing.process)
        return
    _drain_runtime_metadata(spec)
    if _port_open(spec.port):
        raise SystemExit(
            f"Port {spec.port} is already occupied by an unverified process; "
            f"refusing to start Workbench {spec.name}."
        )
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    _protect_runtime_path(RUNTIME_ROOT)
    windows_job = None
    with spec.log_path.open("ab", buffering=0) as log_file:
        kwargs: dict[str, Any] = {
            "cwd": spec.cwd,
            "env": spec.environment,
            "stdin": subprocess.DEVNULL,
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "close_fds": True,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        if os.name == "nt":
            from _emperor_windows_jobs import (
                WindowsJob,
                WindowsJobLimits,
                service_job_object_name,
            )

            windows_job = WindowsJob.create(
                name=service_job_object_name(spec.name, spec.port),
                limits=WindowsJobLimits(
                    memory_bytes=64 * 1024**3,
                    cpu_count=os.cpu_count() or 1,
                    process_count=512,
                ),
            )
            process = windows_job.start_suspended(
                list(spec.command),
                cwd=spec.cwd,
                env=spec.environment,
                stdout=log_file,
                stderr=log_file,
                detached=True,
            )
        else:
            process = subprocess.Popen(list(spec.command), **kwargs)  # noqa: S603
    observed = psutil.Process(process.pid)
    _write_json_atomic(
        spec.metadata_path,
        {
            "argv": list(spec.command),
            "commandIdentity": spec.command_identity,
            "createTime": observed.create_time(),
            "jobName": windows_job.name if windows_job is not None else None,
            "pid": process.pid,
            "port": spec.port,
        },
    )
    _protect_runtime_path(spec.metadata_path)
    _protect_runtime_path(spec.log_path)
    try:
        _wait_ready(spec, observed)
    except BaseException:
        _stop_service(spec, quiet=True)
        raise
    print(f"Workbench {spec.name}: started (PID {process.pid}, port {spec.port})")


def _stop_service(spec: ServiceSpec, *, quiet: bool = False) -> None:
    _reject_runtime_link(spec.metadata_path)
    _reject_runtime_link(spec.log_path)
    observation = _validated_runtime_process(
        spec,
        require_current_command=False,
    )
    if observation is None:
        spec.metadata_path.unlink(missing_ok=True)
        if not quiet:
            print(f"Workbench {spec.name}: stopped")
        return
    _terminate_runtime_process(spec, observation)
    spec.metadata_path.unlink(missing_ok=True)
    _wait_for_stopped_port(spec)
    if not quiet:
        print(f"Workbench {spec.name}: stopped")


def _terminate_windows_service_job(
    name: str | None,
    spec: ServiceSpec,
) -> None:
    if name is None:
        return
    from _emperor_windows_jobs import WindowsJob, service_job_object_name

    if name != service_job_object_name(spec.name, spec.port):
        return

    job = WindowsJob.open(name)
    if job is None:
        return
    try:
        if job.has_processes():
            job.terminate(1)
            try:
                job.wait_empty(5.0)
            except TimeoutError:
                pass
    finally:
        job.close()


def _wait_for_stopped_port(spec: ServiceSpec) -> None:
    deadline = time.monotonic() + 5.0
    while _port_open(spec.port) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _port_open(spec.port):
        raise SystemExit(
            f"Workbench {spec.name} process tree stopped but port "
            f"{spec.port} remains open."
        )


def workbench(action: str) -> int:
    try:
        from filelock import FileLock
    except ImportError as exc:
        raise SystemExit(
            "Workbench lifecycle requires the project environment. "
            "Run: mise run setup --profile cpu"
        ) from exc
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    _protect_runtime_path(RUNTIME_ROOT)
    backend, frontend = service_specs()
    with FileLock(str(RUNTIME_ROOT / "launcher.lock"), timeout=30):
        if action == "start":
            _start_service(backend)
            try:
                _start_service(frontend)
            except BaseException:
                _stop_service(backend, quiet=True)
                raise
        elif action == "stop":
            _stop_service(frontend)
            _stop_service(backend)
        elif action == "status":
            for spec in (backend, frontend):
                observation = _validated_runtime_process(
                    spec,
                    require_current_command=False,
                )
                process = observation.process if observation is not None else None
                ready = process is not None and _http_ready(spec.ready_url)
                state = "running" if ready else ("unhealthy" if process else "stopped")
                suffix = (
                    f" (PID {process.pid}, port {spec.port})"
                    if process is not None
                    else ""
                )
                print(f"Workbench {spec.name}: {state}{suffix}")
        else:
            raise SystemExit(f"Unknown Workbench action: {action}")
    return 0


def _run_runtime_subcommand(arguments: list[str]) -> int:
    if Path(sys.executable).absolute().parent != venv_scripts().absolute():
        return _run_venv([str(Path(__file__).resolve()), *arguments])
    command = arguments[0]
    if command == "workbench":
        return workbench(arguments[1])
    if command == "experiment":
        return _run_venv(["-m", "models.project_cli", *arguments[1:]])
    if command == "test":
        return _run_venv(["-m", "models.project_cli", "test", *arguments[1:]])
    if command == "logs-archive":
        return _run_venv(["-m", "models.project_cli", "logs:archive", *arguments[1:]])
    if command == "python":
        forwarded = arguments[1:]
        if forwarded[:1] == ["--"]:
            forwarded = forwarded[1:]
        return _run_venv(forwarded)
    raise SystemExit(f"Unknown runtime subcommand: {command}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    delegated_commands = {"experiment", "test", "logs-archive", "python"}
    if argv and argv[0] in delegated_commands:
        return argparse.Namespace(command=argv[0], arguments=argv[1:])

    parser = argparse.ArgumentParser(
        description=(
            "Portable setup and Workbench lifecycle launcher. This command uses "
            "only the Python standard library until it re-enters the project "
            "virtual environment for runtime operations."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup_parser = subparsers.add_parser("setup")
    setup_parser.add_argument("--profile", choices=PROFILE_CHOICES, default="cpu")
    dev_parser = subparsers.add_parser("dev")
    dev_parser.add_argument("--profile", choices=PROFILE_CHOICES, default="cpu")
    workbench_parser = subparsers.add_parser("workbench")
    workbench_parser.add_argument("action", choices=("start", "status", "stop"))
    for name in sorted(delegated_commands):
        # Every token after these command names belongs to the delegated CLI.
        # In particular, argparse must not consume ``--help`` before Emperor,
        # unittest, or the archive command sees it.
        runtime = subparsers.add_parser(name, add_help=False)
        runtime.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "setup":
        return setup(args.profile)
    if args.command == "dev":
        setup(args.profile)
        return _run_runtime_subcommand(["workbench", "start"])
    if args.command == "workbench":
        return _run_runtime_subcommand(["workbench", args.action])
    return _run_runtime_subcommand([args.command, *args.arguments])


if __name__ == "__main__":
    raise SystemExit(main())
