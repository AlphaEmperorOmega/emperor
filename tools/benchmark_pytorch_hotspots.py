from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from models.catalog import model_package
from torch.utils.data import DataLoader, TensorDataset

try:
    import resource
except ImportError:  # pragma: no cover - Windows standard library
    resource = None  # type: ignore[assignment]

SCHEMA_VERSION = 1
SEED = 20260710
MODEL_BATCH_SIZE = 32
DATALOADER_BATCH_SIZE = 64
DATALOADER_SAMPLE_COUNT = 2048


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    category: str
    identity: dict[str, Any]
    run: Callable[[], object]
    warmup: int
    repetitions: int


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _timing_summary(samples_ms: list[float]) -> dict[str, float]:
    mean = statistics.fmean(samples_ms)
    stdev = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return {
        "coefficient_of_variation": stdev / mean if mean else 0.0,
        "max_ms": max(samples_ms),
        "mean_ms": mean,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "p95_ms": _percentile(samples_ms, 0.95),
        "stdev_ms": stdev,
    }


def _process_peak_rss_bytes() -> int:
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(peak)
        return int(peak * 1024)
    import psutil

    return int(psutil.Process().memory_info().rss)


def _cpu_model_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"Hardware", "model name"}:
                return value.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _benchmark_case(
    case: BenchmarkCase,
    device: torch.device,
) -> dict[str, Any]:
    for _ in range(case.warmup):
        case.run()
        _synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples_ms = []
    for _ in range(case.repetitions):
        _synchronize(device)
        started_at = time.perf_counter_ns()
        case.run()
        _synchronize(device)
        samples_ms.append((time.perf_counter_ns() - started_at) / 1_000_000)

    tracemalloc.start()
    tracemalloc.reset_peak()
    case.run()
    _synchronize(device)
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    device_peak = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    timing = _timing_summary(samples_ms)
    result = {
        "category": case.category,
        "device": str(device),
        "dtype": "float32",
        "identity": case.identity,
        "memory": {
            "cuda_peak_allocated_bytes": device_peak,
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
            "process_peak_rss_scope": "process high-water mark; cumulative on CPU",
            "python_tracemalloc_peak_bytes": python_peak,
            "python_tracemalloc_scope": "one untimed invocation after timing",
        },
        "name": case.name,
        "repetitions": case.repetitions,
        "samples_ms": samples_ms,
        "status": "passed",
        "synchronization": (
            "torch.cuda.synchronize before and after every sample"
            if device.type == "cuda"
            else "eager CPU execution; no device synchronization required"
        ),
        "timing": timing,
        "warmup": case.warmup,
    }
    if case.category == "dataloader":
        sample_count = int(case.identity["sample_count"])
        result["throughput_samples_per_second"] = (
            sample_count / timing["median_ms"] * 1_000
        )
    return result


def _model_case(
    *,
    device: torch.device,
    model_id: str,
    preset_name: str,
    warmup: int,
    repetitions: int,
    config_overrides: dict[str, object] | None = None,
    scenario_suffix: str = "",
) -> BenchmarkCase:
    package = model_package(model_id)
    if package is None:
        raise ValueError(f"Unknown Model Package: {model_id}")
    preset = package.preset_type[preset_name]
    dataset = package.dataset_metadata[package.default_experiment_task][0]
    config = package.build_configurations(
        preset=preset,
        dataset=dataset,
        config_overrides=config_overrides or {},
    )[0]
    model = package.build_model(config).to(device).eval()
    inputs = torch.randn(
        MODEL_BATCH_SIZE,
        dataset.num_channels,
        dataset.default_height,
        dataset.default_width,
        device=device,
    )

    def run() -> object:
        with torch.no_grad():
            return model(inputs)

    suffix = f"_{scenario_suffix}" if scenario_suffix else ""
    model_name = model_id.replace("/", "_")
    return BenchmarkCase(
        name=f"model_forward_{model_name}_{preset_name.lower()}{suffix}",
        category=("ttt_inner_loop" if "ttt" in scenario_suffix else "routing_halting"),
        identity={
            "batch_size": MODEL_BATCH_SIZE,
            "dataset": dataset.__name__,
            "input_shape": list(inputs.shape),
            "model_package": model_id,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "preset": preset_name,
            "runtime_overrides": config_overrides or {},
            "task": package.default_experiment_task.name,
        },
        run=run,
        warmup=warmup,
        repetitions=repetitions,
    )


def _model_cases(
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> list[BenchmarkCase]:
    cases = [
        _model_case(
            device=device,
            model_id="linears/linear",
            preset_name="BASELINE",
            warmup=warmup,
            repetitions=repetitions,
        ),
        _model_case(
            device=device,
            model_id="linears/linear",
            preset_name="HALTING",
            warmup=warmup,
            repetitions=repetitions,
        ),
        _model_case(
            device=device,
            model_id="experts/linear",
            preset_name="BASELINE",
            warmup=warmup,
            repetitions=repetitions,
        ),
        _model_case(
            device=device,
            model_id="experts/linear",
            preset_name="TOP1_SWITCH_AUX",
            warmup=warmup,
            repetitions=repetitions,
        ),
        _model_case(
            device=device,
            model_id="linears/linear",
            preset_name="MEMORY",
            warmup=warmup,
            repetitions=repetitions,
            scenario_suffix="ttt_disabled",
        ),
    ]
    for inner_steps in (1, 2, 4):
        cases.append(
            _model_case(
                device=device,
                model_id="linears/linear",
                preset_name="MEMORY",
                warmup=warmup,
                repetitions=repetitions,
                config_overrides={
                    "memory_test_time_training_learning_rate": 0.01,
                    "memory_test_time_training_num_inner_steps": inner_steps,
                },
                scenario_suffix=f"ttt_{inner_steps}_steps",
            )
        )
    return cases


def _micro_cases(
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> list[BenchmarkCase]:
    host_images = torch.randn(MODEL_BATCH_SIZE, 1, 28, 28)
    host_labels = torch.randint(0, 10, (MODEL_BATCH_SIZE,))
    device_images = host_images.to(device)
    device_labels = host_labels.to(device)
    scalars = [torch.randn((), device=device) for _ in range(8)]

    def repeated_item() -> object:
        return tuple(scalar.item() for scalar in scalars)

    def stacked_scalar_transfer() -> object:
        return torch.stack(scalars).detach().cpu()

    def host_batch_to_device() -> object:
        return host_images.to(device), host_labels.to(device)

    def forced_batch_copy() -> object:
        return (
            host_images.to(device, copy=True),
            host_labels.to(device, copy=True),
        )

    def same_device_to_once() -> object:
        return device_images.to(device), device_labels.to(device)

    def same_device_to_repeated() -> object:
        images = device_images
        labels = device_labels
        for _ in range(8):
            images = images.to(device)
            labels = labels.to(device)
        return images, labels

    common_identity = {
        "batch_size": MODEL_BATCH_SIZE,
        "image_shape": list(host_images.shape),
        "image_bytes": host_images.numel() * host_images.element_size(),
        "label_bytes": host_labels.numel() * host_labels.element_size(),
    }
    return [
        BenchmarkCase(
            name="scalar_item_repeated_8",
            category="item_synchronization",
            identity={"operation": "eight scalar Tensor.item calls"},
            run=repeated_item,
            warmup=warmup,
            repetitions=repetitions,
        ),
        BenchmarkCase(
            name="scalar_stack_single_host_transfer",
            category="item_synchronization",
            identity={"operation": "stack eight scalars and transfer once to CPU"},
            run=stacked_scalar_transfer,
            warmup=warmup,
            repetitions=repetitions,
        ),
        BenchmarkCase(
            name="batch_placement_host_to_device",
            category="batch_placement",
            identity=common_identity,
            run=host_batch_to_device,
            warmup=warmup,
            repetitions=repetitions,
        ),
        BenchmarkCase(
            name="batch_placement_forced_copy",
            category="batch_placement",
            identity={**common_identity, "copy": True},
            run=forced_batch_copy,
            warmup=warmup,
            repetitions=repetitions,
        ),
        BenchmarkCase(
            name="same_device_to_once",
            category="repeated_to_device",
            identity=common_identity,
            run=same_device_to_once,
            warmup=warmup,
            repetitions=repetitions,
        ),
        BenchmarkCase(
            name="same_device_to_repeated_8",
            category="repeated_to_device",
            identity={**common_identity, "repetitions_inside_sample": 8},
            run=same_device_to_repeated,
            warmup=warmup,
            repetitions=repetitions,
        ),
    ]


def _dataloader_cases(
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> list[BenchmarkCase]:
    images = torch.randn(DATALOADER_SAMPLE_COUNT, 1, 28, 28)
    labels = torch.randint(0, 10, (DATALOADER_SAMPLE_COUNT,))
    dataset = TensorDataset(images, labels)
    cases = []
    for num_workers in (0, 2):
        loader = DataLoader(
            dataset,
            batch_size=DATALOADER_BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=device.type == "cuda",
        )

        def run(loader: DataLoader = loader) -> object:
            return sum(batch_images.shape[0] for batch_images, _ in loader)

        cases.append(
            BenchmarkCase(
                name=f"dataloader_tensor_dataset_workers_{num_workers}",
                category="dataloader",
                identity={
                    "batch_count": math.ceil(
                        DATALOADER_SAMPLE_COUNT / DATALOADER_BATCH_SIZE
                    ),
                    "batch_size": DATALOADER_BATCH_SIZE,
                    "num_workers": num_workers,
                    "persistent_workers": num_workers > 0,
                    "pin_memory": device.type == "cuda",
                    "sample_count": DATALOADER_SAMPLE_COUNT,
                    "sample_shape": [1, 28, 28],
                },
                run=run,
                warmup=min(warmup, 1),
                repetitions=min(repetitions, 5),
            )
        )
    return cases


def _device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device": str(device),
        "dtype": "float32",
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "capability": list(torch.cuda.get_device_capability(device)),
                "compiled_architectures": torch.cuda.get_arch_list(),
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )
    else:
        metadata.update(
            {
                "logical_cpu_count": os.cpu_count(),
                "name": _cpu_model_name(),
                "processor": platform.machine(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "torch_threads": torch.get_num_threads(),
            }
        )
    return metadata


def _cuda_compatibility() -> tuple[dict[str, Any], str | None]:
    metadata: dict[str, Any] = {
        "device": "cuda",
        "dtype": "float32",
    }
    if not torch.cuda.is_available():
        return metadata, "torch.cuda.is_available() returned false"

    try:
        device = torch.device("cuda")
        metadata = _device_metadata(device)
        major, minor = torch.cuda.get_device_capability(device)
        required_architecture = f"sm_{major}{minor}"
        compiled_architectures = torch.cuda.get_arch_list()
        if (
            compiled_architectures
            and required_architecture not in compiled_architectures
        ):
            reason = (
                f"GPU requires {required_architecture}, but this PyTorch build "
                "includes "
                f"{', '.join(compiled_architectures)}"
            )
            return metadata, reason

        torch.ones(1, device=device).add_(1)
        torch.cuda.synchronize(device)
    except Exception as error:  # pragma: no cover - hardware-specific boundary
        return metadata, f"CUDA execution probe failed: {type(error).__name__}: {error}"
    return metadata, None


def _run_device(
    device: torch.device,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    cases = [
        *_micro_cases(device, warmup, repetitions),
        *_dataloader_cases(device, warmup, repetitions),
        *_model_cases(device, warmup, repetitions),
    ]
    results = []
    for case in cases:
        try:
            results.append(_benchmark_case(case, device))
        except Exception as error:  # pragma: no cover - diagnostic boundary
            results.append(
                {
                    "category": case.category,
                    "device": str(device),
                    "error": f"{type(error).__name__}: {error}",
                    "identity": case.identity,
                    "name": case.name,
                    "status": "failed",
                }
            )
    return {
        "metadata": _device_metadata(device),
        "results": results,
        "status": (
            "failed"
            if any(result["status"] == "failed" for result in results)
            else "passed"
        ),
    }


def benchmark(
    *,
    requested_device: str,
    warmup: int,
    repetitions: int,
    threads: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    requested_devices = (
        ("cpu", "cuda") if requested_device == "all" else (requested_device,)
    )
    cuda_metadata: dict[str, Any] | None = None
    cuda_skip_reason: str | None = None
    if "cuda" in requested_devices:
        cuda_metadata, cuda_skip_reason = _cuda_compatibility()
    device_results = []
    for device_name in requested_devices:
        if device_name == "cuda" and cuda_skip_reason is not None:
            device_results.append(
                {
                    "metadata": cuda_metadata,
                    "reason": cuda_skip_reason,
                    "results": [],
                    "status": "skipped",
                }
            )
            continue
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        device_results.append(
            _run_device(
                device,
                warmup=warmup,
                repetitions=repetitions,
            )
        )

    return {
        "conditions": {
            "cuda_sync_policy": "before and after every timed CUDA sample",
            "model_batch_size": MODEL_BATCH_SIZE,
            "repetitions": repetitions,
            "seed": SEED,
            "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_threads": torch.get_num_threads(),
            "warmup": warmup,
        },
        "devices": device_results,
        "environment": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_execution_supported": (
                cuda_skip_reason is None if cuda_metadata is not None else None
            ),
            "cuda_version": torch.version.cuda,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "schema_version": SCHEMA_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible timing baselines for audited PyTorch hot-path candidates."
        )
    )
    parser.add_argument(
        "--device",
        choices=("all", "cpu", "cuda"),
        default="all",
    )
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2 to record variance")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.warmup < 1:
        parser.error("--warmup must be positive")

    result = benchmark(
        requested_device=args.device,
        warmup=args.warmup,
        repetitions=args.repetitions,
        threads=args.threads,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if any(device["status"] == "failed" for device in result["devices"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
