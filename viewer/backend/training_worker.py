from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.experiments.base import GridSearch, RandomSearch
from emperor.experiments.progress import JsonlTrainingProgressCallback

from viewer.backend.inspector.discovery import (
    load_model_parts,
    option_cli_name,
    resolve_dataset,
    resolve_datasets,
    resolve_model_monitors,
)
from viewer.backend.inspector.overrides import parse_override_mapping
from viewer.backend.inspector.search import (
    parse_training_search,
    strip_search_overrides,
)
from viewer.backend.inspector.service import reject_locked_overrides
from viewer.backend.training_events import NeuronClusterGrowthCallback

VIEWER_PROGRESS_STEP_INTERVAL = 25


def search_mode_from_parsed_search(parsed_search):
    if parsed_search is None:
        return None
    if parsed_search.mode == "grid":
        return GridSearch()
    if parsed_search.mode == "random":
        return RandomSearch(parsed_search.random_samples or 10)
    return None


def _resolve_payload_presets(parts, payload: dict) -> tuple[list[str], list]:
    raw_presets = payload.get("presets") or [payload.get("preset")]
    selected = []
    seen = set()
    for raw_preset in raw_presets:
        if not isinstance(raw_preset, str) or not raw_preset.strip():
            continue
        option = parts.experiment_options.get_option(raw_preset)
        if option.name in seen:
            continue
        seen.add(option.name)
        selected.append((option_cli_name(parts.experiment_options, option), option))
    if not selected:
        raise ValueError("Training payload does not include any selected presets.")
    return [name for name, _option in selected], [option for _name, option in selected]


def _materialized_runs_from_plan(parts, payload: dict) -> list[dict] | None:
    run_plan = payload.get("runPlan")
    if not isinstance(run_plan, dict):
        return None
    rows = run_plan.get("runs")
    if not isinstance(rows, list) or not rows:
        return None

    run_total = len(rows)
    materialized_runs = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        materialized_runs.append(
            _materialized_run_from_row(parts, payload, row, index, run_total)
        )
    return materialized_runs


def _materialized_run_from_row(
    parts,
    payload: dict,
    row: dict,
    index: int,
    run_total: int,
) -> dict:
    preset = str(row.get("preset") or "")
    option = parts.experiment_options.get_option(preset)
    dataset_type = resolve_dataset(parts, str(row.get("dataset") or ""))
    config_overrides = parse_override_mapping(
        parts.config_module,
        row.get("overrides") or {},
    )
    reject_locked_overrides(payload["model"], preset, config_overrides)
    return {
        "id": str(row.get("id") or f"run-{index:04d}"),
        "index": int(row.get("index") or index),
        "run_total": run_total,
        "option": option,
        "dataset_type": dataset_type,
        "config_overrides": config_overrides,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a viewer training job.")
    parser.add_argument("--payload", required=True)
    parser.add_argument("--progress", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload_path = Path(args.payload)
    progress_path = Path(args.progress)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    progress = JsonlTrainingProgressCallback(
        progress_path,
        step_interval=VIEWER_PROGRESS_STEP_INTERVAL,
    )
    growth = NeuronClusterGrowthCallback(progress.write_event)
    try:
        progress.write_event(
            {
                "type": "started",
                "status": "running",
                "jobId": payload["id"],
                "model": payload["model"],
                "preset": payload["preset"],
                "presets": payload.get("presets") or [payload["preset"]],
                "datasets": payload["datasets"],
                "monitors": payload.get("monitors") or [],
            }
        )
        parts = load_model_parts(payload["model"])
        selected_preset_names, selected_options = _resolve_payload_presets(
            parts, payload
        )
        selected_datasets = resolve_datasets(parts, payload["datasets"])
        monitor_callbacks = [
            monitor.build_callback()
            for monitor in resolve_model_monitors(parts, payload.get("monitors") or [])
        ]
        parsed_searches = [
            parse_training_search(
                payload["model"],
                selected_preset,
                payload.get("search"),
                dataset_count=len(selected_datasets),
            )
            for selected_preset in selected_preset_names
        ]
        parsed_search = next(
            (candidate for candidate in parsed_searches if candidate is not None),
            None,
        )
        search_model_params: set[str] = set()
        for parsed in parsed_searches:
            if parsed is not None:
                search_model_params.update(parsed.model_params)
        effective_override_payload = strip_search_overrides(
            parts.config_module,
            payload.get("overrides") or {},
            search_model_params,
        )
        config_overrides = parse_override_mapping(
            parts.config_module,
            effective_override_payload,
        )
        for selected_preset in selected_preset_names:
            reject_locked_overrides(payload["model"], selected_preset, config_overrides)
        materialized_runs = _materialized_runs_from_plan(parts, payload)
        search_mode = (
            None
            if materialized_runs is not None
            else search_mode_from_parsed_search(parsed_search)
        )
        experiment_type = parts.presets_module.Experiment
        experiment = experiment_type(selected_options[0])
        experiment.train_model(
            search_mode=search_mode,
            log_folder=payload.get("logFolder"),
            config_overrides=config_overrides,
            search_overrides=(
                parsed_search.search_overrides
                if parsed_search is not None and materialized_runs is None
                else None
            ),
            selected_datasets=selected_datasets,
            selected_options=selected_options,
            callbacks=[progress, growth, *monitor_callbacks],
            materialized_runs=materialized_runs,
        )
        progress.write_event(
            {
                "type": "completed",
                "status": "completed",
                "jobId": payload["id"],
                "preset": selected_preset_names[-1],
                "presets": selected_preset_names,
            }
        )
    except Exception as exc:
        progress.write_event(
            {
                "type": "error",
                "status": "failed",
                "jobId": payload.get("id"),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
