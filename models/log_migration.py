from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from models.catalog import FLAT_TO_PUBLIC_ID, split_model_id


class LogMigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LogMigrationAction:
    source: Path
    destination: Path
    model_id: str


@dataclass(frozen=True)
class LogMetadataMigrationAction:
    path: Path
    model_id: str


def _refuse_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise LogMigrationError(f"Refusing symlinked {label}: {path}")


def _looks_like_model_log_dir(path: Path) -> bool:
    if (path / "best_results.json").is_file():
        return True
    for version_dir in path.rglob("version_*"):
        if not version_dir.is_dir() or version_dir.is_symlink():
            continue
        try:
            relative_parts = version_dir.relative_to(path).parts
        except ValueError:
            continue
        if len(relative_parts) >= 4:
            return True
    return False


def _destination_for(source: Path, model_id: str) -> Path:
    category, model = model_id.split("/", 1)
    return source.parent / category / model


def _already_categorized(source: Path, model_id: str) -> bool:
    category, model = model_id.split("/", 1)
    return source.name == model and source.parent.name == category


def plan_log_migration(logs_root: Path | str = "logs") -> list[LogMigrationAction]:
    root = Path(logs_root)
    if not root.exists():
        return []
    _refuse_symlink(root, "logs root")
    if not root.is_dir():
        raise LogMigrationError(f"Logs root is not a directory: {root}")

    actions: list[LogMigrationAction] = []
    planned_destinations: set[Path] = set()
    for flat_name, model_id in sorted(FLAT_TO_PUBLIC_ID.items()):
        for source in sorted(root.rglob(flat_name)):
            if source.is_symlink():
                raise LogMigrationError(f"Refusing symlinked log path: {source}")
            if not source.is_dir():
                continue
            if _already_categorized(source, model_id):
                continue
            if not _looks_like_model_log_dir(source):
                continue

            destination = _destination_for(source, model_id)
            _refuse_symlink(destination.parent, "destination parent")
            if destination.exists() or destination.is_symlink():
                raise LogMigrationError(
                    f"Destination already exists for {source}: {destination}"
                )
            if destination in planned_destinations:
                raise LogMigrationError(f"Duplicate destination planned: {destination}")
            planned_destinations.add(destination)
            actions.append(
                LogMigrationAction(
                    source=source,
                    destination=destination,
                    model_id=model_id,
                )
            )
    return actions


def apply_log_migration(
    logs_root: Path | str = "logs",
    *,
    apply: bool = False,
) -> list[LogMigrationAction]:
    actions = plan_log_migration(logs_root)
    if not apply:
        return actions
    for action in actions:
        _refuse_symlink(action.source, "log path")
        _refuse_symlink(action.destination.parent, "destination parent")
        if action.destination.exists() or action.destination.is_symlink():
            raise LogMigrationError(
                f"Destination already exists for {action.source}: {action.destination}"
            )
        action.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(action.source), str(action.destination))
    return actions


def _migrate_model_metadata(value: object) -> tuple[object, str | None]:
    if isinstance(value, list):
        changed = False
        model_id: str | None = None
        migrated_items: list[object] = []
        for item in value:
            migrated_item, item_model_id = _migrate_model_metadata(item)
            changed = changed or migrated_item is not item
            model_id = model_id or item_model_id
            migrated_items.append(migrated_item)
        return (migrated_items, model_id) if changed else (value, model_id)

    if not isinstance(value, dict):
        return value, None

    changed = False
    model_id: str | None = None
    migrated: dict[object, object] = {}
    for key, item in value.items():
        migrated_item, item_model_id = _migrate_model_metadata(item)
        changed = changed or migrated_item is not item
        model_id = model_id or item_model_id
        migrated[key] = migrated_item

    raw_model = value.get("model")
    if isinstance(raw_model, str) and "/" in raw_model:
        identity = split_model_id(raw_model)
        if identity is not None:
            migrated["modelType"] = identity.model_type
            migrated["model"] = identity.model
            changed = True
            model_id = identity.catalog_key

    return (migrated, model_id) if changed else (value, model_id)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LogMigrationError(f"Invalid JSON metadata file: {path}") from exc


def plan_log_metadata_migration(
    logs_root: Path | str = "logs",
) -> list[LogMetadataMigrationAction]:
    root = Path(logs_root)
    if not root.exists():
        return []
    _refuse_symlink(root, "logs root")
    if not root.is_dir():
        raise LogMigrationError(f"Logs root is not a directory: {root}")

    actions: list[LogMetadataMigrationAction] = []
    for path in sorted(root.rglob("*.json")):
        if path.is_symlink():
            raise LogMigrationError(f"Refusing symlinked metadata file: {path}")
        if not path.is_file():
            continue
        _migrated, model_id = _migrate_model_metadata(_read_json(path))
        if model_id is not None:
            actions.append(LogMetadataMigrationAction(path=path, model_id=model_id))
    return actions


def apply_log_metadata_migration(
    logs_root: Path | str = "logs",
    *,
    apply: bool = False,
) -> list[LogMetadataMigrationAction]:
    actions = plan_log_metadata_migration(logs_root)
    if not apply:
        return actions
    for action in actions:
        _refuse_symlink(action.path, "metadata file")
        payload = _read_json(action.path)
        migrated, model_id = _migrate_model_metadata(payload)
        if model_id is None:
            continue
        temp_path = action.path.with_name(f"{action.path.name}.tmp")
        _refuse_symlink(temp_path, "temporary metadata file")
        temp_path.write_text(
            json.dumps(migrated, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(action.path)
    return actions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move flat Emperor model log folders into categorized paths and "
            "rewrite legacy model metadata."
        )
    )
    parser.add_argument(
        "--logs-root",
        default="logs",
        help="Logs root to inspect. Defaults to ./logs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move folders. Without this flag, only prints the planned moves.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        folder_actions = apply_log_migration(args.logs_root, apply=args.apply)
        metadata_actions = apply_log_metadata_migration(
            args.logs_root,
            apply=args.apply,
        )
    except LogMigrationError as exc:
        raise SystemExit(str(exc)) from exc

    mode = "Moved" if args.apply else "Would move"
    metadata_mode = "Rewrote" if args.apply else "Would rewrite"
    if not folder_actions and not metadata_actions:
        print("No flat model log folders or legacy metadata to migrate.")
        return
    for action in folder_actions:
        print(f"{mode}: {action.source} -> {action.destination}")
    for action in metadata_actions:
        print(f"{metadata_mode}: {action.path} ({action.model_id})")


if __name__ == "__main__":
    main()
