from __future__ import annotations


def normalize_preset_token(preset: str | None) -> str | None:
    if preset is None:
        return None
    return str(preset).lower().replace("_", "-")


__all__ = ["normalize_preset_token"]
