from __future__ import annotations

from emperor.base.utils import ConfigBase


def abstract_config_class_error(candidate: type) -> str | None:
    try:
        if not issubclass(candidate, ConfigBase):
            return None
    except TypeError:
        return None

    try:
        instance = candidate()
    except TypeError:
        return None

    try:
        instance._registry_owner()
    except (NotImplementedError, ValueError) as exc:
        return str(exc)
    return None
