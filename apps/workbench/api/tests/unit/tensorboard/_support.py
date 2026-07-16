from __future__ import annotations

from unittest.mock import patch

_UNSET = object()


def patch_event_accumulator_loader(
    loader=_UNSET,
    *,
    return_value=_UNSET,
):
    import emperor_workbench.tensorboard._events as implementation

    if return_value is not _UNSET:
        return patch.object(
            implementation,
            "load_event_accumulator",
            return_value=return_value,
        )
    if loader is _UNSET:
        return patch.object(
            implementation,
            "load_event_accumulator",
        )
    return patch.object(
        implementation,
        "load_event_accumulator",
        loader,
    )


__all__ = ["patch_event_accumulator_loader"]
