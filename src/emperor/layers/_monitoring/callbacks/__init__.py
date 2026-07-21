from emperor.layers._monitoring.callbacks._hooks import (
    _extract_hidden_tensor,
    _install_method_replacement,
    _MethodReplacement,
    _remove_hooks,
    _restore_method_replacements,
)
from emperor.layers._monitoring.callbacks.layer_controller import (
    LayerControllerMonitorCallback,
)
from emperor.layers._monitoring.callbacks.recurrent import (
    RecurrentLayerMonitorCallback,
)

__all__ = [
    "LayerControllerMonitorCallback",
    "RecurrentLayerMonitorCallback",
]

for _export in (
    LayerControllerMonitorCallback,
    RecurrentLayerMonitorCallback,
    _MethodReplacement,
    _extract_hidden_tensor,
    _install_method_replacement,
    _remove_hooks,
    _restore_method_replacements,
):
    _export.__module__ = __name__

del _export
