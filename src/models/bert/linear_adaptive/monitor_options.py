from emperor.attention import AttentionMonitorCallback
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterMonitorCallback,
    WeightBankUtilizationMonitorCallback,
)
from emperor.layers import (
    LayerControllerMonitorCallback,
    RecurrentLayerMonitorCallback,
)
from emperor.memory import MemoryMonitorCallback
from emperor.monitoring import MonitorOption

MONITOR_OPTIONS: list[MonitorOption] = [
    MonitorOption(
        name="attention",
        label="Attention",
        description=(
            "Logs Q/K/V norms, attention entropy, max probability, dropout and "
            "mask coverage, auxiliary loss, and attention head visual summaries."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda settings: AttentionMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="recurrent-layer",
        label="Recurrent layers",
        description=(
            "Logs recurrent step count, hidden-state convergence, recurrent gate "
            "openness, halted-state preservation, and step-delta visual summaries."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda settings: RecurrentLayerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="layer-controller",
        label="Layer controllers",
        description=(
            "Logs Layer gate, residual, dropout, layer-norm, and activation "
            "controller statistics without duplicating memory metrics."
        ),
        kinds=["scalar"],
        callback_factory=lambda settings: LayerControllerMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="memory",
        label="Memory modules",
        description=(
            "Logs gating, blend-weight, and state statistics for Emperor memory "
            "modules. Inactive until a memory config is enabled."
        ),
        kinds=["scalar"],
        callback_factory=lambda settings: MemoryMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="adaptive",
        label="Adaptive parameters",
        description=(
            "Logs adaptive weight, bias, diagonal, and mask controller statistics "
            "for adaptive linear layers."
        ),
        kinds=["scalar", "histogram"],
        callback_factory=lambda settings: AdaptiveParameterMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
    MonitorOption(
        name="weight-bank",
        label="Weight banks",
        description=(
            "Logs adaptive parameter bank usage and utilization for bank-backed "
            "dynamic weights and biases."
        ),
        kinds=["scalar", "histogram"],
        callback_factory=lambda settings: WeightBankUtilizationMonitorCallback(
            log_every_n_steps=settings.log_every_n_steps
        ),
    ),
]
