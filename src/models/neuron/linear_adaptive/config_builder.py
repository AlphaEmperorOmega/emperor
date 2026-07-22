from models.neuron.linear_adaptive._neuron_config_builder import NeuronConfigBuilder
from models.neuron.linear_adaptive.runtime_defaults import DEFAULT_RUNTIME
from models.neuron.linear_adaptive.runtime_options import RuntimeOptions


class NeuronLinearAdaptiveConfigBuilder(NeuronConfigBuilder):
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.neuron.linear_adaptive NeuronLinearAdaptiveConfigBuilder "
                "runtime must be RuntimeOptions"
            )
        self.runtime = runtime
        super().__init__(**runtime._as_construction_kwargs())


__all__ = ["NeuronLinearAdaptiveConfigBuilder"]
