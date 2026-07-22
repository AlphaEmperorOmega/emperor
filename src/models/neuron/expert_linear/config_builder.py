from models.neuron.expert_linear._neuron_config_builder import NeuronConfigBuilder
from models.neuron.expert_linear.runtime_defaults import DEFAULT_RUNTIME
from models.neuron.expert_linear.runtime_options import RuntimeOptions


class NeuronExpertLinearConfigBuilder(NeuronConfigBuilder):
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.neuron.expert_linear NeuronExpertLinearConfigBuilder "
                "runtime must be RuntimeOptions"
            )
        self.runtime = runtime
        super().__init__(**runtime._as_construction_kwargs())


__all__ = ["NeuronExpertLinearConfigBuilder"]
