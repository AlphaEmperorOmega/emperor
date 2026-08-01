import models.linears.linear.config as config
import models.linears.linear.dataset_options as dataset_options
from emperor.config import BaseOptions
from emperor.layers import (
    AdditiveResidualConfig,
    AttentionResidualConfig,
    LayerNormPositionOptions,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.memory import (
    ElementWiseWeightedDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,
)
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    ExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase
from models.linears.linear.config_builder import LinearConfigBuilder
from models.linears.linear.model import Model
from models.linears.linear.runtime_defaults import runtime_from_flat


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    GATING = 2
    HALTING = 3
    MEMORY = 4
    GATING_HALTING = 5
    GATING_MEMORY = 6
    HALTING_MEMORY = 7
    GATING_HALTING_MEMORY = 8
    RESIDUAL = 9
    POST_NORM = 10
    RESIDUAL_POST_NORM = 11
    RESIDUAL_GATING = 12
    RESIDUAL_HALTING = 13
    RESIDUAL_MEMORY = 14
    RECURRENT = 15
    RECURRENT_GATING = 16
    RECURRENT_HALTING = 17
    RECURRENT_MEMORY = 18
    RECURRENT_GATING_HALTING = 19
    RECURRENT_GATING_MEMORY = 20
    RECURRENT_HALTING_MEMORY = 21
    RECURRENT_GATING_HALTING_MEMORY = 22
    RECURRENT_RESIDUAL = 23
    RECURRENT_POST_NORM = 24
    WEIGHTED_RESIDUAL = 25
    WEIGHTED_BLEND_RESIDUAL = 26
    ATTENTION_RESIDUAL = 27
    RECURRENT_LAYER_GATING = 28
    RECURRENT_DUAL_GATING = 29
    RECURRENT_LAYER_HALTING = 30
    RECURRENT_DUAL_HALTING = 31
    WEIGHTED_MEMORY = 32
    ELEMENT_WISE_WEIGHTED_MEMORY = 33
    NO_NORM = 34
    PRE_ACTIVATION_NORM = 35


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: PresetDefinition(
        preset_values={},
        description="Default config: a GELU hidden linear stack with pre-layer norm and "
        "dropout.",
    ),
    ExperimentPreset.GATING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
        },
        description="Default config with per-layer gating enabled, so each hidden layer "
        "output is modulated by a learned sigmoid gate.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
        },
        description="Default config with stack halting enabled, so examples can stop early "
        "as they move through the hidden stack.",
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values={
            "memory_flag": True,
        },
        description="Default config with shared stack memory enabled across the hidden "
        "layers.",
    ),
    ExperimentPreset.GATING_HALTING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
        },
        description="Default config with both per-layer gating and stack halting enabled.",
    ),
    ExperimentPreset.GATING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "memory_flag": True,
        },
        description="Default config with both per-layer gating and shared stack memory "
        "enabled.",
    ),
    ExperimentPreset.HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default config with both stack halting and shared stack memory "
        "enabled.",
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default config with per-layer gating, stack halting, and shared stack "
        "memory enabled.",
    ),
    ExperimentPreset.RESIDUAL: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": AdditiveResidualConfig,
        },
        description="Default config with residual skip connections enabled between "
        "same-width hidden layers.",
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description="Default config with layer norm applied after each layer instead of "
        "before it.",
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": AdditiveResidualConfig,
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description="Default config with residual skip connections and post-layer "
        "normalization enabled.",
    ),
    ExperimentPreset.RESIDUAL_GATING: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": AdditiveResidualConfig,
            "stack_gate_flag": True,
        },
        description="Default config with residual skip connections and per-layer gating "
        "enabled.",
    ),
    ExperimentPreset.RESIDUAL_HALTING: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": AdditiveResidualConfig,
            "stack_halting_flag": True,
        },
        description="Default config with residual skip connections and stack halting "
        "enabled.",
    ),
    ExperimentPreset.RESIDUAL_MEMORY: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": AdditiveResidualConfig,
            "memory_flag": True,
        },
        description="Default config with residual skip connections and shared stack memory "
        "enabled.",
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
        },
        description="Default config wrapped in fixed-step recurrence, reusing the hidden "
        "stack for each recurrent step.",
    ),
    ExperimentPreset.RECURRENT_GATING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
        },
        description="Default recurrent config with step-level gating enabled after each "
        "recurrent update.",
    ),
    ExperimentPreset.RECURRENT_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_halting_flag": True,
        },
        description="Default recurrent config with recurrent halting enabled, allowing "
        "early stopping before the max step count.",
    ),
    ExperimentPreset.RECURRENT_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config whose reused hidden stack has shared memory "
        "enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "recurrent_stack_halting_flag": True,
        },
        description="Default recurrent config with both step-level gating and recurrent "
        "halting enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with step-level gating and shared memory in "
        "the reused hidden stack.",
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with recurrent halting and shared memory in "
        "the reused hidden stack.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_stack_gate_flag": True,
            "recurrent_stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with step-level gating, recurrent halting, "
        "and shared memory in the reused hidden stack.",
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "stack_residual_connection_option": AdditiveResidualConfig,
        },
        description="Default recurrent config using a residual hidden stack at each "
        "recurrent step.",
    ),
    ExperimentPreset.RECURRENT_POST_NORM: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        },
        description="Default recurrent config using a post-normalized hidden stack at each "
        "recurrent step.",
    ),
    ExperimentPreset.WEIGHTED_RESIDUAL: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": WeightedResidualConfig,
        },
        description="Default config with a learned tanh-scaled current contribution "
        "composed with the previous hidden state at each hidden layer.",
    ),
    ExperimentPreset.WEIGHTED_BLEND_RESIDUAL: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": WeightedBlendResidualConfig,
        },
        description="Default config with a learned bounded convex blend between current "
        "and previous hidden states at each hidden layer.",
    ),
    ExperimentPreset.ATTENTION_RESIDUAL: PresetDefinition(
        preset_values={
            "stack_residual_connection_option": AttentionResidualConfig,
        },
        description="Default config with depth-local attention over compatible residual "
        "sources in the hidden stack.",
    ),
    ExperimentPreset.RECURRENT_LAYER_GATING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "stack_gate_flag": True,
        },
        description="Recurrent config with per-layer gating inside the reused hidden "
        "stack and no outer recurrent-step gate.",
    ),
    ExperimentPreset.RECURRENT_DUAL_GATING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "stack_gate_flag": True,
            "recurrent_stack_gate_flag": True,
        },
        description="Recurrent config with separate inner per-layer gates and an outer "
        "recurrent-step gate.",
    ),
    ExperimentPreset.RECURRENT_LAYER_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "stack_halting_flag": True,
        },
        description="Recurrent config with halting inside the reused hidden stack while "
        "the outer recurrence retains a fixed maximum step budget.",
    ),
    ExperimentPreset.RECURRENT_DUAL_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "stack_halting_flag": True,
            "recurrent_stack_halting_flag": True,
        },
        description="Recurrent config with separate inner-stack and outer recurrent-step "
        "stick-breaking halting controllers.",
    ),
    ExperimentPreset.WEIGHTED_MEMORY: PresetDefinition(
        preset_values={
            "memory_flag": True,
            "memory_option": WeightedDynamicMemoryConfig,
        },
        description="Default config with shared stack memory using a sample- and "
        "position-level weighted merge.",
    ),
    ExperimentPreset.ELEMENT_WISE_WEIGHTED_MEMORY: PresetDefinition(
        preset_values={
            "memory_flag": True,
            "memory_option": ElementWiseWeightedDynamicMemoryConfig,
        },
        description="Default config with shared stack memory using a per-feature weighted "
        "merge.",
    ),
    ExperimentPreset.NO_NORM: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.DISABLED,
        },
        description="Default config with no layer normalization in the main hidden stack.",
    ),
    ExperimentPreset.PRE_ACTIVATION_NORM: PresetDefinition(
        preset_values={
            "layer_norm_position": LayerNormPositionOptions.DEFAULT,
        },
        description="Default config with normalization after affine and memory work but "
        "before activation, distinct from pre-layer and post-layer normalization.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=LinearConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
        )

    def _preset(self, **kwargs):
        runtime = runtime_from_flat(kwargs)
        return self._builder_type(runtime=runtime).build()


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_preset: ExperimentPreset | None = None,
        experiment_task=None,
        *,
        model_package,
        run_artifacts=None,
    ) -> None:
        super().__init__(
            experiment_preset,
            experiment_task=experiment_task,
            model_package=model_package,
            run_artifacts=run_artifacts,
        )

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        return ExperimentPreset
