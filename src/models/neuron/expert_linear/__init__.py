from __future__ import annotations

from model_runtime.packages import (
    InspectionConstructionLimits,
    InspectionFieldProductLimit,
    ModelIdentity,
    ModelMetadata,
    ModelPackage,
)

_IDENTITY = ModelIdentity("neuron", "expert_linear")


class _ModelPackageAdapter:
    def load_metadata(self) -> ModelMetadata:
        from . import config, dataset_options, monitor_options, search_space
        from ._inspection_metadata import (
            CONFIGURATION_METADATA_SECTIONS,
            SEARCH_METADATA_SECTIONS,
        )

        return ModelMetadata(
            identity=_IDENTITY,
            runtime_defaults=config,
            dataset_options=dataset_options,
            monitor_options_source=monitor_options,
            search_space=search_space,
            configuration_metadata_sections=CONFIGURATION_METADATA_SECTIONS,
            search_metadata_sections=SEARCH_METADATA_SECTIONS,
        )

    def load_runtime_options_type(self) -> type:
        from .runtime_options import RuntimeOptions

        return RuntimeOptions

    def bind_runtime_defaults(self, values):
        from .runtime_defaults import runtime_from_flat

        return runtime_from_flat(values)

    def load_preset_type(self) -> type:
        from .presets import ExperimentPreset

        return ExperimentPreset

    def load_presets(self):
        from .presets import ExperimentPresets

        return ExperimentPresets()

    def build_configuration(self, presets, preset, dataset, **kwargs):
        return presets.get_config(preset, dataset, **kwargs)[0]

    def build_model(self, configuration):
        from .model import Model

        return Model(configuration)

    def build_experiment(
        self,
        preset,
        *,
        experiment_task,
        model_package,
        run_artifacts,
    ):
        from .presets import Experiment

        return Experiment(
            preset,
            experiment_task=experiment_task,
            model_package=model_package,
            run_artifacts=run_artifacts,
        )


_INSPECTION_LIMITS = InspectionConstructionLimits(
    field_maximums={"CLUSTER_BEAM_WIDTH": 64},
    field_product_limits=(
        InspectionFieldProductLimit(
            label="initial neuron count",
            factors=(
                (
                    "CLUSTER_INITIAL_X_AXIS_TOTAL_NEURONS",
                    "CLUSTER_X_AXIS_TOTAL_NEURONS",
                ),
                (
                    "CLUSTER_INITIAL_Y_AXIS_TOTAL_NEURONS",
                    "CLUSTER_Y_AXIS_TOTAL_NEURONS",
                ),
                (
                    "CLUSTER_INITIAL_Z_AXIS_TOTAL_NEURONS",
                    "CLUSTER_Z_AXIS_TOTAL_NEURONS",
                ),
            ),
            maximum=4_096,
            repeats_dense_parameter_estimate=True,
        ),
        InspectionFieldProductLimit(
            label="neuron capacity",
            factors=(
                ("CLUSTER_X_AXIS_TOTAL_NEURONS",),
                ("CLUSTER_Y_AXIS_TOTAL_NEURONS",),
                ("CLUSTER_Z_AXIS_TOTAL_NEURONS",),
            ),
            maximum=4_096,
        ),
    ),
)

MODEL_PACKAGE = ModelPackage(
    _IDENTITY,
    _ModelPackageAdapter(),
    inspection_construction_limits=_INSPECTION_LIMITS,
)

__all__ = ["MODEL_PACKAGE"]
