from __future__ import annotations

from model_runtime.packages import ModelIdentity, ModelMetadata, ModelPackage

_IDENTITY = ModelIdentity("gpt", "expert_linear_adaptive")


class _ModelPackageAdapter:
    def load_metadata(self) -> ModelMetadata:
        from . import config, dataset_options, monitor_options, search_space

        return ModelMetadata(
            identity=_IDENTITY,
            runtime_defaults=config,
            dataset_options=dataset_options,
            monitor_options_source=monitor_options,
            search_space=search_space,
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

    def build_configurations(self, presets, preset, dataset, **kwargs):
        if dataset is None:
            return presets.get_config(preset, **kwargs)
        return presets.get_config(preset, dataset, **kwargs)

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


MODEL_PACKAGE = ModelPackage(_IDENTITY, _ModelPackageAdapter())

__all__ = ["MODEL_PACKAGE"]
