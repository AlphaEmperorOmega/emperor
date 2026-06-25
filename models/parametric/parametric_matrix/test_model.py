import os
import importlib
import runpy
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import torch

import models.parametric.parametric_matrix.config as config

from emperor.experiments.base import GridSearch, RandomSearch
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.parametric import (
    AdaptiveRouterOptions,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
)
from models.parametric.parametric_matrix.config_builder import (
    ParametricMatrixConfigBuilder,
)
from models.parametric.parametric_matrix.experiment_config import ExperimentConfig
from models.parametric.parametric_matrix.model import Model
from models.parametric.parametric_matrix.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
    _preset_locks,
)
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)


class TestParametricMatrixModel(unittest.TestCase):
    def test_public_imports_remain_available(self):
        for module_name in (
            "models.parametric.parametric_matrix.config",
            "models.parametric.parametric_matrix.presets",
            "models.parametric.parametric_matrix.model",
            "models.parametric.parametric_matrix.config_builder",
            "models.parametric.parametric_matrix.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

    def test_experiment_public_model_id_remains_catalog_id(self):
        self.assertEqual(
            Experiment()._public_model_id(),
            "parametric/parametric_matrix",
        )

    def test_module_entrypoint_resolves_cli_without_training(self):
        with (
            patch.object(sys, "argv", ["parametric_matrix", "--preset", "preset"]),
            patch(
                "models.parametric.parametric_matrix.presets.Experiment.train_model",
                autospec=True,
            ) as train_model,
        ):
            runpy.run_module(
                "models.parametric.parametric_matrix.__main__",
                run_name="__main__",
            )

        train_model.assert_called_once()
        experiment = train_model.call_args.args[0]
        kwargs = train_model.call_args.kwargs

        self.assertEqual(experiment.preset, ExperimentPreset.PRESET)
        self.assertIsNone(kwargs["search_mode"])
        self.assertIsNone(kwargs["log_folder"])
        self.assertIsNone(kwargs["search_keys"])
        self.assertEqual(kwargs["config_overrides"], {})
        self.assertEqual(kwargs["search_overrides"], {})
        self.assertEqual(kwargs["selected_datasets"], config.DATASET_OPTIONS)
        self.assertIsNone(kwargs["selected_presets"])

    def test_modern_preset_contract_is_exposed(self):
        self.assertEqual(
            ExperimentPresets.PRESET_OVERRIDES,
            {
                ExperimentPreset.PRESET: {},
                ExperimentPreset.CONFIG: {},
            },
        )
        self.assertEqual(ExperimentPresets.PRESET_LOCKS, {})
        self.assertEqual(
            ExperimentPresets().locked_fields(ExperimentPreset.PRESET),
            {},
        )

    def test_empty_overrides_generate_empty_locks_and_future_locks_reject(self):
        self.assertEqual(_preset_locks(ExperimentPresets.PRESET_OVERRIDES), {})

        locks = _preset_locks(
            {
                ExperimentPreset.PRESET: {
                    "learning_rate": config.LEARNING_RATE,
                },
            }
        )
        original_locks = ExperimentPresets.PRESET_LOCKS
        try:
            ExperimentPresets.PRESET_LOCKS = locks
            with self.assertRaisesRegex(ValueError, "PRESET.*learning_rate"):
                ExperimentPresets().get_config(
                    ExperimentPreset.PRESET,
                    config.DATASET_OPTIONS[0],
                    config_overrides={"learning_rate": config.LEARNING_RATE * 2},
                )
        finally:
            ExperimentPresets.PRESET_LOCKS = original_locks

    def test_builder_returns_boundary_style_experiment_config(self):
        cfg = ParametricMatrixConfigBuilder(
            input_dim=8,
            stack_hidden_dim=4,
            output_dim=3,
        ).build()

        self.assertIsInstance(cfg.experiment_config, ExperimentConfig)
        self.assertIsNotNone(cfg.experiment_config.input_model_config)
        self.assertIsNotNone(cfg.experiment_config.model_config)
        self.assertIsNotNone(cfg.experiment_config.output_model_config)
        self.assertIsInstance(cfg.experiment_config.input_model_config, LayerConfig)
        self.assertIsInstance(cfg.experiment_config.model_config, LayerStackConfig)
        self.assertIsInstance(cfg.experiment_config.output_model_config, LayerConfig)

        model = Model(cfg)
        self.assertEqual(model.input_model.input_dim, 8)
        self.assertEqual(model.input_model.output_dim, 4)
        self.assertEqual(model.model.input_dim, 4)
        self.assertEqual(model.model.hidden_dim, 4)
        self.assertEqual(model.model.output_dim, 4)
        self.assertEqual(model.output_model.input_dim, 4)
        self.assertEqual(model.output_model.output_dim, 3)

    def test_preset_builds_current_parametric_config(self):
        hidden_dim = 5
        cfg = ExperimentPresets()._preset(
            input_dim=8,
            stack_hidden_dim=hidden_dim,
            output_dim=3,
        )
        stack_config = cfg.experiment_config.model_config
        handler_config = stack_config.layer_config
        layer_model_config = handler_config.layer_model_config

        self.assertIsInstance(
            cfg.experiment_config.input_model_config.layer_model_config,
            LinearLayerConfig,
        )
        self.assertIsInstance(
            cfg.experiment_config.output_model_config.layer_model_config,
            LinearLayerConfig,
        )
        self.assertIsInstance(handler_config, ParametricLayerHandlerConfig)
        self.assertIsInstance(layer_model_config, ParametricLayerConfig)
        self.assertIsInstance(
            layer_model_config.weight_mixture_config,
            MatrixWeightsMixtureConfig,
        )
        self.assertIsNone(layer_model_config.bias_mixture_config)
        self.assertEqual(
            layer_model_config.routing_initialization_mode,
            AdaptiveRouterOptions.SHARED_ROUTER,
        )
        self.assertEqual(stack_config.input_dim, hidden_dim)
        self.assertEqual(stack_config.hidden_dim, hidden_dim)
        self.assertEqual(stack_config.output_dim, hidden_dim)
        self.assertTrue(stack_config.apply_output_pipeline_flag)
        self.assertEqual(layer_model_config.input_dim, hidden_dim)
        self.assertEqual(layer_model_config.output_dim, hidden_dim)
        self.assertEqual(layer_model_config.weight_mixture_config.input_dim, hidden_dim)
        self.assertEqual(
            layer_model_config.weight_mixture_config.output_dim,
            hidden_dim,
        )

    def test_bias_option_maps_to_disabled_or_matrix_bias_config(self):
        disabled = ExperimentPresets()._preset(
            input_dim=8,
            stack_hidden_dim=5,
            output_dim=3,
            adaptive_bias_option=None,
        )
        enabled = ExperimentPresets()._preset(
            input_dim=8,
            stack_hidden_dim=5,
            output_dim=3,
            adaptive_bias_option=MatrixBiasMixtureConfig,
        )

        disabled_parametric_config = (
            disabled.experiment_config.model_config.layer_config.layer_model_config
        )
        enabled_parametric_config = (
            enabled.experiment_config.model_config.layer_config.layer_model_config
        )

        self.assertIsNone(disabled_parametric_config.bias_mixture_config)
        self.assertIsInstance(
            enabled_parametric_config.bias_mixture_config,
            MatrixBiasMixtureConfig,
        )

    def test_forward_one_batch_per_dataset(self):
        batch_size = 2
        presets = ExperimentPresets()

        for dataset in config.DATASET_OPTIONS:
            with self.subTest(dataset=dataset.__name__):
                cfg = presets.get_config(ExperimentPreset.PRESET, dataset)[0]
                model = Model(cfg)
                model.eval()
                X = self._fake_batch(dataset, batch_size)

                with torch.no_grad():
                    logits, auxiliary_loss = model(X)

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))
                self.assertEqual(auxiliary_loss.shape, torch.Size([]))

    def test_all_presets_train_one_epoch(self):
        presets = ExperimentPresets()
        dataset = config.DATASET_OPTIONS[0]

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                search_mode = (
                    RandomSearch(num_samples=1)
                    if preset == ExperimentPreset.CONFIG
                    else None
                )
                cfg = presets.get_config(preset, dataset, search_mode)[0]
                model = Model(cfg)
                datamodule = RandomImageClassificationDataModule(dataset)

                tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def test_config_search_space_builds_configs(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.CONFIG,
            config.DATASET_OPTIONS[0],
            RandomSearch(num_samples=2),
        )

        self.assertEqual(len(configs), 2)
        for cfg in configs:
            with self.subTest(hidden_dim=cfg.hidden_dim):
                layer_model_config = (
                    cfg.experiment_config.model_config.layer_config.layer_model_config
                )
                self.assertIsInstance(layer_model_config, ParametricLayerConfig)
                self.assertIsInstance(
                    layer_model_config.weight_mixture_config,
                    MatrixWeightsMixtureConfig,
                )
                self.assertTrue(
                    layer_model_config.bias_mixture_config is None
                    or isinstance(
                        layer_model_config.bias_mixture_config,
                        MatrixBiasMixtureConfig,
                    )
                )

    def test_search_keys_unknown_axis_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ExperimentPresets().get_config(
                ExperimentPreset.CONFIG,
                config.DATASET_OPTIONS[0],
                RandomSearch(num_samples=2),
                search_keys=["bogus_axis"],
            )

        self.assertIn("Unknown", str(ctx.exception))

    def test_preset_accepts_grid_search_over_unlocked_axis(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.PRESET,
            config.DATASET_OPTIONS[0],
            GridSearch(),
            search_keys=["learning_rate"],
        )

        self.assertEqual(len(configs), len(config.SEARCH_SPACE_LEARNING_RATE))
        self.assertEqual(
            {cfg.learning_rate for cfg in configs},
            set(config.SEARCH_SPACE_LEARNING_RATE),
        )

    def test_config_search_applies_matrix_specific_axes(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.CONFIG,
            config.DATASET_OPTIONS[0],
            GridSearch(),
            search_keys=["adaptive_bias_option"],
        )

        self.assertEqual(len(configs), len(config.SEARCH_SPACE_ADAPTIVE_BIAS_OPTION))
        self.assertEqual(
            {
                type(
                    cfg.experiment_config.model_config.layer_config.layer_model_config.bias_mixture_config
                )
                if cfg.experiment_config.model_config.layer_config.layer_model_config.bias_mixture_config
                is not None
                else None
                for cfg in configs
            },
            {None, MatrixBiasMixtureConfig},
        )

    def test_model_step_accepts_tuple_output(self):
        batch_size = 2
        cfg = ExperimentPresets()._preset(input_dim=8, stack_hidden_dim=4, output_dim=3)
        model = Model(cfg)
        X = torch.randn(batch_size, 1, 2, 4)
        y = torch.tensor([0, 2])

        loss, logits, labels = model._model_step((X, y))

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(logits.shape, (batch_size, 3))
        self.assertEqual(labels.shape, (batch_size,))

    def _fake_batch(self, dataset: type, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )


if __name__ == "__main__":
    unittest.main()
