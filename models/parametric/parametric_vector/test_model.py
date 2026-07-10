import importlib
import os
import runpy
import sys
import unittest
from copy import deepcopy
from unittest.mock import patch

import models.parametric.parametric_vector.dataset_options as dataset_options
import models.parametric.parametric_vector.search_space as search_space

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import torch
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.experiments.base import GridSearch, RandomSearch
from emperor.linears.core.config import LinearLayerConfig
from emperor.parametric import (
    AdaptiveRouterOptions,
    ClipParameterOptions,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
    VectorWeightsMixtureConfig,
)

from models.parametric.parametric_vector.config_builder import (
    ParametricVectorConfigBuilder,
)
from models.parametric.parametric_vector.experiment_config import ExperimentConfig
from models.parametric.parametric_vector.model import Model
from models.parametric.parametric_vector.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)
from models.parametric.parametric_vector.runtime_options import (
    ParametricMixtureOptions,
    ParametricRouterOptions,
    ParametricSamplerOptions,
    ParametricStackOptions,
)
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)


class TestParametricVectorModel(unittest.TestCase):
    def test_public_imports_remain_available(self):
        for module_name in (
            "models.parametric.parametric_vector.config",
            "models.parametric.parametric_vector.presets",
            "models.parametric.parametric_vector.model",
            "models.parametric.parametric_vector.config_builder",
            "models.parametric.parametric_vector.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

    def test_experiment_public_model_id_remains_catalog_id(self):
        self.assertEqual(
            Experiment()._public_model_id(),
            "parametric/parametric_vector",
        )

    def test_module_entrypoint_resolves_cli_without_training(self):
        with (
            patch.object(sys, "argv", ["parametric_vector", "--preset", "preset"]),
            patch(
                "models.parametric.parametric_vector.presets.Experiment.train_model",
                autospec=True,
            ) as train_model,
        ):
            runpy.run_module(
                "models.parametric.parametric_vector.__main__",
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
        self.assertEqual(
            kwargs["selected_datasets"],
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ],
        )
        self.assertIsNone(kwargs["selected_presets"])

    def test_modern_preset_contract_is_exposed(self):
        presets = ExperimentPresets()

        self.assertEqual(
            {
                preset: presets.overrides_for_preset(preset)
                for preset in ExperimentPreset
            },
            {
                ExperimentPreset.PRESET: {},
                ExperimentPreset.CONFIG: {},
            },
        )
        self.assertEqual(
            presets.locked_fields(ExperimentPreset.PRESET),
            {},
        )

    def test_empty_overrides_generate_empty_locks(self):
        presets = ExperimentPresets()

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                self.assertEqual(presets.locks_for_preset(preset), {})
                self.assertEqual(presets.locked_fields(preset), {})

    def test_builder_returns_boundary_style_experiment_config(self):
        cfg = ParametricVectorConfigBuilder(
            input_dim=8,
            hidden_dim=4,
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

    def test_option_group_build_matches_flat_kwargs(self):
        stack_options = ParametricStackOptions(
            hidden_dim=7,
            num_layers=2,
            activation=ActivationOptions.MISH,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.15,
        )
        mixture_options = ParametricMixtureOptions(
            top_k=2,
            num_experts=4,
            weighted_parameters_flag=True,
            clip_parameter_option=ClipParameterOptions.AFTER,
            clip_range=0.7,
        )
        sampler_options = ParametricSamplerOptions(
            threshold=0.2,
            filter_above_threshold=True,
            num_topk_samples=3,
            normalize_probabilities_flag=False,
            noisy_topk_flag=True,
            coefficient_of_variation_loss_weight=0.01,
            switch_loss_weight=0.02,
            zero_centred_loss_weight=0.03,
            mutual_information_loss_weight=0.04,
        )
        router_options = ParametricRouterOptions(
            activation=stack_options.activation,
        )
        flat_kwargs = {
            "batch_size": 3,
            "learning_rate": 0.02,
            "input_dim": 8,
            "hidden_dim": stack_options.hidden_dim,
            "output_dim": 5,
            "stack_num_layers": stack_options.num_layers,
            "stack_activation": stack_options.activation,
            "stack_residual_connection_option": (
                stack_options.residual_connection_option
            ),
            "stack_dropout_probability": stack_options.dropout_probability,
            "adaptive_mixture_top_k": mixture_options.top_k,
            "adaptive_mixture_num_experts": mixture_options.num_experts,
            "adaptive_mixture_weighted_parameters_flag": (
                mixture_options.weighted_parameters_flag
            ),
            "adaptive_mixture_clip_parameter_option": (
                mixture_options.clip_parameter_option
            ),
            "adaptive_mixture_clip_range": mixture_options.clip_range,
            "sampler_threshold": sampler_options.threshold,
            "sampler_filter_above_threshold": (sampler_options.filter_above_threshold),
            "sampler_num_topk_samples": sampler_options.num_topk_samples,
            "sampler_normalize_probabilities_flag": (
                sampler_options.normalize_probabilities_flag
            ),
            "sampler_noisy_topk_flag": sampler_options.noisy_topk_flag,
            "sampler_coefficient_of_variation_loss_weight": (
                sampler_options.coefficient_of_variation_loss_weight
            ),
            "sampler_switch_loss_weight": sampler_options.switch_loss_weight,
            "sampler_zero_centred_loss_weight": (
                sampler_options.zero_centred_loss_weight
            ),
            "sampler_mutual_information_loss_weight": (
                sampler_options.mutual_information_loss_weight
            ),
        }
        grouped_kwargs = {
            "batch_size": 3,
            "learning_rate": 0.02,
            "input_dim": 8,
            "output_dim": 5,
            "stack_options": stack_options,
            "mixture_options": mixture_options,
            "sampler_options": sampler_options,
            "router_options": router_options,
        }
        flat_cfg = ParametricVectorConfigBuilder(**flat_kwargs).build()
        grouped_cfg = ParametricVectorConfigBuilder(**grouped_kwargs).build()

        self.assertEqual(flat_cfg, grouped_cfg)

    def test_preset_builds_current_parametric_config(self):
        hidden_dim = 5
        cfg = ExperimentPresets()._preset(
            input_dim=8,
            hidden_dim=hidden_dim,
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
            VectorWeightsMixtureConfig,
        )
        self.assertIsNone(layer_model_config.bias_mixture_config)
        self.assertEqual(
            layer_model_config.routing_initialization_mode,
            AdaptiveRouterOptions.INDEPENDENT_ROUTER,
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

    def test_vector_shared_router_is_rejected(self):
        cfg = ExperimentPresets()._preset(input_dim=8, hidden_dim=4, output_dim=3)
        parametric_config = deepcopy(
            cfg.experiment_config.model_config.layer_config.layer_model_config
        )
        parametric_config.routing_initialization_mode = (
            AdaptiveRouterOptions.SHARED_ROUTER
        )

        with self.assertRaises(ValueError):
            parametric_config.build()

    def test_router_and_sampler_defaults_match_expected_structure(self):
        summary = self._router_sampler_summary(
            ParametricVectorConfigBuilder(
                input_dim=8,
                hidden_dim=9,
                output_dim=3,
            ).build()
        )

        self.assertEqual(
            summary,
            {
                "router_input_dim": 9,
                "router_num_experts": 2,
                "router_noisy_topk_flag": False,
                "router_model_input_dim": 9,
                "router_hidden_dim": 9,
                "router_model_output_dim": 2,
                "router_num_layers": 1,
                "router_last_layer_bias_option": LastLayerBiasOptions.DEFAULT,
                "router_apply_output_pipeline_flag": False,
                "router_activation": ActivationOptions.GELU,
                "router_residual_connection_option": (
                    ResidualConnectionOptions.DISABLED
                ),
                "router_dropout_probability": 0.0,
                "router_layer_norm_position": LayerNormPositionOptions.DISABLED,
                "router_linear_input_dim": 9,
                "router_linear_output_dim": 2,
                "router_linear_bias_flag": True,
                "sampler_top_k": 1,
                "sampler_threshold": 0.0,
                "sampler_filter_above_threshold": False,
                "sampler_num_topk_samples": 0,
                "sampler_normalize_probabilities_flag": False,
                "sampler_noisy_topk_flag": False,
                "sampler_num_experts": 2,
                "sampler_coefficient_of_variation_loss_weight": 0.0,
                "sampler_switch_loss_weight": 0.0,
                "sampler_zero_centred_loss_weight": 0.0,
                "sampler_mutual_information_loss_weight": 0.0,
                "sampler_router_config": None,
            },
        )

    def test_forward_one_batch_per_dataset(self):
        batch_size = 2
        presets = ExperimentPresets()

        for dataset in dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]:
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
        dataset = dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ][0]

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
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
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
                    VectorWeightsMixtureConfig,
                )
                self.assertIsNone(layer_model_config.bias_mixture_config)

    def test_preset_accepts_grid_search_over_unlocked_axis(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.PRESET,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            GridSearch(),
            search_keys=["learning_rate"],
        )

        self.assertEqual(len(configs), len(search_space.SEARCH_SPACE_LEARNING_RATE))
        self.assertEqual(
            {cfg.learning_rate for cfg in configs},
            set(search_space.SEARCH_SPACE_LEARNING_RATE),
        )

    def test_config_search_applies_parametric_axes(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.CONFIG,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            GridSearch(),
            search_keys=["adaptive_mixture_num_experts"],
        )

        self.assertEqual(
            len(configs),
            len(search_space.SEARCH_SPACE_ADAPTIVE_MIXTURE_NUM_EXPERTS),
        )
        self.assertEqual(
            {
                cfg.experiment_config.model_config.layer_config.layer_model_config.weight_mixture_config.num_experts
                for cfg in configs
            },
            set(search_space.SEARCH_SPACE_ADAPTIVE_MIXTURE_NUM_EXPERTS),
        )

    def test_search_keys_unknown_axis_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ExperimentPresets().get_config(
                ExperimentPreset.CONFIG,
                dataset_options.DATASET_OPTIONS_BY_TASK[
                    dataset_options.DEFAULT_EXPERIMENT_TASK
                ][0],
                RandomSearch(num_samples=2),
                search_keys=["bogus_axis"],
            )

        self.assertIn("Unknown", str(ctx.exception))

    def test_model_step_accepts_tuple_output(self):
        batch_size = 2
        cfg = ExperimentPresets()._preset(input_dim=8, hidden_dim=4, output_dim=3)
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

    def _router_sampler_summary(self, cfg) -> dict:
        parametric_config = (
            cfg.experiment_config.model_config.layer_config.layer_model_config
        )
        router_config = parametric_config.router_config
        router_model_config = router_config.model_config
        router_layer_config = router_model_config.layer_config
        router_linear_config = router_layer_config.layer_model_config
        sampler_config = parametric_config.sampler_config

        return {
            "router_input_dim": router_config.input_dim,
            "router_num_experts": router_config.num_experts,
            "router_noisy_topk_flag": router_config.noisy_topk_flag,
            "router_model_input_dim": router_model_config.input_dim,
            "router_hidden_dim": router_model_config.hidden_dim,
            "router_model_output_dim": router_model_config.output_dim,
            "router_num_layers": router_model_config.num_layers,
            "router_last_layer_bias_option": (
                router_model_config.last_layer_bias_option
            ),
            "router_apply_output_pipeline_flag": (
                router_model_config.apply_output_pipeline_flag
            ),
            "router_activation": router_layer_config.activation,
            "router_residual_connection_option": (
                router_layer_config.residual_connection_option
            ),
            "router_dropout_probability": router_layer_config.dropout_probability,
            "router_layer_norm_position": router_layer_config.layer_norm_position,
            "router_linear_input_dim": router_linear_config.input_dim,
            "router_linear_output_dim": router_linear_config.output_dim,
            "router_linear_bias_flag": router_linear_config.bias_flag,
            "sampler_top_k": sampler_config.top_k,
            "sampler_threshold": sampler_config.threshold,
            "sampler_filter_above_threshold": (sampler_config.filter_above_threshold),
            "sampler_num_topk_samples": sampler_config.num_topk_samples,
            "sampler_normalize_probabilities_flag": (
                sampler_config.normalize_probabilities_flag
            ),
            "sampler_noisy_topk_flag": sampler_config.noisy_topk_flag,
            "sampler_num_experts": sampler_config.num_experts,
            "sampler_coefficient_of_variation_loss_weight": (
                sampler_config.coefficient_of_variation_loss_weight
            ),
            "sampler_switch_loss_weight": sampler_config.switch_loss_weight,
            "sampler_zero_centred_loss_weight": (
                sampler_config.zero_centred_loss_weight
            ),
            "sampler_mutual_information_loss_weight": (
                sampler_config.mutual_information_loss_weight
            ),
            "sampler_router_config": sampler_config.router_config,
        }


if __name__ == "__main__":
    unittest.main()
