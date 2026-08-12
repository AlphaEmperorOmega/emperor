from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from importlib import import_module

from emperor.halting import SoftHaltingConfig
from emperor.layers import GateConfig
from emperor.neuron import NeuronClusterConfig
from models.catalog import discover_model_packages, model_package


def _contains_instance(root: object, expected_type: type) -> bool:
    pending = [root]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, expected_type):
            return True
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, field.name) for field in fields(value))
        elif isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            pending.extend(value)
    return False


def _find_instance(root: object, expected_type: type):
    pending = [root]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, expected_type):
            return value
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, field.name) for field in fields(value))
        elif isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            pending.extend(value)
    return None


class ModelPackageConfigPropagationContractTests(unittest.TestCase):
    def test_neuron_outer_runtime_resolvers_accept_one_flat_mapping(self) -> None:
        overrides = {
            "cluster_x_axis_total_neurons": 7,
            "cluster_terminal_top_k": 2,
            "cluster_halting_threshold": 0.37,
        }
        for catalog_key in (
            "neuron/linear",
            "neuron/linear_adaptive",
            "neuron/expert_linear",
            "neuron/expert_linear_adaptive",
        ):
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None

            configuration = package.build_configuration(config_overrides=overrides)
            cluster_config = _find_instance(configuration, NeuronClusterConfig)

            with self.subTest(model_package=catalog_key):
                self.assertIsNotNone(cluster_config)
                assert cluster_config is not None
                self.assertEqual(cluster_config.x_axis_total_neurons, 7)
                self.assertEqual(cluster_config.halting_config.threshold, 0.37)
                self.assertEqual(
                    cluster_config.neuron_config.terminal_config.sampler_config.top_k,
                    2,
                )

    def test_adaptive_runtime_groups_preserve_flat_override_behavior(
        self,
    ) -> None:
        cases = (
            (
                "models.experts.linear_adaptive.runtime_defaults",
                "models.experts.linear_adaptive.config",
            ),
            (
                "models.neuron.expert_linear_adaptive._hidden.runtime_defaults",
                "models.neuron.expert_linear_adaptive.config",
            ),
        )
        for runtime_module_name, config_module_name in cases:
            runtime_from_flat = import_module(runtime_module_name).runtime_from_flat
            config_module = import_module(config_module_name)
            runtime = runtime_from_flat(
                {
                    "hidden_dim": 37,
                    "top_k": 3,
                    "expert_gate_stack_hidden_dim": 23,
                    "router_memory_flag": True,
                    "recurrent_max_steps": 7,
                    "residual_stack_independent_flag": True,
                    "residual_stack_hidden_dim": 19,
                    "weight_option_flag": True,
                    "weight_option": config_module.LowRankDynamicWeightConfig,
                    "input_layer_mask_threshold": 0.25,
                    "router_weight_option_flag": True,
                    "router_weight_option": config_module.LowRankDynamicWeightConfig,
                }
            )

            with self.subTest(runtime=runtime_module_name):
                self.assertEqual(runtime.stack_options.hidden_dim, 37)
                self.assertEqual(runtime.mixture_options.top_k, 3)
                self.assertEqual(
                    runtime.expert_layer_controller_options.gate_stack_source.hidden_dim,
                    23,
                )
                self.assertTrue(runtime.router_dynamic_memory_options.memory_flag)
                self.assertEqual(
                    runtime.recurrent_controller_options.recurrent_max_steps,
                    7,
                )
                self.assertEqual(
                    runtime.stack_options.residual_stack_options.hidden_dim,
                    19,
                )
                self.assertTrue(runtime.hidden_adaptive_weight_options.option_flag)
                self.assertEqual(runtime.input_boundary_options.mask_threshold, 0.25)
                self.assertTrue(runtime.router_adaptive_weight_options.option_flag)

    def test_expert_controller_flat_updates_reach_the_named_stack(
        self,
    ) -> None:
        for catalog_key in (
            "bert/expert_linear_adaptive",
            "gpt/expert_linear_adaptive",
        ):
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None

            configuration = package.build_configuration(
                config_overrides={
                    "expert_stack_gate_flag": True,
                    "expert_gate_stack_independent_flag": True,
                    "expert_gate_stack_hidden_dim": 37,
                }
            )
            gate_config = _find_instance(configuration, GateConfig)

            with self.subTest(model_package=catalog_key):
                self.assertIsNotNone(gate_config)
                assert gate_config is not None
                self.assertEqual(gate_config.model_config.hidden_dim, 37)

    def test_adaptive_expert_runtime_state_stays_behind_package_interface(
        self,
    ) -> None:
        for catalog_key in (
            "experts/linear_adaptive",
            "neuron/expert_linear_adaptive",
        ):
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None

            runtime = package.bind_runtime_defaults({"hidden_dim": 37})
            configuration = package.build_configuration(
                config_overrides={"hidden_dim": 37}
            )

            with self.subTest(model_package=catalog_key):
                self.assertIs(type(runtime), package.runtime_options_type)
                self.assertFalse(hasattr(runtime, "_resolved_state"))
                self.assertEqual(configuration.hidden_dim, 37)

    def test_neuron_cluster_lifecycle_controls_reach_final_construction(self) -> None:
        expected = {
            "beam_width": 2,
            "growth_cooldown_steps": 3,
            "max_total_growths": 4,
            "growth_warmup_steps": 5,
            "pruning_threshold": 6,
            "escape_driven_growth_flag": True,
            "mitosis_initialization_flag": True,
        }
        overrides = {f"cluster_{field}": value for field, value in expected.items()}
        for catalog_key in (
            "neuron/linear",
            "neuron/linear_adaptive",
            "neuron/expert_linear",
            "neuron/expert_linear_adaptive",
        ):
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            with self.subTest(model_package=catalog_key):
                advertised = set(package.runtime_defaults_spec.supported_keys)
                self.assertEqual(
                    set(),
                    {key.upper() for key in overrides} - advertised,
                )
                configuration = package.build_configuration(
                    config_overrides=overrides,
                )
                cluster_config = _find_instance(configuration, NeuronClusterConfig)
                self.assertIsNotNone(cluster_config)
                assert cluster_config is not None
                for field, value in expected.items():
                    self.assertEqual(getattr(cluster_config, field), value)

    def test_transformer_packages_expose_common_run_controls(self) -> None:
        common_run_controls = {
            "TRAINER_ACCELERATOR",
            "TRAINER_DEVICES",
            "TRAINER_ACCUMULATE_GRAD_BATCHES",
            "TRAINER_PRECISION",
            "TRAINER_BENCHMARK",
            "TRAINER_MAX_STEPS",
            "TRAINER_MAX_TIME",
            "TRAINER_VAL_CHECK_INTERVAL",
            "TRAINER_LIMIT_TRAIN_BATCHES",
            "TRAINER_LIMIT_VAL_BATCHES",
            "TRAINER_OVERFIT_BATCHES",
            "TRAINER_NUM_SANITY_VAL_STEPS",
            "TRAINER_ENABLE_PROGRESS_BAR",
            "TRAINER_ENABLE_CHECKPOINTING",
            "TRAINER_ENABLE_MODEL_SUMMARY",
            "TRAINER_PROFILER",
            "MONITOR_LOG_EVERY_N_STEPS",
        }
        for catalog_key in (
            "transformer/linear",
            "transformer/linear_adaptive",
            "transformer/expert_linear",
            "transformer/expert_linear_adaptive",
        ):
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            advertised = set(package.runtime_defaults_spec.supported_keys)
            with self.subTest(model_package=catalog_key):
                self.assertEqual(set(), common_run_controls - advertised)

    def test_bert_nsp_output_dimension_is_not_a_runtime_option(self) -> None:
        for catalog_key in (
            "bert/linear",
            "bert/linear_adaptive",
            "bert/expert_linear",
            "bert/expert_linear_adaptive",
        ):
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            with self.subTest(model_package=catalog_key):
                self.assertNotIn(
                    "NSP_OUTPUT_DIM",
                    package.runtime_defaults_spec.supported_keys,
                )
                if catalog_key == "bert/linear":
                    with self.assertRaisesRegex(
                        ValueError,
                        "next-sentence prediction.*exactly 2",
                    ):
                        package.build_configuration(
                            config_overrides={"nsp_output_dim": 3}
                        )
                else:
                    with self.assertRaisesRegex(
                        ValueError,
                        "unknown Runtime Defaults field.*nsp_output_dim",
                    ):
                        package.bind_runtime_defaults({"nsp_output_dim": 3})

    def test_neuron_cluster_halting_strategy_reaches_final_construction(self) -> None:
        for catalog_key in (
            "neuron/linear",
            "neuron/linear_adaptive",
            "neuron/expert_linear",
            "neuron/expert_linear_adaptive",
        ):
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            with self.subTest(model_package=catalog_key):
                configuration = package.build_configuration(
                    config_overrides={
                        "cluster_halting_option": SoftHaltingConfig,
                    },
                )
                self.assertTrue(_contains_instance(configuration, SoftHaltingConfig))

    def test_neuron_hidden_halting_strategies_reach_final_construction(self) -> None:
        package_roles = {
            "neuron/linear": ("",),
            "neuron/expert_linear": ("", "expert_"),
            "neuron/expert_linear_adaptive": ("", "expert_", "router_"),
        }
        for catalog_key, roles in package_roles.items():
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            for role in roles:
                with self.subTest(model_package=catalog_key, role=role or "layer"):
                    configuration = package.build_configuration(
                        config_overrides={
                            f"{role}stack_halting_flag": True,
                            f"{role}halting_option": SoftHaltingConfig,
                        },
                    )
                    self.assertTrue(
                        _contains_instance(configuration, SoftHaltingConfig)
                    )

    def test_neuron_hidden_recurrent_halting_strategies_reach_construction(
        self,
    ) -> None:
        package_roles = {
            "neuron/linear": ("",),
            "neuron/expert_linear": ("", "expert_"),
            "neuron/expert_linear_adaptive": ("", "expert_", "router_"),
        }
        for catalog_key, roles in package_roles.items():
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            for role in roles:
                with self.subTest(model_package=catalog_key, role=role or "layer"):
                    configuration = package.build_configuration(
                        config_overrides={
                            f"{role}recurrent_flag": True,
                            f"{role}recurrent_stack_halting_flag": True,
                            f"{role}recurrent_halting_option": SoftHaltingConfig,
                        },
                    )
                    self.assertTrue(
                        _contains_instance(configuration, SoftHaltingConfig)
                    )

    def test_expert_halting_strategies_reach_final_construction(self) -> None:
        package_roles = {
            "experts/linear": ("", "expert_"),
            "experts/linear_adaptive": ("", "expert_", "router_"),
        }
        for catalog_key, roles in package_roles.items():
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            for role in roles:
                with self.subTest(model_package=catalog_key, role=role or "layer"):
                    configuration = package.build_configuration(
                        config_overrides={
                            f"{role}stack_halting_flag": True,
                            f"{role}halting_option": SoftHaltingConfig,
                        },
                    )
                    self.assertTrue(
                        _contains_instance(configuration, SoftHaltingConfig)
                    )

    def test_expert_recurrent_halting_strategies_reach_construction(self) -> None:
        package_roles = {
            "experts/linear": ("", "expert_"),
            "experts/linear_adaptive": ("", "expert_", "router_"),
        }
        for catalog_key, roles in package_roles.items():
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            for role in roles:
                with self.subTest(model_package=catalog_key, role=role or "layer"):
                    configuration = package.build_configuration(
                        config_overrides={
                            f"{role}recurrent_flag": True,
                            f"{role}recurrent_stack_halting_flag": True,
                            f"{role}recurrent_halting_option": SoftHaltingConfig,
                        },
                    )
                    self.assertTrue(
                        _contains_instance(configuration, SoftHaltingConfig)
                    )

    def test_fixed_transformer_halting_gate_dimension_is_not_advertised(self) -> None:
        for catalog_key in (
            "bert/expert_linear",
            "gpt/expert_linear",
            "vit/expert_linear",
        ):
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            with self.subTest(model_package=catalog_key):
                self.assertNotIn(
                    "HALTING_OUTPUT_DIM",
                    package.runtime_defaults_spec.supported_keys,
                )

    def test_transformer_linear_halting_strategies_reach_final_construction(
        self,
    ) -> None:
        roles = ("", "attn_", "ff_")
        for catalog_key in (
            "bert/linear",
            "bert/expert_linear",
            "gpt/linear",
            "gpt/expert_linear",
        ):
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            for role in roles:
                with self.subTest(model_package=catalog_key, role=role or "layer"):
                    overrides = {
                        f"{role}stack_halting_flag": True,
                        f"{role}halting_option": SoftHaltingConfig,
                    }
                    configuration = package.build_configuration(
                        config_overrides=overrides,
                    )
                    self.assertTrue(
                        _contains_instance(configuration, SoftHaltingConfig)
                    )

    def test_transformer_linear_recurrent_halting_strategies_reach_construction(
        self,
    ) -> None:
        roles = ("", "attn_", "ff_")
        for catalog_key in (
            "bert/linear",
            "bert/expert_linear",
            "gpt/linear",
            "gpt/expert_linear",
        ):
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            for role in roles:
                with self.subTest(model_package=catalog_key, role=role or "layer"):
                    overrides = {
                        f"{role}recurrent_flag": True,
                        f"{role}recurrent_stack_halting_flag": True,
                        f"{role}recurrent_halting_option": SoftHaltingConfig,
                    }
                    configuration = package.build_configuration(
                        config_overrides=overrides,
                    )
                    self.assertTrue(
                        _contains_instance(configuration, SoftHaltingConfig)
                    )

    def test_linear_halting_strategy_reaches_final_construction(self) -> None:
        package = model_package("linears/linear")
        self.assertIsNotNone(package)
        assert package is not None
        overrides = {
            "stack_halting_flag": True,
            "halting_option": SoftHaltingConfig,
        }

        runtime = package.bind_runtime_defaults(overrides)
        configuration = package.build_configuration(config_overrides=overrides)
        experiment_config = configuration.experiment_config
        self.assertIsNotNone(experiment_config)
        assert experiment_config is not None
        model_config = experiment_config.model_config
        self.assertIsNotNone(model_config)
        assert model_config is not None

        self.assertIs(runtime.halting_option, SoftHaltingConfig)
        self.assertIsInstance(
            model_config.layer_config.halting_config,
            SoftHaltingConfig,
        )

    def test_linear_recurrent_halting_strategy_reaches_construction(self) -> None:
        package = model_package("linears/linear")
        self.assertIsNotNone(package)
        assert package is not None
        overrides = {
            "recurrent_flag": True,
            "recurrent_stack_halting_flag": True,
            "recurrent_halting_option": SoftHaltingConfig,
        }

        runtime = package.bind_runtime_defaults(overrides)
        configuration = package.build_configuration(config_overrides=overrides)
        experiment_config = configuration.experiment_config
        self.assertIsNotNone(experiment_config)
        assert experiment_config is not None
        model_config = experiment_config.model_config
        self.assertIsNotNone(model_config)
        assert model_config is not None

        self.assertIs(runtime.recurrent_halting_option, SoftHaltingConfig)
        self.assertIsInstance(model_config.halting_config, SoftHaltingConfig)

    def test_linear_adaptive_halting_strategy_reaches_final_construction(self) -> None:
        package = model_package("linears/linear_adaptive")
        self.assertIsNotNone(package)
        assert package is not None
        overrides = {
            "stack_halting_flag": True,
            "halting_option": SoftHaltingConfig,
        }

        runtime = package.bind_runtime_defaults(overrides)
        configuration = package.build_configuration(config_overrides=overrides)
        experiment_config = configuration.experiment_config
        self.assertIsNotNone(experiment_config)
        assert experiment_config is not None
        model_config = experiment_config.model_config
        self.assertIsNotNone(model_config)
        assert model_config is not None

        self.assertIs(runtime.halting_option, SoftHaltingConfig)
        self.assertIsInstance(
            model_config.layer_config.halting_config,
            SoftHaltingConfig,
        )

    def test_linear_adaptive_recurrent_halting_strategy_reaches_construction(
        self,
    ) -> None:
        package = model_package("linears/linear_adaptive")
        self.assertIsNotNone(package)
        assert package is not None
        overrides = {
            "recurrent_flag": True,
            "recurrent_stack_halting_flag": True,
            "recurrent_halting_option": SoftHaltingConfig,
        }

        runtime = package.bind_runtime_defaults(overrides)
        configuration = package.build_configuration(config_overrides=overrides)
        experiment_config = configuration.experiment_config
        self.assertIsNotNone(experiment_config)
        assert experiment_config is not None
        model_config = experiment_config.model_config
        self.assertIsNotNone(model_config)
        assert model_config is not None

        self.assertIs(runtime.recurrent_halting_option, SoftHaltingConfig)
        self.assertIsInstance(model_config.halting_config, SoftHaltingConfig)

    def test_neuron_linear_adaptive_halting_strategy_reaches_hidden_block(self) -> None:
        package = model_package("neuron/linear_adaptive")
        self.assertIsNotNone(package)
        assert package is not None
        overrides = {
            "stack_halting_flag": True,
            "halting_option": SoftHaltingConfig,
        }

        configuration = package.build_configuration(config_overrides=overrides)
        experiment_config = configuration.experiment_config
        self.assertIsNotNone(experiment_config)
        assert experiment_config is not None
        cluster_config = experiment_config.neuron_cluster_config
        self.assertIsNotNone(cluster_config)
        assert cluster_config is not None
        neuron_config = cluster_config.neuron_config
        self.assertIsNotNone(neuron_config)
        assert neuron_config is not None
        nucleus_config = neuron_config.nucleus_config
        self.assertIsNotNone(nucleus_config)
        assert nucleus_config is not None
        hidden_block = nucleus_config.model_config
        self.assertIsNotNone(hidden_block)
        assert hidden_block is not None
        hidden_model = hidden_block.model_config
        self.assertIsNotNone(hidden_model)
        assert hidden_model is not None

        self.assertIsInstance(
            hidden_model.layer_config.halting_config,
            SoftHaltingConfig,
        )

    def test_neuron_linear_adaptive_recurrent_halting_reaches_hidden_block(
        self,
    ) -> None:
        package = model_package("neuron/linear_adaptive")
        self.assertIsNotNone(package)
        assert package is not None
        overrides = {
            "recurrent_flag": True,
            "recurrent_stack_halting_flag": True,
            "recurrent_halting_option": SoftHaltingConfig,
        }

        configuration = package.build_configuration(config_overrides=overrides)
        experiment_config = configuration.experiment_config
        self.assertIsNotNone(experiment_config)
        assert experiment_config is not None
        cluster_config = experiment_config.neuron_cluster_config
        self.assertIsNotNone(cluster_config)
        assert cluster_config is not None
        neuron_config = cluster_config.neuron_config
        self.assertIsNotNone(neuron_config)
        assert neuron_config is not None
        nucleus_config = neuron_config.nucleus_config
        self.assertIsNotNone(nucleus_config)
        assert nucleus_config is not None
        hidden_block = nucleus_config.model_config
        self.assertIsNotNone(hidden_block)
        assert hidden_block is not None
        hidden_model = hidden_block.model_config
        self.assertIsNotNone(hidden_model)
        assert hidden_model is not None

        self.assertIsInstance(hidden_model.halting_config, SoftHaltingConfig)

    def test_transformer_seed_is_run_owned_not_a_model_builder_argument(self) -> None:
        package = model_package("transformer/linear")
        self.assertIsNotNone(package)
        assert package is not None

        configuration = package.build_configuration(
            config_overrides={"seed": 17},
        )

        self.assertGreater(configuration.input_dim, 0)
        self.assertGreater(configuration.output_dim, 0)

    def test_every_model_package_exposes_run_seed_without_changing_defaults(
        self,
    ) -> None:
        for package in discover_model_packages():
            with self.subTest(model_package=package.catalog_key):
                self.assertIn(
                    "SEED",
                    package.runtime_defaults_spec.supported_keys,
                )
                expected_default = (
                    0 if package.catalog_key.startswith("transformer/") else None
                )
                self.assertEqual(
                    package.runtime_defaults_spec.current_value("SEED"),
                    expected_default,
                )
                configuration = package.build_configuration(
                    config_overrides={"seed": 17},
                )
                self.assertGreater(configuration.input_dim, 0)


if __name__ == "__main__":
    unittest.main()
