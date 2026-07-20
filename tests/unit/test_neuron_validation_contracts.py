import copy
import re
import unittest

import torch

from emperor.neuron import AxonsConfig, NeuronClusterConfig, NucleusConfig
from unit.test_memory import make_memory_config
from unit.test_neuron import NeuronTestCase


class TestNeuronCompositionValidation(NeuronTestCase):
    def cluster_config(self, **overrides) -> NeuronClusterConfig:
        values = {
            "x_axis_total_neurons": 1,
            "y_axis_total_neurons": 1,
            "z_axis_total_neurons": 1,
            "max_steps": 1,
            "growth_threshold": None,
            "neuron_config": self.neuron_config(),
        }
        values.update(overrides)
        return NeuronClusterConfig(**values)

    def test_standalone_neuron_rejects_nucleus_dimension_mismatch_before_rng(
        self,
    ) -> None:
        config = self.neuron_config()
        config.nucleus_config.model_config.output_dim = self.input_dim - 1
        torch.manual_seed(20260718)
        rng_before = torch.random.get_rng_state().clone()

        with self.assertRaisesRegex(
            ValueError,
            "nucleus_config.model_config must preserve the terminal feature dimension",
        ):
            config.build()

        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_cluster_accepts_deferred_terminal_positions_without_mutating_config(
        self,
    ) -> None:
        config = self.cluster_config(
            x_axis_total_neurons=2,
            y_axis_total_neurons=2,
            z_axis_total_neurons=2,
        )
        terminal_config = config.neuron_config.terminal_config
        terminal_config.x_axis_position = None
        terminal_config.y_axis_position = None
        terminal_config.z_axis_position = None

        cluster = config.build()

        self.assertIsNone(terminal_config.x_axis_position)
        self.assertIsNone(terminal_config.y_axis_position)
        self.assertIsNone(terminal_config.z_axis_position)
        for name, neuron in cluster.cluster.items():
            _, x_value, y_value, z_value = name.split("_")
            self.assertEqual(neuron.terminal.x_axis_position, int(x_value))
            self.assertEqual(neuron.terminal.y_axis_position, int(y_value))
            self.assertEqual(neuron.terminal.z_axis_position, int(z_value))

    def test_cluster_validation_preserves_partial_deferred_terminal_positions(
        self,
    ) -> None:
        config = self.cluster_config(
            x_axis_total_neurons=2,
            y_axis_total_neurons=2,
            z_axis_total_neurons=2,
        )
        terminal_config = config.neuron_config.terminal_config
        terminal_config.x_axis_position = 1
        terminal_config.y_axis_position = None
        terminal_config.z_axis_position = None

        cluster = config.build()

        self.assertEqual(terminal_config.x_axis_position, 1)
        self.assertIsNone(terminal_config.y_axis_position)
        self.assertIsNone(terminal_config.z_axis_position)
        for name, neuron in cluster.cluster.items():
            _, x_value, y_value, z_value = name.split("_")
            self.assertEqual(neuron.terminal.x_axis_position, int(x_value))
            self.assertEqual(neuron.terminal.y_axis_position, int(y_value))
            self.assertEqual(neuron.terminal.z_axis_position, int(z_value))

    def test_standalone_neuron_rejects_axons_memory_dimension_mismatch_before_rng(
        self,
    ) -> None:
        config = self.neuron_config()
        config.axons_config = AxonsConfig(
            memory_config=make_memory_config(input_dim=3, output_dim=3)
        )
        torch.manual_seed(20260718)
        rng_before = torch.random.get_rng_state().clone()

        with self.assertRaisesRegex(
            ValueError,
            "axons_config.memory_config.input_dim must preserve the terminal feature",
        ):
            config.build()

        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_axons_preflight_preserves_type_error_and_rejects_missing_dimension(
        self,
    ) -> None:
        invalid_type = self.neuron_config()
        invalid_type.axons_config.memory_config = object()
        missing_dimension = self.neuron_config()
        missing_memory_config = make_memory_config(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        missing_memory_config.input_dim = None
        missing_dimension.axons_config.memory_config = missing_memory_config

        cases = (
            (
                invalid_type,
                TypeError,
                "memory_config must be an instance of DynamicMemoryConfig",
            ),
            (
                missing_dimension,
                ValueError,
                "axons_config.memory_config.input_dim is required",
            ),
        )
        for config, error_type, message in cases:
            with self.subTest(message=message):
                torch.manual_seed(20260718)
                rng_before = torch.random.get_rng_state().clone()
                with self.assertRaisesRegex(error_type, message):
                    config.build()
                torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_nested_terminal_count_mismatch_fails_before_random_nucleus(self) -> None:
        config = self.neuron_config()
        config.nucleus_config = NucleusConfig(
            model_config=self.router_config(self.input_dim, self.input_dim).model_config
        )
        config.terminal_config.sampler_config.num_experts -= 1
        torch.manual_seed(20260718)
        rng_before = torch.random.get_rng_state().clone()

        with self.assertRaisesRegex(
            ValueError,
            "sampler_config.num_experts must equal Terminal total_neuron_connections",
        ):
            config.build()

        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_nested_sampler_contracts_fail_before_random_nucleus(self) -> None:
        cases = (
            ("top_k", 0, "top_k must be a positive integer"),
            ("threshold", -0.5, "threshold must be between 0.0 and 1.0"),
            (
                "num_topk_samples",
                99,
                "num_topk_samples cannot exceed top_k",
            ),
        )
        for field_name, invalid_value, message in cases:
            with self.subTest(field_name=field_name):
                config = self.neuron_config()
                config.nucleus_config = NucleusConfig(
                    model_config=self.router_config(
                        self.input_dim,
                        self.input_dim,
                    ).model_config
                )
                setattr(
                    config.terminal_config.sampler_config,
                    field_name,
                    invalid_value,
                )
                torch.manual_seed(20260718)
                rng_before = torch.random.get_rng_state().clone()
                with self.assertRaisesRegex(ValueError, message):
                    config.build()
                torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_schema_validation_owns_child_config_type_errors(self) -> None:
        invalid_nucleus = self.neuron_config()
        invalid_nucleus.nucleus_config.model_config = object()
        invalid_terminal = self.terminal_config()
        invalid_terminal.sampler_config = object()

        cases = (
            (
                invalid_nucleus.build,
                "model_config must be ConfigBase for NucleusConfig, got object",
            ),
            (
                invalid_terminal.build,
                "sampler_config must be SamplerConfig for TerminalConfig, got object",
            ),
        )
        for build, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, re.escape(message)):
                    build()

    def test_disabled_embedding_accepts_dimensionless_identity_nucleus(self) -> None:
        config = self.neuron_config(coordinate_embedding_flag=False)
        config.nucleus_config.model_config = AxonsConfig(memory_config=None)
        neuron = config.build()
        input_tensor = torch.randn(2, self.input_dim)

        output = neuron.process_signal(input_tensor)

        self.assertIsNone(neuron.coordinate_embedding)
        self.assertIs(output, input_tensor)

    def test_explicit_entry_sampler_validation_and_copy_contract(self) -> None:
        explicit_sampler = self.sampler_config(
            input_dim=self.input_dim,
            num_experts=3,
            top_k=2,
        )
        explicit_sampler.normalize_probabilities_flag = True
        original_sampler = copy.deepcopy(explicit_sampler)

        model = self.cluster_config(
            x_axis_total_neurons=3,
            entry_sampler_config=explicit_sampler,
        ).build()

        self.assertIsNot(model.entry_sampler_config, explicit_sampler)
        self.assertEqual(model.entry_sampler_config, original_sampler)
        self.assertEqual(explicit_sampler, original_sampler)
        self.assertIsNot(
            model.entry_sampler_config.router_config,
            explicit_sampler.router_config,
        )
        self.assertIsNot(
            model.entry_sampler_config.router_config.model_config,
            explicit_sampler.router_config.model_config,
        )
        self.assertTrue(model.entry_sampler.sampler_model.normalize_probabilities_flag)

    def test_explicit_logits_only_entry_sampler_runs_when_dimensions_match(
        self,
    ) -> None:
        entry_sampler = self.sampler_config(
            input_dim=self.input_dim,
            num_experts=self.input_dim,
            top_k=1,
            router_config=None,
        )
        model = self.cluster_config(
            x_axis_total_neurons=self.input_dim,
            entry_sampler_config=entry_sampler,
        ).build()

        output, auxiliary_loss = model(torch.randn(2, self.input_dim))

        self.assertEqual(output.shape, (2, self.input_dim))
        self.assertEqual(auxiliary_loss.shape, ())
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(auxiliary_loss))

    def test_invalid_explicit_entry_sampler_contracts_are_precise(self) -> None:
        top_k_too_large = self.sampler_config(num_experts=1, top_k=2)
        invalid_router_type = self.sampler_config(num_experts=1, top_k=1)
        invalid_router_type.router_config = object()
        router_count_mismatch = self.sampler_config(num_experts=1, top_k=1)
        router_count_mismatch.router_config.num_experts = 2
        cases = (
            (
                object(),
                TypeError,
                "entry_sampler_config must be a SamplerConfig for "
                "NeuronClusterConfig, got object.",
            ),
            (
                top_k_too_large,
                ValueError,
                "entry_sampler_config.top_k cannot exceed the initialized entry "
                "coordinate count, received top_k=2 and entry_coordinate_count=1.",
            ),
            (
                invalid_router_type,
                TypeError,
                "entry_sampler_config.router_config must be a RouterConfig for "
                "NeuronClusterConfig, got object.",
            ),
            (
                router_count_mismatch,
                ValueError,
                "entry_sampler_config.router_config.num_experts must equal the "
                "initialized entry coordinate count, received num_experts=2 and "
                "entry_coordinate_count=1.",
            ),
        )
        for entry_sampler_config, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, re.escape(message)):
                    self.cluster_config(
                        entry_sampler_config=entry_sampler_config
                    ).build()

    def test_invalid_explicit_entry_sampler_fails_before_rng(self) -> None:
        entry_sampler = self.sampler_config(
            input_dim=self.input_dim,
            num_experts=1,
            top_k=1,
        )
        entry_sampler.threshold = -0.5
        config = self.cluster_config(entry_sampler_config=entry_sampler)
        config.neuron_config.nucleus_config = NucleusConfig(
            model_config=self.router_config(self.input_dim, self.input_dim).model_config
        )
        torch.manual_seed(20260718)
        rng_before = torch.random.get_rng_state().clone()

        with self.assertRaisesRegex(
            ValueError,
            "threshold must be between 0.0 and 1.0",
        ):
            config.build()

        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)


if __name__ == "__main__":
    unittest.main()
