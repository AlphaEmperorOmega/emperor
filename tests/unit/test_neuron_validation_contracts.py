import copy
import re
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from emperor.halting import HaltingBase, HaltingConfig
from emperor.neuron import AxonsConfig, NeuronClusterConfig, NucleusConfig
from unit.test_memory import make_memory_config
from unit.test_neuron import NeuronTestCase


@dataclass
class InvalidOwnerHaltingConfig(HaltingConfig):
    def _registry_owner(self):
        return object()


class CustomHaltingValidator:
    @staticmethod
    def validate_config(cfg) -> None:
        return None

    @staticmethod
    def validate(model) -> None:
        if model.derived_width != model.cfg.input_dim:
            raise ValueError("derived_width must match input_dim")


class CustomHalting(HaltingBase):
    VALIDATOR = CustomHaltingValidator

    def __init__(self, cfg, overrides=None) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.derived_width = self.cfg.input_dim
        self.VALIDATOR.validate(self)

    def update_halting_state(self, previous_state, model_hidden_state):
        state = SimpleNamespace(
            halt_mask=torch.zeros(
                model_hidden_state.shape[0],
                dtype=torch.bool,
                device=model_hidden_state.device,
            )
        )
        return state, model_hidden_state

    def finalize_weighted_accumulation(self, state, current_hidden):
        return current_hidden, current_hidden.new_zeros(current_hidden.shape[:-1])


class NestedConfigMutatingHalting(CustomHalting):
    @classmethod
    def validate_resolved_config(cls, cfg) -> None:
        cfg.halting_gate_config.layer_config.dropout_probability = 0.75


@dataclass
class CustomHaltingConfig(HaltingConfig):
    def _registry_owner(self):
        return CustomHalting


@dataclass
class NestedConfigMutatingHaltingConfig(CustomHaltingConfig):
    def _registry_owner(self):
        return NestedConfigMutatingHalting


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

    def test_cluster_requires_hard_route_limit_before_rng(self) -> None:
        config = self.cluster_config(max_steps=None)
        torch.manual_seed(20260719)
        rng_before = torch.random.get_rng_state().clone()

        with self.assertRaisesRegex(
            ValueError,
            "max_steps is required for NeuronClusterConfig, received None",
        ):
            config.build()

        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

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

    def test_terminal_sampler_and_runtime_dimension_errors_are_precise(self) -> None:
        outer_count_mismatch = self.terminal_config()
        outer_count_mismatch.sampler_config.num_experts = 17
        invalid_router_type = self.terminal_config()
        invalid_router_type.sampler_config.router_config = object()
        router_count_mismatch = self.terminal_config()
        router_count_mismatch.sampler_config.router_config.num_experts = 17

        invalid_builds = (
            (
                outer_count_mismatch,
                ValueError,
                "sampler_config.num_experts must equal Terminal "
                "total_neuron_connections, received num_experts=17 and "
                "total_neuron_connections=18.",
            ),
            (
                invalid_router_type,
                TypeError,
                "sampler_config.router_config must be a RouterConfig for Terminal, "
                "got object.",
            ),
            (
                router_count_mismatch,
                ValueError,
                "sampler_config.router_config.num_experts must equal Terminal "
                "total_neuron_connections, received num_experts=17 and "
                "total_neuron_connections=18.",
            ),
        )
        for config, error_type, message in invalid_builds:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, re.escape(message)):
                    config.build()

        terminal = self.terminal_config().build()
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Terminal input feature dimension must match input_dim, received "
                "input_dim=4 and input shape (3, 5)."
            ),
        ):
            terminal(torch.zeros(3, 5))

    def test_bool_cluster_width_and_non_tensor_runtime_inputs_are_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            TypeError,
            re.escape("beam_width must be an integer, received bool."),
        ):
            self.cluster_config(beam_width=True).build()

        nucleus = self.neuron_config().nucleus_config.build()
        with self.assertRaisesRegex(
            TypeError,
            re.escape("Nucleus input must be a Tensor, received list."),
        ):
            nucleus([1.0, 2.0])

        cluster = self.cluster_config().build()
        with self.assertRaisesRegex(
            TypeError,
            re.escape("NeuronCluster input must be a Tensor, received list."),
        ):
            cluster([])

    def test_axons_and_neuron_rank_errors_name_the_public_owner_exactly(self) -> None:
        invalid_input = torch.zeros(2, 1, self.input_dim)
        cases = (
            (
                self.neuron_config().axons_config.build(),
                "Axons input must be a 2D tensor, received a 3D tensor "
                "with shape (2, 1, 4).",
            ),
            (
                self.neuron_config().build(),
                "Neuron input must be a 2D tensor, received a 3D tensor "
                "with shape (2, 1, 4).",
            ),
        )

        for model, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                with self.assertRaises(ValueError) as raised:
                    model(invalid_input)
                self.assertEqual(str(raised.exception), expected_message)

    def test_routerless_terminal_dimension_error_is_exact_and_precedes_rng(
        self,
    ) -> None:
        config = self.terminal_config()
        config.sampler_config.router_config = None
        expected_message = (
            "sampler_config.router_config is required when Terminal input_dim "
            "does not equal total_neuron_connections, received input_dim=4 and "
            "total_neuron_connections=18."
        )
        torch.manual_seed(20260719)
        rng_before = torch.random.get_rng_state().clone()

        with self.assertRaises(ValueError) as raised:
            config.build()

        self.assertEqual(str(raised.exception), expected_message)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

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

    def test_derived_entry_sampler_clamps_top_k_without_mutating_terminal(self) -> None:
        terminal_sampler = self.sampler_config(
            input_dim=self.input_dim,
            num_experts=self.terminal_total_connections(),
            top_k=2,
        )
        terminal_sampler.normalize_probabilities_flag = True
        config = self.cluster_config()
        config.neuron_config.terminal_config.sampler_config = terminal_sampler

        model = config.build()

        self.assertEqual(terminal_sampler.top_k, 2)
        self.assertTrue(terminal_sampler.normalize_probabilities_flag)
        self.assertEqual(model.entry_sampler_config.top_k, 1)
        self.assertEqual(model.entry_sampler_config.num_topk_samples, 0)
        self.assertFalse(model.entry_sampler_config.normalize_probabilities_flag)
        self.assertEqual(model.entry_sampler_config.num_experts, 1)
        self.assertEqual(model.entry_sampler_config.router_config.num_experts, 1)

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

    def test_growth_flag_and_abstract_halting_config_fail_before_rng(self) -> None:
        cases = (
            (
                self.cluster_config(
                    growth_threshold=1,
                    escape_driven_growth_flag=1,
                ),
                TypeError,
                "escape_driven_growth_flag must be a bool for "
                "NeuronClusterConfig, got int.",
            ),
            (
                self.cluster_config(
                    halting_config=HaltingConfig(input_dim=self.input_dim)
                ),
                ValueError,
                "halting_config must be a concrete halting config for "
                "NeuronClusterConfig",
            ),
        )
        for config, error_type, message in cases:
            with self.subTest(message=message):
                torch.manual_seed(20260718)
                rng_before = torch.random.get_rng_state().clone()
                with self.assertRaisesRegex(error_type, re.escape(message)):
                    config.build()
                torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_invalid_concrete_halting_config_fails_before_rng(self) -> None:
        halting_config = self.halting_config(input_dim=self.input_dim)
        halting_config.threshold = 2.0
        config = self.cluster_config(halting_config=halting_config)
        torch.manual_seed(20260718)
        rng_before = torch.random.get_rng_state().clone()

        with self.assertRaisesRegex(
            ValueError,
            "threshold must be finite and between 0.0.*received 2.0",
        ):
            config.build()

        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_deferred_stick_breaking_defaults_are_resolved_on_a_copy(self) -> None:
        halting_config = self.halting_config(input_dim=self.input_dim)
        halting_config.input_dim = None
        halting_config.threshold = None

        model = self.cluster_config(halting_config=halting_config).build()

        self.assertIsNone(halting_config.input_dim)
        self.assertIsNone(halting_config.threshold)
        self.assertEqual(model.halting_model.input_dim, self.input_dim)
        self.assertEqual(
            model.halting_model.threshold,
            halting_config.DEFAULT_THRESHOLD,
        )

    def test_halting_preflight_cannot_mutate_nested_caller_configuration(self) -> None:
        source = self.halting_config(input_dim=self.input_dim)
        halting_config = NestedConfigMutatingHaltingConfig(
            input_dim=source.input_dim,
            threshold=source.threshold,
            dropout_probability=source.dropout_probability,
            hidden_state_mode=source.hidden_state_mode,
            halting_gate_config=source.halting_gate_config,
        )

        model = self.cluster_config(halting_config=halting_config).build()

        self.assertEqual(
            halting_config.halting_gate_config.layer_config.dropout_probability,
            0.0,
        )
        self.assertEqual(
            model.halting_model.cfg.halting_gate_config.layer_config.dropout_probability,
            0.0,
        )

    def test_custom_halting_runtime_validator_receives_real_initialized_model(
        self,
    ) -> None:
        config = self.cluster_config(
            halting_config=CustomHaltingConfig(input_dim=self.input_dim)
        )

        model = config.build()

        self.assertIsInstance(model.halting_model, CustomHalting)
        self.assertEqual(model.halting_model.derived_width, self.input_dim)

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

    def test_new_neuron_context_skips_integer_parameters_when_resolving_dtype(
        self,
    ) -> None:
        cluster = self.cluster_config().build()
        integer_marker = torch.nn.Parameter(
            torch.tensor(7, dtype=torch.int64),
            requires_grad=False,
        )
        cluster.register_parameter("integer_marker", integer_marker)
        cluster.double()

        new_neuron = cluster._initialize_neuron(1, 1, 1)

        self.assertEqual(cluster.integer_marker.dtype, torch.int64)
        self.assertEqual(
            {
                parameter.dtype
                for parameter in new_neuron.parameters()
                if parameter.is_floating_point()
            },
            {torch.float64},
        )
        self.assertEqual(
            {parameter.device for parameter in new_neuron.parameters()},
            {integer_marker.device},
        )

    def test_invalid_halting_registry_owner_has_stable_configuration_error(
        self,
    ) -> None:
        config = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            halting_config=InvalidOwnerHaltingConfig(input_dim=self.input_dim),
            neuron_config=self.neuron_config(),
        )
        torch.manual_seed(20260718)
        rng_before = torch.random.get_rng_state().clone()

        with self.assertRaisesRegex(
            ValueError,
            "does not implement the HaltingBase lifecycle",
        ):
            config.build()

        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)


if __name__ == "__main__":
    import unittest

    unittest.main()
