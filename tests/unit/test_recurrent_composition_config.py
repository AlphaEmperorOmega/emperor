import importlib
import unittest
from dataclasses import fields

from emperor.layers import (
    HierarchicalReasoningModelRecurrentConfig,
    RecurrentCompositionConfig,
    RecurrentLayer,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
)
from emperor.layers._composition.recurrent.base import (
    RecurrentCompositionAbstract,
)
from emperor.layers._composition.recurrent.validation import (
    HierarchicalReasoningModelRecurrentValidator,
    RecurrentLayerValidator,
    TinyRecursiveModelRecurrentValidator,
)
from emperor.layers._composition.recurrent.variants.hierarchical_reasoning_model import (
    HierarchicalReasoningModelRecurrent,
)
from emperor.layers._composition.recurrent.variants.tiny_recursive_model import (
    TinyRecursiveModelRecurrent,
)


def _declared_protected_method_names(owner: type) -> set[str]:
    private_method_prefix = f"_{owner.__name__.lstrip('_')}__"
    return {
        name
        for name, member in vars(owner).items()
        if name.startswith("_")
        and not name.startswith("__")
        and not name.startswith(private_method_prefix)
        and (callable(member) or isinstance(member, (classmethod, staticmethod)))
    }


class TestRecurrentCompositionConfig(unittest.TestCase):
    def test_abstract_config_cannot_be_built(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "RecurrentCompositionConfig is abstract.*concrete recurrent config",
        ):
            RecurrentCompositionConfig().build()

    def test_standard_config_owns_the_canonical_standard_runtime(self) -> None:
        owner = RecurrentLayerConfig().registry_owner()

        self.assertIs(owner, RecurrentLayer)
        self.assertTrue(issubclass(owner, RecurrentCompositionAbstract))
        self.assertEqual(
            owner.__module__,
            "emperor.layers._composition.recurrent.variants.standard",
        )

    def test_recurrent_validators_live_in_dedicated_validation_modules(self) -> None:
        cases = (
            (
                RecurrentLayerValidator,
                "emperor.layers._composition.recurrent.validation.standard",
            ),
            (
                TinyRecursiveModelRecurrentValidator,
                "emperor.layers._composition.recurrent.validation.tiny_recursive_model",
            ),
            (
                HierarchicalReasoningModelRecurrentValidator,
                "emperor.layers._composition.recurrent.validation."
                "hierarchical_reasoning_model",
            ),
        )

        for validator, expected_module in cases:
            with self.subTest(validator=validator.__name__):
                self.assertEqual(validator.__module__, expected_module)

    def test_recurrent_controller_fields_belong_to_the_family_config(self) -> None:
        controller_field_names = {
            "recurrent_layer_norm_position",
            "gate_config",
            "residual_config",
            "halting_config",
            "memory_config",
        }

        self.assertTrue(
            controller_field_names
            <= {field.name for field in fields(RecurrentCompositionConfig)}
        )
        self.assertTrue(
            controller_field_names.isdisjoint(RecurrentLayerConfig.__annotations__)
        )

    def test_fixed_input_reinjection_control_belongs_only_to_standard_config(
        self,
    ) -> None:
        field_name = "reinject_original_hidden_flag"

        self.assertIn(
            field_name, {field.name for field in fields(RecurrentLayerConfig)}
        )
        for config_type in (
            RecurrentCompositionConfig,
            TinyRecursiveModelRecurrentConfig,
            HierarchicalReasoningModelRecurrentConfig,
        ):
            with self.subTest(config_type=config_type.__name__):
                self.assertNotIn(
                    field_name,
                    {field.name for field in fields(config_type)},
                )

    def test_transition_seam_ignores_unconfigured_resources(self) -> None:
        config = RecurrentLayerConfig(block_config=None)

        config._map_transition_configs(
            lambda _transition: self.fail("None transition must not be mapped")
        )

        self.assertEqual(config._transition_config_items(), ())
        self.assertEqual(config._transition_configs(), ())
        self.assertEqual(config._missing_transition_config_fields(), ("block_config",))

    def test_only_subclass_extension_seams_are_protected(self) -> None:
        expected_protected_methods = {
            RecurrentCompositionAbstract: {
                "_accumulate_recurrent_losses",
                "_build_recurrent_residual_schedule",
                "_build_transition_model",
                "_expand_recurrent_initial",
                "_finalize_recurrent_halting",
                "_initialize_transition_gradient_window",
                "_new_recurrent_initial_buffer",
                "_observe_recurrent_step",
                "_recurrent_row_layout_for_transitions",
                "_run_recurrent_transition",
                "_set_recurrent_diagnostic_observer",
                "_starts_gradient_suffix",
                "_transition_gradient_context",
            },
            RecurrentCompositionConfig: {
                "_map_transition_configs",
                "_missing_transition_config_fields",
                "_registry_owner",
                "_transition_config_items",
                "_transition_configs",
            },
            RecurrentLayerConfig: {"_registry_owner"},
            TinyRecursiveModelRecurrentConfig: {"_registry_owner"},
            HierarchicalReasoningModelRecurrentConfig: {"_registry_owner"},
            RecurrentLayerValidator: {"_validate_integer_field"},
            TinyRecursiveModelRecurrentValidator: set(),
            HierarchicalReasoningModelRecurrentValidator: set(),
            RecurrentLayer: set(),
            TinyRecursiveModelRecurrent: set(),
            HierarchicalReasoningModelRecurrent: set(),
        }

        for owner, expected_method_names in expected_protected_methods.items():
            with self.subTest(owner=owner.__name__):
                self.assertEqual(
                    _declared_protected_method_names(owner),
                    expected_method_names,
                )

    def test_old_private_runtime_and_validator_modules_are_retired(self) -> None:
        for module_name in (
            "emperor.layers._recurrent",
            "emperor.layers._validation.recurrent",
            "emperor.layers._composition.recurrent.variants.hrm",
            "emperor.layers._composition.recurrent.variants.trm",
        ):
            with (
                self.subTest(module_name=module_name),
                self.assertRaises(ModuleNotFoundError),
            ):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
