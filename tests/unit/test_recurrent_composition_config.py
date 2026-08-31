import ast
import importlib
import inspect
import textwrap
import unittest
from dataclasses import fields
from pathlib import Path

from torch import nn

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig
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
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentIterationSchedule,
)
from emperor.layers._composition.recurrent.validation import (
    HierarchicalReasoningModelRecurrentValidator,
    RecurrentExecutionValidator,
    RecurrentIterationScheduleValidator,
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
    def test_usage_monitoring_does_not_enter_base_or_variant_contracts(self) -> None:
        emperor_root = Path(__file__).parents[2] / "src" / "emperor"
        recurrent_root = emperor_root / "layers" / "_composition" / "recurrent"
        source_paths = (
            emperor_root / "halting" / "_base.py",
            *(emperor_root / "halting" / "_variants").glob("*.py"),
            recurrent_root / "base.py",
            *(recurrent_root / "variants").glob("*.py"),
        )

        for source_path in source_paths:
            source = source_path.read_text(encoding="utf-8")
            for monitoring_term in (
                "HaltingUsageTracker",
                "_usage_tracker",
                "suppress_usage_tracking",
            ):
                with self.subTest(
                    source_path=source_path,
                    monitoring_term=monitoring_term,
                ):
                    self.assertNotIn(monitoring_term, source)

    def test_recurrent_execution_interface_is_separate_from_implementation(
        self,
    ) -> None:
        recurrent_root = (
            Path(__file__).parents[2]
            / "src"
            / "emperor"
            / "layers"
            / "_composition"
            / "recurrent"
        )
        execution_package_path = recurrent_root / "runtime" / "execution"
        package_interface_path = execution_package_path / "__init__.py"
        execution_path = execution_package_path / "executor.py"
        interface_path = execution_package_path / "interface.py"
        runtime_state_path = execution_package_path / "runtime_state.py"
        execution_validation_path = recurrent_root / "validation" / "execution.py"

        self.assertTrue(package_interface_path.is_file())
        self.assertTrue(execution_path.is_file())
        self.assertTrue(interface_path.is_file())
        self.assertTrue(runtime_state_path.is_file())
        self.assertTrue(execution_validation_path.is_file())
        self.assertFalse((recurrent_root / "runtime" / "execution.py").exists())
        self.assertFalse(
            (recurrent_root / "runtime" / "execution_interface.py").exists()
        )
        execution_source = execution_path.read_text(encoding="utf-8")
        interface_source = interface_path.read_text(encoding="utf-8")
        runtime_state_source = runtime_state_path.read_text(encoding="utf-8")
        base_source = (recurrent_root / "base.py").read_text(encoding="utf-8")

        self.assertNotIn("@dataclass", execution_source)
        self.assertNotIn("Protocol", execution_source)

        for declaration in (
            "class PreparedRecurrentTransition",
            "class RecurrentExecutionAdapter",
            "class RecurrentExecutionResult",
            "class RecurrentExecutionState",
            "class RecurrentTransitionResult",
            "class _RecurrentExecutionOwner",
        ):
            with self.subTest(declaration=declaration):
                self.assertNotIn(declaration, execution_source)

        for declaration in (
            "class PreparedRecurrentTransition",
            "class RecurrentExecutionAdapter",
            "class RecurrentExecutionResult",
            "class RecurrentExecutionState",
        ):
            with self.subTest(interface_declaration=declaration):
                self.assertIn(declaration, interface_source)

        self.assertNotIn("class _RecurrentExecutionOwner", interface_source)
        self.assertNotIn(
            "_rollback_recurrent_runtime_state_on_failure", interface_source
        )
        self.assertNotIn("class RecurrentTransitionResult", interface_source)
        self.assertIn("class RecurrentTransitionResult", base_source)
        self.assertNotIn("class _RecurrentTransitionResult", base_source)
        self.assertIn("class RecurrentRuntimeStateGuard", runtime_state_source)
        for runtime_state_declaration in (
            "class _BufferSnapshot",
            "class _NamedBufferSnapshot",
            "class _ModuleBufferSnapshot",
            "class _RecurrentRuntimeStateSnapshot",
        ):
            with self.subTest(runtime_state_declaration=runtime_state_declaration):
                self.assertIn(runtime_state_declaration, runtime_state_source)
                self.assertNotIn(runtime_state_declaration, base_source)
        self.assertIn("isolate_provisional_branch", base_source)
        self.assertIn("rollback_handoff_on_failure", execution_source)
        self.assertIn(
            "self.VALIDATOR.validate_adapter_is_module(adapter)",
            execution_source,
        )
        self.assertNotIn("__require_recurrent_module", execution_source)
        self.assertNotIn("isinstance(adapter, nn.Module)", execution_source)

        execution_package_module_name = (
            "emperor.layers._composition.recurrent.runtime.execution"
        )
        execution_package = importlib.import_module(execution_package_module_name)
        expected_execution_symbol_modules = {
            "RecurrentExecution": (f"{execution_package_module_name}.executor"),
            "PreparedRecurrentTransition": (
                f"{execution_package_module_name}.interface"
            ),
            "RecurrentExecutionAdapter": (f"{execution_package_module_name}.interface"),
            "RecurrentExecutionResult": (f"{execution_package_module_name}.interface"),
            "RecurrentExecutionState": (f"{execution_package_module_name}.interface"),
        }
        self.assertEqual(
            set(execution_package.__all__),
            set(expected_execution_symbol_modules),
        )
        for (
            symbol_name,
            expected_module_name,
        ) in expected_execution_symbol_modules.items():
            with self.subTest(symbol_name=symbol_name):
                symbol = getattr(execution_package, symbol_name)
                self.assertEqual(symbol.__module__, expected_module_name)

        recurrent_base_module_name = "emperor.layers._composition.recurrent.base"
        recurrent_base_module = importlib.import_module(recurrent_base_module_name)
        self.assertEqual(
            recurrent_base_module.RecurrentTransitionResult.__module__,
            recurrent_base_module_name,
        )

        base_initializer_tree = ast.parse(
            textwrap.dedent(inspect.getsource(RecurrentCompositionAbstract.__init__))
        )
        base_initializer_calls = [
            node
            for node in ast.walk(base_initializer_tree)
            if isinstance(node, ast.Call)
        ]
        call_lines = {
            ast.unparse(call.func): call.lineno for call in base_initializer_calls
        }
        self.assertLess(
            call_lines["self.VALIDATOR.validate"],
            call_lines["self.__initialize_from_config"],
        )
        self.assertLess(
            call_lines["self.__initialize_from_config"],
            call_lines["self.__initialize_delegates"],
        )
        delegate_initializer = vars(RecurrentCompositionAbstract)[
            "_RecurrentCompositionAbstract__initialize_delegates"
        ]
        delegate_initializer_tree = ast.parse(
            textwrap.dedent(inspect.getsource(delegate_initializer))
        )
        self.assertTrue(
            any(
                isinstance(call.func, ast.Name)
                and call.func.id == "RecurrentIterationSchedule"
                for call in ast.walk(delegate_initializer_tree)
                if isinstance(call, ast.Call)
            )
        )

    def test_variant_classes_are_adapters_for_shared_recurrent_execution(
        self,
    ) -> None:
        recurrent_variants = (
            RecurrentLayer,
            TinyRecursiveModelRecurrent,
            HierarchicalReasoningModelRecurrent,
        )
        adapter_method_names = {
            "_apply_recurrent_transition_result",
            "_detach_recurrent_execution_state",
            "_fork_recurrent_handoff_state",
            "_initialize_recurrent_execution_state",
            "_prepare_recurrent_transition",
            "_recurrent_branch_loss",
        }

        recurrent_root = (
            Path(__file__).parents[2]
            / "src"
            / "emperor"
            / "layers"
            / "_composition"
            / "recurrent"
        )
        variant_specific_execution_files = [
            path.name
            for owner_directory in (
                recurrent_root / "runtime",
                recurrent_root / "variants",
            )
            for path in owner_directory.rglob("*_execution.py")
        ]
        self.assertEqual(variant_specific_execution_files, [])

        execution_source = (
            recurrent_root / "runtime" / "execution" / "executor.py"
        ).read_text(encoding="utf-8")
        interface_source = (
            recurrent_root / "runtime" / "execution" / "interface.py"
        ).read_text(encoding="utf-8")
        shared_execution_source = "\n".join((execution_source, interface_source))
        for variant_term in (
            "StandardRecurrent",
            "TinyRecursiveModel",
            "HierarchicalReasoningModel",
            "reinject_original_hidden_flag",
            "latent_updates_per_answer_update",
            "low_cycles",
        ):
            with self.subTest(variant_term=variant_term):
                self.assertNotIn(variant_term, shared_execution_source)

        for recurrent_variant in recurrent_variants:
            with self.subTest(recurrent_variant=recurrent_variant.__name__):
                recurrent_variant_source = inspect.getsource(recurrent_variant)
                initializer_source = inspect.getsource(recurrent_variant.__init__)
                self.assertTrue(adapter_method_names <= vars(recurrent_variant).keys())
                self.assertNotIn("TYPE_CHECKING", initializer_source)
                self.assertNotIn("assert ", initializer_source)
                self.assertNotIn(
                    "EXECUTION_ADAPTER",
                    vars(recurrent_variant),
                )
                normalized_source = " ".join(recurrent_variant_source.split())
                self.assertIn(
                    "return self.__recurrent_execution.execute( self, state, "
                    "self.recurrent_iteration_schedule, )",
                    normalized_source,
                )
                self.assertLess(
                    normalized_source.index("self.VALIDATOR.validate_state"),
                    normalized_source.index("self.__recurrent_execution.execute"),
                )
                for delegated_lifecycle_statement in (
                    "execution_plan()",
                    "record_successful_forward()",
                    "state.hidden =",
                    "state.loss =",
                ):
                    with self.subTest(
                        delegated_lifecycle_statement=delegated_lifecycle_statement,
                    ):
                        self.assertNotIn(
                            delegated_lifecycle_statement,
                            recurrent_variant_source,
                        )

        for shared_lifecycle_statement in (
            "iteration_schedule.execution_plan()",
            "layer_state.hidden = execution_result.hidden",
            "layer_state.loss = execution_result.loss",
            "iteration_schedule.record_successful_forward()",
            "return layer_state",
        ):
            with self.subTest(
                shared_lifecycle_statement=shared_lifecycle_statement,
            ):
                self.assertIn(shared_lifecycle_statement, execution_source)

    def test_halting_floor_policy_does_not_live_in_the_recurrent_package(self) -> None:
        recurrent_package = (
            Path(__file__).parents[2]
            / "src"
            / "emperor"
            / "layers"
            / "_composition"
            / "recurrent"
        )
        prohibited_policy_names = (
            "min_steps",
            "is_update_eligible",
            "owner_step",
        )
        offenders = {}
        for path in sorted(recurrent_package.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            leaked_policy_names = tuple(
                policy_name
                for policy_name in prohibited_policy_names
                if policy_name in source
            )
            if leaked_policy_names:
                offenders[str(path.relative_to(recurrent_package))] = (
                    leaked_policy_names
                )

        self.assertEqual(offenders, {})

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

    def test_all_recurrent_configs_own_the_shared_iteration_schedule(self) -> None:
        for config_type in (
            RecurrentLayerConfig,
            TinyRecursiveModelRecurrentConfig,
            HierarchicalReasoningModelRecurrentConfig,
        ):
            with self.subTest(config_type=config_type.__name__):
                config = config_type(
                    initial_iterations=2,
                    gradient_transition_count=3,
                    iteration_increment=4,
                    forward_calls_before_iteration_increment=5,
                    smooth_iteration_growth_flag=True,
                )

                self.assertEqual(config.initial_iterations, 2)
                self.assertEqual(config.gradient_transition_count, 3)
                self.assertEqual(config.iteration_increment, 4)
                self.assertEqual(config.forward_calls_before_iteration_increment, 5)
                self.assertTrue(config.smooth_iteration_growth_flag)

        self.assertIn(
            "smooth_iteration_growth_flag",
            RecurrentCompositionConfig.__annotations__,
        )
        self.assertNotIn(
            "smooth_iteration_growth_flag",
            RecurrentLayerConfig.__annotations__,
        )

        self.assertIs(RecurrentLayerConfig().registry_owner(), RecurrentLayer)

    def test_recurrent_config_module_contains_only_owner_backed_configs(self) -> None:
        config_module = importlib.import_module(
            "emperor.layers._composition.recurrent.config"
        )
        config_types = {
            value
            for value in vars(config_module).values()
            if isinstance(value, type)
            and issubclass(value, ConfigBase)
            and value.__module__ == config_module.__name__
        }

        self.assertEqual(
            config_types,
            {
                RecurrentCompositionConfig,
                RecurrentLayerConfig,
                TinyRecursiveModelRecurrentConfig,
                HierarchicalReasoningModelRecurrentConfig,
            },
        )
        for config_type in config_types - {RecurrentCompositionConfig}:
            with self.subTest(config_type=config_type.__name__):
                self.assertIsInstance(config_type().registry_owner(), type)

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
            (
                RecurrentIterationScheduleValidator,
                "emperor.layers._composition.recurrent.validation.iteration_schedule",
            ),
            (
                RecurrentExecutionValidator,
                "emperor.layers._composition.recurrent.validation.execution",
            ),
        )

        for validator, expected_module in cases:
            with self.subTest(validator=validator.__name__):
                self.assertEqual(validator.__module__, expected_module)

    def test_recurrent_execution_validator_only_checks_module_ownership(
        self,
    ) -> None:
        self.assertIsNone(
            RecurrentExecutionValidator.validate_adapter_is_module(nn.Identity())
        )
        with self.assertRaisesRegex(
            TypeError,
            "Recurrent Execution Adapter must be an nn.Module",
        ):
            RecurrentExecutionValidator.validate_adapter_is_module(object())

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

    def test_minimum_steps_belongs_to_the_halting_contract(self) -> None:
        recurrent_field_names = {field.name for field in fields(RecurrentLayerConfig)}
        halting_field_names = [field.name for field in fields(HaltingConfig)]

        self.assertNotIn("min_steps", recurrent_field_names)
        self.assertGreater(
            halting_field_names.index("min_steps"),
            halting_field_names.index("halting_gate_config"),
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
                "_blend_recurrent_branch_losses",
                "_new_recurrent_initial_buffer",
                "_observe_recurrent_step",
                "_recurrent_row_layout_for_transitions",
                "_run_recurrent_transition",
                "_run_shared_handoff_boundary_transition",
                "_set_recurrent_diagnostic_observer",
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
            RecurrentIterationScheduleValidator: set(),
            RecurrentExecutionValidator: set(),
            RecurrentIterationSchedule: set(),
            RecurrentLayer: {
                "_apply_recurrent_transition_result",
                "_detach_recurrent_execution_state",
                "_fork_recurrent_handoff_state",
                "_initialize_recurrent_execution_state",
                "_prepare_recurrent_transition",
                "_recurrent_branch_loss",
            },
            TinyRecursiveModelRecurrent: {
                "_apply_recurrent_transition_result",
                "_detach_recurrent_execution_state",
                "_fork_recurrent_handoff_state",
                "_initialize_recurrent_execution_state",
                "_prepare_recurrent_transition",
                "_recurrent_branch_loss",
            },
            HierarchicalReasoningModelRecurrent: {
                "_apply_recurrent_transition_result",
                "_detach_recurrent_execution_state",
                "_fork_recurrent_handoff_state",
                "_initialize_recurrent_execution_state",
                "_prepare_recurrent_transition",
                "_recurrent_branch_loss",
            },
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
