from __future__ import annotations

import inspect
import subprocess
import sys
import unittest
from typing import Any, cast
from unittest.mock import Mock

from model_runtime.packages import (
    DEFAULT_INSPECTION_CONSTRUCTION_LIMITS,
    InspectionConstructionLimits,
    ModelIdentity,
    ModelPackage,
)


class _CorePackageAdapter:
    @property
    def checkpoint_config_overrides(self) -> object:
        raise AssertionError("legacy reflective checkpoint hook was inspected")


class _HashableInspectionLimits(InspectionConstructionLimits):
    def __hash__(self) -> int:
        return 1


def _package(
    interpreter: object | None = None,
) -> ModelPackage:
    arguments: dict[str, Any] = {}
    if interpreter is not None:
        arguments["_checkpoint_config_interpreter"] = interpreter
    return ModelPackage(
        ModelIdentity("test", "checkpoint_capability"),
        cast(Any, _CorePackageAdapter()),
        **arguments,
    )


class ModelPackageCheckpointCapabilityTests(unittest.TestCase):
    def test_absent_explicit_capability_ignores_legacy_adapter_hook(self) -> None:
        package = _package()

        self.assertEqual(package.checkpoint_config_overrides({}), {})

    def test_capable_interpreter_receives_identity_input_and_returns_fresh_mapping(
        self,
    ) -> None:
        tensor_shapes = {"model.weight": (2, 3)}
        interpreted = {"hidden_dim": 3}
        interpreter = Mock(return_value=interpreted)
        package = _package(interpreter)

        result = package.checkpoint_config_overrides(tensor_shapes)

        interpreter.assert_called_once_with(tensor_shapes)
        self.assertIs(interpreter.call_args.args[0], tensor_shapes)
        self.assertEqual(result, interpreted)
        self.assertIsNot(result, interpreted)

    def test_interpreter_exceptions_propagate_unchanged(self) -> None:
        failure = LookupError("checkpoint layout is unknown")
        package = _package(Mock(side_effect=failure))

        with self.assertRaises(LookupError) as raised:
            package.checkpoint_config_overrides({"model.weight": (2, 3)})

        self.assertIs(raised.exception, failure)

    def test_invalid_explicit_capabilities_keep_exact_errors(self) -> None:
        cases = (
            (
                object(),
                "Model package 'test/checkpoint_capability' has an invalid "
                "checkpoint interpreter.",
            ),
            (
                lambda _shapes: [],
                "Model package 'test/checkpoint_capability' returned invalid "
                "checkpoint configuration overrides.",
            ),
            (
                lambda _shapes: {1: "invalid"},
                "Model package 'test/checkpoint_capability' returned invalid "
                "checkpoint configuration overrides.",
            ),
        )

        for interpreter, message in cases:
            with self.subTest(interpreter=interpreter):
                package = _package(interpreter)
                with self.assertRaises(ValueError) as raised:
                    package.checkpoint_config_overrides({})
                self.assertEqual(str(raised.exception), message)

    def test_constructor_repr_equality_and_hash_remain_compatible(self) -> None:
        signature = inspect.signature(ModelPackage)
        parameters = signature.parameters
        self.assertEqual(
            tuple(parameters),
            (
                "identity",
                "_adapter",
                "inspection_construction_limits",
                "_checkpoint_config_interpreter",
            ),
        )
        for name in ("identity", "_adapter", "inspection_construction_limits"):
            self.assertIs(
                parameters[name].kind,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        self.assertIs(
            parameters["_checkpoint_config_interpreter"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )

        identity = ModelIdentity("test", "constructor")
        hashable_limits = _HashableInspectionLimits()
        positional = ModelPackage(
            identity,
            cast(Any, _CorePackageAdapter()),
            hashable_limits,
        )
        keyword = ModelPackage(
            identity=identity,
            _adapter=cast(Any, _CorePackageAdapter()),
            inspection_construction_limits=hashable_limits,
            _checkpoint_config_interpreter=lambda _shapes: {"first": 1},
        )
        other_capability = ModelPackage(
            identity,
            cast(Any, _CorePackageAdapter()),
            hashable_limits,
            _checkpoint_config_interpreter=lambda _shapes: {"second": 2},
        )

        with self.assertRaises(TypeError):
            ModelPackage(  # type: ignore[call-arg]
                identity,
                cast(Any, _CorePackageAdapter()),
                DEFAULT_INSPECTION_CONSTRUCTION_LIMITS,
                lambda _shapes: {},
            )
        self.assertNotIn("checkpoint", repr(keyword))
        self.assertEqual(positional, keyword)
        self.assertEqual(keyword, other_capability)
        self.assertEqual(hash(positional), hash(keyword))
        self.assertEqual(hash(keyword), hash(other_capability))

    def test_capable_package_imports_checkpoint_metadata_only_on_use(self) -> None:
        module_name = "models.linears.linear.checkpoint_metadata"
        script = (
            "import sys\n"
            "from models.linears.linear import MODEL_PACKAGE\n"
            f"module_name = {module_name!r}\n"
            "if module_name in sys.modules:\n"
            "    raise SystemExit('checkpoint metadata loaded eagerly')\n"
            "result = MODEL_PACKAGE.checkpoint_config_overrides({\n"
            "    'input_model.model.weight_params': (784, 12),\n"
            "    'main_model.layers.0.model.weight_params': (12, 12),\n"
            "    'output_model.model.weight_params': (12, 10),\n"
            "})\n"
            "expected = {\n"
            "    'input_dim': 784,\n"
            "    'output_dim': 10,\n"
            "    'hidden_dim': 12,\n"
            "    'stack_num_layers': 1,\n"
            "}\n"
            "if result != expected:\n"
            "    raise SystemExit(f'unexpected overrides: {result!r}')\n"
            "if module_name not in sys.modules:\n"
            "    raise SystemExit('checkpoint metadata was not loaded on use')\n"
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
