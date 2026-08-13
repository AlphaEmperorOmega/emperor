from __future__ import annotations

import importlib
import inspect
import unittest
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from enum import Enum
from types import MappingProxyType, ModuleType, NoneType, UnionType
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Literal,
    NotRequired,
    Protocol,
    Required,
    TypedDict,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from unittest.mock import patch

import pytest
from typing_extensions import ReadOnly as ExtensionsReadOnly
from typing_extensions import TypedDict as ExtensionsTypedDict

from emperor.layers import AdditiveResidualConfig, ResidualConfig
from model_runtime.packages import RuntimeDefaultsError
from model_runtime.packages.runtime_values import (
    RuntimeValueConstraints,
    validate_runtime_default_value_types,
    validate_runtime_default_values,
)
from models.catalog import discover_model_packages, model_package

_UNSET = object()
_BARE_LIST_ALIAS = importlib.import_module("typing").List


class _RuntimeMode(Enum):
    SAFE = "safe"


class _RuntimeBase:
    pass


class _RuntimeChild(_RuntimeBase):
    pass


class _RuntimePayload(TypedDict):
    value: int


class _ExtensionsRuntimePayload(ExtensionsTypedDict):
    value: int


class _RuntimeProtocol(Protocol):
    def runtime_value(self) -> int: ...


def _runtime_config_module(
    annotation: object,
    *,
    constraints: object = _UNSET,
) -> ModuleType:
    module = ModuleType(f"fixture_runtime_values_{id(annotation)}_{id(object())}")
    module.VALUE = None
    module.__annotations__ = {"VALUE": annotation}
    if constraints is not _UNSET:
        module.RUNTIME_VALUE_CONSTRAINTS = constraints
    return module


def _validate_runtime_annotation(annotation: object, value: object) -> None:
    validate_runtime_default_value_types(
        {"value": value},
        package="fixture",
        config_module=_runtime_config_module(annotation),
    )


def _annotation_options(annotation: object) -> tuple[object, ...]:
    if get_origin(annotation) in {UnionType, Union}:
        return get_args(annotation)
    return (annotation,)


def _representative_key(
    annotations: dict[str, Any],
    supported_keys: tuple[str, ...],
    predicate,
) -> str:
    return next(
        key
        for key in supported_keys
        if predicate(_annotation_options(annotations[key]))
    )


def _check_parametric_runtime_rejects_wrong_declared_type_before_construction() -> None:
    package = model_package("parametric/parametric_vector")
    assert package is not None

    with pytest.raises(TypeError) as error:
        package.bind_runtime_defaults({"batch_size": "bad"})

    assert str(error.value) == (
        "models.parametric.parametric_vector: runtime key 'batch_size' has type "
        "str; expected int"
    )


def _check_runtime_boolean_fields_do_not_accept_integer_values() -> None:
    package = model_package("parametric/parametric_vector")
    assert package is not None

    with pytest.raises(TypeError) as error:
        package.bind_runtime_defaults({"stack_residual_model_flag": 1})

    assert str(error.value) == (
        "models.parametric.parametric_vector: runtime key "
        "'stack_residual_model_flag' has type int; expected bool"
    )


_EXTERNAL_TYPE_CASES = (
    ("parametric/parametric_vector", "learning_rate", "fast", "float"),
    (
        "parametric/parametric_vector",
        "stack_activation",
        "GELU",
        "ActivationOptions",
    ),
    (
        "parametric/parametric_vector",
        "stack_residual_connection_option",
        object(),
        "type[ResidualConfig] | None",
    ),
    ("neuron/linear", "cluster_x_axis_total_neurons", "many", "int"),
    (
        "transformer/linear",
        "positional_embedding_option",
        object(),
        "type[AbsolutePositionalEmbeddingConfig]",
    ),
    (
        "transformer/linear_adaptive",
        "stack_residual_connection_option",
        object(),
        "type[ResidualConfig] | None",
    ),
)


def _check_runtime_defaults_enforce_declared_external_types(
    catalog_key: str,
    runtime_key: str,
    invalid_value: object,
    expected_type: str,
) -> None:
    package = model_package(catalog_key)
    assert package is not None

    with pytest.raises(TypeError) as error:
        package.bind_runtime_defaults({runtime_key: invalid_value})

    package_name = "models." + catalog_key.replace("/", ".")
    assert str(error.value) == (
        f"{package_name}: runtime key {runtime_key!r} has type "
        f"{type(invalid_value).__name__}; expected {expected_type}"
    )


def _check_runtime_override_parser_rejects_wrong_exported_configuration_class() -> None:
    package = model_package("parametric/parametric_vector")
    assert package is not None

    with pytest.raises(RuntimeDefaultsError) as error:
        package.runtime_defaults_spec.parse_overrides(
            {"stack_residual_connection_option": "ActivationOptions"}
        )

    assert str(error.value) == (
        "models.parametric.parametric_vector: runtime key "
        "'stack_residual_connection_option' has type EnumType; expected "
        "type[ResidualConfig] | None"
    )


def _check_runtime_override_parser_rejects_none_for_non_optional_value() -> None:
    package = model_package("parametric/parametric_vector")
    assert package is not None

    with pytest.raises(RuntimeDefaultsError) as error:
        package.runtime_defaults_spec.parse_overrides({"learning_rate": "None"})

    assert str(error.value) == (
        "models.parametric.parametric_vector: runtime key 'learning_rate' has "
        "type NoneType; expected float"
    )


def _check_direct_neuron_runtime_resolution_uses_declared_type_contract() -> None:
    from models.neuron.linear.runtime_defaults import runtime_from_flat

    with pytest.raises(TypeError) as error:
        runtime_from_flat({"cluster_x_axis_total_neurons": "many"})

    assert str(error.value) == (
        "models.neuron.linear: runtime key 'cluster_x_axis_total_neurons' has "
        "type str; expected int"
    )


def _check_unknown_runtime_keys_retain_precedence_over_known_value_types() -> None:
    package = model_package("parametric/parametric_vector")
    assert package is not None

    with pytest.raises(ValueError) as error:
        package.bind_runtime_defaults({"unknown_field": 1, "batch_size": "bad"})

    assert str(error.value) == (
        "models.parametric.parametric_vector: unknown Runtime Defaults field(s): "
        "'unknown_field'"
    )


def _check_runtime_defaults_reject_abstract_configuration_classes() -> None:
    package = model_package("parametric/parametric_vector")
    assert package is not None

    with pytest.raises(ValueError) as error:
        package.bind_runtime_defaults(
            {"stack_residual_connection_option": ResidualConfig}
        )

    assert str(error.value) == (
        "models.parametric.parametric_vector: runtime key "
        "'stack_residual_connection_option' selects invalid config class "
        "ResidualConfig: ResidualConfig is abstract and has no registered residual "
        "connection; instantiate a concrete residual config instead."
    )


def _check_declared_numeric_union_accepts_typed_branch(value: int | float) -> None:
    package = model_package("parametric/parametric_vector")
    assert package is not None

    package.runtime_defaults_spec.validate_typed_values(
        {"trainer_overfit_batches": value}
    )


def _check_all_packages_reject_representative_declared_type_violations() -> None:
    failures: list[str] = []
    for package in discover_model_packages():
        spec = package.runtime_defaults_spec
        annotations = get_type_hints(spec._config_module)
        supported = tuple(key for key in spec.supported_keys if key in annotations)
        probes = (
            (
                _representative_key(
                    annotations,
                    supported,
                    lambda options: options == (int,),
                ),
                True,
            ),
            (
                _representative_key(
                    annotations,
                    supported,
                    lambda options: options == (bool,),
                ),
                1,
            ),
            (
                _representative_key(
                    annotations,
                    supported,
                    lambda options: options == (float,),
                ),
                "not-a-float",
            ),
            (
                _representative_key(
                    annotations,
                    supported,
                    lambda options: any(
                        inspect.isclass(option) and issubclass(option, Enum)
                        for option in options
                    ),
                ),
                "not-an-enum",
            ),
            (
                _representative_key(
                    annotations,
                    supported,
                    lambda options: any(
                        get_origin(option) is type for option in options
                    ),
                ),
                object(),
            ),
            (
                _representative_key(
                    annotations,
                    supported,
                    lambda options: (
                        NoneType in options
                        and any(option is not NoneType for option in options)
                    ),
                ),
                object(),
            ),
        )
        runtime_module = importlib.import_module(
            f"models.{package.catalog_key.replace('/', '.')}.runtime_defaults"
        )
        runtime_from_flat = runtime_module.runtime_from_flat
        for config_key, invalid_value in probes:
            runtime_key = config_key.lower()
            for interface, operation in (
                ("package", package.bind_runtime_defaults),
                ("direct", runtime_from_flat),
            ):
                try:
                    operation({runtime_key: invalid_value})
                except (TypeError, ValueError):
                    continue
                failures.append(
                    f"{package.catalog_key}:{interface}:{runtime_key}: accepted "
                    f"{invalid_value!r}"
                )

    assert not failures, "Declared type violations accepted:\n" + "\n".join(failures)


def _check_all_packages_reject_nonpositive_core_model_widths() -> None:
    failures: list[str] = []
    for package in discover_model_packages():
        config_key = next(
            key
            for key in ("HIDDEN_DIM", "MODEL_DIM")
            if key in package.runtime_defaults_spec.supported_keys
        )
        runtime_key = config_key.lower()
        constraints = getattr(
            package.runtime_defaults_spec._config_module,
            "RUNTIME_VALUE_CONSTRAINTS",
            None,
        )
        if not isinstance(constraints, RuntimeValueConstraints):
            failures.append(
                f"{package.catalog_key}: missing RuntimeValueConstraints declaration"
            )
        elif config_key not in constraints.positive:
            failures.append(
                f"{package.catalog_key}: {config_key} is not declared positive"
            )
        runtime_module = importlib.import_module(
            f"models.{package.catalog_key.replace('/', '.')}.runtime_defaults"
        )
        for interface, operation in (
            ("package", package.bind_runtime_defaults),
            ("direct", runtime_module.runtime_from_flat),
        ):
            try:
                operation({runtime_key: 0})
            except ValueError:
                continue
            failures.append(
                f"{package.catalog_key}:{interface}:{runtime_key}: accepted 0"
            )

    assert not failures, "Nonpositive core widths accepted:\n" + "\n".join(failures)


@pytest.mark.parametrize(
    ("annotation", "value"),
    (
        (list[int], [1, 2]),
        (list[list[int]], [[1], [2, 3]]),
        (tuple[int, str], (1, "two")),
        (tuple[int, ...], (1, 2, 3)),
        (tuple[()], ()),
        (dict[str, int], {"one": 1}),
        (Mapping[str, int], MappingProxyType({"one": 1})),
        (list[int | str | None], [1, "two", None]),
        (Annotated[list[int], "runtime-values"], [1, 2]),
    ),
)
def test_supported_recursive_runtime_annotations_accept_valid_values(
    annotation: object,
    value: object,
) -> None:
    _validate_runtime_annotation(annotation, value)


@pytest.mark.parametrize(
    ("annotation", "value", "message"),
    (
        (
            list[int],
            [1, "bad"],
            "fixture: runtime key 'value' at $[1] has type str; expected int",
        ),
        (
            list[list[int]],
            [[1], ["bad"]],
            "fixture: runtime key 'value' at $[1][0] has type str; expected int",
        ),
        (
            list[int],
            [True],
            "fixture: runtime key 'value' at $[0] has type bool; expected int",
        ),
        (
            list[bool],
            [1],
            "fixture: runtime key 'value' at $[0] has type int; expected bool",
        ),
        (
            list[int | str],
            [1, 2.0],
            "fixture: runtime key 'value' at $[1] has type float; expected int | str",
        ),
    ),
)
def test_recursive_runtime_annotations_report_exact_nested_paths(
    annotation: object,
    value: object,
    message: str,
) -> None:
    with pytest.raises(TypeError) as error:
        _validate_runtime_annotation(annotation, value)

    assert str(error.value) == message


@pytest.mark.parametrize(
    ("annotation", "value", "message"),
    (
        (
            tuple[int, str],
            (1,),
            "fixture: runtime key 'value' has tuple length 1; expected tuple[int, str]",
        ),
        (
            tuple[int, str],
            (1, "two", 3),
            "fixture: runtime key 'value' has tuple length 3; expected tuple[int, str]",
        ),
        (
            tuple[()],
            (1,),
            "fixture: runtime key 'value' has tuple length 1; expected tuple[()]",
        ),
        (
            tuple[int, ...],
            (1, "bad"),
            "fixture: runtime key 'value' at $[1] has type str; expected int",
        ),
    ),
)
def test_tuple_annotations_enforce_arity_and_member_types(
    annotation: object,
    value: object,
    message: str,
) -> None:
    with pytest.raises(TypeError) as error:
        _validate_runtime_annotation(annotation, value)

    assert str(error.value) == message


def test_mapping_annotations_validate_keys_before_values_in_iteration_order() -> None:
    with pytest.raises(TypeError) as key_error:
        _validate_runtime_annotation(dict[str, int], {"one": 1, 2: "bad"})

    assert str(key_error.value) == (
        "fixture: runtime key 'value' at $.keys[1] has type int; expected str"
    )

    with pytest.raises(TypeError) as value_error:
        _validate_runtime_annotation(
            Mapping[str, int],
            MappingProxyType({"one": 1, "two": "bad"}),
        )

    assert str(value_error.value) == (
        "fixture: runtime key 'value' at $.values[1] has type str; expected int"
    )


@pytest.mark.parametrize(
    ("annotation", "value", "expected"),
    (
        (dict[str, int], [("one", 1)], "dict[str, int]"),
        (Mapping[str, int], [("one", 1)], "Mapping[str, int]"),
        (tuple[int, ...], [1, 2], "tuple[int, ...]"),
        (list[int], (1, 2), "list[int]"),
    ),
)
def test_recursive_annotations_require_the_declared_container_origin(
    annotation: object,
    value: object,
    expected: str,
) -> None:
    with pytest.raises(TypeError) as error:
        _validate_runtime_annotation(annotation, value)

    assert str(error.value) == (
        f"fixture: runtime key 'value' has type {type(value).__name__}; "
        f"expected {expected}"
    )


@pytest.mark.parametrize(
    ("annotation", "value"),
    (
        (Any, object()),
        (object, object()),
        (NoneType, None),
        (_RuntimeMode, _RuntimeMode.SAFE),
        (_RuntimeBase, _RuntimeChild()),
        (type, _RuntimeChild),
        (type[Any], _RuntimeChild),
        (type[object], _RuntimeChild),
        (type[_RuntimeBase], _RuntimeChild),
        (type[Annotated[Any, "runtime-values"]], _RuntimeChild),
        (type[Annotated[object, "runtime-values"]], _RuntimeChild),
        (type[Annotated[_RuntimeBase, "runtime-values"]], _RuntimeChild),
    ),
)
def test_existing_wildcard_nominal_and_type_annotations_remain_supported(
    annotation: object,
    value: object,
) -> None:
    _validate_runtime_annotation(annotation, value)


def test_nested_type_annotation_rejects_abstract_configuration_class() -> None:
    for annotation in (
        list[type[ResidualConfig]],
        list[type[Annotated[ResidualConfig, "runtime-values"]]],
    ):
        with pytest.raises(ValueError) as error:
            _validate_runtime_annotation(annotation, [ResidualConfig])

        assert str(error.value) == (
            "fixture: runtime key 'value' at $[0] selects invalid config class "
            "ResidualConfig: ResidualConfig is abstract and has no registered residual "
            "connection; instantiate a concrete residual config instead."
        )


@pytest.mark.parametrize(
    "annotation",
    (
        list[Any | type[ResidualConfig]],
        list[type[ResidualConfig] | Any],
        list[object | type[ResidualConfig]],
        list[type[ResidualConfig] | object],
    ),
)
def test_nested_abstract_class_validation_is_union_order_independent(
    annotation: object,
) -> None:
    with pytest.raises(ValueError) as error:
        _validate_runtime_annotation(annotation, [ResidualConfig])

    assert str(error.value) == (
        "fixture: runtime key 'value' at $[0] selects invalid config class "
        "ResidualConfig: ResidualConfig is abstract and has no registered residual "
        "connection; instantiate a concrete residual config instead."
    )


def test_valid_root_configuration_class_is_semantically_probed_once() -> None:
    with patch(
        "model_runtime.packages.runtime_values.abstract_config_class_error",
        return_value=None,
    ) as semantic_probe:
        _validate_runtime_annotation(
            type[ResidualConfig],
            AdditiveResidualConfig,
        )

    semantic_probe.assert_called_once_with(AdditiveResidualConfig)


_RUNTIME_TYPE_VARIABLE = TypeVar("_RUNTIME_TYPE_VARIABLE")


@pytest.mark.parametrize(
    ("annotation", "value", "label"),
    (
        (Literal["safe"], "safe", "Literal[safe]"),
        (Sequence[int], [1], "Sequence[int]"),
        (Sequence[int], "123", "Sequence[int]"),
        (set[int], set(), "set[int]"),
        (frozenset[int], frozenset(), "frozenset[int]"),
        (Callable[[int], str], lambda value: str(value), "Callable"),
        (_RUNTIME_TYPE_VARIABLE, 1, "_RUNTIME_TYPE_VARIABLE"),
        (_BARE_LIST_ALIAS, [], "list"),
        (type[int | str], int, "type[int | str]"),
        (Required[int], 1, "Required[int]"),
        (NotRequired[int], 1, "NotRequired[int]"),
        (ExtensionsReadOnly[int], 1, "ReadOnly[int]"),
    ),
)
def test_unsupported_runtime_annotations_fail_before_value_matching(
    annotation: object,
    value: object,
    label: str,
) -> None:
    with pytest.raises(TypeError) as error:
        _validate_runtime_annotation(annotation, value)

    assert str(error.value).startswith(
        "fixture: runtime key 'value' declares unsupported annotation "
    )
    assert label in str(error.value)


@pytest.mark.parametrize(
    ("annotation", "value", "label"),
    (
        (list[Callable[[int], str]], [], "Callable"),
        (dict[str, Literal["safe"]], {}, "Literal[safe]"),
        (int | Literal["safe"], 1, "Literal[safe]"),
        (list[Required[int]], [], "Required[int]"),
        (int | NotRequired[int], 1, "NotRequired[int]"),
        (dict[str, ExtensionsReadOnly[int]], {}, "ReadOnly[int]"),
    ),
)
def test_unsupported_nested_annotations_cannot_hide_behind_runtime_values(
    annotation: object,
    value: object,
    label: str,
) -> None:
    with pytest.raises(TypeError) as error:
        _validate_runtime_annotation(annotation, value)

    assert str(error.value).startswith(
        "fixture: runtime key 'value' declares unsupported annotation "
    )
    assert label in str(error.value)


@pytest.mark.parametrize(
    ("annotation", "value", "label"),
    (
        (_RuntimePayload, {"value": 1}, "_RuntimePayload"),
        (list[_RuntimePayload], [], "_RuntimePayload"),
        (_RuntimeProtocol, object(), "_RuntimeProtocol"),
        (list[_RuntimeProtocol], [], "_RuntimeProtocol"),
        (type[_RuntimePayload], _RuntimePayload, "type[_RuntimePayload]"),
        (type[_RuntimeProtocol], _RuntimeProtocol, "type[_RuntimeProtocol]"),
        (Literal[[1]], [1], "Literal[[1]]"),
    ),
)
def test_class_shaped_and_unhashable_typing_forms_are_rejected_deterministically(
    annotation: object,
    value: object,
    label: str,
) -> None:
    with pytest.raises(TypeError) as error:
        _validate_runtime_annotation(annotation, value)

    assert str(error.value) == (
        "fixture: runtime key 'value' declares unsupported annotation " + label
    )


@pytest.mark.parametrize(
    ("annotation", "value", "label"),
    (
        (_ExtensionsRuntimePayload, {"value": 1}, "_ExtensionsRuntimePayload"),
        (list[_ExtensionsRuntimePayload], [], "_ExtensionsRuntimePayload"),
        (
            type[_ExtensionsRuntimePayload],
            _ExtensionsRuntimePayload,
            "type[_ExtensionsRuntimePayload]",
        ),
    ),
)
def test_extensions_typed_dict_is_rejected_before_value_matching(
    annotation: object,
    value: object,
    label: str,
) -> None:
    with pytest.raises(TypeError) as error:
        _validate_runtime_annotation(annotation, value)

    assert str(error.value) == (
        "fixture: runtime key 'value' declares unsupported annotation " + label
    )


def test_finite_annotations_terminate_for_cyclic_values() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)

    with pytest.raises(TypeError) as error:
        _validate_runtime_annotation(list[list[int]], cyclic)

    assert str(error.value) == (
        "fixture: runtime key 'value' at $[0][0] has type list; expected int"
    )

    _validate_runtime_annotation(list[Any], cyclic)


def test_recursive_annotation_alias_is_rejected_before_value_traversal() -> None:
    recursive = list[ForwardRef("RecursiveRuntimeValue")]
    module = _runtime_config_module(recursive)
    module.__dict__["RecursiveRuntimeValue"] = recursive

    with pytest.raises(TypeError) as error:
        validate_runtime_default_value_types(
            {"value": []},
            package="fixture",
            config_module=module,
        )

    assert str(error.value) == (
        "fixture: runtime key 'value' declares unsupported annotation "
        "ForwardRef('RecursiveRuntimeValue')"
    )


def test_unknown_keys_bypass_annotation_resolution_and_constraints() -> None:
    sentinel = RuntimeError("annotation resolution should not run")
    module = _runtime_config_module(int, constraints=object())

    with patch(
        "model_runtime.packages.runtime_values.get_type_hints",
        side_effect=sentinel,
    ) as resolver:
        validate_runtime_default_value_types(
            {"unknown": 1, "value": "bad"},
            package="fixture",
            config_module=module,
        )

    resolver.assert_not_called()


def test_full_validator_preserves_unknown_field_precedence() -> None:
    sentinel = RuntimeError("annotation resolution should not run")
    module = _runtime_config_module(int, constraints=object())

    with patch(
        "model_runtime.packages.runtime_values.get_type_hints",
        side_effect=sentinel,
    ) as resolver:
        with pytest.raises(ValueError) as error:
            validate_runtime_default_values(
                {"unknown": 1, "value": "bad"},
                package="fixture",
                config_module=module,
            )

    assert str(error.value) == ("fixture: unknown Runtime Defaults field(s): 'unknown'")
    resolver.assert_not_called()


def test_known_keys_preserve_annotation_resolution_exception_identity() -> None:
    sentinel = RuntimeError("unresolved annotation")
    module = _runtime_config_module(int)

    with patch(
        "model_runtime.packages.runtime_values.get_type_hints",
        side_effect=sentinel,
    ):
        with pytest.raises(RuntimeError) as error:
            validate_runtime_default_value_types(
                {"value": 1},
                package="fixture",
                config_module=module,
            )

    assert error.value is sentinel


def test_empty_values_do_not_resolve_annotations() -> None:
    module = _runtime_config_module(int)
    with patch(
        "model_runtime.packages.runtime_values.get_type_hints",
        side_effect=AssertionError("annotation resolution should not run"),
    ) as resolver:
        validate_runtime_default_value_types(
            {},
            package="fixture",
            config_module=module,
        )

    resolver.assert_not_called()


def test_runtime_defaults_spec_preserves_nested_type_error_as_cause() -> None:
    package = model_package("parametric/parametric_vector")
    assert package is not None
    config_module = _runtime_config_module(list[int])
    spec = replace(
        package.runtime_defaults_spec,
        _config_module=config_module,
        supported_keys=("VALUE",),
        keys_by_alias=MappingProxyType({"value": "VALUE"}),
        annotations=MappingProxyType({"VALUE": list[int]}),
    )

    with pytest.raises(RuntimeDefaultsError) as error:
        spec.validate_typed_overrides({"value": ["bad"]})

    assert str(error.value) == (
        "models.parametric.parametric_vector: runtime key 'value' at $[0] has "
        "type str; expected int"
    )
    assert isinstance(error.value.__cause__, TypeError)


def test_all_catalog_defaults_remain_valid_under_the_closed_grammar() -> None:
    for package in discover_model_packages():
        package.runtime_defaults_spec.validate_typed_values(
            dict(package.runtime_defaults_spec.default_items())
        )


class RuntimeDefaultValueValidationTests(unittest.TestCase):
    def test_parametric_runtime_rejects_wrong_type_before_construction(self) -> None:
        _check_parametric_runtime_rejects_wrong_declared_type_before_construction()

    def test_runtime_boolean_fields_do_not_accept_integer_values(self) -> None:
        _check_runtime_boolean_fields_do_not_accept_integer_values()

    def test_runtime_defaults_enforce_declared_external_types(self) -> None:
        for case in _EXTERNAL_TYPE_CASES:
            with self.subTest(catalog_key=case[0], runtime_key=case[1]):
                _check_runtime_defaults_enforce_declared_external_types(*case)

    def test_override_parser_rejects_wrong_exported_config_class(self) -> None:
        _check_runtime_override_parser_rejects_wrong_exported_configuration_class()

    def test_override_parser_rejects_none_for_non_optional_value(self) -> None:
        _check_runtime_override_parser_rejects_none_for_non_optional_value()

    def test_direct_neuron_resolution_uses_declared_type_contract(self) -> None:
        _check_direct_neuron_runtime_resolution_uses_declared_type_contract()

    def test_unknown_keys_retain_precedence_over_known_value_types(self) -> None:
        _check_unknown_runtime_keys_retain_precedence_over_known_value_types()

    def test_runtime_defaults_reject_abstract_config_classes(self) -> None:
        _check_runtime_defaults_reject_abstract_configuration_classes()

    def test_declared_numeric_union_accepts_each_typed_branch(self) -> None:
        for value in (2, 2.5):
            with self.subTest(value=value):
                _check_declared_numeric_union_accepts_typed_branch(value)

    def test_all_packages_reject_representative_type_violations(self) -> None:
        _check_all_packages_reject_representative_declared_type_violations()

    def test_all_packages_reject_nonpositive_core_model_widths(self) -> None:
        _check_all_packages_reject_nonpositive_core_model_widths()

    def test_recursive_annotation_contract_is_in_authoritative_suite(self) -> None:
        valid_cases = (
            (list[int], [1, 2]),
            (list[list[int]], [[1], [2, 3]]),
            (tuple[int, str], (1, "two")),
            (tuple[int, ...], (1, 2, 3)),
            (tuple[()], ()),
            (dict[str, int], {"one": 1}),
            (Mapping[str, int], MappingProxyType({"one": 1})),
            (list[int | str | None], [1, "two", None]),
            (Annotated[list[int], "runtime-values"], [1, 2]),
        )
        for annotation, value in valid_cases:
            with self.subTest(annotation=annotation, value=value):
                test_supported_recursive_runtime_annotations_accept_valid_values(
                    annotation,
                    value,
                )

    def test_nested_mismatch_paths_are_in_authoritative_suite(self) -> None:
        cases = (
            (
                list[int],
                [1, "bad"],
                "fixture: runtime key 'value' at $[1] has type str; expected int",
            ),
            (
                list[list[int]],
                [[1], ["bad"]],
                "fixture: runtime key 'value' at $[1][0] has type str; expected int",
            ),
            (
                list[int],
                [True],
                "fixture: runtime key 'value' at $[0] has type bool; expected int",
            ),
            (
                list[bool],
                [1],
                "fixture: runtime key 'value' at $[0] has type int; expected bool",
            ),
            (
                list[int | str],
                [1, 2.0],
                "fixture: runtime key 'value' at $[1] has type float; expected "
                "int | str",
            ),
        )
        for case in cases:
            with self.subTest(annotation=case[0], value=case[1]):
                test_recursive_runtime_annotations_report_exact_nested_paths(*case)

    def test_tuple_contract_is_in_authoritative_suite(self) -> None:
        cases = (
            (
                tuple[int, str],
                (1,),
                "fixture: runtime key 'value' has tuple length 1; expected "
                "tuple[int, str]",
            ),
            (
                tuple[int, str],
                (1, "two", 3),
                "fixture: runtime key 'value' has tuple length 3; expected "
                "tuple[int, str]",
            ),
            (
                tuple[()],
                (1,),
                "fixture: runtime key 'value' has tuple length 1; expected tuple[()]",
            ),
            (
                tuple[int, ...],
                (1, "bad"),
                "fixture: runtime key 'value' at $[1] has type str; expected int",
            ),
        )
        for case in cases:
            with self.subTest(annotation=case[0], value=case[1]):
                test_tuple_annotations_enforce_arity_and_member_types(*case)

    def test_mapping_and_container_contract_is_in_authoritative_suite(self) -> None:
        test_mapping_annotations_validate_keys_before_values_in_iteration_order()
        cases = (
            (dict[str, int], [("one", 1)], "dict[str, int]"),
            (Mapping[str, int], [("one", 1)], "Mapping[str, int]"),
            (tuple[int, ...], [1, 2], "tuple[int, ...]"),
            (list[int], (1, 2), "list[int]"),
        )
        for case in cases:
            with self.subTest(annotation=case[0], value=case[1]):
                test_recursive_annotations_require_the_declared_container_origin(*case)

    def test_nominal_and_type_contract_is_in_authoritative_suite(self) -> None:
        cases = (
            (Any, object()),
            (object, object()),
            (NoneType, None),
            (_RuntimeMode, _RuntimeMode.SAFE),
            (_RuntimeBase, _RuntimeChild()),
            (type, _RuntimeChild),
            (type[Any], _RuntimeChild),
            (type[object], _RuntimeChild),
            (type[_RuntimeBase], _RuntimeChild),
            (type[Annotated[Any, "runtime-values"]], _RuntimeChild),
            (type[Annotated[object, "runtime-values"]], _RuntimeChild),
            (type[Annotated[_RuntimeBase, "runtime-values"]], _RuntimeChild),
        )
        for case in cases:
            with self.subTest(annotation=case[0], value=case[1]):
                test_existing_wildcard_nominal_and_type_annotations_remain_supported(
                    *case
                )
        test_nested_type_annotation_rejects_abstract_configuration_class()
        for annotation in (
            list[Any | type[ResidualConfig]],
            list[type[ResidualConfig] | Any],
            list[object | type[ResidualConfig]],
            list[type[ResidualConfig] | object],
        ):
            with self.subTest(annotation=annotation):
                test_nested_abstract_class_validation_is_union_order_independent(
                    annotation
                )
        test_valid_root_configuration_class_is_semantically_probed_once()

    def test_unsupported_annotation_contract_is_in_authoritative_suite(self) -> None:
        cases = (
            (Literal["safe"], "safe", "Literal[safe]"),
            (Sequence[int], [1], "Sequence[int]"),
            (Sequence[int], "123", "Sequence[int]"),
            (set[int], set(), "set[int]"),
            (frozenset[int], frozenset(), "frozenset[int]"),
            (Callable[[int], str], str, "Callable"),
            (_RUNTIME_TYPE_VARIABLE, 1, "_RUNTIME_TYPE_VARIABLE"),
            (_BARE_LIST_ALIAS, [], "list"),
            (type[int | str], int, "type[int | str]"),
            (Required[int], 1, "Required[int]"),
            (NotRequired[int], 1, "NotRequired[int]"),
            (ExtensionsReadOnly[int], 1, "ReadOnly[int]"),
        )
        for case in cases:
            with self.subTest(annotation=case[0], value=case[1]):
                test_unsupported_runtime_annotations_fail_before_value_matching(*case)

    def test_hidden_unsupported_annotations_are_in_authoritative_suite(self) -> None:
        cases = (
            (list[Callable[[int], str]], [], "Callable"),
            (dict[str, Literal["safe"]], {}, "Literal[safe]"),
            (int | Literal["safe"], 1, "Literal[safe]"),
            (list[Required[int]], [], "Required[int]"),
            (int | NotRequired[int], 1, "NotRequired[int]"),
            (dict[str, ExtensionsReadOnly[int]], {}, "ReadOnly[int]"),
        )
        for case in cases:
            with self.subTest(annotation=case[0], value=case[1]):
                test_unsupported_nested_annotations_cannot_hide_behind_runtime_values(
                    *case
                )

    def test_class_shaped_typing_forms_are_in_authoritative_suite(self) -> None:
        cases = (
            (_RuntimePayload, {"value": 1}, "_RuntimePayload"),
            (list[_RuntimePayload], [], "_RuntimePayload"),
            (_RuntimeProtocol, object(), "_RuntimeProtocol"),
            (list[_RuntimeProtocol], [], "_RuntimeProtocol"),
            (type[_RuntimePayload], _RuntimePayload, "type[_RuntimePayload]"),
            (type[_RuntimeProtocol], _RuntimeProtocol, "type[_RuntimeProtocol]"),
            (Literal[[1]], [1], "Literal[[1]]"),
        )
        for case in cases:
            with self.subTest(annotation=case[0], value=case[1]):
                test_class_shaped_and_unhashable_typing_forms_are_rejected_deterministically(
                    *case
                )
        extension_cases = (
            (_ExtensionsRuntimePayload, {"value": 1}, "_ExtensionsRuntimePayload"),
            (list[_ExtensionsRuntimePayload], [], "_ExtensionsRuntimePayload"),
            (
                type[_ExtensionsRuntimePayload],
                _ExtensionsRuntimePayload,
                "type[_ExtensionsRuntimePayload]",
            ),
        )
        for case in extension_cases:
            with self.subTest(annotation=case[0], value=case[1]):
                test_extensions_typed_dict_is_rejected_before_value_matching(*case)

    def test_cycle_and_alias_contract_is_in_authoritative_suite(self) -> None:
        test_finite_annotations_terminate_for_cyclic_values()
        test_recursive_annotation_alias_is_rejected_before_value_traversal()

    def test_precedence_and_laziness_contract_is_in_authoritative_suite(self) -> None:
        test_unknown_keys_bypass_annotation_resolution_and_constraints()
        test_full_validator_preserves_unknown_field_precedence()
        test_known_keys_preserve_annotation_resolution_exception_identity()
        test_empty_values_do_not_resolve_annotations()

    def test_boundary_and_catalog_contract_is_in_authoritative_suite(self) -> None:
        test_runtime_defaults_spec_preserves_nested_type_error_as_cause()
        test_all_catalog_defaults_remain_valid_under_the_closed_grammar()


if __name__ == "__main__":
    unittest.main()
