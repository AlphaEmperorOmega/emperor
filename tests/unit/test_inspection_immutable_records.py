from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any, cast

from model_runtime.cli import (
    WireCodecError,
    configuration_schema_to_wire,
    inspection_result_to_wire,
    search_space_to_wire,
)
from model_runtime.inspection import (
    ConfigurationField,
    ConfigurationFieldCondition,
    ConfigurationSchema,
    GraphConfiguration,
    GraphConfigurationField,
    GraphEdge,
    GraphNode,
    InspectionRequest,
    InspectionResult,
    ModelGraph,
    ParsedOverrides,
    SearchAxis,
    SearchSpace,
)
from model_runtime.packages import ModelIdentity


def _identity() -> ModelIdentity:
    return ModelIdentity("linears", "linear")


def _graph_node(
    *,
    details: Mapping[str, Any] | None = None,
    configuration: GraphConfiguration | None = None,
) -> GraphNode:
    return GraphNode(
        id="__root__",
        type_name="Model",
        description="Root model.",
        path="model",
        graph_role="architecture",
        parameter_count=1,
        parameter_size_bytes=4,
        details={} if details is None else details,
        configuration=configuration,
    )


def _inspection_result(node: GraphNode) -> InspectionResult:
    return InspectionResult(
        identity=_identity(),
        preset="baseline",
        parameter_count=1,
        parameter_size_bytes=4,
        nodes=(node,),
        edges=(),
    )


def _nested_mapping(seed: int) -> dict[str, Any]:
    return {
        "nested": {
            "values": [seed],
            "tags": {seed},
        }
    }


def _mutate_nested_mapping(value: dict[str, Any]) -> None:
    nested = cast(dict[str, Any], value["nested"])
    cast(list[int], nested["values"]).append(99)
    cast(set[int], nested["tags"]).add(99)
    value["added"] = []


def _assert_fresh_json_containers(
    case: unittest.TestCase,
    first: object,
    second: object,
) -> None:
    if isinstance(first, dict):
        case.assertIsInstance(second, dict)
        case.assertIsNot(first, second)
        second_mapping = cast(dict[object, object], second)
        case.assertEqual(tuple(first), tuple(second_mapping))
        for key, value in first.items():
            _assert_fresh_json_containers(case, value, second_mapping[key])
        return
    if isinstance(first, list):
        case.assertIsInstance(second, list)
        case.assertIsNot(first, second)
        second_list = cast(list[object], second)
        case.assertEqual(len(first), len(second_list))
        for left, right in zip(first, second_list, strict=True):
            _assert_fresh_json_containers(case, left, right)
        return
    case.assertEqual(first, second)


class InspectionImmutableRecordTests(unittest.TestCase):
    def test_every_collection_membership_is_snapshotted_as_a_tuple(self) -> None:
        condition_values = [1]
        condition = ConfigurationFieldCondition(
            key="MODE",
            values=cast(Any, condition_values),
        )
        section_path = ["General"]
        choices = [1]
        conditions = [condition]
        configuration_field = ConfigurationField(
            key="COUNT",
            flag="--count",
            section_path=cast(Any, section_path),
            description="Count.",
            value_type="integer",
            default=1,
            nullable=False,
            choices=cast(Any, choices),
            applicable_when=cast(Any, conditions),
        )
        schema_fields = [configuration_field]
        schema = ConfigurationSchema(
            identity=_identity(),
            fields=cast(Any, schema_fields),
        )

        search_values = [1]
        locked_by_presets = ["baseline"]
        lock_reasons = ["fixed"]
        search_axis = SearchAxis(
            key="COUNT",
            search_key="SEARCH_SPACE_COUNT",
            section="General",
            value_type="integer",
            values=cast(Any, search_values),
            locked_by_presets=cast(Any, locked_by_presets),
            lock_reasons=cast(Any, lock_reasons),
        )
        search_axes = [search_axis]
        search_space = SearchSpace(
            identity=_identity(),
            preset="baseline",
            axes=cast(Any, search_axes),
        )

        graph_field = GraphConfigurationField("count", 1)
        graph_fields = [graph_field]
        graph_configuration = GraphConfiguration(
            "Config",
            cast(Any, graph_fields),
        )
        node = _graph_node(configuration=graph_configuration)
        edge = GraphEdge("loop-free", "__root__", "__root__")
        graph_nodes = [node]
        graph_edges = [edge]
        graph = ModelGraph(
            nodes=cast(Any, graph_nodes),
            edges=cast(Any, graph_edges),
        )
        result_nodes = [node]
        result_edges = [edge]
        result = InspectionResult(
            identity=_identity(),
            preset="baseline",
            parameter_count=1,
            parameter_size_bytes=4,
            nodes=cast(Any, result_nodes),
            edges=cast(Any, result_edges),
        )

        condition_values.append(2)
        section_path.append("Mutated")
        choices.append(2)
        conditions.clear()
        schema_fields.clear()
        search_values.append(2)
        locked_by_presets.append("other")
        lock_reasons.append("other")
        search_axes.clear()
        graph_fields.clear()
        graph_nodes.clear()
        graph_edges.clear()
        result_nodes.clear()
        result_edges.clear()

        self.assertEqual(condition.values, (1,))
        self.assertEqual(configuration_field.section_path, ("General",))
        self.assertEqual(configuration_field.choices, (1,))
        self.assertEqual(configuration_field.applicable_when, (condition,))
        self.assertEqual(schema.fields, (configuration_field,))
        self.assertEqual(search_axis.values, (1,))
        self.assertEqual(search_axis.locked_by_presets, ("baseline",))
        self.assertEqual(search_axis.lock_reasons, ("fixed",))
        self.assertEqual(search_space.axes, (search_axis,))
        self.assertEqual(graph_configuration.fields, (graph_field,))
        self.assertEqual(graph.nodes, (node,))
        self.assertEqual(graph.edges, (edge,))
        self.assertEqual(result.nodes, (node,))
        self.assertEqual(result.edges, (edge,))

    def test_every_arbitrary_value_owner_recursively_freezes_sources(self) -> None:
        sources = [_nested_mapping(index) for index in range(1, 5)]
        request = InspectionRequest(preset="baseline", overrides=sources[0])
        parsed = ParsedOverrides(sources[1])
        node = _graph_node(details=sources[2])
        configuration_field = GraphConfigurationField("nested", sources[3])
        stored_values = (
            request.overrides,
            parsed.values,
            node.details,
            configuration_field.value,
        )

        for source in sources:
            _mutate_nested_mapping(source)

        for index, stored in enumerate(stored_values, start=1):
            with self.subTest(owner=index):
                mapping = cast(Mapping[str, Any], stored)
                nested = cast(Mapping[str, Any], mapping["nested"])
                self.assertEqual(nested["values"], (index,))
                self.assertEqual(nested["tags"], frozenset({index}))
                self.assertNotIn("added", mapping)
                with self.assertRaises(TypeError):
                    mapping["forbidden"] = True  # type: ignore[index]

        colliding: dict[object, Any] = {
            1: ["first"],
            "1": ["second"],
            "tail": {3},
        }
        colliding_field = GraphConfigurationField("colliding", colliding)
        cast(list[str], colliding["1"]).append("mutated")
        cast(set[int], colliding["tail"]).add(4)

        frozen = cast(Mapping[str, Any], colliding_field.value)
        self.assertEqual(tuple(frozen), ("1", "tail"))
        self.assertEqual(frozen["1"], ("second",))
        self.assertEqual(frozen["tail"], frozenset({3}))

    def test_wire_projections_are_recursively_fresh(self) -> None:
        condition = ConfigurationFieldCondition("MODE", ("enabled",))
        configuration_field = ConfigurationField(
            key="COUNT",
            flag="--count",
            section_path=("General",),
            description="Count.",
            value_type="integer",
            default=1,
            nullable=False,
            choices=(1, 2),
            applicable_when=(condition,),
        )
        schema = ConfigurationSchema(_identity(), (configuration_field,))
        first_schema = configuration_schema_to_wire(schema)
        second_schema = configuration_schema_to_wire(schema)
        _assert_fresh_json_containers(self, first_schema, second_schema)
        first_schema["fields"][0]["section_path"].append("mutated")
        first_schema["fields"][0]["applicableWhen"][0]["values"].append("mutated")
        self.assertEqual(configuration_schema_to_wire(schema), second_schema)

        axis = SearchAxis(
            key="COUNT",
            search_key="SEARCH_SPACE_COUNT",
            section="General",
            value_type="integer",
            values=(1, 2),
            locked_by_presets=("baseline",),
            lock_reasons=("fixed",),
        )
        search_space = SearchSpace(_identity(), "baseline", (axis,))
        first_search = search_space_to_wire(search_space)
        second_search = search_space_to_wire(search_space)
        _assert_fresh_json_containers(self, first_search, second_search)
        first_search["axes"][0]["values"].append(3)
        first_search["axes"][0]["locked_by_presets"].append("other")
        self.assertEqual(search_space_to_wire(search_space), second_search)

        graph_configuration = GraphConfiguration(
            "Config",
            (
                GraphConfigurationField(
                    "nested",
                    {"values": [1]},
                ),
            ),
        )
        result = _inspection_result(
            _graph_node(
                details={"nested": {"values": [2]}},
                configuration=graph_configuration,
            )
        )
        first_result = inspection_result_to_wire(result)
        second_result = inspection_result_to_wire(result)
        _assert_fresh_json_containers(self, first_result, second_result)
        first_result["nodes"][0]["details"]["nested"]["values"].append(3)
        first_result["nodes"][0]["configuration"]["fields"][0]["value"][
            "values"
        ].append(3)
        first_result["nodes"].clear()
        self.assertEqual(inspection_result_to_wire(result), second_result)

    def test_non_json_sets_remain_rejected_at_the_wire_boundary(self) -> None:
        set_in_details = _inspection_result(_graph_node(details={"bad": {1}}))
        set_in_configuration = _inspection_result(
            _graph_node(
                configuration=GraphConfiguration(
                    "Config",
                    (GraphConfigurationField("bad", {1}),),
                )
            )
        )

        for result in (set_in_details, set_in_configuration):
            with (
                self.subTest(location=result.nodes[0].configuration is not None),
                self.assertRaisesRegex(WireCodecError, "unsupported value"),
            ):
                inspection_result_to_wire(result)


if __name__ == "__main__":
    unittest.main()
