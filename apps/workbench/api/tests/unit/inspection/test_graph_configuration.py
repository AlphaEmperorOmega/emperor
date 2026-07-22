from __future__ import annotations

import os
import unittest

from emperor.layers import ResidualConfig

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import models.experts.linear.config as experts_linear_config
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
)
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    RecurrentLayerConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from torch import nn

from tests.support.inspection import (
    inspect_model,
    serialize_graph,
)
from tests.unit.inspection._graph_support import (
    ConfiguredModule,
    DocumentedModule,
    PlainConfig,
    config_fields,
    nodes_by_id,
)


class InspectionGraphConfigurationTests(unittest.TestCase):
    def test_graph_serializer_reports_direct_weight_and_bias_shapes(self) -> None:
        nodes, _edges = serialize_graph(nn.Sequential(nn.Linear(4, 3), nn.ReLU()))
        details_by_id = {node["id"]: node["details"] for node in nodes}

        self.assertNotIn("weightShape", details_by_id["__root__"])
        self.assertNotIn("biasShape", details_by_id["__root__"])
        self.assertEqual(details_by_id["0"]["weightShape"], "3 x 4")
        self.assertEqual(details_by_id["0"]["biasShape"], "3")
        self.assertNotIn("weightShape", details_by_id["1"])
        self.assertNotIn("biasShape", details_by_id["1"])

    def test_graph_serializer_includes_linear_config_fields(self) -> None:
        model = LinearLayerConfig(
            input_dim=4,
            output_dim=3,
            bias_flag=False,
        ).build()
        nodes, _edges = serialize_graph(nn.Sequential(model))
        linear_node = {node["id"]: node for node in nodes}["0"]

        self.assertEqual(linear_node["config"]["typeName"], "LinearLayerConfig")
        self.assertEqual(
            linear_node["description"],
            (
                "Applies a learned linear projection with configured input/output "
                "dimensions and optional bias."
            ),
        )
        self.assertEqual(
            config_fields(linear_node),
            {"input_dim": 4, "output_dim": 3, "bias_flag": False},
        )
        field_by_key = {
            field["key"]: field for field in linear_node["config"]["fields"]
        }
        self.assertEqual(
            field_by_key["input_dim"]["description"],
            "Input feature dimension.",
        )
        self.assertEqual(
            field_by_key["output_dim"]["description"],
            "Output feature dimension.",
        )
        self.assertEqual(
            field_by_key["bias_flag"]["description"],
            "Add a learnable bias to the output.",
        )
        self.assertIn("weightShape", linear_node["details"])
        self.assertNotIn("biasShape", linear_node["details"])

    def test_graph_serializer_uses_docstring_description_fallback(self) -> None:
        nodes, _edges = serialize_graph(DocumentedModule())

        self.assertEqual(nodes[0]["description"], "Documented component description.")

    def test_graph_serializer_omits_unknown_descriptions_quietly(self) -> None:
        nodes, _edges = serialize_graph(ConfiguredModule(PlainConfig(width=7)))
        root = nodes[0]

        self.assertNotIn("description", root)
        self.assertEqual(
            root["config"],
            {
                "typeName": "PlainConfig",
                "fields": [{"key": "width", "value": 7}],
            },
        )

    def test_graph_serializer_serializes_adaptive_config_references(self) -> None:
        no_augmentation = ConfiguredModule(
            AdaptiveLinearLayerConfig(
                input_dim=4,
                output_dim=3,
                bias_flag=True,
                adaptive_augmentation_config=None,
            )
        )
        with_augmentation = ConfiguredModule(
            AdaptiveLinearLayerConfig(
                input_dim=4,
                output_dim=3,
                bias_flag=True,
                adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                    input_dim=4,
                    output_dim=3,
                ),
            )
        )

        nodes, _edges = serialize_graph(
            nn.Sequential(no_augmentation, with_augmentation)
        )
        node_by_id = {node["id"]: node for node in nodes}

        self.assertIsNone(
            config_fields(node_by_id["0"])["adaptive_augmentation_config"]
        )
        self.assertEqual(
            config_fields(node_by_id["1"])["adaptive_augmentation_config"],
            "AdaptiveParameterAugmentationConfig",
        )

    def test_graph_serializer_includes_layer_config_fields(self) -> None:
        cfg = LayerConfig(
            input_dim=4,
            output_dim=3,
            activation=ActivationOptions.GELU,
            residual_config=None,
            dropout_probability=0.1,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=4,
                output_dim=3,
                bias_flag=True,
            ),
        )
        nodes, _edges = serialize_graph(ConfiguredModule(cfg))
        root = nodes[0]

        self.assertEqual(root["config"]["typeName"], "LayerConfig")
        self.assertEqual(
            [field["key"] for field in root["config"]["fields"]],
            [
                "input_dim",
                "output_dim",
                "activation",
                "residual_connection_option",
                "dropout_probability",
                "layer_norm_position",
                "gate_config",
                "halting_config",
                "memory_config",
                "layer_model_config",
                "residual_model_config",
            ],
        )
        self.assertEqual(config_fields(root)["activation"], "GELU")
        self.assertEqual(config_fields(root)["layer_norm_position"], "BEFORE")
        self.assertEqual(config_fields(root)["layer_model_config"], "LinearLayerConfig")
        self.assertIsNone(config_fields(root)["gate_config"])
        self.assertIsNone(config_fields(root)["halting_config"])
        self.assertIsNone(config_fields(root)["memory_config"])
        self.assertIsNone(config_fields(root)["residual_model_config"])

    def test_graph_serializer_reports_layer_gate_option_details(self) -> None:
        gate_model_config = LayerStackConfig(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=4,
                output_dim=4,
                activation=ActivationOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=4,
                    output_dim=4,
                    bias_flag=True,
                ),
            ),
        )
        cfg = LayerConfig(
            input_dim=4,
            output_dim=4,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=GateConfig(
                model_config=gate_model_config,
                option=LayerGateOptions.MULTIPLIER,
            ),
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=4,
                output_dim=4,
                bias_flag=True,
            ),
        )

        nodes, _edges = serialize_graph(Layer(cfg))
        root = nodes[0]

        self.assertEqual(root["details"]["gateOption"], "MULTIPLIER")
        self.assertTrue(root["details"]["gate"])

    def test_graph_serializer_reports_recurrent_gate_option_details(self) -> None:
        block_config = LayerConfig(
            input_dim=4,
            output_dim=4,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=4,
                output_dim=4,
                bias_flag=True,
            ),
        )
        cfg = RecurrentLayerConfig(
            input_dim=4,
            output_dim=4,
            max_steps=2,
            recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
            block_config=block_config,
            gate_config=GateConfig(
                model_config=LayerStackConfig(
                    input_dim=4,
                    hidden_dim=4,
                    output_dim=4,
                    num_layers=1,
                    last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                    apply_output_pipeline_flag=False,
                    layer_config=block_config,
                ),
                option=LayerGateOptions.MULTIPLIER,
            ),
            residual_config=None,
        )

        nodes, _edges = serialize_graph(cfg.build())
        root = nodes[0]

        self.assertEqual(root["details"]["recurrent"]["gateOption"], "MULTIPLIER")
        self.assertEqual(root["details"]["recurrent"]["layerNorm"], "AFTER")
        self.assertTrue(root["details"]["recurrent"]["gate"])

    def test_graph_serializer_omits_absent_layer_residual_module(self) -> None:
        def layer_config(
            residual_connection_option: ResidualConnectionOptions | None,
        ) -> LayerConfig:
            return LayerConfig(
                input_dim=4,
                output_dim=4,
                activation=ActivationOptions.DISABLED,
                residual_config=None
                if residual_connection_option is None
                else ResidualConfig(option=residual_connection_option),
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=4,
                    output_dim=4,
                    bias_flag=True,
                ),
            )

        disabled_layer = Layer(layer_config(None))
        enabled_layer = Layer(layer_config(ResidualConnectionOptions.RESIDUAL))

        disabled_nodes, _disabled_edges = serialize_graph(disabled_layer)
        enabled_nodes, _enabled_edges = serialize_graph(enabled_layer)

        self.assertIsNone(disabled_layer.residual_connection)
        self.assertNotIn(
            "ResidualConnection",
            {node["typeName"] for node in disabled_nodes},
        )
        self.assertIn(
            "ResidualConnection",
            {node["typeName"] for node in enabled_nodes},
        )

    def test_graph_serializer_preserves_layer_stack_config_on_layer_stack(self) -> None:
        stack_config = LayerStackConfig(
            input_dim=4,
            hidden_dim=5,
            output_dim=2,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=True,
            layer_config=LayerConfig(
                input_dim=4,
                output_dim=5,
                activation=ActivationOptions.GELU,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=4,
                    output_dim=5,
                    bias_flag=True,
                ),
            ),
        )

        model = stack_config.build()
        nodes, _edges = serialize_graph(model)
        root = nodes[0]

        self.assertIsInstance(model, LayerStack)
        self.assertEqual(root["config"]["typeName"], "LayerStackConfig")
        self.assertEqual(config_fields(root)["input_dim"], 4)
        self.assertEqual(config_fields(root)["hidden_dim"], 5)
        self.assertEqual(config_fields(root)["output_dim"], 2)
        self.assertEqual(config_fields(root)["num_layers"], 2)
        self.assertEqual(config_fields(root)["last_layer_bias_option"], "DISABLED")
        self.assertTrue(config_fields(root)["apply_output_pipeline_flag"])
        self.assertIsNone(config_fields(root)["shared_gate_config"])
        self.assertEqual(config_fields(root)["layer_config"], "LayerConfig")

    def test_graph_serializer_reports_shared_gate_on_layer_details(self) -> None:
        gate_config = LayerStackConfig(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=4,
                output_dim=4,
                activation=ActivationOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=4,
                    output_dim=4,
                    bias_flag=True,
                ),
            ),
        )
        stack_config = LayerStackConfig(
            input_dim=5,
            hidden_dim=4,
            output_dim=4,
            num_layers=3,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            shared_gate_config=GateConfig(
                model_config=gate_config,
                option=LayerGateOptions.MULTIPLIER,
            ),
            layer_config=LayerConfig(
                input_dim=5,
                output_dim=4,
                activation=ActivationOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=5,
                    output_dim=4,
                    bias_flag=True,
                ),
            ),
        )

        nodes, _edges = serialize_graph(stack_config.build())
        node_by_id = nodes_by_id(nodes)

        self.assertEqual(
            config_fields(node_by_id["__root__"])["shared_gate_config"],
            "GateConfig",
        )
        for node_id in ("layers.0", "layers.1", "layers.2"):
            with self.subTest(node_id=node_id):
                self.assertTrue(node_by_id[node_id]["details"]["gate"])

    def test_graph_serializer_reports_expert_metadata(self) -> None:
        result = inspect_model("experts/linear", "baseline")
        expert_model = next(
            node for node in result["nodes"] if node["typeName"] == "MixtureOfExperts"
        )
        expert_stack_model = next(
            node
            for node in result["nodes"]
            if node["typeName"] == "MixtureOfExpertsModel"
        )

        for node in (expert_model, expert_stack_model):
            with self.subTest(node=node["id"]):
                self.assertEqual(
                    node["details"]["topK"],
                    experts_linear_config.TOP_K,
                )
                self.assertEqual(
                    node["details"]["numExperts"],
                    experts_linear_config.NUM_EXPERTS,
                )
                self.assertEqual(node["details"]["routingMode"], "LAYER")


if __name__ == "__main__":
    unittest.main()
