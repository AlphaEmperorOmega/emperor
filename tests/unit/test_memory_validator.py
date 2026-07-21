import unittest

import torch
from torch import nn

from emperor.layers import Layer, LayerStackConfig
from emperor.memory import (
    AttentionDynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
)
from emperor.memory._validation import (
    AdaptiveGeneratorValidatorBase,
    DynamicMemoryValidator,
)
from emperor.memory._variants.gated_residual import GatedResidualDynamicMemory
from unit.test_memory import make_layer_stack_config, make_memory_config


class MemoryValidatorBehaviorTests(unittest.TestCase):
    def test_validate_requires_a_memory_model_with_dynamic_config(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "^DynamicMemoryValidator.validate expected a DynamicMemoryAbstract "
            "with a DynamicMemoryConfig cfg, received object\\.$",
        ):
            DynamicMemoryValidator.validate(object())

    def test_validate_routes_dimension_failures_with_exact_field_context(self) -> None:
        for field_name in ("input_dim", "output_dim"):
            with self.subTest(field_name=field_name):
                cfg = make_memory_config(
                    config_cls=GatedResidualDynamicMemoryConfig,
                    input_dim=2,
                    output_dim=3,
                )
                setattr(cfg, field_name, 0)
                model = GatedResidualDynamicMemory.__new__(GatedResidualDynamicMemory)
                torch.nn.Module.__init__(model)
                model.cfg = cfg

                with self.assertRaisesRegex(
                    ValueError,
                    f"^{field_name} must be greater than 0 for "
                    "GatedResidualDynamicMemoryConfig, received 0\\.$",
                ):
                    DynamicMemoryValidator.validate(model)

    def test_custom_dimension_position_and_model_errors_are_exact(self) -> None:
        cfg = make_memory_config(
            config_cls=GatedResidualDynamicMemoryConfig,
            input_dim=2,
            output_dim=3,
        )

        for field_name, value, error_type, message in (
            (
                "input_dim",
                True,
                TypeError,
                "^input_dim must be a positive integer for "
                "GatedResidualDynamicMemoryConfig, received bool\\.$",
            ),
            (
                "output_dim",
                1.5,
                TypeError,
                "^output_dim must be a positive integer for "
                "GatedResidualDynamicMemoryConfig, received float\\.$",
            ),
            (
                "input_dim",
                0,
                ValueError,
                "^input_dim must be greater than 0 for "
                "GatedResidualDynamicMemoryConfig, received 0\\.$",
            ),
        ):
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaisesRegex(error_type, message):
                    DynamicMemoryValidator._validate_positive_int_field(
                        cfg,
                        field_name,
                        value,
                    )

        cfg.memory_position_option = "after"
        with self.assertRaisesRegex(
            TypeError,
            "^memory_position_option must be a MemoryPositionOptions value for "
            "GatedResidualDynamicMemoryConfig, received str\\.$",
        ):
            DynamicMemoryValidator._validate_memory_position_option(cfg)

        cfg.model_config = None
        with self.assertRaisesRegex(
            ValueError,
            "^model_config is required for GatedResidualDynamicMemoryConfig, "
            "received None\\.$",
        ):
            DynamicMemoryValidator._validate_model_config(cfg)

        cfg.model_config = object()
        with self.assertRaisesRegex(
            TypeError,
            "^model_config must be a LayerStackConfig for "
            "GatedResidualDynamicMemoryConfig, received object\\.$",
        ):
            DynamicMemoryValidator._validate_model_config(cfg)

    def test_generator_config_without_layer_config_has_no_nested_controls(
        self,
    ) -> None:
        config = LayerStackConfig(
            input_dim=2,
            hidden_dim=3,
            output_dim=2,
            layer_config=None,
            shared_memory_config=None,
            shared_halting_config=None,
        )

        DynamicMemoryValidator._validate_generator_config(config)

    def test_every_forbidden_generator_control_has_an_exact_error(self) -> None:
        cases = (
            (
                "shared_memory_config",
                "^model_config\\.shared_memory_config must be None, "
                "shared memory is not allowed in memory generators\\.$",
            ),
            (
                "shared_halting_config",
                "^model_config\\.shared_halting_config must be None, "
                "shared halting is not allowed in memory generators\\.$",
            ),
            (
                "layer_config.gate_config",
                "^model_config\\.layer_config\\.gate_config must be None, "
                "nested gates are not allowed in memory generators\\.$",
            ),
            (
                "layer_config.halting_config",
                "^model_config\\.layer_config\\.halting_config must be None, "
                "halting is not allowed in memory generators\\.$",
            ),
            (
                "layer_config.memory_config",
                "^model_config\\.layer_config\\.memory_config must be None, "
                "nested memory is not allowed in memory generators\\.$",
            ),
        )

        for field_path, message in cases:
            with self.subTest(field_path=field_path):
                config = make_layer_stack_config(
                    input_dim=2,
                    hidden_dim=3,
                    output_dim=2,
                )
                owner = config
                field_name = field_path
                if field_path.startswith("layer_config."):
                    owner = config.layer_config
                    field_name = field_path.removeprefix("layer_config.")
                setattr(owner, field_name, object())

                with self.assertRaisesRegex(ValueError, message):
                    DynamicMemoryValidator._validate_generator_config(config)

    def test_generator_sequence_and_layer_rejections_are_exact(self) -> None:
        invalid_sequence = nn.Sequential(nn.Linear(2, 2))
        with self.assertRaisesRegex(
            TypeError,
            "^Expected each generator sequence item to be a Layer, received Linear\\.$",
        ):
            AdaptiveGeneratorValidatorBase.validate_generator_model(invalid_sequence)

        invalid_layer = Layer.__new__(Layer)
        torch.nn.Module.__init__(invalid_layer)
        invalid_layer.model = nn.Identity()
        with self.assertRaisesRegex(
            TypeError,
            "^Expected each generator Layer to wrap a LinearLayer, "
            "received Identity\\.$",
        ):
            AdaptiveGeneratorValidatorBase.validate_generator_model(invalid_layer)

    def test_attention_slot_bool_has_its_distinct_error_contract(self) -> None:
        cfg = make_memory_config(
            config_cls=AttentionDynamicMemoryConfig,
            input_dim=2,
            output_dim=2,
            num_memory_slots=True,
        )

        with self.assertRaisesRegex(
            TypeError,
            "^num_memory_slots must be int for AttentionDynamicMemoryConfig, "
            "received bool\\.$",
        ):
            DynamicMemoryValidator.validate_attention_num_memory_slots(cfg)

    def test_attention_slot_none_type_and_boundary_errors_are_exact(self) -> None:
        for value, error_type, message in (
            (
                None,
                ValueError,
                "^num_memory_slots is required for "
                "AttentionDynamicMemoryConfig, received None\\.$",
            ),
            (
                "two",
                TypeError,
                "^num_memory_slots must be int for "
                "AttentionDynamicMemoryConfig, received str\\.$",
            ),
            (
                0,
                ValueError,
                "^num_memory_slots must be greater than 0 for "
                "AttentionDynamicMemoryConfig, received 0\\.$",
            ),
        ):
            with self.subTest(value=value):
                cfg = make_memory_config(
                    config_cls=AttentionDynamicMemoryConfig,
                    input_dim=2,
                    output_dim=2,
                    num_memory_slots=value,
                )
                with self.assertRaisesRegex(error_type, message):
                    DynamicMemoryValidator.validate_attention_num_memory_slots(cfg)

    def test_ttt_configuration_errors_are_exact(self) -> None:
        cases = (
            (
                0.1,
                None,
                ValueError,
                "^test_time_training_learning_rate and "
                "test_time_training_num_inner_steps must be provided together "
                "for GatedResidualDynamicMemoryConfig\\.$",
            ),
            (
                "fast",
                1,
                TypeError,
                "^test_time_training_learning_rate must be a positive float for "
                "GatedResidualDynamicMemoryConfig, received str\\.$",
            ),
            (
                float("inf"),
                1,
                ValueError,
                "^test_time_training_learning_rate must be finite for "
                "GatedResidualDynamicMemoryConfig, received inf\\.$",
            ),
            (
                0.0,
                1,
                ValueError,
                "^test_time_training_learning_rate must be greater than 0 for "
                "GatedResidualDynamicMemoryConfig, received 0\\.0\\.$",
            ),
            (
                0.1,
                1.5,
                TypeError,
                "^test_time_training_num_inner_steps must be a positive integer "
                "for GatedResidualDynamicMemoryConfig, received float\\.$",
            ),
            (
                0.1,
                0,
                ValueError,
                "^test_time_training_num_inner_steps must be greater than 0 for "
                "GatedResidualDynamicMemoryConfig, received 0\\.$",
            ),
        )

        for learning_rate, steps, error_type, message in cases:
            with self.subTest(learning_rate=learning_rate, steps=steps):
                cfg = make_memory_config(
                    config_cls=GatedResidualDynamicMemoryConfig,
                    input_dim=2,
                    output_dim=2,
                    test_time_training_learning_rate=learning_rate,
                    test_time_training_num_inner_steps=steps,
                )
                with self.assertRaisesRegex(error_type, message):
                    DynamicMemoryValidator._validate_test_time_training_config(cfg)

    def test_forward_input_errors_include_exact_rank_and_dimensions(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "^logits must be a torch.Tensor for DynamicMemory forward, "
            "received list\\.$",
        ):
            DynamicMemoryValidator.validate_forward_inputs([1.0, 2.0], 2)
        with self.assertRaisesRegex(
            ValueError,
            "^logits must have rank >= 2 for DynamicMemory forward, "
            r"received rank 1 with shape \(2,\)\.$",
        ):
            DynamicMemoryValidator.validate_forward_inputs(torch.ones(2), 2)
        with self.assertRaisesRegex(
            ValueError,
            "^logits final dimension must match the configured memory_dim "
            "for DynamicMemory forward, expected 2, received 3 "
            r"with shape \(4, 3\)\.$",
        ):
            DynamicMemoryValidator.validate_forward_inputs(torch.ones(4, 3), 2)
        with self.assertRaisesRegex(
            ValueError,
            "^logits final dimension must match the configured memory_dim "
            "for DynamicMemory forward, expected 2, received 3 "
            r"with shape \(2, 4, 3\)\.$",
        ):
            DynamicMemoryValidator.validate_forward_inputs(
                torch.ones(2, 4, 3),
                2,
            )

    def test_real_generator_config_still_validates_after_direct_helper_cases(
        self,
    ) -> None:
        DynamicMemoryValidator._validate_generator_config(
            make_layer_stack_config(
                input_dim=2,
                hidden_dim=3,
                output_dim=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
