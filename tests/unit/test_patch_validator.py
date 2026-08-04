import unittest

import torch

from emperor.convs import Conv2dLayerConfig
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    MirroredLayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.patch import (
    ConvPatchEmbeddingConfig,
    LinearPatchEmbeddingConfig,
    PatchBase,
    PatchConfig,
    PatchEmbeddingConv,
    PatchEmbeddingLinear,
)
from emperor.patch._validation import PatchValidator


class DerivedLayerStackConfig(LayerStackConfig):
    pass


class BuildOnlyCollaborator:
    def build(self, overrides=None):
        raise AssertionError("invalid collaborator must not be built")


def linear_stack_config(
    stack_config_type: type[LayerStackConfig] = LayerStackConfig,
) -> LayerStackConfig:
    return stack_config_type(
        input_dim=1,
        hidden_dim=4,
        output_dim=4,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def convolutional_stack_config(
    *,
    kernel_size: int = 2,
    stride: int = 1,
    num_layers: int = 1,
    stack_config_type: type[LayerStackConfig] = LayerStackConfig,
) -> LayerStackConfig:
    return stack_config_type(
        input_dim=1,
        hidden_dim=4,
        output_dim=4,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=Conv2dLayerConfig(
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias_flag=True,
            ),
        ),
    )


def linear_config(**overrides) -> object:
    values = {
        "embedding_dim": 4,
        "num_input_channels": 1,
        "patch_size": 2,
        "dropout_probability": 0.0,
        "stride": 2,
        "padding": 0,
        "embedding_stack_config": linear_stack_config(),
    }
    values.update(overrides)
    return LinearPatchEmbeddingConfig(**values)


def conv_config(**overrides) -> ConvPatchEmbeddingConfig:
    values = {
        "embedding_dim": 4,
        "num_input_channels": 1,
        "patch_size": 2,
        "dropout_probability": 0.0,
        "conv_stack_config": linear_stack_config(),
    }
    values.update(overrides)
    return ConvPatchEmbeddingConfig(**values)


class PatchValidatorBehaviorTests(unittest.TestCase):
    def test_invalid_scalar_geometry_returns_none_before_arithmetic(self) -> None:
        invalid_values = (True, 0, -1, 1.5, (1,))
        fields = ("num_layers", "kernel_size", "stride")

        for field_name in fields:
            for invalid_value in invalid_values:
                with self.subTest(field_name=field_name, value=invalid_value):
                    stack_config = convolutional_stack_config(
                        kernel_size=3,
                        stride=2,
                        num_layers=2,
                    )
                    layer_model_config = stack_config.layer_config.layer_model_config
                    owner = (
                        stack_config
                        if field_name == "num_layers"
                        else layer_model_config
                    )
                    setattr(owner, field_name, invalid_value)
                    snapshot = (
                        stack_config.num_layers,
                        layer_model_config.kernel_size,
                        layer_model_config.stride,
                    )
                    torch.manual_seed(700)
                    rng_state = torch.random.get_rng_state()

                    result = PatchValidator._effective_convolutional_patch_size(
                        stack_config,
                        layer_model_config,
                    )

                    self.assertIsNone(result)
                    self.assertEqual(
                        (
                            stack_config.num_layers,
                            layer_model_config.kernel_size,
                            layer_model_config.stride,
                        ),
                        snapshot,
                    )
                    torch.testing.assert_close(
                        torch.random.get_rng_state(),
                        rng_state,
                        rtol=0,
                        atol=0,
                    )

        ordinary = convolutional_stack_config(
            kernel_size=3,
            stride=2,
            num_layers=2,
        )
        mirrored = convolutional_stack_config(
            kernel_size=3,
            stride=2,
            num_layers=2,
            stack_config_type=MirroredLayerStackConfig,
        )
        self.assertEqual(
            PatchValidator._effective_convolutional_patch_size(
                ordinary,
                ordinary.layer_config.layer_model_config,
            ),
            7,
        )
        self.assertEqual(
            PatchValidator._effective_convolutional_patch_size(
                mirrored,
                mirrored.layer_config.layer_model_config,
            ),
            31,
        )

    def test_real_patch_modules_use_the_real_validator(self) -> None:
        for module_type in (PatchBase, PatchEmbeddingLinear, PatchEmbeddingConv):
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, PatchValidator)

    def test_stack_collaborators_reject_invalid_nominal_types_before_build(
        self,
    ) -> None:
        invalid_values = (object(), LinearLayerConfig(), BuildOnlyCollaborator())
        cases = (
            (
                PatchEmbeddingLinear,
                linear_config,
                "embedding_stack_config",
            ),
            (
                PatchEmbeddingConv,
                conv_config,
                "conv_stack_config",
            ),
        )

        for module_type, config_factory, field_name in cases:
            for invalid_value in invalid_values:
                with self.subTest(
                    module_type=module_type.__name__,
                    invalid_type=type(invalid_value).__name__,
                ):
                    torch.manual_seed(701)
                    rng_state = torch.random.get_rng_state()
                    config = config_factory(**{field_name: invalid_value})

                    with self.assertRaises(TypeError) as error:
                        module_type(config)

                    self.assertEqual(
                        str(error.exception),
                        f"{field_name} must be an instance of LayerStackConfig for "
                        f"{type(config).__name__}, got "
                        f"{type(invalid_value).__name__}",
                    )
                    torch.testing.assert_close(
                        torch.random.get_rng_state(),
                        rng_state,
                        rtol=0,
                        atol=0,
                    )

    def test_linear_patch_accepts_layer_stack_config_subclasses(self) -> None:
        stack_config = linear_stack_config(DerivedLayerStackConfig)
        model = PatchEmbeddingLinear(linear_config(embedding_stack_config=stack_config))

        output = model(torch.ones(2, 1, 4, 4))

        self.assertIs(model.cfg.embedding_stack_config, stack_config)
        self.assertEqual(output.shape, (2, 5, 4))

    def test_convolutional_patch_rejects_non_convolutional_stack_topology(
        self,
    ) -> None:
        config = conv_config(conv_stack_config=linear_stack_config())

        with self.assertRaises(TypeError) as error:
            PatchEmbeddingConv(config)

        self.assertEqual(
            str(error.exception),
            "conv_stack_config.layer_config.layer_model_config must be an instance "
            "of Conv2dLayerConfig for ConvPatchEmbeddingConfig, got "
            "LinearLayerConfig",
        )

    def test_convolutional_patch_rejects_mismatched_effective_geometry_before_rng(
        self,
    ) -> None:
        cases = (
            (
                2,
                convolutional_stack_config(kernel_size=3),
                3,
            ),
            (
                3,
                convolutional_stack_config(
                    kernel_size=2,
                    stride=2,
                    num_layers=2,
                ),
                4,
            ),
            (
                4,
                convolutional_stack_config(
                    kernel_size=2,
                    num_layers=2,
                    stack_config_type=MirroredLayerStackConfig,
                ),
                5,
            ),
        )

        for patch_size, stack_config, effective_patch_size in cases:
            with self.subTest(
                patch_size=patch_size,
                effective_patch_size=effective_patch_size,
                stack_type=type(stack_config).__name__,
            ):
                torch.manual_seed(702)
                rng_state = torch.random.get_rng_state()
                config = conv_config(
                    patch_size=patch_size,
                    conv_stack_config=stack_config,
                )

                with self.assertRaises(ValueError) as error:
                    PatchEmbeddingConv(config)

                self.assertEqual(
                    str(error.exception),
                    "patch_size must match the effective convolutional receptive "
                    "field for ConvPatchEmbeddingConfig, got "
                    f"patch_size={patch_size} and "
                    f"effective_patch_size={effective_patch_size}",
                )
                torch.testing.assert_close(
                    torch.random.get_rng_state(),
                    rng_state,
                    rtol=0,
                    atol=0,
                )

    def test_required_common_field_error_is_exact(self) -> None:
        with self.assertRaises(ValueError) as error:
            PatchBase(
                PatchConfig(
                    num_input_channels=1,
                    patch_size=2,
                    dropout_probability=0.0,
                )
            )

        self.assertEqual(
            str(error.exception),
            "embedding_dim is required for PatchConfig, received None",
        )

    def test_common_field_type_errors_are_exact(self) -> None:
        invalid_cases = (
            ("embedding_dim", "4", "embedding_dim must be int"),
            ("num_input_channels", 1.0, "num_input_channels must be int"),
            ("patch_size", True, "patch_size must be int"),
            (
                "dropout_probability",
                0,
                "dropout_probability must be float",
            ),
        )

        for field_name, value, message_prefix in invalid_cases:
            values = {
                "embedding_dim": 4,
                "num_input_channels": 1,
                "patch_size": 2,
                "dropout_probability": 0.0,
            }
            values[field_name] = value
            with self.subTest(field_name=field_name):
                with self.assertRaises(TypeError) as error:
                    PatchBase(PatchConfig(**values))
                self.assertEqual(
                    str(error.exception),
                    f"{message_prefix} for PatchConfig, got {type(value).__name__}",
                )

    def test_every_common_dimension_is_validated_with_exact_message(self) -> None:
        for field_name in ("embedding_dim", "num_input_channels", "patch_size"):
            values = {
                "embedding_dim": 4,
                "num_input_channels": 1,
                "patch_size": 2,
                "dropout_probability": 0.0,
            }
            values[field_name] = 0
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError) as error:
                    PatchBase(PatchConfig(**values))
                self.assertEqual(
                    str(error.exception),
                    f"{field_name} must be greater than 0, received 0",
                )

    def test_dropout_accepts_both_boundaries_and_rejects_outside_them(self) -> None:
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability):
                model = PatchBase(
                    PatchConfig(
                        embedding_dim=4,
                        num_input_channels=1,
                        patch_size=2,
                        dropout_probability=probability,
                    )
                )
                self.assertEqual(model.dropout.p, probability)

        for probability in (-0.1, 1.1):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError) as error:
                    PatchBase(
                        PatchConfig(
                            embedding_dim=4,
                            num_input_channels=1,
                            patch_size=2,
                            dropout_probability=probability,
                        )
                    )
                self.assertEqual(
                    str(error.exception),
                    "dropout_probability must be in [0.0, 1.0], "
                    f"received {probability}",
                )

    def test_class_token_flag_accepts_bool_or_none_and_rejects_other_values(
        self,
    ) -> None:
        for flag in (True, False, None):
            with self.subTest(flag=flag):
                model = PatchBase(
                    PatchConfig(
                        embedding_dim=4,
                        num_input_channels=1,
                        patch_size=2,
                        dropout_probability=0.0,
                        class_token_flag=flag,
                    )
                )
                self.assertEqual(model.class_token_flag, flag is not False)

        for value in (0, 1, "false"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError) as error:
                    PatchBase(
                        PatchConfig(
                            embedding_dim=4,
                            num_input_channels=1,
                            patch_size=2,
                            dropout_probability=0.0,
                            class_token_flag=value,
                        )
                    )
                self.assertEqual(
                    str(error.exception),
                    "class_token_flag must be bool or None for PatchConfig, got "
                    f"{type(value).__name__}",
                )

    def test_linear_stride_must_be_positive_at_construction(self) -> None:
        for stride in (0, -1):
            with self.subTest(stride=stride):
                with self.assertRaises(ValueError) as error:
                    PatchEmbeddingLinear(linear_config(stride=stride))
                self.assertEqual(
                    str(error.exception),
                    f"stride must be greater than 0, received {stride}",
                )

    def test_linear_padding_must_be_nonnegative_at_construction(self) -> None:
        with self.assertRaises(ValueError) as error:
            PatchEmbeddingLinear(linear_config(padding=-1))

        self.assertEqual(
            str(error.exception),
            "padding must be greater than or equal to 0, received -1",
        )

    def test_forward_input_error_contracts_are_exact(self) -> None:
        model = PatchEmbeddingLinear(linear_config())
        invalid_cases = (
            (
                [1, 2, 3],
                TypeError,
                "Input Error: forward input must be a Tensor, received list.",
            ),
            (
                torch.ones(1, 4, 4),
                ValueError,
                "Input Error: PatchBase expects a 4D input tensor "
                "(batch, channels, height, width), received a 3D tensor "
                "with shape (1, 4, 4).",
            ),
            (
                torch.ones(2, 3, 4, 5),
                ValueError,
                "Input Error: input channel dimension must match "
                "'num_input_channels', received num_input_channels=1 and "
                "input shape (2, 3, 4, 5).",
            ),
        )

        for value, error_type, expected_message in invalid_cases:
            with self.subTest(error_type=error_type.__name__):
                with self.assertRaises(error_type) as error:
                    model(value)
                self.assertEqual(str(error.exception), expected_message)


if __name__ == "__main__":
    unittest.main()
