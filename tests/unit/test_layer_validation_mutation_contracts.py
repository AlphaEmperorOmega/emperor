import unittest
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

import torch

from emperor.config import ConfigBase, optional_field
from emperor.convs import Conv2dLayerConfig
from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    SoftHaltingConfig,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
    RecurrentLayerConfig,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.layers._validation.common import _matches_config_contract
from emperor.layers._validation.gate import LayerGateValidator
from emperor.layers._validation.layer import LayerValidator
from emperor.layers._validation.recurrent import RecurrentLayerValidator
from emperor.layers._validation.stack import LayerStackValidator
from emperor.linears import LinearLayerConfig
from emperor.memory import (
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,
)


def _layer_config(
    dim: int = 2,
    *,
    input_dim: int | None = None,
    output_dim: int | None = None,
    gate_config: GateConfig | None = None,
    halting_config: object | None = None,
    memory_config: object | None = None,
    model_config: ConfigBase | None = None,
) -> LayerConfig:
    resolved_input = dim if input_dim is None else input_dim
    resolved_output = dim if output_dim is None else output_dim
    return LayerConfig(
        input_dim=resolved_input,
        output_dim=resolved_output,
        activation=ActivationOptions.DISABLED,
        residual_config=None,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=gate_config,
        halting_config=halting_config,
        memory_config=memory_config,
        layer_model_config=(
            LinearLayerConfig(
                input_dim=resolved_input,
                output_dim=resolved_output,
                bias_flag=True,
            )
            if model_config is None
            else model_config
        ),
    )


def _stack_config(
    dim: int = 2,
    *,
    input_dim: int | None = None,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
    num_layers: int = 2,
    layer_config: LayerConfig | None = None,
    shared_gate_config: object | None = None,
    shared_halting_config: object | None = None,
    shared_memory_config: object | None = None,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=dim if input_dim is None else input_dim,
        hidden_dim=dim if hidden_dim is None else hidden_dim,
        output_dim=dim if output_dim is None else output_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        shared_gate_config=shared_gate_config,
        shared_halting_config=shared_halting_config,
        shared_memory_config=shared_memory_config,
        layer_config=_layer_config(dim) if layer_config is None else layer_config,
    )


def _gate_config(
    dim: int = 2,
    *,
    option: LayerGateOptions | None = LayerGateOptions.MULTIPLIER,
    activation: ActivationOptions | None = ActivationOptions.SIGMOID,
    model_config: LayerStackConfig | None = None,
) -> GateConfig:
    return GateConfig(
        gate_dim=dim,
        option=option,
        activation=activation,
        model_config=_stack_config(dim, num_layers=1)
        if model_config is None
        else model_config,
    )


def _halting_config(dim: int = 2) -> StickBreakingConfig:
    return StickBreakingConfig(
        input_dim=dim,
        threshold=0.99,
        dropout_probability=0.0,
        hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
        halting_gate_config=_stack_config(
            dim,
            output_dim=2,
            num_layers=1,
        ),
    )


def _soft_halting_config(dim: int = 2) -> SoftHaltingConfig:
    return SoftHaltingConfig(
        input_dim=dim,
        threshold=0.99,
        dropout_probability=0.0,
        hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
        halting_gate_config=_stack_config(
            dim,
            output_dim=2,
            num_layers=1,
        ),
    )


def _memory_config(dim: int = 2) -> WeightedDynamicMemoryConfig:
    return WeightedDynamicMemoryConfig(
        input_dim=dim,
        output_dim=dim,
        memory_position_option=MemoryPositionOptions.AFTER_AFFINE,
        test_time_training_learning_rate=None,
        test_time_training_num_inner_steps=None,
        model_config=_stack_config(dim, num_layers=1),
    )


def _recurrent_config(
    dim: int = 2,
    *,
    input_dim: object | None = None,
    output_dim: object | None = None,
    max_steps: object = 3,
    norm: object = LayerNormPositionOptions.DISABLED,
    block_config: object | None = None,
    gate_config: GateConfig | None = None,
    residual: object = None,
    halting_config: object | None = None,
    memory_config: object | None = None,
) -> RecurrentLayerConfig:
    return RecurrentLayerConfig(
        input_dim=dim if input_dim is None else input_dim,
        output_dim=dim if output_dim is None else output_dim,
        max_steps=max_steps,
        recurrent_layer_norm_position=norm,
        block_config=_layer_config(dim) if block_config is None else block_config,
        gate_config=gate_config,
        residual_config=None if residual is None else ResidualConfig(option=residual),
        halting_config=halting_config,
        memory_config=memory_config,
    )


@dataclass
class DerivedLayerConfig(LayerConfig):
    pass


@dataclass
class InputOnlyBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input dimension.")


@dataclass
class OutputOnlyBlockConfig(ConfigBase):
    output_dim: int | None = optional_field("Output dimension.")


class TestLayerValidationMutationContracts(unittest.TestCase):
    def assert_raises_exact(
        self,
        exception_type: type[BaseException],
        message: str,
        function,
        *args,
        **kwargs,
    ) -> None:
        with self.assertRaises(exception_type) as caught:
            function(*args, **kwargs)
        self.assertEqual(str(caught.exception), message)

    def test_gate_option_activation_and_runtime_errors_are_exact(self) -> None:
        self.assertEqual(LayerGateValidator.option_names(), "MULTIPLIER, ADDITION")

        required_suffix = (
            " is required when gate_config is provided; pass one of "
            "LayerGateOptions: MULTIPLIER, ADDITION. Use gate_config=None "
            "to disable the gate."
        )
        self.assert_raises_exact(
            ValueError,
            "gate_config.option" + required_suffix,
            LayerGateValidator.validate_option,
            None,
        )
        self.assert_raises_exact(
            ValueError,
            "Owner.option" + required_suffix,
            LayerGateValidator.validate_option,
            None,
            owner_name="Owner.option",
        )
        self.assert_raises_exact(
            TypeError,
            "gate_config.option must be a LayerGateOptions value, got object. "
            "Valid values are: MULTIPLIER, ADDITION.",
            LayerGateValidator.validate_option,
            object(),
        )
        self.assert_raises_exact(
            TypeError,
            "Owner.option must be a LayerGateOptions value, got object. "
            "Valid values are: MULTIPLIER, ADDITION.",
            LayerGateValidator.validate_option,
            object(),
            owner_name="Owner.option",
        )
        LayerGateValidator.validate_option(LayerGateOptions.ADDITION)

        LayerGateValidator.validate_activation(None)
        LayerGateValidator.validate_activation(ActivationOptions.TANH)
        self.assert_raises_exact(
            TypeError,
            "gate_config.activation must be an ActivationOptions value or None, "
            "got object.",
            LayerGateValidator.validate_activation,
            object(),
        )
        self.assert_raises_exact(
            TypeError,
            "Owner.activation must be an ActivationOptions value or None, got object.",
            LayerGateValidator.validate_activation,
            object(),
            owner_name="Owner.activation",
        )

        self.assert_raises_exact(
            ValueError,
            "LayerGate requires a gate model when enabled.",
            LayerGateValidator.validate_gate_model,
            None,
        )
        LayerGateValidator.validate_gate_model(torch.nn.Identity())
        current = torch.zeros(2, 2)
        self.assert_raises_exact(
            TypeError,
            "LayerGate model must return a Tensor or LayerState.hidden Tensor, "
            "got object.",
            LayerGateValidator.validate_gate_output,
            object(),
            current,
            LayerGateOptions.MULTIPLIER,
        )
        self.assert_raises_exact(
            ValueError,
            "MULTIPLIER requires gate output and current shapes to match, "
            "got gate output shape (1, 2) and current shape (2, 2).",
            LayerGateValidator.validate_gate_output,
            torch.zeros(1, 2),
            current,
            LayerGateOptions.MULTIPLIER,
        )
        LayerGateValidator.validate_gate_output(
            torch.zeros_like(current),
            current,
            LayerGateOptions.ADDITION,
        )

        malformed_gate = SimpleNamespace(
            cfg=replace(_gate_config(), option=None),
        )
        self.assert_raises_exact(
            ValueError,
            "GateConfig.option" + required_suffix,
            LayerGateValidator.validate,
            malformed_gate,
        )

    def test_gate_config_contract_and_owner_paths_are_exact(self) -> None:
        self.assert_raises_exact(
            ValueError,
            "gate_config.model_config is required when gate_config is provided",
            LayerGateValidator.validate_gate_model_config,
            None,
        )
        self.assert_raises_exact(
            ValueError,
            "Owner.model_config is required when gate_config is provided",
            LayerGateValidator.validate_gate_model_config,
            None,
            owner_name="Owner",
        )
        self.assert_raises_exact(
            TypeError,
            "gate_config.model_config must be an instance of LayerStackConfig, "
            "got object",
            LayerGateValidator.validate_gate_model_config,
            object(),
        )
        self.assert_raises_exact(
            TypeError,
            "Owner.model_config must be an instance of LayerStackConfig for Owner, "
            "got object",
            LayerGateValidator.validate_gate_model_config,
            object(),
            owner_name="Owner",
        )

        missing_layer = _stack_config(num_layers=1)
        missing_layer.layer_config = None
        self.assert_raises_exact(
            ValueError,
            "Owner.model_config.layer_config is required when gate_config is provided",
            LayerGateValidator.validate_gate_model_config,
            missing_layer,
            owner_name="Owner",
        )

        base_layer = _layer_config()
        derived_layer = DerivedLayerConfig(
            **{
                field.name: getattr(base_layer, field.name)
                for field in fields(base_layer)
            }
        )
        derived_stack = _stack_config(layer_config=derived_layer, num_layers=1)
        self.assert_raises_exact(
            TypeError,
            "Owner.model_config.layer_config must be exactly LayerConfig for Owner, "
            "got DerivedLayerConfig. Configured gate stacks must be tensor-only "
            "controllers and must not depend on caller LayerState fields.",
            LayerGateValidator.validate_gate_model_config,
            derived_stack,
            owner_name="Owner",
        )

        nested_layer_stack = _stack_config(
            layer_config=_layer_config(gate_config=_gate_config()),
            num_layers=1,
        )
        self.assert_raises_exact(
            ValueError,
            "Owner.model_config.layer_config.gate_config must be inactive, "
            "nested gates are not allowed",
            LayerGateValidator.validate_gate_model_config,
            nested_layer_stack,
            owner_name="Owner",
        )
        malformed_nested_layer_stack = _stack_config(
            layer_config=_layer_config(gate_config=_gate_config(option=None)),
            num_layers=1,
        )
        self.assert_raises_exact(
            ValueError,
            "Owner.model_config.layer_config.gate_config.option is required when "
            "gate_config is provided; pass one of LayerGateOptions: MULTIPLIER, "
            "ADDITION. Use gate_config=None to disable the gate.",
            LayerGateValidator.validate_gate_model_config,
            malformed_nested_layer_stack,
            owner_name="Owner",
        )
        nested_shared_stack = _stack_config(
            shared_gate_config=_gate_config(),
            num_layers=1,
        )
        self.assert_raises_exact(
            ValueError,
            "Owner.model_config.shared_gate_config must be inactive, "
            "nested gates are not allowed",
            LayerGateValidator.validate_gate_model_config,
            nested_shared_stack,
            owner_name="Owner",
        )
        malformed_nested_shared_gate = _gate_config()
        malformed_nested_shared_gate.activation = object()
        malformed_nested_shared_stack = _stack_config(
            shared_gate_config=malformed_nested_shared_gate,
            num_layers=1,
        )
        self.assert_raises_exact(
            TypeError,
            "Owner.model_config.shared_gate_config.activation must be an "
            "ActivationOptions value or None, got object.",
            LayerGateValidator.validate_gate_model_config,
            malformed_nested_shared_stack,
            owner_name="Owner",
        )
        layer_halting_stack = _stack_config(
            layer_config=_layer_config(halting_config=_halting_config()),
            num_layers=2,
        )
        self.assert_raises_exact(
            ValueError,
            "Owner.model_config.layer_config.halting_config must be None, "
            "halting is not allowed in gates",
            LayerGateValidator.validate_gate_model_config,
            layer_halting_stack,
            owner_name="Owner",
        )
        shared_halting_stack = _stack_config(
            shared_halting_config=_halting_config(),
            num_layers=2,
        )
        self.assert_raises_exact(
            ValueError,
            "Owner.model_config.shared_halting_config must be None, "
            "halting is not allowed in gates",
            LayerGateValidator.validate_gate_model_config,
            shared_halting_stack,
            owner_name="Owner",
        )
        LayerGateValidator.validate_gate_model_config(
            _stack_config(num_layers=1),
            owner_name="Owner",
        )

    def test_layer_and_recurrent_gate_adapters_preserve_all_errors(self) -> None:
        for validator in (
            LayerGateValidator.validate_layer_gate_config,
            LayerGateValidator.validate_recurrent_gate_config,
        ):
            with self.subTest(validator=validator.__name__, case="none"):
                validator(None)
            with self.subTest(validator=validator.__name__, case="type_default"):
                self.assert_raises_exact(
                    TypeError,
                    "gate_config must be an instance of GateConfig, got object",
                    validator,
                    object(),
                )
            with self.subTest(validator=validator.__name__, case="type_owner"):
                self.assert_raises_exact(
                    TypeError,
                    "gate_config must be an instance of GateConfig for Owner, "
                    "got object",
                    validator,
                    object(),
                    owner_name="Owner",
                )
            with self.subTest(validator=validator.__name__, case="option"):
                self.assert_raises_exact(
                    ValueError,
                    "Owner.option is required when gate_config is provided; pass "
                    "one of LayerGateOptions: MULTIPLIER, ADDITION. Use "
                    "gate_config=None to disable the gate.",
                    validator,
                    _gate_config(option=None),
                    owner_name="Owner",
                )
            invalid_activation = _gate_config()
            invalid_activation.activation = object()
            with self.subTest(
                validator=validator.__name__,
                case="activation_default_owner",
            ):
                self.assert_raises_exact(
                    TypeError,
                    "gate_config.activation must be an ActivationOptions value or "
                    "None, got object.",
                    validator,
                    invalid_activation,
                )
            with self.subTest(validator=validator.__name__, case="activation"):
                self.assert_raises_exact(
                    TypeError,
                    "Owner.activation must be an ActivationOptions value or None, "
                    "got object.",
                    validator,
                    invalid_activation,
                    owner_name="Owner",
                )
            invalid_model = _gate_config()
            invalid_model.model_config = object()
            with self.subTest(validator=validator.__name__, case="model"):
                self.assert_raises_exact(
                    TypeError,
                    "Owner.model_config must be an instance of LayerStackConfig "
                    "for Owner, got object",
                    validator,
                    invalid_model,
                    owner_name="Owner",
                )
            validator(_gate_config(), owner_name="Owner")

    def test_config_contract_detection_requires_type_and_every_field(self) -> None:
        required_fields = (
            "input_dim",
            "threshold",
            "dropout_probability",
            "hidden_state_mode",
            "halting_gate_config",
        )
        self.assertTrue(_matches_config_contract(_halting_config(), required_fields))
        self.assertFalse(_matches_config_contract(ConfigBase(), required_fields))
        all_attributes_wrong_type = SimpleNamespace(
            **{field_name: None for field_name in required_fields}
        )
        self.assertFalse(
            _matches_config_contract(all_attributes_wrong_type, required_fields)
        )

    def test_layer_validator_dimension_and_controller_errors_are_exact(self) -> None:
        for field_name in ("input_dim", "output_dim"):
            config = _layer_config()
            setattr(config, field_name, 0)
            with self.subTest(field_name=field_name):
                self.assert_raises_exact(
                    ValueError,
                    f"{field_name} must be greater than 0, received 0",
                    LayerValidator.validate,
                    SimpleNamespace(cfg=config),
                )

        self.assert_raises_exact(
            ValueError,
            "input_dim and output_dim must be equal when "
            "residual_config.option is ResidualConnectionOptions.RESIDUAL, "
            "got input_dim=2 and output_dim=3.",
            LayerValidator._validate_residual_dimensions,
            2,
            3,
            ResidualConfig(option=ResidualConnectionOptions.RESIDUAL),
        )
        self.assert_raises_exact(
            ValueError,
            "layer_model_config is required, Layer needs it to build the model",
            LayerValidator._validate_model_config,
            None,
        )
        self.assert_raises_exact(
            TypeError,
            "model_config must be an instance of ConfigBase, got object",
            LayerValidator._validate_model_config,
            object(),
        )

        strided = _layer_config()
        strided.residual_config = ResidualConfig(
            option=ResidualConnectionOptions.RESIDUAL
        )
        strided.layer_model_config = Conv2dLayerConfig(
            input_dim=2,
            output_dim=2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_flag=True,
        )
        self.assert_raises_exact(
            ValueError,
            "residual_config.option cannot be "
            "ResidualConnectionOptions.RESIDUAL when layer_model_config has "
            "stride > 1 (received stride=2). Spatial reduction breaks the "
            "residual connection shape contract.",
            LayerValidator._validate_residual_with_strided_model,
            strided,
        )
        unit_stride = replace(
            strided,
            layer_model_config=replace(strided.layer_model_config, stride=1),
        )
        LayerValidator._validate_residual_with_strided_model(unit_stride)
        self.assert_raises_exact(
            TypeError,
            "halting_config must be an instance of HaltingConfig, got object",
            LayerValidator._validate_halting_config,
            object(),
        )
        self.assert_raises_exact(
            TypeError,
            "memory_config must be an instance of DynamicMemoryConfig, got object.",
            LayerValidator._validate_memory_config,
            object(),
        )
        self.assert_raises_exact(
            ValueError,
            "input_dim and output_dim must be equal when halting_config is "
            "provided, got input_dim=2 and output_dim=3. Halting accumulates "
            "hidden states across steps, which requires consistent dimensions.",
            LayerValidator._validate_halting_dimensions,
            2,
            3,
            _halting_config(),
        )
        LayerValidator._validate_halting_dimensions(2, 3, None)

    def test_stack_validator_gate_halting_and_memory_errors_are_exact(self) -> None:
        for field_name in ("input_dim", "hidden_dim", "output_dim", "num_layers"):
            config = _stack_config()
            setattr(config, field_name, 0)
            with self.subTest(field_name=field_name):
                self.assert_raises_exact(
                    ValueError,
                    f"{field_name} must be greater than 0, received 0",
                    LayerStackValidator.validate,
                    SimpleNamespace(cfg=config),
                )

        invalid_shared_gate = _stack_config(shared_gate_config=object())
        self.assert_raises_exact(
            TypeError,
            "shared_gate_config must be an instance of GateConfig for "
            "LayerStackConfig, got object",
            LayerStackValidator._validate_gate_config,
            invalid_shared_gate,
        )
        gate_conflict = _stack_config(
            layer_config=_layer_config(gate_config=_gate_config()),
            shared_gate_config=_gate_config(),
        )
        self.assert_raises_exact(
            ValueError,
            "shared_gate_config and layer_config.gate_config are mutually "
            "exclusive. Put shared gate controllers on LayerStackConfig and "
            "per-layer gate controllers on LayerConfig.",
            LayerStackValidator._validate_gate_config,
            gate_conflict,
        )
        malformed_layer_gate = _stack_config(
            layer_config=_layer_config(gate_config=_gate_config(option=None)),
        )
        self.assert_raises_exact(
            ValueError,
            "LayerStackConfig.layer_config.option is required when gate_config is "
            "provided; pass one of LayerGateOptions: MULTIPLIER, ADDITION. Use "
            "gate_config=None to disable the gate.",
            LayerStackValidator._validate_gate_config,
            malformed_layer_gate,
        )
        malformed_shared_gate = _stack_config(
            shared_gate_config=_gate_config(option=None),
        )
        self.assert_raises_exact(
            ValueError,
            "LayerStackConfig.shared_gate_config.option is required when "
            "gate_config is provided; pass one of LayerGateOptions: MULTIPLIER, "
            "ADDITION. Use gate_config=None to disable the gate.",
            LayerStackValidator._validate_gate_config,
            malformed_shared_gate,
        )
        shared_gate_dims = _stack_config(
            hidden_dim=3,
            output_dim=4,
            shared_gate_config=_gate_config(4),
        )
        self.assert_raises_exact(
            ValueError,
            "hidden_dim and output_dim must be equal when shared_gate_config is "
            "provided, got hidden_dim=3 and output_dim=4. Shared gates use one "
            "module across all layer outputs and require one gate dimension.",
            LayerStackValidator._validate_gate_config,
            shared_gate_dims,
        )

        self.assert_raises_exact(
            TypeError,
            "shared_halting_config must be an instance of HaltingConfig for "
            "LayerStackConfig, got object",
            LayerStackValidator._validate_shared_halting_config_type,
            object(),
        )
        halting_conflict = _stack_config(
            layer_config=_layer_config(halting_config=_halting_config()),
            shared_halting_config=_halting_config(),
        )
        self.assert_raises_exact(
            ValueError,
            "shared_halting_config and layer_config.halting_config are mutually "
            "exclusive. Put shared halting controllers on LayerStackConfig and "
            "per-layer halting controllers on LayerConfig.",
            LayerStackValidator._validate_halting_config,
            halting_conflict,
        )
        one_step_halting = _stack_config(
            num_layers=1,
            layer_config=_layer_config(halting_config=_halting_config()),
        )
        self.assert_raises_exact(
            ValueError,
            "num_layers must be at least 2 when halting_config is provided, got "
            "1. The halting mechanism requires multiple steps to accumulate "
            "halting probabilities across layers.",
            LayerStackValidator._validate_halting_config,
            one_step_halting,
        )
        mismatched_halting = _stack_config(
            input_dim=2,
            hidden_dim=3,
            output_dim=4,
            layer_config=_layer_config(halting_config=_halting_config()),
        )
        self.assert_raises_exact(
            ValueError,
            "input_dim, hidden_dim, and output_dim must all be equal when "
            "halting_config is provided, got input_dim=2, hidden_dim=3, "
            "output_dim=4. Halting accumulates hidden states across steps, "
            "which requires consistent dimensions.",
            LayerStackValidator._validate_halting_config,
            mismatched_halting,
        )

        self.assert_raises_exact(
            TypeError,
            "shared_memory_config must be an instance of DynamicMemoryConfig for "
            "LayerStackConfig, got object",
            LayerStackValidator._validate_shared_memory_config_type,
            object(),
        )
        memory_conflict = _stack_config(
            layer_config=_layer_config(memory_config=_memory_config()),
            shared_memory_config=_memory_config(),
        )
        self.assert_raises_exact(
            ValueError,
            "shared_memory_config and layer_config.memory_config are mutually "
            "exclusive. Put shared memory controllers on LayerStackConfig and "
            "per-layer memory controllers on LayerConfig.",
            LayerStackValidator._validate_memory_config,
            memory_conflict,
        )
        mismatched_memory = _stack_config(
            input_dim=2,
            hidden_dim=3,
            output_dim=4,
            shared_memory_config=_memory_config(),
        )
        self.assert_raises_exact(
            ValueError,
            "input_dim, hidden_dim, and output_dim must all be equal when "
            "shared_memory_config is provided, got input_dim=2, hidden_dim=3, "
            "output_dim=4. Shared memory uses one module across all layers and "
            "requires consistent dimensions.",
            LayerStackValidator._validate_memory_config,
            mismatched_memory,
        )
        for input_dim, hidden_dim, output_dim in (
            (2, 3, 3),
            (2, 2, 3),
        ):
            with self.subTest(
                controller="shared_memory",
                dimensions=(input_dim, hidden_dim, output_dim),
            ):
                one_mismatched_memory_dimension = _stack_config(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    shared_memory_config=_memory_config(),
                )
                self.assert_raises_exact(
                    ValueError,
                    "input_dim, hidden_dim, and output_dim must all be equal when "
                    "shared_memory_config is provided, "
                    f"got input_dim={input_dim}, hidden_dim={hidden_dim}, "
                    f"output_dim={output_dim}. Shared memory uses one module "
                    "across all layers and requires consistent dimensions.",
                    LayerStackValidator._validate_memory_config,
                    one_mismatched_memory_dimension,
                )

    def test_recurrent_validator_configuration_errors_are_exact(self) -> None:
        for field_name in ("input_dim", "output_dim", "max_steps"):
            config = _recurrent_config()
            setattr(config, field_name, 2.5)
            with self.subTest(field_name=field_name, case="type"):
                self.assert_raises_exact(
                    TypeError,
                    f"{field_name} must be int for RecurrentLayerConfig, got float",
                    RecurrentLayerValidator.validate,
                    SimpleNamespace(cfg=config),
                )
            config = _recurrent_config()
            setattr(config, field_name, 0)
            with self.subTest(field_name=field_name, case="dimension"):
                self.assert_raises_exact(
                    ValueError,
                    f"{field_name} must be greater than 0, received 0",
                    RecurrentLayerValidator.validate,
                    SimpleNamespace(cfg=config),
                )

        self.assert_raises_exact(
            ValueError,
            "input_dim and output_dim must be equal for RecurrentLayerConfig, "
            "got input_dim=2 and output_dim=3.",
            RecurrentLayerValidator._validate_stable_dimensions,
            2,
            3,
        )
        self.assert_raises_exact(
            TypeError,
            "recurrent_layer_norm_position must be a LayerNormPositionOptions "
            "value for RecurrentLayerConfig, got object",
            RecurrentLayerValidator._validate_recurrent_layer_norm_position,
            object(),
        )
        self.assert_raises_exact(
            TypeError,
            "block_config must be an instance of ConfigBase for "
            "RecurrentLayerConfig, got object",
            RecurrentLayerValidator._validate_block_config,
            object(),
        )
        for block_config, missing in (
            (ConfigBase(), "input_dim, output_dim"),
            (InputOnlyBlockConfig(input_dim=2), "output_dim"),
            (OutputOnlyBlockConfig(output_dim=2), "input_dim"),
        ):
            with self.subTest(missing=missing):
                self.assert_raises_exact(
                    TypeError,
                    "block_config must declare dataclass fields input_dim and "
                    "output_dim for RecurrentLayerConfig; "
                    f"{type(block_config).__name__} is missing {missing}",
                    RecurrentLayerValidator._validate_block_config,
                    block_config,
                )
        self.assert_raises_exact(
            TypeError,
            "residual_config must be an instance of ResidualConfig for "
            "RecurrentLayerConfig, got object",
            RecurrentLayerValidator._validate_residual_config,
            object(),
        )
        self.assert_raises_exact(
            TypeError,
            "halting_config must be an instance of HaltingConfig for "
            "RecurrentLayerConfig, got object",
            RecurrentLayerValidator._validate_halting_config,
            object(),
        )
        self.assert_raises_exact(
            ValueError,
            "halting_config must be a concrete halting config for RecurrentLayerConfig",
            RecurrentLayerValidator._validate_halting_config,
            HaltingConfig(),
        )
        self.assert_raises_exact(
            ValueError,
            "halting_config SoftHaltingConfig builds SoftHalting, which does not "
            "implement the HaltingInterface required by RecurrentLayerConfig",
            RecurrentLayerValidator._validate_halting_config,
            _soft_halting_config(),
        )
        RecurrentLayerValidator._validate_halting_config(_halting_config())
        self.assert_raises_exact(
            TypeError,
            "memory_config must be an instance of DynamicMemoryConfig for "
            "RecurrentLayerConfig, got object",
            RecurrentLayerValidator._validate_memory_config,
            object(),
        )

    def test_recurrent_state_hidden_and_candidate_errors_are_exact(self) -> None:
        self.assert_raises_exact(
            TypeError,
            "state must be an instance of LayerState for RecurrentLayer, got object",
            RecurrentLayerValidator.validate_state,
            object(),
            3,
        )
        self.assert_raises_exact(
            ValueError,
            "state.hidden must have rank >= 2 with feature-last layout, got 1D "
            "tensor with shape (3,)",
            RecurrentLayerValidator.validate_state,
            LayerState(hidden=torch.zeros(3)),
            3,
        )
        self.assert_raises_exact(
            ValueError,
            "state.hidden last dimension must be 3 for RecurrentLayer, got 4 "
            "with shape (2, 4)",
            RecurrentLayerValidator.validate_state,
            LayerState(hidden=torch.zeros(2, 4)),
            3,
        )
        self.assert_raises_exact(
            ValueError,
            "hidden must have rank >= 2 with feature-last layout, got 1D tensor "
            "with shape (4,)",
            RecurrentLayerValidator.validate_hidden,
            torch.zeros(4),
            4,
        )
        self.assert_raises_exact(
            ValueError,
            "candidate last dimension must be 3 for RecurrentLayer, got 4 with "
            "shape (2, 4)",
            RecurrentLayerValidator.validate_hidden,
            torch.zeros(2, 4),
            3,
            "candidate",
        )
        self.assert_raises_exact(
            ValueError,
            "recurrent block must preserve hidden shape, got candidate shape "
            "(3, 3) and previous shape (2, 3)",
            RecurrentLayerValidator.validate_candidate,
            torch.zeros(3, 3),
            torch.zeros(2, 3),
            3,
        )
        self.assert_raises_exact(
            ValueError,
            "hidden last dimension must be 3 for RecurrentLayer, got 4 with "
            "shape (2, 4)",
            RecurrentLayerValidator.validate_candidate,
            torch.zeros(2, 4),
            torch.zeros(2, 3),
            3,
        )


if __name__ == "__main__":
    unittest.main()
