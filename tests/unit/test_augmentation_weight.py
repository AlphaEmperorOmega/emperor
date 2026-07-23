import copy
import unittest

import torch
import torch.nn.functional as F

from emperor.augmentations.adaptive_parameters import (
    BankExpansionFactorOptions,
    DualModelDynamicWeightConfig,
    DynamicDepthOptions,
    DynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters._weights.variants.dual_model import (
    DualModelDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.hypernetwork import (
    HypernetworkDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.layered_weighted_bank import (  # noqa: E501
    LayeredWeightedBankDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.low_rank import (
    LowRankDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.single_model import (
    SingleModelDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.soft_weighted_bank import (  # noqa: E501
    SoftWeightedBankDynamicWeight,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.nn import Module


class TestWeightHandlerForward(unittest.TestCase):
    def preset(
        self,
        config_cls: type[DynamicWeightConfig] = DualModelDynamicWeightConfig,
        input_dim: int = 12,
        hidden_dim: int = 36,
        output_dim: int = 24,
        bias_flag: bool = True,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
        normalization_option: WeightNormalizationOptions = (
            WeightNormalizationOptions.L2_SCALE
        ),
        normalization_position_option: WeightNormalizationPositionOptions = (
            WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
        ),
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_dropout_probability: float = 0.1,
        bank_expansion_factor: BankExpansionFactorOptions | None = None,
        decay_schedule: WeightDecayScheduleOptions = (
            WeightDecayScheduleOptions.DISABLED
        ),
        decay_rate: float = 0.0,
        decay_warmup_batches: int = 0,
    ) -> DynamicWeightConfig:
        common_kwargs = dict(
            input_dim=input_dim,
            output_dim=output_dim,
            generator_depth=generator_depth,
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
            model_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=2,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_pipeline_flag=apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    activation=stack_activation,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=stack_dropout_probability,
                    gate_config=None,
                    halting_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                    ),
                ),
            ),
        )
        if config_cls in {
            SingleModelDynamicWeightConfig,
            DualModelDynamicWeightConfig,
        }:
            return config_cls(
                **common_kwargs,
                normalization_option=normalization_option,
                normalization_position_option=normalization_position_option,
            )
        if config_cls in {
            LowRankDynamicWeightConfig,
            HypernetworkDynamicWeightConfig,
        }:
            return config_cls(
                **common_kwargs,
                normalization_option=normalization_option,
            )
        if config_cls in {
            LayeredWeightedBankDynamicWeightConfig,
            SoftWeightedBankDynamicWeightConfig,
        }:
            return config_cls(
                **common_kwargs,
                bank_expansion_factor=bank_expansion_factor,
            )
        return config_cls(**common_kwargs)

    def weight_cases(
        self,
    ) -> list[tuple[type[DynamicWeightConfig], type, int]]:
        input_dim = 12
        output_dim = 24
        return [
            (
                SingleModelDynamicWeightConfig,
                SingleModelDynamicWeight,
                input_dim,
            ),
            (
                DualModelDynamicWeightConfig,
                DualModelDynamicWeight,
                output_dim,
            ),
            (
                LowRankDynamicWeightConfig,
                LowRankDynamicWeight,
                output_dim,
            ),
            (
                HypernetworkDynamicWeightConfig,
                HypernetworkDynamicWeight,
                output_dim,
            ),
            (
                LayeredWeightedBankDynamicWeightConfig,
                LayeredWeightedBankDynamicWeight,
                output_dim,
            ),
            (
                SoftWeightedBankDynamicWeightConfig,
                SoftWeightedBankDynamicWeight,
                output_dim,
            ),
        ]

    def test_single_model_handler_forward(self):
        batch_size = 2
        dim = 12
        cfg = self.preset(
            config_cls=SingleModelDynamicWeightConfig,
            input_dim=dim,
            output_dim=dim,
        )
        weight_params = Module()._init_parameter_bank((dim, dim))
        input_tensor = torch.randn(batch_size, dim)
        model = SingleModelDynamicWeight(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, dim, dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))

    def test_dual_model_handler_forward(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            config_cls=DualModelDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        input_tensor = torch.randn(batch_size, input_dim)
        model = DualModelDynamicWeight(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))

    def test_low_rank_handler_forward(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            config_cls=LowRankDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        input_tensor = torch.randn(batch_size, input_dim)
        model = LowRankDynamicWeight(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))

    def test_hypernetwork_handler_forward(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            config_cls=HypernetworkDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        input_tensor = torch.randn(batch_size, input_dim)
        model = HypernetworkDynamicWeight(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))

    def test_weighted_bank_handler_forward(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        bank_expansion_factor = BankExpansionFactorOptions.FACTOR_OF_THREE
        cfg = self.preset(
            config_cls=LayeredWeightedBankDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
            generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
            bank_expansion_factor=bank_expansion_factor,
        )
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        input_tensor = torch.randn(batch_size, input_dim)
        model = LayeredWeightedBankDynamicWeight(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))

    def test_soft_weighted_bank_handler_forward(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        bank_expansion_factor = BankExpansionFactorOptions.FACTOR_OF_THREE
        cfg = self.preset(
            config_cls=SoftWeightedBankDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
            generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
            bank_expansion_factor=bank_expansion_factor,
        )
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        input_tensor = torch.randn(batch_size, input_dim)
        model = SoftWeightedBankDynamicWeight(cfg)
        output = model(weight_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(torch.all(output == 0))

    def test_compute_raw_outer_product_values(self):
        batch_size = 2
        generator_depth = 3
        input_dim = 4
        output_dim = 5
        cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
        model = DualModelDynamicWeight(cfg)
        input_vectors = torch.randn(batch_size, generator_depth, input_dim)
        output_vectors = torch.randn(batch_size, generator_depth, output_dim)
        result = model._compute_raw_outer_product(input_vectors, output_vectors)
        self.assertEqual(
            result.shape, (batch_size, generator_depth, input_dim, output_dim)
        )
        for b in range(batch_size):
            for k in range(generator_depth):
                expected = input_vectors[b, k].unsqueeze(1) * output_vectors[
                    b, k
                ].unsqueeze(0)
                self.assertTrue(
                    torch.allclose(result[b, k], expected, atol=1e-6),
                    f"batch={b}, depth={k}",
                )

    def test_compute_dynamic_weights_sums_over_depth(self):
        batch_size = 2
        generator_depth = 3
        input_dim = 4
        output_dim = 5
        outer_product = torch.randn(batch_size, generator_depth, input_dim, output_dim)
        cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
        model = DualModelDynamicWeight(cfg)
        result = model._compute_dynamic_weights(outer_product)
        expected = outer_product.sum(dim=1)
        self.assertEqual(result.shape, (batch_size, input_dim, output_dim))
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_compute_outer_product_raises_on_unknown_position(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            normalization_option=WeightNormalizationOptions.L2_SCALE,
            normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
        )
        model = DualModelDynamicWeight(cfg)
        model.normalization_position_option = "invalid_position"
        input_vectors = torch.randn(batch_size, generator_depth, input_dim)
        output_vectors = torch.randn(batch_size, generator_depth, output_dim)
        with self.assertRaises(ValueError):
            model._compute_outer_product(input_vectors, output_vectors)

    def test_normalization_position_after_outer_product(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        valid_normalizations = [
            WeightNormalizationOptions.L2_SCALE,
            WeightNormalizationOptions.RMS,
            WeightNormalizationOptions.CLAMP,
            WeightNormalizationOptions.SOFT_CLAMP,
            WeightNormalizationOptions.SIGMOID_SCALE,
        ]
        for normalization in valid_normalizations:
            message = f"normalization={normalization}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    normalization_option=normalization,
                    normalization_position_option=WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
                )
                model = DualModelDynamicWeight(cfg)
                input_vectors = torch.randn(batch_size, generator_depth, input_dim)
                output_vectors = torch.randn(batch_size, generator_depth, output_dim)
                result = model._compute_outer_product(input_vectors, output_vectors)
                raw = model._compute_raw_outer_product(input_vectors, output_vectors)
                expected = model._apply_normalization_transform(raw)
                self.assertTrue(torch.equal(result, expected))

    def test_normalization_position_before_outer_product(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        valid_normalizations = [
            WeightNormalizationOptions.L2_SCALE,
            WeightNormalizationOptions.RMS,
            WeightNormalizationOptions.CLAMP,
            WeightNormalizationOptions.SOFT_CLAMP,
            WeightNormalizationOptions.SIGMOID_SCALE,
        ]
        for normalization in valid_normalizations:
            message = f"normalization={normalization}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    normalization_option=normalization,
                    normalization_position_option=WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT,
                )
                model = DualModelDynamicWeight(cfg)
                input_vectors = torch.randn(batch_size, generator_depth, input_dim)
                output_vectors = torch.randn(batch_size, generator_depth, output_dim)
                result = model._compute_outer_product(input_vectors, output_vectors)
                normalized_input = model._apply_normalization_transform(input_vectors)
                normalized_output = model._apply_normalization_transform(output_vectors)
                expected = model._compute_raw_outer_product(
                    normalized_input, normalized_output
                )
                self.assertTrue(torch.equal(result, expected))

    def test_normalization_position_disabled_uses_raw_outer_product(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            normalization_option=WeightNormalizationOptions.L2_SCALE,
            normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
        )
        model = DualModelDynamicWeight(cfg)
        input_vectors = torch.randn(batch_size, generator_depth, input_dim)
        output_vectors = torch.randn(batch_size, generator_depth, output_dim)
        result = model._compute_outer_product(input_vectors, output_vectors)
        expected = model._compute_raw_outer_product(input_vectors, output_vectors)
        self.assertTrue(torch.equal(result, expected))

    def test_build_creates_model_for_each_leaf_config(self):
        input_dim = 12
        for config_cls, model_cls, effective_output_dim in self.weight_cases():
            with self.subTest(config_cls=config_cls.__name__):
                bank_factor = (
                    BankExpansionFactorOptions.FACTOR_OF_TWO
                    if config_cls
                    in {
                        LayeredWeightedBankDynamicWeightConfig,
                        SoftWeightedBankDynamicWeightConfig,
                    }
                    else None
                )
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=input_dim,
                    output_dim=effective_output_dim,
                    bank_expansion_factor=bank_factor,
                )
                model = cfg.build()
                self.assertIsInstance(model, model_cls)
                weight_shape = (input_dim, effective_output_dim)
                expected_shape = (2, input_dim, effective_output_dim)
                weight_params = Module()._init_parameter_bank(weight_shape)
                input_tensor = torch.randn(2, input_dim)
                output = model(weight_params, input_tensor)
                self.assertEqual(output.shape, expected_shape)

    def test_abstract_config_cannot_build(self):
        cfg = self.preset(config_cls=DynamicWeightConfig)
        with self.assertRaises(ValueError) as error:
            cfg.build()
        self.assertEqual(
            str(error.exception),
            "DynamicWeightConfig is abstract and has no registered "
            "DynamicWeight class; instantiate a concrete leaf config instead.",
        )

    def test_gradients_flow(self):
        batch_size = 2
        input_dim = 12
        for config_cls, _, effective_output_dim in self.weight_cases():
            with self.subTest(config_cls=config_cls.__name__):
                effective_weight_params = Module()._init_parameter_bank(
                    (input_dim, effective_output_dim)
                )
                expected_shape = (batch_size, input_dim, effective_output_dim)
                bank_factor = (
                    BankExpansionFactorOptions.FACTOR_OF_TWO
                    if config_cls
                    in {
                        LayeredWeightedBankDynamicWeightConfig,
                        SoftWeightedBankDynamicWeightConfig,
                    }
                    else None
                )
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=input_dim,
                    output_dim=effective_output_dim,
                    bank_expansion_factor=bank_factor,
                )
                model = cfg.build()
                input_tensor = torch.randn(batch_size, input_dim, requires_grad=True)
                output = model(effective_weight_params, input_tensor)
                self.assertEqual(output.shape, expected_shape)
                output.sum().backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad]
                non_none_grads = [g for g in grads if g is not None]
                self.assertTrue(len(non_none_grads) > 0)

    def test_each_leaf_preserves_float64_and_backpropagates_to_both_inputs(self):
        input_dim = 2
        batch_size = 3
        cases = (
            (SingleModelDynamicWeightConfig, SingleModelDynamicWeight, input_dim),
            (DualModelDynamicWeightConfig, DualModelDynamicWeight, 3),
            (LowRankDynamicWeightConfig, LowRankDynamicWeight, 3),
            (HypernetworkDynamicWeightConfig, HypernetworkDynamicWeight, 3),
            (
                LayeredWeightedBankDynamicWeightConfig,
                LayeredWeightedBankDynamicWeight,
                3,
            ),
            (
                SoftWeightedBankDynamicWeightConfig,
                SoftWeightedBankDynamicWeight,
                3,
            ),
        )

        for config_type, model_type, output_dim in cases:
            with self.subTest(model_type=model_type.__name__):
                torch.manual_seed(17)
                is_bank = config_type in {
                    LayeredWeightedBankDynamicWeightConfig,
                    SoftWeightedBankDynamicWeightConfig,
                }
                config = self.preset(
                    config_cls=config_type,
                    input_dim=input_dim,
                    hidden_dim=4,
                    output_dim=output_dim,
                    generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
                    apply_output_pipeline_flag=False,
                    normalization_option=WeightNormalizationOptions.DISABLED,
                    normalization_position_option=(
                        WeightNormalizationPositionOptions.DISABLED
                    ),
                    stack_activation=ActivationOptions.DISABLED,
                    stack_dropout_probability=0.0,
                    bank_expansion_factor=(
                        BankExpansionFactorOptions.FACTOR_OF_TWO if is_bank else None
                    ),
                )
                model = model_type(config).double()
                for state_value in model.state_dict().values():
                    if torch.is_floating_point(state_value):
                        self.assertEqual(state_value.dtype, torch.float64)
                        self.assertEqual(state_value.device.type, "cpu")
                weight_params = torch.randn(
                    input_dim,
                    output_dim,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                input_tensor = torch.randn(
                    batch_size,
                    input_dim,
                    dtype=torch.float64,
                    requires_grad=True,
                )

                output = model(weight_params, input_tensor)
                output.square().sum().backward()

                self.assertEqual(output.dtype, torch.float64)
                self.assertEqual(output.device.type, "cpu")
                self.assertTrue(torch.isfinite(output).all())
                for gradient in (weight_params.grad, input_tensor.grad):
                    self.assertIsNotNone(gradient)
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertTrue(torch.any(gradient != 0))
                active_model_gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if name not in {"scale", "clamp_limit"}
                    and parameter.grad is not None
                ]
                self.assertTrue(active_model_gradients)
                self.assertTrue(
                    any(torch.any(gradient != 0) for gradient in active_model_gradients)
                )
                if is_bank:
                    self.assertIsNotNone(model.weight_bank.grad)
                    self.assertTrue(torch.isfinite(model.weight_bank.grad).all())
                    self.assertTrue(torch.any(model.weight_bank.grad != 0))

    def test_active_normalization_parameters_receive_finite_nonzero_gradients(self):
        config = self.preset(
            input_dim=3,
            output_dim=4,
            normalization_option=WeightNormalizationOptions.DISABLED,
        )
        model = DualModelDynamicWeight(config).double()
        vectors = torch.tensor(
            [[[-3.0, -0.5, 1.5], [0.25, 2.0, -4.0]]],
            dtype=torch.float64,
        )
        cases = (
            (WeightNormalizationOptions.SIGMOID_SCALE, model.scale),
            (WeightNormalizationOptions.SOFT_CLAMP, model.clamp_limit),
        )

        for option, active_parameter in cases:
            with self.subTest(option=option):
                model.zero_grad(set_to_none=True)
                model.normalization_option = option
                differentiable_vectors = vectors.clone().requires_grad_()

                transformed = model._apply_normalization_transform(
                    differentiable_vectors
                )
                transformed.square().sum().backward()

                self.assertIsNotNone(active_parameter.grad)
                self.assertTrue(torch.isfinite(active_parameter.grad).all())
                self.assertTrue(torch.any(active_parameter.grad != 0))
                self.assertIsNotNone(differentiable_vectors.grad)
                self.assertTrue(torch.isfinite(differentiable_vectors.grad).all())
                self.assertTrue(torch.any(differentiable_vectors.grad != 0))

    def test_each_leaf_strictly_restores_model_and_adam_state(self):
        input_dim = 2
        cases = (
            (SingleModelDynamicWeightConfig, input_dim),
            (DualModelDynamicWeightConfig, 3),
            (LowRankDynamicWeightConfig, 3),
            (HypernetworkDynamicWeightConfig, 3),
            (LayeredWeightedBankDynamicWeightConfig, 3),
            (SoftWeightedBankDynamicWeightConfig, 3),
        )

        for config_type, output_dim in cases:
            with self.subTest(config_type=config_type.__name__):
                torch.manual_seed(23)
                is_bank = config_type in {
                    LayeredWeightedBankDynamicWeightConfig,
                    SoftWeightedBankDynamicWeightConfig,
                }
                config = self.preset(
                    config_cls=config_type,
                    input_dim=input_dim,
                    hidden_dim=4,
                    output_dim=output_dim,
                    generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
                    apply_output_pipeline_flag=False,
                    normalization_option=WeightNormalizationOptions.DISABLED,
                    normalization_position_option=(
                        WeightNormalizationPositionOptions.DISABLED
                    ),
                    stack_activation=ActivationOptions.DISABLED,
                    stack_dropout_probability=0.0,
                    bank_expansion_factor=(
                        BankExpansionFactorOptions.FACTOR_OF_TWO if is_bank else None
                    ),
                    decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
                    decay_rate=0.2,
                    decay_warmup_batches=0,
                )
                source = config.build().double()
                source_optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
                weight_params = torch.tensor(
                    [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]],
                    dtype=torch.float64,
                )[:, :output_dim]
                first_input = torch.tensor(
                    [[1.0, 2.0], [-0.5, 3.0]],
                    dtype=torch.float64,
                )
                continuation_input = torch.tensor(
                    [[2.0, -1.0], [0.75, 0.5]],
                    dtype=torch.float64,
                )

                def training_step(
                    model,
                    optimizer,
                    input_tensor,
                    base_weight_params,
                ):
                    optimizer.zero_grad()
                    output = model(base_weight_params, input_tensor)
                    output.square().mean().backward()
                    optimizer.step()
                    return output.detach()

                training_step(source, source_optimizer, first_input, weight_params)
                model_state = copy.deepcopy(source.state_dict())
                optimizer_state = copy.deepcopy(source_optimizer.state_dict())

                torch.manual_seed(101)
                restored = config.build().double()
                incompatible = restored.load_state_dict(model_state, strict=True)
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])
                restored_optimizer = torch.optim.Adam(
                    restored.parameters(),
                    lr=0.01,
                )
                restored_optimizer.load_state_dict(optimizer_state)

                self.assertTupleEqual(
                    tuple(source.state_dict()),
                    tuple(restored.state_dict()),
                )
                self.assertTupleEqual(
                    tuple(dict(source.named_parameters())),
                    tuple(dict(restored.named_parameters())),
                )
                source_output = training_step(
                    source,
                    source_optimizer,
                    continuation_input,
                    weight_params,
                )
                restored_output = training_step(
                    restored,
                    restored_optimizer,
                    continuation_input,
                    weight_params,
                )
                torch.testing.assert_close(restored_output, source_output)
                for name, source_value in source.state_dict().items():
                    torch.testing.assert_close(
                        restored.state_dict()[name],
                        source_value,
                    )

                source_optimizer_state = source_optimizer.state_dict()
                restored_optimizer_state = restored_optimizer.state_dict()
                self.assertEqual(
                    source_optimizer_state["param_groups"],
                    restored_optimizer_state["param_groups"],
                )
                for source_values, restored_values in zip(
                    source_optimizer_state["state"].values(),
                    restored_optimizer_state["state"].values(),
                    strict=True,
                ):
                    self.assertEqual(source_values.keys(), restored_values.keys())
                    for key, source_value in source_values.items():
                        restored_value = restored_values[key]
                        if torch.is_tensor(source_value):
                            torch.testing.assert_close(restored_value, source_value)
                        else:
                            self.assertEqual(restored_value, source_value)

    def test_generator_depth_options(self):
        batch_size = 2
        input_dim = 12
        for config_cls, _, effective_output_dim in self.weight_cases():
            is_bank_type = config_cls in {
                LayeredWeightedBankDynamicWeightConfig,
                SoftWeightedBankDynamicWeightConfig,
            }
            bank_factor = (
                BankExpansionFactorOptions.FACTOR_OF_TWO if is_bank_type else None
            )
            weight_shape = (input_dim, effective_output_dim)
            expected_shape = (batch_size, input_dim, effective_output_dim)
            weight_params = Module()._init_parameter_bank(weight_shape)
            for depth in DynamicDepthOptions:
                if depth == DynamicDepthOptions.DISABLED:
                    continue
                msg = f"config_cls={config_cls.__name__}, depth={depth}"
                with self.subTest(msg=msg):
                    cfg = self.preset(
                        config_cls=config_cls,
                        input_dim=input_dim,
                        output_dim=effective_output_dim,
                        generator_depth=depth,
                        bank_expansion_factor=bank_factor,
                    )
                    model = cfg.build()
                    input_tensor = torch.randn(batch_size, input_dim)
                    output = model(weight_params, input_tensor)
                    self.assertEqual(output.shape, expected_shape)

    def test_bank_expansion_factor_variants(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        bank_config_classes = [
            LayeredWeightedBankDynamicWeightConfig,
            SoftWeightedBankDynamicWeightConfig,
        ]
        for config_cls in bank_config_classes:
            for factor in BankExpansionFactorOptions:
                is_disabled = factor == BankExpansionFactorOptions.DISABLED
                msg = f"config_cls={config_cls.__name__}, factor={factor}"
                with self.subTest(msg=msg):
                    cfg = self.preset(
                        config_cls=config_cls,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bank_expansion_factor=factor,
                    )
                    if is_disabled:
                        with self.assertRaises(ValueError):
                            cfg.build()
                    else:
                        model = cfg.build()
                        input_tensor = torch.randn(batch_size, input_dim)
                        output = model(weight_params, input_tensor)
                        expected_shape = (
                            batch_size,
                            input_dim,
                            output_dim,
                        )
                        self.assertEqual(output.shape, expected_shape)

    def test_bank_expansion_factor_field_absent_on_non_bank_leaves(self):
        non_bank_leaf_classes = [
            SingleModelDynamicWeightConfig,
            DualModelDynamicWeightConfig,
            LowRankDynamicWeightConfig,
            HypernetworkDynamicWeightConfig,
        ]
        for leaf_cls in non_bank_leaf_classes:
            with self.subTest(leaf=leaf_cls.__name__):
                with self.assertRaises(TypeError):
                    leaf_cls(
                        bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO
                    )

    def test_normalization_fields_absent_on_bank_leaf_configs(self):
        bank_leaf_classes = [
            LayeredWeightedBankDynamicWeightConfig,
            SoftWeightedBankDynamicWeightConfig,
        ]
        for leaf_cls in bank_leaf_classes:
            with self.subTest(leaf=leaf_cls.__name__, field="normalization_option"):
                with self.assertRaises(TypeError):
                    leaf_cls(normalization_option=WeightNormalizationOptions.L2_SCALE)
            with self.subTest(
                leaf=leaf_cls.__name__, field="normalization_position_option"
            ):
                with self.assertRaises(TypeError):
                    leaf_cls(
                        normalization_position_option=WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
                    )

    def test_normalization_position_field_absent_on_low_rank_and_hypernetwork(self):
        leaf_classes = [
            LowRankDynamicWeightConfig,
            HypernetworkDynamicWeightConfig,
        ]
        for leaf_cls in leaf_classes:
            with self.subTest(leaf=leaf_cls.__name__):
                with self.assertRaises(TypeError):
                    leaf_cls(
                        normalization_position_option=WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
                    )

    def test_soft_weighted_bank_squashes_all_expanded_rows_per_output_row(self):
        input_dim = 2
        output_dim = 3
        batch_size = 2
        depth = DynamicDepthOptions.DEPTH_OF_ONE
        bank_factor = BankExpansionFactorOptions.FACTOR_OF_TWO
        expanded_bank_rows = input_dim * bank_factor.value
        cfg = self.preset(
            config_cls=SoftWeightedBankDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=6,
            generator_depth=depth,
            bank_expansion_factor=bank_factor,
        )
        model = SoftWeightedBankDynamicWeight(cfg)
        weight_params = torch.zeros(input_dim, output_dim)
        input_tensor = torch.randn(batch_size, input_dim)
        generated_logits = model.model(input_tensor)
        self.assertEqual(
            generated_logits.shape,
            (
                batch_size,
                depth.value,
                input_dim * expanded_bank_rows,
            ),
        )
        self.assertEqual(
            model.weight_bank.shape,
            (depth.value, expanded_bank_rows, output_dim),
        )
        raw_logits = torch.tensor(
            [[[-8.0, -8.0, -8.0, 8.0, 8.0, -8.0, -8.0, -8.0]]],
            dtype=weight_params.dtype,
        )

        class StaticDepthMapper(torch.nn.Module):
            def __init__(self, logits: torch.Tensor):
                super().__init__()
                self.logits = logits

            def forward(self, X: torch.Tensor) -> torch.Tensor:
                return self.logits.expand(X.size(0), -1, -1)

        model.model = StaticDepthMapper(raw_logits)
        with torch.no_grad():
            model.weight_bank.copy_(
                torch.tensor(
                    [
                        [
                            [1.0, 2.0, 3.0],
                            [10.0, 20.0, 30.0],
                            [4.0, 5.0, 6.0],
                            [40.0, 50.0, 60.0],
                        ]
                    ],
                    dtype=weight_params.dtype,
                )
            )

        output = model(weight_params, input_tensor)
        bank_logits = model.model(input_tensor).view(
            batch_size,
            depth.value,
            input_dim,
            expanded_bank_rows,
        )
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        expected_update = torch.einsum(
            "bdim,dmo->bdio", bank_distribution, model.weight_bank
        ).sum(dim=1)
        torch.testing.assert_close(output, expected_update)
        torch.testing.assert_close(
            bank_distribution.sum(dim=-1),
            torch.ones(batch_size, depth.value, input_dim),
        )

    def test_layered_weighted_bank_forward_applies_depth_and_factor_reduction(self):
        input_dim = 2
        output_dim = 3
        batch_size = 2
        depth = DynamicDepthOptions.DEPTH_OF_ONE
        bank_factor = BankExpansionFactorOptions.FACTOR_OF_TWO
        cfg = self.preset(
            config_cls=LayeredWeightedBankDynamicWeightConfig,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=6,
            generator_depth=depth,
            bank_expansion_factor=bank_factor,
        )
        model = LayeredWeightedBankDynamicWeight(cfg)
        weight_params = torch.zeros(input_dim, output_dim)
        input_tensor = torch.randn(batch_size, input_dim)
        raw_logits = torch.tensor([[[2.0, -1.0, -0.5, 1.5]]])

        class StaticDepthMapper(torch.nn.Module):
            def __init__(self, logits: torch.Tensor):
                super().__init__()
                self.logits = logits

            def forward(self, X: torch.Tensor) -> torch.Tensor:
                return self.logits.expand(X.size(0), -1, -1)

        model.model = StaticDepthMapper(raw_logits)
        model.weight_bank.data = torch.tensor(
            [
                [
                    [
                        [1.0, 2.0, 3.0],
                        [10.0, 20.0, 30.0],
                        [4.0, 5.0, 6.0],
                        [40.0, 50.0, 60.0],
                    ]
                ]
            ]
        )

        output = model(weight_params, input_tensor)
        bank_distribution = torch.softmax(model.model(input_tensor), dim=-1)
        weighted_bank = model.weight_bank * bank_distribution.unsqueeze(-1)
        expected = weighted_bank.view(
            batch_size,
            depth.value,
            input_dim,
            bank_factor.value,
            output_dim,
        ).sum(dim=(1, 3))

        torch.testing.assert_close(output, expected)

    def test_decay_schedule_mathematical_correctness(self):
        input_dim = 12
        output_dim = 24
        decay_rate = 0.3
        num_steps = 5
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))

        schedule_expected_factor = {
            WeightDecayScheduleOptions.EXPONENTIAL: lambda step: torch.exp(
                torch.tensor(-decay_rate * step)
            ),
            WeightDecayScheduleOptions.LINEAR: lambda step: torch.clamp(
                torch.tensor(1.0 - decay_rate * step), min=0.0
            ),
            WeightDecayScheduleOptions.MULTIPLICATIVE: lambda step: torch.pow(
                torch.tensor(1.0 - decay_rate), torch.tensor(float(step))
            ),
        }

        for schedule, expected_factor_fn in schedule_expected_factor.items():
            message = f"schedule={schedule}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    decay_schedule=schedule,
                    decay_rate=decay_rate,
                )
                model = cfg.build()
                for step in range(num_steps):
                    result = model._maybe_apply_weight_decay(weight_params)
                    expected_factor = expected_factor_fn(step)
                    expected = weight_params * expected_factor
                    self.assertTrue(
                        torch.allclose(result, expected, atol=1e-6),
                        f"{message}, step={step}: expected factor="
                        f"{expected_factor.item():.6f}",
                    )

    def test_exponential_weight_decay_saturates_huge_rate_to_active_dtype(self):
        weight_params = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        cfg = self.preset(
            input_dim=2,
            output_dim=2,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=1.0e100,
        )
        model = cfg.build()

        initial = model._maybe_apply_weight_decay(weight_params)
        decayed = model._maybe_apply_weight_decay(weight_params)

        torch.testing.assert_close(initial, weight_params)
        torch.testing.assert_close(decayed, torch.zeros_like(weight_params))
        self.assertEqual(initial.dtype, weight_params.dtype)
        self.assertEqual(decayed.dtype, weight_params.dtype)
        self.assertTrue(torch.isfinite(initial).all())
        self.assertTrue(torch.isfinite(decayed).all())

    def test_decay_schedule_disabled_leaves_weights_unchanged(self):
        input_dim = 12
        output_dim = 24
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.3,
        )
        model = cfg.build()
        baseline_decay_step = model.decay_step.clone()
        baseline_warmup_step = model.warmup_step.clone()

        for _ in range(3):
            result = model._maybe_apply_weight_decay(weight_params)
            self.assertTrue(torch.equal(result, weight_params))

        self.assertTrue(torch.equal(model.decay_step, baseline_decay_step))
        self.assertTrue(torch.equal(model.warmup_step, baseline_warmup_step))

    def test_invalid_decay_parameters_raise(self):
        invalid_cases = [
            ("missing_rate", WeightDecayScheduleOptions.EXPONENTIAL, None, 0),
            ("zero_rate", WeightDecayScheduleOptions.EXPONENTIAL, 0.0, 0),
            ("negative_rate", WeightDecayScheduleOptions.EXPONENTIAL, -0.1, 0),
            ("linear_rate_too_large", WeightDecayScheduleOptions.LINEAR, 1.0, 0),
            (
                "multiplicative_rate_too_large",
                WeightDecayScheduleOptions.MULTIPLICATIVE,
                1.0,
                0,
            ),
            ("negative_warmup", WeightDecayScheduleOptions.EXPONENTIAL, 0.1, -1),
        ]

        for name, schedule, rate, warmup_batches in invalid_cases:
            with self.subTest(case=name):
                cfg = self.preset(
                    decay_schedule=schedule,
                    decay_rate=rate,
                    decay_warmup_batches=warmup_batches,
                )
                with self.assertRaises(ValueError):
                    cfg.build()

    def test_weight_decay_warmup_delays_decay(self):
        input_dim = 12
        output_dim = 24
        warmup_batches = 3
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=0.1,
            decay_warmup_batches=warmup_batches,
        )
        model = cfg.build()

        for step in range(warmup_batches):
            message = f"warmup_step={step}"
            with self.subTest(msg=message):
                result = model._maybe_apply_weight_decay(weight_params)
                self.assertTrue(torch.equal(result, weight_params))

        # first call after warmup: step=0, factor=1.0, no decay yet
        result = model._maybe_apply_weight_decay(weight_params)
        self.assertTrue(torch.equal(result, weight_params))
        # second call: step=1, factor < 1.0, decay applied
        result = model._maybe_apply_weight_decay(weight_params)
        self.assertFalse(torch.equal(result, weight_params))

    def test_decay_counters_frozen_in_eval_mode(self):
        input_dim = 12
        output_dim = 24
        warmup_batches = 2
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=0.1,
            decay_warmup_batches=warmup_batches,
        )
        model = cfg.build()

        model.eval()
        for _ in range(3):
            model._maybe_apply_weight_decay(weight_params)
        self.assertEqual(model.warmup_step.item(), 0)
        self.assertEqual(model.decay_step.item(), 0)

        model.train()
        for _ in range(warmup_batches):
            model._maybe_apply_weight_decay(weight_params)
        model._maybe_apply_weight_decay(weight_params)
        frozen_decay_step = model.decay_step.clone()
        frozen_warmup_step = model.warmup_step.clone()

        model.eval()
        baseline = model._maybe_apply_weight_decay(weight_params)
        for _ in range(3):
            result = model._maybe_apply_weight_decay(weight_params)
            self.assertTrue(torch.equal(result, baseline))
        self.assertTrue(torch.equal(model.decay_step, frozen_decay_step))
        self.assertTrue(torch.equal(model.warmup_step, frozen_warmup_step))

    def test_weight_decay_schedule_raises_on_unknown_schedule(self):
        input_dim = 12
        output_dim = 24
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=0.1,
        )
        model = cfg.build()
        model.decay_schedule_option = "invalid_schedule"
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        with self.assertRaises(ValueError):
            model._maybe_apply_weight_decay(weight_params)

    def test_apply_normalization_transform_all_options(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        vectors = torch.randn(batch_size, generator_depth, input_dim)

        for normalization in WeightNormalizationOptions:
            message = f"normalization={normalization}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    normalization_option=normalization,
                )
                model = DualModelDynamicWeight(cfg)
                result = model._apply_normalization_transform(vectors)
                self.assertEqual(result.shape, vectors.shape)
                if normalization == WeightNormalizationOptions.DISABLED:
                    self.assertTrue(torch.equal(result, vectors))

    def test_l2_and_rms_normalization_remain_finite_at_zero(self):
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            for normalization in (
                WeightNormalizationOptions.L2_SCALE,
                WeightNormalizationOptions.RMS,
            ):
                message = f"dtype={dtype}, normalization={normalization}"
                with self.subTest(msg=message):
                    cfg = self.preset(normalization_option=normalization)
                    model = DualModelDynamicWeight(cfg).to(dtype=dtype)
                    vectors = torch.zeros(
                        1,
                        1,
                        2,
                        dtype=dtype,
                        requires_grad=True,
                    )

                    result = model._apply_normalization_transform(vectors)
                    (vector_gradient,) = torch.autograd.grad(result.sum(), vectors)

                    self.assertTrue(torch.isfinite(result).all().item())
                    self.assertTrue(torch.isfinite(vector_gradient).all().item())
                    self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_l2_and_rms_normalization_preserve_extreme_vector_direction(self):
        expected_value_by_normalization = {
            WeightNormalizationOptions.L2_SCALE: 2.0**-0.5,
            WeightNormalizationOptions.RMS: 1.0,
        }

        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            maximum_finite_value = torch.finfo(dtype).max
            for (
                normalization,
                expected_value,
            ) in expected_value_by_normalization.items():
                message = f"dtype={dtype}, normalization={normalization}"
                with self.subTest(msg=message):
                    cfg = self.preset(normalization_option=normalization)
                    model = DualModelDynamicWeight(cfg).to(dtype=dtype)
                    vectors = torch.full(
                        (1, 1, 2),
                        maximum_finite_value / 2,
                        dtype=dtype,
                        requires_grad=True,
                    )

                    result = model._apply_normalization_transform(vectors)
                    (vector_gradient,) = torch.autograd.grad(result.sum(), vectors)

                    expected = torch.full_like(result, expected_value)
                    torch.testing.assert_close(result, expected)
                    self.assertTrue(torch.isfinite(vector_gradient).all().item())

    def test_soft_clamp_remains_finite_at_degenerate_limits(self):
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            for clamp_limit in (0.0, torch.finfo(dtype).tiny / 2):
                message = f"dtype={dtype}, clamp_limit={clamp_limit}"
                with self.subTest(msg=message):
                    cfg = self.preset(
                        normalization_option=WeightNormalizationOptions.SOFT_CLAMP,
                    )
                    model = DualModelDynamicWeight(cfg).to(dtype=dtype)
                    model.clamp_limit.data.fill_(clamp_limit)
                    vectors = torch.tensor(
                        [[[0.0, 1.0, -1.0]]],
                        dtype=dtype,
                        requires_grad=True,
                    )

                    result = model._apply_normalization_transform(vectors)
                    vector_gradient, limit_gradient = torch.autograd.grad(
                        result.sum(),
                        (vectors, model.clamp_limit),
                    )

                    self.assertTrue(torch.isfinite(result).all().item())
                    self.assertTrue(torch.isfinite(vector_gradient).all().item())
                    self.assertTrue(torch.isfinite(limit_gradient).all().item())
                    self.assertEqual(result[..., 0].item(), 0.0)

    def test_clamp_uses_the_learned_limit_as_a_magnitude(self):
        cfg = self.preset(
            normalization_option=WeightNormalizationOptions.CLAMP,
        )
        model = DualModelDynamicWeight(cfg)
        model.clamp_limit.data.fill_(-1.0)
        vectors = torch.tensor([[[-3.0, 0.0, 3.0]]])

        result = model._apply_normalization_transform(vectors)

        expected = torch.tensor([[[-1.0, 0.0, 1.0]]])
        torch.testing.assert_close(result, expected)

    def test_apply_normalization_transform_exact_math(self):
        cfg = self.preset(
            normalization_option=WeightNormalizationOptions.DISABLED,
        )
        model = DualModelDynamicWeight(cfg)
        vectors = torch.tensor([[[-2.0, -0.5, 0.5, 2.0]]])
        model.scale.data.fill_(2.0)
        model.clamp_limit.data.fill_(1.0)

        expected_by_option = {
            WeightNormalizationOptions.CLAMP: torch.clamp(vectors, -1.0, 1.0),
            WeightNormalizationOptions.L2_SCALE: F.normalize(vectors, dim=-1) * 2.0,
            WeightNormalizationOptions.SOFT_CLAMP: torch.tanh(vectors),
            WeightNormalizationOptions.RMS: vectors
            / (vectors.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8)
            * 2.0,
            WeightNormalizationOptions.SIGMOID_SCALE: (
                torch.sigmoid(vectors) * 2.0 - 1.0
            )
            * 2.0,
            WeightNormalizationOptions.DISABLED: vectors,
        }

        for option, expected in expected_by_option.items():
            with self.subTest(option=option):
                model.normalization_option = option
                result = model._apply_normalization_transform(vectors)
                torch.testing.assert_close(result, expected)

    def test_apply_normalization_transform_raises_on_unknown_option(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
        model = DualModelDynamicWeight(cfg)
        model.normalization_option = "invalid_normalization"
        vectors = torch.randn(batch_size, generator_depth, input_dim)
        with self.assertRaises(ValueError):
            model._apply_normalization_transform(vectors)

    def test_init_model_accepts_depth_mapping_handler_override(self):
        input_dim = 12
        output_dim = 24
        cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
        model = DualModelDynamicWeight(cfg)

        overrides = DepthMappingHandlerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        generator = model._init_model(overrides)

        self.assertIsInstance(generator, DepthMappingLayerStack)
        self.assertEqual(generator.input_dim, input_dim)
        self.assertEqual(generator.output_dim, output_dim)

    def test_weighted_bank_requires_bank_expansion_factor_enum(self):
        bank_cases = (
            (
                LayeredWeightedBankDynamicWeightConfig,
                "LayeredWeightedBankDynamicWeight",
            ),
            (
                SoftWeightedBankDynamicWeightConfig,
                "SoftWeightedBankDynamicWeight",
            ),
        )
        for config_type, model_name in bank_cases:
            for factor in (None, 0, -1):
                with self.subTest(config_type=config_type.__name__, factor=factor):
                    config = self.preset(
                        config_cls=config_type,
                        bank_expansion_factor=factor,
                    )
                    with self.assertRaises(ValueError) as error:
                        config.build()
                    self.assertEqual(
                        str(error.exception),
                        f"{model_name} requires bank_expansion_factor to be a "
                        "BankExpansionFactorOptions value, "
                        f"received {factor!r}.",
                    )

            with self.subTest(
                config_type=config_type.__name__,
                factor=BankExpansionFactorOptions.DISABLED,
            ):
                config = self.preset(
                    config_cls=config_type,
                    bank_expansion_factor=BankExpansionFactorOptions.DISABLED,
                )
                with self.assertRaises(ValueError) as error:
                    config.build()
                self.assertEqual(
                    str(error.exception),
                    f"{model_name} requires bank_expansion_factor > 0, received "
                    "BankExpansionFactorOptions.DISABLED. Use FACTOR_OF_ONE, "
                    "FACTOR_OF_TWO, FACTOR_OF_THREE, or FACTOR_OF_FOUR.",
                )

    def test_single_model_raises_when_dims_not_square(self):
        cfg = self.preset(input_dim=12, output_dim=24)
        with self.assertRaises(ValueError):
            SingleModelDynamicWeight(cfg)
