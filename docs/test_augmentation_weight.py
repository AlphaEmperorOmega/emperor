import torch
import unittest

from emperor.base.utils import Module
from emperor.linears.options import LinearOptions
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.linears.core.config import LinearLayerConfig
from emperor.augmentations.adaptive_parameters import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    DynamicWeightOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters.core.depth_mapper import (
    DepthMappingHandlerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeight,
    HypernetworkDynamicWeight,
    LayeredWeightedBankDynamicWeight,
    LowRankDynamicWeight,
    SingleModelDynamicWeight,
    SoftWeightedBankDynamicWeight,
    DynamicWeightConfig,
    DynamicWeightAbstract,
)


class TestWeightHandlerForward(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        hidden_dim: int = 36,
        output_dim: int = 24,
        bias_flag: bool = True,
        model_type: DynamicWeightOptions = DynamicWeightOptions.DUAL_MODEL,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
        stack_dropout_probability: float = 0.0,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
        normalization_option: WeightNormalizationOptions = WeightNormalizationOptions.L2_SCALE,
        normalization_position_option: WeightNormalizationPositionOptions = WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT,
        bank_expansion_factor: BankExpansionFactorOptions | None = None,
        decay_schedule: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED,
        decay_rate: float = 0.0,
        decay_warmup_batches: int = 0,
    ) -> DynamicWeightConfig:
        return DynamicWeightConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=model_type,
            normalization_option=normalization_option,
            normalization_position_option=normalization_position_option,
            generator_depth=generator_depth,
            bank_expansion_factor=bank_expansion_factor,
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
                    activation=ActivationOptions.RELU,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_flag=False,
                    dropout_probability=0.1,
                    gate_config=None,
                    halting_config=None,
                    shared_halting_flag=False,
                    layer_model_config=LinearLayerConfig(
                        model_type=LinearOptions.LINEAR,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                    ),
                ),
            ),
        )

    def test_single_model_handler_forward(self):
        batch_size = 2
        dim = 12
        cfg = self.preset(
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
        cfg = self.preset(input_dim, output_dim)
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
        cfg = self.preset(input_dim, output_dim)
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
        flattened_weight_dim = input_dim * output_dim
        cfg = self.preset(input_dim, flattened_weight_dim)
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
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=DynamicWeightOptions.LAYERED_WEIGHTED_BANK,
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
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=DynamicWeightOptions.SOFT_WEIGHTED_BANK,
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

    def test_compute_outer_product_dispatches_by_position(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        positions = [
            WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT,
            WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
            WeightNormalizationPositionOptions.DISABLED,
        ]
        for position in positions:
            message = f"position={position}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    normalization_option=WeightNormalizationOptions.L2_SCALE,
                    normalization_position_option=position,
                )
                model = DualModelDynamicWeight(cfg)
                input_vectors = torch.randn(batch_size, generator_depth, input_dim)
                output_vectors = torch.randn(batch_size, generator_depth, output_dim)
                result = model._compute_outer_product(input_vectors, output_vectors)

                if position == WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT:
                    expected = model._compute_prenormalized_outer_product(
                        input_vectors, output_vectors
                    )
                elif position == WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT:
                    expected = model._compute_postnormalized_outer_product(
                        input_vectors, output_vectors
                    )
                else:
                    expected = model._compute_raw_outer_product(
                        input_vectors, output_vectors
                    )
                self.assertTrue(torch.equal(result, expected))

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
        setattr(model, "normalization_position_option", "invalid_position")
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

    def test_build_creates_model_for_each_option(self):
        input_dim = 12
        output_dim = 24
        flattened_weight_dim = input_dim * output_dim

        for option in DynamicWeightOptions:
            if option == DynamicWeightOptions.DISABLED:
                with self.subTest(msg="option=DISABLED raises ValueError"):
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        model_type=option,
                    )
                    with self.assertRaises(ValueError):
                        cfg.build()
                continue
            message = f"option={option}"
            with self.subTest(msg=message):
                if option == DynamicWeightOptions.SINGLE_MODEL:
                    effective_output_dim = input_dim
                elif option == DynamicWeightOptions.HYPERNETWORK:
                    effective_output_dim = flattened_weight_dim
                else:
                    effective_output_dim = output_dim
                bank_factor = (
                    BankExpansionFactorOptions.FACTOR_OF_TWO
                    if option
                    in {
                        DynamicWeightOptions.LAYERED_WEIGHTED_BANK,
                        DynamicWeightOptions.SOFT_WEIGHTED_BANK,
                    }
                    else None
                )
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=effective_output_dim,
                    model_type=option,
                    bank_expansion_factor=bank_factor,
                )
                model = cfg.build()
                self.assertIsInstance(model, DynamicWeightAbstract)

    def test_gradients_flow(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        flattened_weight_dim = input_dim * output_dim

        for option in DynamicWeightOptions:
            if option == DynamicWeightOptions.SOFT_WEIGHTED_BANK:
                # TODO: removem this when you have implemented SoftWeightedBankDynamicWeight
                continue

            if option == DynamicWeightOptions.DISABLED:
                with self.subTest(msg="option=DISABLED raises ValueError"):
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        model_type=option,
                    )
                    with self.assertRaises(ValueError):
                        cfg.build()
                continue
            message = f"option={option}"
            with self.subTest(msg=message):
                if option == DynamicWeightOptions.SINGLE_MODEL:
                    effective_output_dim = input_dim
                elif option == DynamicWeightOptions.HYPERNETWORK:
                    effective_output_dim = flattened_weight_dim
                else:
                    effective_output_dim = output_dim
                effective_weight_params = Module()._init_parameter_bank(
                    (input_dim, effective_output_dim)
                )
                bank_factor = (
                    BankExpansionFactorOptions.FACTOR_OF_TWO
                    if option
                    in {
                        DynamicWeightOptions.LAYERED_WEIGHTED_BANK,
                        DynamicWeightOptions.SOFT_WEIGHTED_BANK,
                    }
                    else None
                )
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=effective_output_dim,
                    model_type=option,
                    bank_expansion_factor=bank_factor,
                )
                model = cfg.build()
                input_tensor = torch.randn(batch_size, input_dim, requires_grad=True)
                output = model(effective_weight_params, input_tensor)
                output.sum().backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad]
                non_none_grads = [g for g in grads if g is not None]
                self.assertTrue(len(non_none_grads) > 0)

    def test_generator_depth_options(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        flattened_weight_dim = input_dim * output_dim

        for weight_option in DynamicWeightOptions:
            if weight_option == DynamicWeightOptions.DISABLED:
                continue
            if weight_option == DynamicWeightOptions.SINGLE_MODEL:
                effective_output_dim = input_dim
            elif weight_option == DynamicWeightOptions.HYPERNETWORK:
                effective_output_dim = flattened_weight_dim
            else:
                effective_output_dim = output_dim
            is_bank_type = weight_option in {
                DynamicWeightOptions.LAYERED_WEIGHTED_BANK,
                DynamicWeightOptions.SOFT_WEIGHTED_BANK,
            }
            bank_factor = (
                BankExpansionFactorOptions.FACTOR_OF_TWO if is_bank_type else None
            )
            weight_params = Module()._init_parameter_bank(
                (input_dim, effective_output_dim)
            )
            for depth in DynamicDepthOptions:
                if depth == DynamicDepthOptions.DISABLED:
                    continue
                msg = f"option={weight_option}, depth={depth}"
                with self.subTest(msg=msg):
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=effective_output_dim,
                        model_type=weight_option,
                        generator_depth=depth,
                        bank_expansion_factor=bank_factor,
                    )
                    model = cfg.build()
                    input_tensor = torch.randn(batch_size, input_dim)
                    output = model(weight_params, input_tensor)
                    expected_shape = (
                        batch_size,
                        input_dim,
                        effective_output_dim,
                    )
                    self.assertEqual(output.shape, expected_shape)

    def test_bank_expansion_factor_variants(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        for option in DynamicWeightOptions:
            if option == DynamicWeightOptions.DISABLED:
                continue
            is_bank_type = option in {
                DynamicWeightOptions.LAYERED_WEIGHTED_BANK,
                DynamicWeightOptions.SOFT_WEIGHTED_BANK,
            }
            effective_output_dim = (
                input_dim if option == DynamicWeightOptions.SINGLE_MODEL else output_dim
            )
            for factor in BankExpansionFactorOptions:
                is_disabled = factor == BankExpansionFactorOptions.DISABLED
                msg = f"option={option}, factor={factor}"
                with self.subTest(msg=msg):
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=effective_output_dim,
                        model_type=option,
                        bank_expansion_factor=factor,
                    )
                    if not is_bank_type or is_disabled:
                        with self.assertRaises(ValueError):
                            cfg.build()
                    elif is_bank_type:
                        model = cfg.build()
                        input_tensor = torch.randn(batch_size, input_dim)
                        output = model(weight_params, input_tensor)
                        expected_shape = (
                            batch_size,
                            input_dim,
                            output_dim,
                        )
                        self.assertEqual(output.shape, expected_shape)

    def test_weight_decay_schedule_options(self):
        input_dim = 12
        output_dim = 24
        decay_rate = 0.4
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))

        for schedule in WeightDecayScheduleOptions:
            message = f"decay_schedule={schedule}"
            with self.subTest(msg=message):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    decay_schedule=schedule,
                    decay_rate=decay_rate,
                )
                model = cfg.build()

                if schedule == WeightDecayScheduleOptions.DISABLED:
                    result = model._maybe_apply_weight_decay(weight_params)
                    self.assertTrue(torch.equal(result, weight_params))
                else:
                    # step=0: all schedules produce factor=1.0, no change
                    result = model._maybe_apply_weight_decay(weight_params)
                    self.assertTrue(torch.equal(result, weight_params))
                    # step=1: decay factor < 1.0, output differs
                    result = model._maybe_apply_weight_decay(weight_params)
                    self.assertEqual(result.shape, weight_params.shape)
                    self.assertFalse(torch.equal(result, weight_params))

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
                        f"{message}, step={step}: expected factor={expected_factor.item():.6f}",
                    )

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
        setattr(model, "decay_schedule_option", "invalid_schedule")
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

    def test_apply_normalization_transform_raises_on_unknown_option(self):
        batch_size = 2
        generator_depth = 1
        input_dim = 12
        output_dim = 24
        cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
        model = DualModelDynamicWeight(cfg)
        setattr(model, "normalization_option", "invalid_normalization")
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

    def test_single_model_raises_when_dims_not_square(self):
        cfg = self.preset(input_dim=12, output_dim=24)
        with self.assertRaises(ValueError):
            SingleModelDynamicWeight(cfg)
