import torch
import unittest

from torch.types import Tensor
from torch.nn import Sequential
from Emperor.base.layer import Layer
from Emperor.config import ModelConfig
from Emperor.behaviours.model import AdaptiveParameterBehaviour
from Emperor.adaptive.utils.stack import AdaptiveParameterLayerStack
from Emperor.adaptive.utils.presets import AdaptiveParameterLayerPresets
from Emperor.adaptive.utils.mixtures.base import AdaptiveMixtureBase
from Emperor.adaptive.utils.mixtures.types.vector import VectorWeightsMixture
from Emperor.adaptive.utils.layers import (
    AdaptiveRouterOptions,
    AdaptiveParameterLayer,
)
from Emperor.adaptive.utils.mixtures.options import (
    AdaptiveBiasOptions,
    AdaptiveWeightOptions,
)
from Emperor.adaptive.utils.mixtures.types.generator import (
    GeneratorBiasMixture,
    GeneratorWeightsMixture,
)
from Emperor.adaptive.utils.mixtures.types.matrix import (
    MatrixBiasMixture,
    MatrixWeightsMixture,
)
from Emperor.experts.utils.enums import InitSamplerOptions


class TestAdaptiveParameterLayer(unittest.TestCase):
    def setUp(self):
        self.cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset()

    def test_init(self):
        for adaptive_router_otpion in AdaptiveRouterOptions:
            for adaptive_weight_otpion in AdaptiveWeightOptions:
                for adaptive_bias_option in AdaptiveBiasOptions:
                    message = f"Testing configuration: Weight Option: {adaptive_weight_otpion}, Bias Option: {adaptive_bias_option}, Router Option: {adaptive_router_otpion}"
                    with self.subTest(i=message):
                        overrides = {}
                        if adaptive_weight_otpion == AdaptiveWeightOptions.VECTOR:
                            overrides = {
                                "input_dim": 8,
                                "output_dim": 8,
                            }
                        cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset(
                            adaptive_weight_option=adaptive_weight_otpion,
                            adaptive_bias_option=adaptive_bias_option,
                            init_sampler_model_option=adaptive_router_otpion,
                            **overrides,
                        )

                        if (
                            adaptive_router_otpion
                            == AdaptiveRouterOptions.SHARED_ROUTER
                            and adaptive_weight_otpion == AdaptiveWeightOptions.VECTOR
                        ):
                            with self.assertRaises(
                                ValueError,
                                msg="Router option is disabled, but weight option cannot be disabled.",
                            ):
                                AdaptiveParameterLayer(cfg)
                            continue

                        m = AdaptiveParameterLayer(cfg)

                        self.assertIsInstance(m, AdaptiveParameterLayer)
                        self.assertEqual(m.input_dim, cfg.input_dim)
                        self.assertEqual(m.output_dim, cfg.output_dim)
                        self.assertEqual(
                            m.adaptive_weight_option, cfg.adaptive_weight_option
                        )
                        self.assertEqual(
                            m.adaptive_bias_option, cfg.adaptive_bias_option
                        )
                        self.assertEqual(
                            m.init_sampler_model_option, cfg.init_sampler_model_option
                        )
                        self.assertIsInstance(
                            m.adaptive_behaviour, AdaptiveParameterBehaviour
                        )
                        self.assertIsInstance(
                            m.weight_parameter_model, AdaptiveMixtureBase
                        )
                        if adaptive_bias_option == AdaptiveBiasOptions.DISABLED:
                            self.assertIsNone(m.bias_parameter_model)
                        else:
                            self.assertIsInstance(
                                m.bias_parameter_model, AdaptiveMixtureBase
                            )

    def test__init_adaptive_behaviour(self):
        behaviour_config_options = [None, self.cfg.adaptive_behaviour_config]

        for adaptive_behaviour_config in behaviour_config_options:
            message = f"Test failed for the inputs: adaptive_behaviour_config: {adaptive_behaviour_config}"
            with self.subTest(error=message):
                cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset()
                cfg.adaptive_behaviour_config = adaptive_behaviour_config
                m = AdaptiveParameterLayer(cfg)
                behaviour_model = m._AdaptiveParameterLayer__init_adaptive_behaviour()

                if adaptive_behaviour_config is not None:
                    self.assertIsInstance(behaviour_model, AdaptiveParameterBehaviour)
                    continue
                self.assertIsNone(behaviour_model)

    def test__init_weight_model(self):
        for adaptive_weight_option in AdaptiveWeightOptions:
            message = f"Test failed for the inputs: adaptive_weight_option: {adaptive_weight_option}"
            with self.subTest(error=message):
                options = {}
                if adaptive_weight_option == AdaptiveWeightOptions.VECTOR:
                    options = {
                        "input_dim": 8,
                        "output_dim": 8,
                        "init_sampler_model_option": AdaptiveRouterOptions.INDEPENTENT_ROUTER,
                    }
                cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset(
                    adaptive_weight_option=adaptive_weight_option,
                    **options,
                )
                m = AdaptiveParameterLayer(cfg)
                model = m._AdaptiveParameterLayer__init_weight_model()

                self.assertIsInstance(model, AdaptiveMixtureBase)
                if adaptive_weight_option == AdaptiveWeightOptions.VECTOR:
                    self.assertIsInstance(model, VectorWeightsMixture)
                elif adaptive_weight_option == AdaptiveWeightOptions.MATRIX:
                    self.assertIsInstance(model, MatrixWeightsMixture)
                elif adaptive_weight_option == AdaptiveWeightOptions.GENERATOR:
                    self.assertIsInstance(model, GeneratorWeightsMixture)

    def test__init_bias_model(self):
        for adaptive_bias_option in AdaptiveBiasOptions:
            message = f"Test failed for the inputs: adaptive_bias_option: {adaptive_bias_option}"
            with self.subTest(error=message):
                cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset(
                    adaptive_bias_option=adaptive_bias_option,
                )
                m = AdaptiveParameterLayer(cfg)
                model = m._AdaptiveParameterLayer__init_bias_model()

                if adaptive_bias_option == AdaptiveBiasOptions.MATRIX:
                    self.assertIsInstance(model, MatrixBiasMixture)
                elif adaptive_bias_option == AdaptiveBiasOptions.GENERATOR:
                    self.assertIsInstance(model, GeneratorBiasMixture)
                elif adaptive_bias_option == AdaptiveBiasOptions.DISABLED:
                    self.assertIsNone(model)

    def test_sample_probabilities_and_indices_for_weights(self):
        for init_sampler_model_flag in AdaptiveRouterOptions:
            for adaptive_weight_option in AdaptiveWeightOptions:
                message = f"Testing inputs: init_sampler_model_flag: {init_sampler_model_flag}, adaptive_weight_option: {adaptive_weight_option}"
                with self.subTest(error=message):
                    options = {}
                    if adaptive_weight_option == AdaptiveWeightOptions.VECTOR:
                        options = {
                            "input_dim": 8,
                            "output_dim": 8,
                        }
                    cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset(
                        init_sampler_model_option=init_sampler_model_flag,
                        adaptive_weight_option=adaptive_weight_option,
                        **options,
                    )

                    batch_size = 5
                    input_tensor = torch.randn(batch_size, cfg.input_dim)

                    if (
                        adaptive_weight_option == AdaptiveWeightOptions.VECTOR
                        and init_sampler_model_flag
                        == AdaptiveRouterOptions.SHARED_ROUTER
                    ):
                        with self.assertRaises(
                            ValueError,
                            msg="Router option is disabled, but weight option cannot be disabled.",
                        ):
                            AdaptiveParameterLayer(cfg)
                        continue

                    m = AdaptiveParameterLayer(cfg)
                    probabilities, indices, skip_mask, loss = (
                        m._AdaptiveParameterLayer__sample_weight_probabilities_and_indices(
                            input_tensor
                        )
                    )

                    if m.weights_router is None:
                        self.assertIsNone(probabilities)
                        self.assertIsNone(indices)
                        self.assertIsInstance(loss, Tensor)
                        continue
                    self.assertIsInstance(probabilities, Tensor)
                    self.assertIsInstance(indices, Tensor)
                    self.assertIsInstance(loss, Tensor)

    def test_sample_probabilities_and_indices_for_bias(self):
        for init_sampler_model_flag in AdaptiveRouterOptions:
            for adaptive_bias_option in AdaptiveBiasOptions:
                message = f"Testing inputs: init_sampler_model_flag: {init_sampler_model_flag}, adaptive_bias_option: {adaptive_bias_option}"
                with self.subTest(error=message):
                    cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset(
                        init_sampler_model_option=init_sampler_model_flag,
                        adaptive_bias_option=adaptive_bias_option,
                    )

                    batch_size = 5
                    input_tensor = torch.randn(batch_size, cfg.input_dim)

                    m = AdaptiveParameterLayer(cfg)
                    probabilities, indices, skip_mask, loss = (
                        m._AdaptiveParameterLayer__sample_bias_probabilities_and_indices(
                            input_tensor
                        )
                    )

                    if m.bias_router is None:
                        self.assertIsNone(probabilities)
                        self.assertIsNone(indices)
                        self.assertIsInstance(loss, Tensor)
                        continue
                    self.assertIsInstance(probabilities, Tensor)
                    self.assertIsInstance(indices, Tensor)
                    self.assertIsInstance(loss, Tensor)

    def test__generate_weight_parameters(self):
        for adaptive_weight_option in AdaptiveWeightOptions:
            message = f"Test failed for the inputs: adaptive_weight_option: {adaptive_weight_option}"
            with self.subTest(error=message):
                options = {}
                cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset(
                    input_dim=8,
                    output_dim=8,
                    experts_compute_expert_mixture_flag=True,
                    adaptive_weight_option=adaptive_weight_option,
                    init_sampler_model_option=AdaptiveRouterOptions.INDEPENTENT_ROUTER,
                    experts_init_sampler_option=InitSamplerOptions.LAYER,
                    **options,
                )

                batch_size = 5
                input_tensor = torch.randn(batch_size, cfg.input_dim)
                m = AdaptiveParameterLayer(cfg)
                probabilities, indices, skip_mask, loss = (
                    m._AdaptiveParameterLayer__sample_weight_probabilities_and_indices(
                        input_tensor
                    )
                )
                weight_parameters, loss = (
                    m._AdaptiveParameterLayer__generate_weight_parameters(
                        probabilities, indices, input_tensor
                    )
                )

                expected_weight_shape = (batch_size, cfg.input_dim, cfg.output_dim)
                self.assertIsInstance(weight_parameters, Tensor)
                self.assertEqual(weight_parameters.shape, expected_weight_shape)
                self.assertIsInstance(loss, Tensor)

    def test__generate_bias_parameters(self):
        for adaptive_bias_option in AdaptiveBiasOptions:
            message = f"Test failed for the inputs: adaptive_bias_option: {adaptive_bias_option}"
            with self.subTest(error=message):
                options = {}
                cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset(
                    input_dim=8,
                    output_dim=8,
                    adaptive_bias_option=adaptive_bias_option,
                    init_sampler_model_option=AdaptiveRouterOptions.INDEPENTENT_ROUTER,
                    experts_init_sampler_option=InitSamplerOptions.LAYER,
                    experts_compute_expert_mixture_flag=True,
                    **options,
                )

                batch_size = 5
                input_tensor = torch.randn(batch_size, cfg.output_dim)
                m = AdaptiveParameterLayer(cfg)
                probabilities, indices, skip_mask, loss = (
                    m._AdaptiveParameterLayer__sample_bias_probabilities_and_indices(
                        input_tensor
                    )
                )
                bias_parameters, loss = (
                    m._AdaptiveParameterLayer__generate_bias_parameters(
                        input_tensor, skip_mask, probabilities, indices
                    )
                )

                expected_bias_shape = (batch_size, cfg.output_dim)
                if adaptive_bias_option == AdaptiveBiasOptions.DISABLED:
                    self.assertIsNone(bias_parameters)
                    self.assertIsInstance(loss, Tensor)
                    self.assertEqual(loss, torch.tensor(0.0))
                    continue
                self.assertIsInstance(bias_parameters, Tensor)
                self.assertEqual(bias_parameters.shape, expected_bias_shape)
                self.assertIsInstance(loss, Tensor)

    def test_generate_parameters(self):
        for adaptive_weight_option in AdaptiveWeightOptions:
            for adaptive_bias_option in AdaptiveBiasOptions:
                for init_sampler_model_option in AdaptiveRouterOptions:
                    message = f"Test failed for the inputs: adaptive_weight_option: {adaptive_weight_option}, adaptive_bias_option: {adaptive_bias_option}, init_sampler_model_option: {init_sampler_model_option}"
                    with self.subTest(error=message):
                        options = {}
                        if adaptive_weight_option == AdaptiveWeightOptions.VECTOR:
                            options = {
                                "input_dim": 8,
                                "output_dim": 8,
                            }

                            shared_option = AdaptiveRouterOptions.SHARED_ROUTER
                            if init_sampler_model_option == shared_option:
                                cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset(
                                    experts_compute_expert_mixture_flag=True,
                                    adaptive_weight_option=adaptive_weight_option,
                                    adaptive_bias_option=adaptive_bias_option,
                                    init_sampler_model_option=init_sampler_model_option,
                                    experts_init_sampler_option=InitSamplerOptions.LAYER,
                                    **options,
                                )
                                with self.assertRaises(ValueError):
                                    AdaptiveParameterLayer(cfg)
                                continue

                        if (
                            adaptive_weight_option == AdaptiveWeightOptions.GENERATOR
                            or adaptive_bias_option == AdaptiveBiasOptions.GENERATOR
                        ):
                            if (
                                init_sampler_model_option
                                == AdaptiveRouterOptions.INDEPENTENT_ROUTER
                            ):
                                options["experts_init_sampler_option"] = (
                                    InitSamplerOptions.LAYER
                                )

                        cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset(
                            experts_compute_expert_mixture_flag=True,
                            adaptive_weight_option=adaptive_weight_option,
                            adaptive_bias_option=adaptive_bias_option,
                            init_sampler_model_option=init_sampler_model_option,
                            **options,
                        )
                        batch_size = 5
                        input_tensor = torch.randn(batch_size, cfg.input_dim)

                        m = AdaptiveParameterLayer(cfg)
                        weight_parameters, bias_parameters, loss = (
                            m._generate_parameters(
                                input_tensor,
                            )
                        )

                        expected_weight_shape = (
                            batch_size,
                            cfg.input_dim,
                            cfg.output_dim,
                        )
                        expected_bias_shape = (batch_size, cfg.output_dim)
                        self.assertIsInstance(weight_parameters, Tensor)
                        self.assertEqual(weight_parameters.shape, expected_weight_shape)
                        if adaptive_bias_option == AdaptiveBiasOptions.DISABLED:
                            self.assertIsNone(bias_parameters)
                        else:
                            self.assertIsInstance(bias_parameters, Tensor)
                            self.assertEqual(bias_parameters.shape, expected_bias_shape)
                        self.assertIsInstance(loss, Tensor)

    def test__apply_generated_parameters(self):
        cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset()
        m = AdaptiveParameterLayer(cfg)

        batch_size = 5
        input = torch.randn(batch_size, cfg.input_dim)
        weight_parameters = torch.randn(batch_size, cfg.input_dim, cfg.output_dim)

        output = m._AdaptiveParameterLayer__apply_generated_weights(
            input, weight_parameters
        )

        expected_output_shape = (batch_size, cfg.output_dim)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, expected_output_shape)
        for i in range(batch_size):
            input_i = input[i].unsqueeze(0)
            weight_parameters_i = weight_parameters[i]
            output_i = torch.mm(input_i, weight_parameters_i)
            self.assertTrue(torch.equal(output[i].round(decimals=4), output_i.squeeze(0).round(decimals=4)))

    def test__apply_generated_biases(self):
        cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset()
        m = AdaptiveParameterLayer(cfg)

        batch_size = 5
        input = torch.ones(batch_size, cfg.output_dim)
        bias_parameters = torch.randn(batch_size, cfg.output_dim)

        output = m._AdaptiveParameterLayer__apply_generated_biases(
            input, bias_parameters
        )

        expected_output_shape = (batch_size, cfg.output_dim)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, expected_output_shape)

    def test_forward(self):
        for adaptive_weight_option in AdaptiveWeightOptions:
            for adaptive_bias_option in AdaptiveBiasOptions:
                for init_sampler_model_option in AdaptiveRouterOptions:
                    message = f"Test failed for the inputs: adaptive_weight_option: {adaptive_weight_option}, adaptive_bias_option: {adaptive_bias_option}, init_sampler_model_option: {init_sampler_model_option}"
                    with self.subTest(error=message):
                        options = {}
                        if adaptive_weight_option == AdaptiveWeightOptions.VECTOR:
                            options = {
                                "input_dim": 8,
                                "output_dim": 8,
                            }

                            shared_option = AdaptiveRouterOptions.SHARED_ROUTER
                            if init_sampler_model_option == shared_option:
                                cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset(
                                    experts_compute_expert_mixture_flag=True,
                                    adaptive_weight_option=adaptive_weight_option,
                                    adaptive_bias_option=adaptive_bias_option,
                                    init_sampler_model_option=init_sampler_model_option,
                                    experts_init_sampler_option=InitSamplerOptions.LAYER,
                                    **options,
                                )
                                with self.assertRaises(ValueError):
                                    AdaptiveParameterLayer(cfg)
                                continue

                        if (
                            adaptive_weight_option == AdaptiveWeightOptions.GENERATOR
                            or adaptive_bias_option == AdaptiveBiasOptions.GENERATOR
                        ):
                            if (
                                init_sampler_model_option
                                == AdaptiveRouterOptions.INDEPENTENT_ROUTER
                            ):
                                options["experts_init_sampler_option"] = (
                                    InitSamplerOptions.LAYER
                                )

                        cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_preset(
                            experts_compute_expert_mixture_flag=True,
                            adaptive_weight_option=adaptive_weight_option,
                            adaptive_bias_option=adaptive_bias_option,
                            init_sampler_model_option=init_sampler_model_option,
                            **options,
                        )
                        m = AdaptiveParameterLayer(cfg)

                        batch_size = 5
                        input_tensor = torch.randn(batch_size, cfg.input_dim)
                        output, skip_mask, loss = m.forward(input_tensor)

                        expected_shape = (batch_size, cfg.input_dim, cfg.output_dim)
                        expected_shape = (batch_size, cfg.output_dim)
                        self.assertIsInstance(output, Tensor)
                        self.assertEqual(output.shape, expected_shape)
                        self.assertIsInstance(loss, Tensor)


class TestAdaptiveParameterLayerStack(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: ModelConfig | None = None):
        self.cfg = (
            AdaptiveParameterLayerPresets.adaptive_parameter_layer_stack_preset(
                return_model_config_flag=True,
            )
            if config is None
            else config
        )

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

    def test_init_with_default_config(self):
        num_layer_options = [1, 2, 3]

        for adaptive_weight_option in AdaptiveWeightOptions:
            for adaptive_bias_option in AdaptiveBiasOptions:
                for num_layers in num_layer_options:
                    message = f"Testing configuration with num_layers={num_layers}"
                    with self.subTest(msg=message):
                        cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_stack_preset(
                            input_dim=8,
                            hidden_dim=8,
                            output_dim=8,
                            num_layers=num_layers,
                            adaptive_weight_option=adaptive_weight_option,
                            adaptive_bias_option=adaptive_bias_option,
                            adaptive_init_sampler_model_option=AdaptiveRouterOptions.INDEPENTENT_ROUTER,
                            experts_init_sampler_option=InitSamplerOptions.LAYER,
                        )
                        m = AdaptiveParameterLayerStack(cfg).build_model()

                        if num_layers == 1:
                            self.assertIsInstance(m, Layer)
                        else:
                            self.assertIsInstance(m, Sequential)
                            self.assertEqual(len(m), num_layers)

    def test_forward(self):
        top_k_options = [1, 3, 6]
        num_layers_options = [1, 2, 3]

        for num_layers in num_layers_options:
            for adaptive_weight_option in AdaptiveWeightOptions:
                for adaptive_bias_option in AdaptiveBiasOptions:
                    for top_k in top_k_options:
                        message = f"Testing with layer_stack_option={adaptive_weight_option.name}, weighting_position_option={adaptive_bias_option.name}, init_sampler_model_flag={AdaptiveRouterOptions.__members__}, compute_expert_mixture_flag={True or False}, weighted_parameters_flag={True or False}, top_k={top_k}, num_layers={num_layers}"
                        with self.subTest(msg=message):
                            cfg = AdaptiveParameterLayerPresets.adaptive_parameter_layer_stack_preset(
                                input_dim=8,
                                hidden_dim=8,
                                output_dim=8,
                                num_layers=num_layers,
                                experts_compute_expert_mixture_flag=True,
                                adaptive_weight_option=adaptive_weight_option,
                                adaptive_bias_option=adaptive_bias_option,
                                adaptive_init_sampler_model_option=AdaptiveRouterOptions.INDEPENTENT_ROUTER,
                                experts_init_sampler_option=InitSamplerOptions.LAYER,
                            )
                            m = AdaptiveParameterLayerStack(cfg).build_model()

                            batch_size = 10

                            input_tensor = torch.randn(batch_size, cfg.input_dim)
                            output, skip_mask, loss = m(input_tensor)
                            expected_shape = (batch_size, cfg.output_dim)
                            self.assertEqual(output.shape, expected_shape)
