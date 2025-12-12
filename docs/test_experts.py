import torch
import unittest

from torch.nn import Sequential
from Emperor.base.layer import Layer
from Emperor.config import ModelConfig
from Emperor.experts.utils.layers import MixtureOfExperts
from Emperor.experts.utils.config import MixtureOfExpertsConfigs
from Emperor.experts.utils.enums import ExpertWeightingPositionOptions, LayerRoleOptions
from Emperor.experts.utils.stack import MixtureOfExpertsStack
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.sampler.model import SamplerModel
from Emperor.sampler.utils.routers import RouterModel
from Emperor.sampler.utils.samplers import SamplerFull, SamplerSparse, SamplerTopk


class TestMixtureOfExperts(unittest.TestCase):
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
            MixtureOfExpertsConfigs.linear_adaptive_layer_preset()
            if config is None
            else config
        )

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

    def test_init_with_different_configs(self):
        top_k_options = [1, 3, 6]
        num_experts = 6
        bool_flags = [True, False]

        for layer_stack_option in LinearLayerStackOptions:
            for layer_role_option in LayerRoleOptions:
                for top_k in top_k_options:
                    for init_sampler_model_flag in bool_flags:
                        message = f"Testing configuration with num_experts={num_experts}, top_k={top_k}, layer_stack_option={layer_stack_option}, layer_role_option={layer_role_option}, and init_sampler_model_flag={init_sampler_model_flag}"
                        with self.subTest(msg=message):
                            c = MixtureOfExpertsConfigs.linear_adaptive_layer_preset(
                                layer_stack_option=layer_stack_option,
                                num_experts=num_experts,
                                layer_role_option=layer_role_option,
                                top_k=top_k,
                                init_sampler_model_flag=init_sampler_model_flag,
                            )

                            m = MixtureOfExperts(c)
                            cfg = m.cfg
                            self.assertIsInstance(m, MixtureOfExperts)
                            self.assertEqual(
                                m.layer_stack_model.value, layer_stack_option.value
                            )
                            self.assertEqual(m.top_k, top_k)
                            self.assertEqual(m.num_experts, num_experts)
                            self.assertEqual(
                                m.compute_expert_mixture_flag,
                                cfg.compute_expert_mixture_flag,
                            )
                            self.assertEqual(
                                m.weighted_parameters_flag, cfg.weighted_parameters_flag
                            )
                            self.assertEqual(
                                m.init_sampler_model_flag, cfg.init_sampler_model_flag
                            )
                            self.assertEqual(
                                m.weighting_position_option,
                                cfg.weighting_position_option,
                            )
                            self.assertEqual(m.layer_role_option, cfg.layer_role_option)
                            self.assertEqual(
                                m.router_model_config, cfg.router_model_config
                            )
                            self.assertEqual(
                                m.sampler_model_config, cfg.sampler_model_config
                            )

    def test__create_experts(self):
        for layer_stack_option in LinearLayerStackOptions:
            message = f"Testing configuration with layer_stack_option={layer_stack_option.name}"
            with self.subTest(msg=message):
                c = MixtureOfExpertsConfigs.linear_adaptive_layer_preset(
                    layer_stack_option=layer_stack_option,
                )

                m = MixtureOfExperts(c)
                expert_models = m._MixtureOfExperts__create_experts()

                self.assertEqual(len(m.expert_modules), m.num_experts)
                for expert in expert_models:
                    self.assertIsInstance(expert, Sequential)
                    for layer in expert:
                        self.assertIsInstance(layer, Layer)

    def test__maybe_create_router_and_sampler(self):
        num_experts = 6
        expert_options = [1, 3, 6]
        init_sampler_model_flag_options = [True, False]
        sampler_options = [SamplerSparse, SamplerTopk, SamplerFull]

        for init_sampler_model_flag in init_sampler_model_flag_options:
            for sampler_option, expert_option in zip(sampler_options, expert_options):
                message = f"Testing configuration with sampler_option={sampler_option.__name__}, num_experts={num_experts}, top_k={expert_option}"
                with self.subTest(msg=message):
                    c = MixtureOfExpertsConfigs.linear_adaptive_layer_preset(
                        init_sampler_model_flag=init_sampler_model_flag,
                        num_experts=num_experts,
                        top_k=expert_option,
                    )

                    m = MixtureOfExperts(c)
                    router, sampler = (
                        m._MixtureOfExperts__maybe_create_router_and_sampler()
                    )
                    if init_sampler_model_flag:
                        self.assertIsInstance(router, RouterModel)
                        self.assertIsInstance(sampler, SamplerModel)
                        self.assertIsInstance(sampler.sampler_model, sampler_option)
                        self.assertEqual(sampler.sampler_model.top_k, expert_option)
                        continue
                    self.assertIsNone(router)
                    self.assertIsNone(sampler)

    def test__maybe_compute_expert_indices(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        init_sampler_model_flag_options = [True, False]

        for top_k in top_k_options:
            for init_sampler_model_flag in init_sampler_model_flag_options:
                message = f"Testing configuration with init_sampler_model_flag={init_sampler_model_flag}, top_k={top_k}"
                with self.subTest(msg=message):
                    c = MixtureOfExpertsConfigs.linear_adaptive_layer_preset(
                        init_sampler_model_flag=init_sampler_model_flag,
                        num_experts=num_experts,
                        top_k=top_k,
                    )

                    m = MixtureOfExperts(c)
                    if init_sampler_model_flag:
                        inputs = torch.randn(5, c.input_dim)
                        input_indices = None
                        input_probabilities = None
                        probabilities, indices, sampler_loss = (
                            m._MixtureOfExperts__maybe_compute_expert_indices(
                                inputs, input_probabilities, input_indices
                            )
                        )
                        if top_k == num_experts:
                            self.assertIsNone(indices)
                        else:
                            self.assertIsInstance(indices, torch.Tensor)
                        self.assertIsInstance(probabilities, torch.Tensor)
                        self.assertIsInstance(sampler_loss, torch.Tensor)
                        self.assertEqual(sampler_loss.item(), 0.0)
                        continue

                    inputs = torch.randn(5, c.input_dim)
                    indices_input = torch.randint(0, m.num_experts, (5, top_k))
                    indices, probabilities, sampler_loss = (
                        m._MixtureOfExperts__maybe_compute_expert_indices(
                            inputs, indices_input
                        )
                    )
                    self.assertTrue(torch.allclose(indices, indices_input))
                    self.assertIsNone(probabilities)
                    self.assertEqual(sampler_loss.item(), 0.0)

    def test__get_expert_indices(self):
        top_k = 3
        num_experts = 6

        for expert_index in range(num_experts):
            message = f"Testing configuration with expert_index={expert_index}"
            with self.subTest(msg=message):
                c = MixtureOfExpertsConfigs.linear_adaptive_layer_preset(
                    num_experts=num_experts,
                )

                m = MixtureOfExperts(c)

                indices = torch.randint(0, m.num_experts, (10, top_k))
                probabilities = torch.randn(10, top_k)
                sample_indices_for_expert = m._MixtureOfExperts__get_expert_indices(
                    indices, probabilities, expert_index
                )

                self.assertIsInstance(sample_indices_for_expert, torch.Tensor)

    def test__get_expert_probabilities(self):
        top_k = 3
        num_experts = 6

        for expert_index in range(num_experts):
            message = f"Testing configuration with expert_index={expert_index}"
            with self.subTest(msg=message):
                c = MixtureOfExpertsConfigs.linear_adaptive_layer_preset(
                    num_experts=num_experts,
                )

                m = MixtureOfExperts(c)

                indices = torch.randint(0, m.num_experts, (10, top_k))
                probabilities = torch.randn(10, top_k)
                probabilities = m._MixtureOfExperts__get_expert_probabilities(
                    indices, probabilities, expert_index
                )

                self.assertIsInstance(probabilities, torch.Tensor)

    def test__compute_experts_output(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        weighted_parameters_flag_options = [True, False]

        for weighting_position_option in ExpertWeightingPositionOptions:
            for top_k in top_k_options:
                for weighted_parameters_flag in weighted_parameters_flag_options:
                    message = f"Testing configuration with weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}"
                    with self.subTest(msg=message):
                        c = MixtureOfExpertsConfigs.linear_adaptive_layer_preset(
                            weighted_parameters_flag=weighted_parameters_flag,
                            weighting_position_option=weighting_position_option,
                            num_experts=num_experts,
                            top_k=top_k,
                        )

                        m = MixtureOfExperts(c)

                        expert_model = m.expert_modules[0]
                        input_batch = torch.randn(10, c.input_dim)
                        indices = torch.randint(0, m.num_experts, (10 * top_k,))
                        pribabilities = torch.randn(10 * top_k)

                        output, loss = m._MixtureOfExperts__compute_expert_output(
                            expert_model, input_batch, indices, pribabilities
                        )

                        self.assertIsInstance(output, torch.Tensor)
                        self.assertEqual(output.shape, (10 * top_k, c.output_dim))
                        self.assertTrue(torch.isfinite(output).all())
                        self.assertEqual(loss.item(), 0.0)

    def test__maybe_apply_probabilities(self):
        num_experts = 6
        top_k_options = [1, 3, 6]

        for top_k in top_k_options:
            for weighted_parameters_flag in [True, False]:
                message = f"Testing configuration with weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}"
                with self.subTest(msg=message):
                    c = MixtureOfExpertsConfigs.linear_adaptive_layer_preset(
                        weighted_parameters_flag=weighted_parameters_flag,
                        num_experts=num_experts,
                        top_k=top_k,
                    )

                    m = MixtureOfExperts(c)
                    logits = torch.randn(10 * top_k, c.input_dim)
                    pribabilities = torch.randn(10 * top_k)
                    output = m._MixtureOfExperts__maybe_apply_probabilities(
                        logits, pribabilities
                    )

                    if weighted_parameters_flag:
                        self.assertIsInstance(output, torch.Tensor)
                        expected_output = logits * pribabilities.view(-1, 1)
                        self.assertTrue(torch.allclose(output, expected_output))
                        continue
                    self.assertIsInstance(output, torch.Tensor)
                    self.assertTrue(torch.allclose(output, logits))

    def test__compute_expert_mixture(self):
        num_experts = 6
        top_k_options = [1, 3, 6]

        for top_k in top_k_options:
            for compute_expert_mixture_flag in [True, False]:
                for weighted_parameters_flag in [True, False]:
                    message = f"Testing configuration with weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}"
                    with self.subTest(msg=message):
                        c = MixtureOfExpertsConfigs.linear_adaptive_layer_preset(
                            weighted_parameters_flag=weighted_parameters_flag,
                            compute_expert_mixture_flag=compute_expert_mixture_flag,
                            num_experts=num_experts,
                            top_k=top_k,
                        )

                        m = MixtureOfExperts(c)

                        batch_size = 8
                        experts_output = torch.randn(batch_size * top_k, c.output_dim)
                        sample_indices = torch.randint(0, top_k, (batch_size * top_k,))
                        probabilities = torch.randn(batch_size * top_k)

                        output = m._MixtureOfExperts__compute_expert_mixture(
                            experts_output, sample_indices, probabilities
                        )

                        expected_shape = (batch_size * top_k, c.output_dim)
                        if compute_expert_mixture_flag:
                            expected_shape = (batch_size, c.output_dim)
                        self.assertEqual(output.shape, expected_shape)

    def test_forward(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        flag_options = [True, False]
        num_layers_options = [1, 2, 3]

        for num_layers in num_layers_options:
            for layer_stack_option in LinearLayerStackOptions:
                for weighting_position_option in ExpertWeightingPositionOptions:
                    for top_k in top_k_options:
                        for init_sampler_model_flag in flag_options:
                            for compute_expert_mixture_flag in flag_options:
                                for weighted_parameters_flag in flag_options:
                                    message = f"Testing with layer_stack_option={layer_stack_option.name}, weighting_position_option={weighting_position_option.name}, init_sampler_model_flag={init_sampler_model_flag}, compute_expert_mixture_flag={compute_expert_mixture_flag}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}"
                                    with self.subTest(msg=message):
                                        c = MixtureOfExpertsConfigs.linear_adaptive_layer_preset(
                                            batch_size=10,
                                            layer_stack_option=layer_stack_option,
                                            top_k=top_k,
                                            weighting_position_option=weighting_position_option,
                                            init_sampler_model_flag=init_sampler_model_flag,
                                            weighted_parameters_flag=weighted_parameters_flag,
                                            compute_expert_mixture_flag=compute_expert_mixture_flag,
                                            num_experts=num_experts,
                                            stack_num_layers=num_layers,
                                        )

                                        m = MixtureOfExperts(c)

                                        input = torch.randn(c.batch_size, c.input_dim)
                                        indices = probabilities = None
                                        if not init_sampler_model_flag:
                                            router_cfg = c.mixture_of_experts_config.router_model_config
                                            sampler_cfg = c.mixture_of_experts_config.sampler_model_config
                                            router = RouterModel(router_cfg)
                                            sampler = SamplerModel(sampler_cfg)

                                            logits = router.compute_logit_scores(input)
                                            probabilities, indices, _, _ = (
                                                sampler.sample_probabilities_and_indices(
                                                    logits
                                                )
                                            )

                                        output, total_loss = m.forward(
                                            input, probabilities, indices
                                        )

                                        expected_shape = (
                                            c.batch_size * top_k,
                                            c.output_dim,
                                        )
                                        if compute_expert_mixture_flag:
                                            expected_shape = (
                                                c.batch_size,
                                                c.output_dim,
                                            )
                                        self.assertEqual(output.shape, expected_shape)
                                        self.assertEqual(total_loss.item(), 0.0)


class TestMixtureOfExpertsStack(unittest.TestCase):
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
            MixtureOfExpertsConfigs.linear_adaptive_layer_preset()
            if config is None
            else config
        )

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

    def test_init_with_default_config(self):
        num_layer_options = [1, 2, 3]

        for num_layers in num_layer_options:
            message = f"Testing configuration with num_layers={num_layers}"
            with self.subTest(msg=message):
                c = MixtureOfExpertsConfigs.experts_stack_config(
                    stack_num_layers=num_layers,
                )
                m = MixtureOfExpertsStack(c).build_model()
                if num_layers == 1:
                    self.assertIsInstance(m, Layer)
                else:
                    self.assertIsInstance(m, Sequential)

    def test_forward(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        flag_options = [True, False]
        num_layers_options = [1, 2, 3]

        for num_layers in num_layers_options:
            for layer_stack_option in LinearLayerStackOptions:
                for weighting_position_option in ExpertWeightingPositionOptions:
                    for top_k in top_k_options:
                        for init_sampler_model_flag in flag_options:
                            for compute_expert_mixture_flag in flag_options:
                                for weighted_parameters_flag in flag_options:
                                    message = f"Testing with layer_stack_option={layer_stack_option.name}, weighting_position_option={weighting_position_option.name}, init_sampler_model_flag={init_sampler_model_flag}, compute_expert_mixture_flag={compute_expert_mixture_flag}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}"
                                    with self.subTest(msg=message):
                                        c = MixtureOfExpertsConfigs.experts_stack_config(
                                            layer_stack_option=layer_stack_option,
                                            top_k=top_k,
                                            weighting_position_option=weighting_position_option,
                                            init_sampler_model_flag=init_sampler_model_flag,
                                            weighted_parameters_flag=weighted_parameters_flag,
                                            compute_expert_mixture_flag=True,
                                            num_experts=num_experts,
                                            stack_num_layers=num_layers,
                                        )
                                        m = MixtureOfExpertsStack(c).build_model()

                                        batch_size = 10

                                        input = torch.randn(batch_size, c.input_dim)
                                        indices = probabilities = None
                                        if not init_sampler_model_flag:
                                            router_cfg = (
                                                c.override_config.router_model_config
                                            )
                                            sampler_cfg = (
                                                c.override_config.sampler_model_config
                                            )
                                            router = RouterModel(router_cfg)
                                            sampler = SamplerModel(sampler_cfg)

                                            logits = router.compute_logit_scores(input)
                                            probabilities, indices, _, _ = (
                                                sampler.sample_probabilities_and_indices(
                                                    logits
                                                )
                                            )

                                        loss = torch.tensor(0.0)
                                        inputs = {
                                            "input_batch": input,
                                            "probabilities": probabilities,
                                            "indices": indices,
                                            "loss": loss,
                                        }
                                        output, loss = m(inputs)

                                        expected_shape = (
                                            batch_size,
                                            c.output_dim,
                                        )
                                        self.assertEqual(output.shape, expected_shape)
                                        self.assertEqual(loss.item(), 0.0)


#  class TestMixtureOfExpertsFeedForward(unittest.TestCase):
# def setUp(self):
#     # MODEL WISE CONFI
#     BATCH_SIZE = 2
#     INPUT_DIM = 4
#     HIDDEN_DIM = 5
#     OUTPUT_DIM = 6
#     GATHER_FREQUENCY_FLAG = False
#
#     # PARAMETER GENRETOR ROUTER OPITONS
#     ROUTER_INPUT_DIM = HIDDEN_DIM
#     ROUTER_HIDDEN_DIM = 8
#     ROUTER_OUTPUT_DIM = 9
#     ROUTER_NOISY_TOPK_FLAG = False
#     ROUTER_ACTIVATION_FUNCTION = nn.ReLU()
#     ROUTER_NUM_LAYERNUM_LAYERSS = 5
#     ROUTER_DYNAMIC_LINEAR_MODEL_FLAG = False
#
#     # PARAMETER GENRETOR SAMPLER OPITONS
#     SAMPLER_TOP_K = 3
#     SAMPLER_THRESHOLD = 0.0
#     SAMPLER_FILTER_THRESHOLD = False
#     SAMPLER_DYNAMIC_TOPK_THRESHOLD = (1 / ROUTER_OUTPUT_DIM) * 1e-6
#     SAMPLER_NUM_TOPK_SAMPLES = 0
#     SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
#     SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
#     SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
#     SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT = 0.1
#     SAMPLER_SWITCH_WEIGHT = 0.1
#     SAMPLER_ZERO_CENTRED_WEIGHT = 0.1
#     SAMPLER_MUTUAL_INFORMATION_WEIGHT = 0.0
#
#     # PARAMETER GENRETOR MIXTURE OPITONS
#     MIXTURE_INPUT_DIM = ROUTER_INPUT_DIM
#     MIXTURE_OUTPUT_DIM = ROUTER_OUTPUT_DIM
#     MIXTURE_DEPTH_DIM = ROUTER_OUTPUT_DIM
#     MIXTURE_TOP_K = SAMPLER_TOP_K
#     MIXTURE_WEIGHTED_PARAMETERS_FLAG = False
#     MIXTURE_BIAS_PARAMETERS_FLAG = False
#     MIXTURE_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
#     MIXTURE_CROSS_DIAGONAL_FLAG = False
#
#     # PARAMETER GENERATOR OPTIONS
#     PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG = MIXTURE_WEIGHTED_PARAMETERS_FLAG
#
#     self.cfg = ModelConfig(
#         batch_size=BATCH_SIZE,
#         input_dim=INPUT_DIM,
#         hidden_dim=HIDDEN_DIM,
#         output_dim=OUTPUT_DIM,
#         gather_frequency_flag=GATHER_FREQUENCY_FLAG,
#         router_model_config=RouterConfig(
#             input_dim=ROUTER_INPUT_DIM,
#             hidden_dim=ROUTER_HIDDEN_DIM,
#             num_experts=ROUTER_OUTPUT_DIM,
#             noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
#             activation=ROUTER_ACTIVATION_FUNCTION,
#             num_layers=ROUTER_NUM_LAYERNUM_LAYERSS,
#             diagonal_model_type_flag=ROUTER_DYNAMIC_LINEAR_MODEL_FLAG,
#             residual_flag=False,
#         ),
#         sampler_model_config=SamplerConfig(
#             top_k=SAMPLER_TOP_K,
#             threshold=SAMPLER_THRESHOLD,
#             filter_above_threshold=SAMPLER_FILTER_THRESHOLD,
#             num_topk_samples=SAMPLER_NUM_TOPK_SAMPLES,
#             normalize_probabilities_flag=SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
#             noisy_topk_flag=SAMPLER_NOISY_TOPK_FLAG,
#             num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
#             coefficient_of_variation_loss_weight=SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT,
#             switch_loss_weight=SAMPLER_SWITCH_WEIGHT,
#             zero_centred_loss_weight=SAMPLER_ZERO_CENTRED_WEIGHT,
#             mutual_information_loss_weight=SAMPLER_MUTUAL_INFORMATION_WEIGHT,
#         ),
#         mixture_model_config=MixtureConfig(
#             input_dim=MIXTURE_INPUT_DIM,
#             output_dim=MIXTURE_OUTPUT_DIM,
#             depth_dim=MIXTURE_DEPTH_DIM,
#             top_k=MIXTURE_TOP_K,
#             bias_parameters_flag=MIXTURE_BIAS_PARAMETERS_FLAG,
#             weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
#             num_experts=MIXTURE_ROUTER_OUTPUT_DIM,
#             dynamic_diagonal_params_flag=MIXTURE_CROSS_DIAGONAL_FLAG,
#         ),
#         parameter_generator_model_config=ParameterLayerConfig(
#             bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
#             time_tracker_flag=False,
#             dynamic_diagonal_params_flag=False,
#         ),
#         mixture_of_experts_config=MixtureOfExpertsFeedForwardConfig(
#             weighted_parameters_flag=True,
#         ),
#         input_moe_layer_config=MixtureOfExpertsConfig(
#             input_dim=ROUTER_INPUT_DIM,
#             output_dim=64,
#             top_k=MIXTURE_TOP_K,
#             dropout_probability=0.1,
#             layer_norm_flag=True,
#             activation=ActivationOptions.GELU,
#             model_type=LinearLayerTypes.DYNAMIC,
#             num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
#             compute_expert_mixture_flag=False,
#             weighted_parameters_flag=False,
#             init_sampler_model_flag=False,
#         ),
#         output_moe_layer_config=MixtureOfExpertsConfig(
#             input_dim=64,
#             output_dim=ROUTER_INPUT_DIM,
#             top_k=MIXTURE_TOP_K,
#             dropout_probability=0.1,
#             layer_norm_flag=True,
#             activation=ActivationOptions.GELU,
#             model_type=LinearLayerTypes.DYNAMIC,
#             num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
#             compute_expert_mixture_flag=True,
#             weighted_parameters_flag=True,
#             init_sampler_model_flag=False,
#         ),
#     )
#
# def test__init_input_layer_with_default_config(self):
#     c = copy.deepcopy(self.cfg)
#     config = c.mixture_of_experts_config
#     m = MixtureOfExpertsFeedForward(c)
#
#     self.assertIsInstance(m, MixtureOfExpertsFeedForward)
#     self.assertEqual(m.weighted_parameters_flag, config.weighted_parameters_flag)
#
# def test__prepare_inputs(self):
#     c = copy.deepcopy(self.cfg)
#     m = MixtureOfExpertsFeedForward(c)
#
#     batch_size = 5
#     sequence_length = 6
#     embedding_dim = 7
#     input_batch = torch.randn(batch_size, sequence_length, embedding_dim)
#     output, skip_mask = m._prepare_inputs(input_batch)
#     output_combined_dims, output_embedding_dim = output.shape
#
#     self.assertEqual(output_combined_dims, batch_size * sequence_length)
#     self.assertEqual(output_embedding_dim, embedding_dim)
#     self.assertIsNone(skip_mask)
#
# def test__prepare_inputs__with__skip_mask(self):
#     c = copy.deepcopy(self.cfg)
#     m = MixtureOfExpertsFeedForward(c)
#
#     batch_size = 5
#     sequence_length = 6
#     embedding_dim = 7
#     input_batch = torch.randn(batch_size, sequence_length, embedding_dim)
#     skip_mask = torch.randint(0, 2, (batch_size, sequence_length))
#     output, reshaped_skip_ask = m._prepare_inputs(input_batch, skip_mask)
#     output_combined_dims, output_embedding_dim = output.shape
#
#     self.assertEqual(output_combined_dims, batch_size * sequence_length)
#     self.assertEqual(output_embedding_dim, embedding_dim)
#     self.assertTrue(torch.allclose(reshaped_skip_ask, skip_mask.view(-1, 1)))
#
# def test__resolve_output_shape__2d_input_tensor(self):
#     c = copy.deepcopy(self.cfg)
#     m = MixtureOfExpertsFeedForward(c)
#
#     batch_size = 5
#     embedding_dim = 7
#     input_batch = torch.randn(batch_size, embedding_dim)
#     m._MixtureOfExpertsFeedForward__resolve_output_shape(input_batch)
#
#     self.assertEqual(m.batch_size, batch_size)
#     self.assertEqual(m.sequence_length, 1)
#     self.assertEqual(m.output_shape, [batch_size, -1])
#
# def test__resolve_output_shape__3d_input_tensor(self):
#     c = copy.deepcopy(self.cfg)
#     m = MixtureOfExpertsFeedForward(c)
#
#     batch_size = 5
#     sequence_length = 6
#     embedding_dim = 7
#     input_batch = torch.randn(batch_size, sequence_length, embedding_dim)
#     m._MixtureOfExpertsFeedForward__resolve_output_shape(input_batch)
#
#     self.assertEqual(batch_size, m.batch_size)
#     self.assertEqual(sequence_length, m.sequence_length)
#     self.assertEqual(m.output_shape, [batch_size, sequence_length, -1])
#
# def test__forward__LinearLayer(self):
#     c = copy.deepcopy(self.cfg)
#     c.input_moe_layer_config.model_type = LinearLayerTypes.BASE
#     m = MixtureOfExpertsFeedForward(c)
#
#     batch_size = 7
#     sequence_length = 6
#     embedding_dim = 5
#     input_batch = torch.randn(batch_size, sequence_length, embedding_dim)
#     output, skip_mask, loss = m(input_batch)
#
#     self.assertIsInstance(output, torch.Tensor)
#     self.assertEqual(
#         list(output.shape), [batch_size, sequence_length, embedding_dim]
#     )
#     self.assertIsNone(skip_mask)
#     self.assertTrue(loss > 0.0)
#
# def test__forward__AdaptiveLinearLayer(self):
#     c = copy.deepcopy(self.cfg)
#     c.input_moe_layer_config.model_type = LinearLayerTypes.DYNAMIC
#     m = MixtureOfExpertsFeedForward(c)
#
#     batch_size = 7
#     sequence_length = 6
#     embedding_dim = 5
#     input_batch = torch.randn(batch_size, sequence_length, embedding_dim)
#     output, skip_mask, loss = m(input_batch)
#
#     self.assertIsInstance(output, torch.Tensor)
#     self.assertEqual(
#         list(output.shape), [batch_size, sequence_length, embedding_dim]
#     )
#     self.assertIsNone(skip_mask)
#     self.assertTrue(loss > 0.0)
#
# def test__forward__VectorParameterLayer(self):
#     c = copy.deepcopy(self.cfg)
#     c.input_moe_layer_config.model_type = ParameterGeneratorTypes.VECTOR
#     m = MixtureOfExpertsFeedForward(c)
#
#     batch_size = 7
#     sequence_length = 6
#     embedding_dim = 5
#     input_batch = torch.randn(batch_size, sequence_length, embedding_dim)
#     output, skip_mask, loss = m(input_batch)
#
#     self.assertIsInstance(output, torch.Tensor)
#     self.assertEqual(
#         list(output.shape), [batch_size, sequence_length, embedding_dim]
#     )
#     self.assertIsNone(skip_mask)
#     self.assertTrue(loss > 0.0)
#
# def test__forward__MatrixParameterLayer(self):
#     c = copy.deepcopy(self.cfg)
#     c.input_moe_layer_config.model_type = ParameterGeneratorTypes.MATRIX
#     m = MixtureOfExpertsFeedForward(c)
#
#     batch_size = 7
#     sequence_length = 6
#     embedding_dim = 5
#     input_batch = torch.randn(batch_size, sequence_length, embedding_dim)
#     output, skip_mask, loss = m(input_batch)
#
#     self.assertIsInstance(output, torch.Tensor)
#     self.assertEqual(
#         list(output.shape), [batch_size, sequence_length, embedding_dim]
#     )
#     self.assertIsNone(skip_mask)
#     self.assertTrue(loss > 0.0)
#
# def test__forward__GeneratorParameterLayer(self):
#     c = copy.deepcopy(self.cfg)
#     c.input_moe_layer_config.model_type = ParameterGeneratorTypes.GENERATOR
#     m = MixtureOfExpertsFeedForward(c)
#
#     batch_size = 7
#     sequence_length = 6
#     embedding_dim = 5
#     input_batch = torch.randn(batch_size, sequence_length, embedding_dim)
#     output, skip_mask, loss = m(input_batch)
#
#     self.assertIsInstance(output, torch.Tensor)
#     self.assertEqual(
#         list(output.shape), [batch_size, sequence_length, embedding_dim]
#     )
#     self.assertIsNone(skip_mask)
#     self.assertTrue(loss > 0.0)
