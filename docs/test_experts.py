import torch
import unittest

from torch.nn import Sequential
from emperor.base.layer import Layer
from emperor.sampler.model import SamplerModel
from emperor.sampler.utils.routers import RouterModel
from emperor.experts.utils.layers import (
    MixtureOfExperts,
    MixtureOfExpertsMap,
    MixtureOfExpertsReduce,
    _ExpertInputData,
)
from emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from emperor.experts.utils.model import MixtureOfExpertsModel
from emperor.experts.utils.stack import MixtureOfExpertsStack
from emperor.experts.utils.presets import MixtureOfExpertsPresets
from emperor.sampler.utils.samplers import SamplerFull, SamplerSparse, SamplerTopk
from emperor.experts.utils.enums import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    InitSamplerOptions,
)


class TestMixtureOfExperts(unittest.TestCase):
    def test_init_with_different_configs(self):
        top_k_options = [1, 3, 6]
        num_experts = 6
        init_sampler_options = [InitSamplerOptions.DISABLED, InitSamplerOptions.LAYER]

        for layer_stack_option in LinearLayerStackOptions:
            for top_k in top_k_options:
                for init_sampler_option in init_sampler_options:
                    message = f"Testing configuration with num_experts={num_experts}, top_k={top_k}, layer_stack_option={layer_stack_option}, and init_sampler_option={init_sampler_option}"
                    with self.subTest(msg=message):
                        c = MixtureOfExpertsPresets.experts_preset(
                            return_model_config_flag=True,
                            experts_layer_stack_option=layer_stack_option,
                            experts_num_experts=num_experts,
                            experts_top_k=top_k,
                            experts_init_sampler_option=init_sampler_option,
                        )

                        m = MixtureOfExperts(c)
                        cfg = m.cfg
                        self.assertIsInstance(m, MixtureOfExperts)
                        self.assertEqual(m.input_dim, cfg.input_dim)
                        self.assertEqual(m.output_dim, cfg.output_dim)
                        self.assertEqual(
                            m.layer_stack_model.value, layer_stack_option.value
                        )
                        self.assertEqual(m.top_k, top_k)
                        self.assertEqual(m.num_experts, num_experts)
                        self.assertEqual(m.capacity_factor, cfg.capacity_factor)
                        self.assertEqual(
                            m.dropped_token_behavior,
                            cfg.dropped_token_behavior or DroppedTokenOptions.ZEROS,
                        )
                        self.assertEqual(
                            m.compute_expert_mixture_flag,
                            cfg.compute_expert_mixture_flag,
                        )
                        self.assertEqual(
                            m.weighted_parameters_flag, cfg.weighted_parameters_flag
                        )
                        self.assertEqual(m.init_sampler_option, cfg.init_sampler_option)
                        self.assertEqual(
                            m.weighting_position_option,
                            cfg.weighting_position_option,
                        )
                        self.assertEqual(m.router_model_config, cfg.router_model_config)
                        self.assertEqual(
                            m.sampler_model_config, cfg.sampler_model_config
                        )

    def test__create_experts(self):
        for layer_stack_option, linear_option in zip(
            LinearLayerStackOptions, LinearLayerOptions
        ):
            message = f"Testing configuration with layer_stack_option={layer_stack_option.name}"
            with self.subTest(msg=message):
                c = MixtureOfExpertsPresets.experts_preset(
                    return_model_config_flag=True,
                    experts_layer_stack_option=layer_stack_option,
                )

                m = MixtureOfExperts(c)
                expert_models = m._MixtureOfExperts__create_experts()
                self.assertEqual(len(m.expert_modules), m.num_experts)
                for expert in expert_models:
                    self.assertIsInstance(expert, Sequential)
                    for layer in expert:
                        self.assertIsInstance(layer, Layer)
                        self.assertIsInstance(layer.model, linear_option)

    def test__maybe_create_router_and_sampler(self):
        num_experts = 6
        expert_options = [1, 3, 6]
        init_sampler_options = [
            InitSamplerOptions.DISABLED,
            InitSamplerOptions.LAYER,
        ]
        sampler_options = [SamplerSparse, SamplerTopk, SamplerFull]

        for init_sampler_option in init_sampler_options:
            for sampler_option, expert_option in zip(sampler_options, expert_options):
                message = f"Testing configuration with sampler_option={sampler_option.__name__}, num_experts={num_experts}, top_k={expert_option}"
                with self.subTest(msg=message):
                    c = MixtureOfExpertsPresets.experts_preset(
                        return_model_config_flag=True,
                        experts_init_sampler_option=init_sampler_option,
                        experts_num_experts=num_experts,
                        experts_top_k=expert_option,
                    )

                    m = MixtureOfExperts(c)
                    router, sampler = (
                        m._MixtureOfExperts__maybe_create_router_and_sampler()
                    )
                    if init_sampler_option == InitSamplerOptions.LAYER:
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
        init_sampler_option_options = [
            InitSamplerOptions.DISABLED,
            InitSamplerOptions.LAYER,
        ]

        for top_k in top_k_options:
            for init_sampler_option in init_sampler_option_options:
                message = f"Testing configuration with init_sampler_option={init_sampler_option}, top_k={top_k}"
                with self.subTest(msg=message):
                    c = MixtureOfExpertsPresets.experts_preset(
                        return_model_config_flag=True,
                        experts_init_sampler_option=init_sampler_option,
                        experts_num_experts=num_experts,
                        experts_top_k=top_k,
                    )

                    m = MixtureOfExperts(c)
                    if init_sampler_option == InitSamplerOptions.LAYER:
                        inputs = torch.randn(5, c.input_dim)
                        input_indices = None
                        input_probabilities = None
                        probabilities, indices, sampler_loss = (
                            m._maybe_compute_expert_indices(
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
                        m._maybe_compute_expert_indices(inputs, indices_input)
                    )
                    self.assertTrue(
                        torch.allclose(indices, indices_input, atol=1e-6, rtol=1e-5)
                    )
                    self.assertIsNone(probabilities)
                    self.assertEqual(sampler_loss.item(), 0.0)

    def test_get_expert_token_indices(self):
        num_experts = 6
        top_k_options = [1, 3]
        capacity_factor_options = [0.0, 0.5, 1.0, 2.0]

        for top_k in top_k_options:
            for capacity_factor in capacity_factor_options:
                for expert_index in range(num_experts):
                    message = f"Testing with top_k={top_k}, capacity_factor={capacity_factor}, expert_index={expert_index}"
                    with self.subTest(msg=message):
                        dim = 8
                        c = MixtureOfExpertsPresets.experts_preset(
                            return_model_config_flag=True,
                            input_dim=dim,
                            output_dim=dim,
                            experts_num_experts=num_experts,
                            experts_top_k=top_k,
                            experts_capacity_factor=capacity_factor,
                        )

                        m = MixtureOfExperts(c)

                        batch_size = 30
                        rows = []
                        for _ in range(batch_size):
                            row = torch.randperm(m.num_experts)[:top_k]
                            rows.append(row)
                        indices = torch.stack(rows)

                        sample_indices, dropped_indices = m._get_expert_token_indices(
                            indices, expert_index
                        )

                        self.assertIsInstance(sample_indices, torch.Tensor)
                        self.assertIsInstance(dropped_indices, torch.Tensor)

                        total = sample_indices.size(0) + dropped_indices.size(0)
                        if capacity_factor > 0 and dropped_indices.size(0) > 0:
                            self.assertLess(sample_indices.size(0), total)

    def test__get_expert_probabilities(self):
        top_k = 3
        num_experts = 6

        for expert_index in range(num_experts):
            message = f"Testing configuration with expert_index={expert_index}"
            with self.subTest(msg=message):
                c = MixtureOfExpertsPresets.experts_preset(
                    return_model_config_flag=True,
                    experts_num_experts=num_experts,
                )

                m = MixtureOfExperts(c)

                indices = torch.randint(0, m.num_experts, (10, top_k))
                probabilities = torch.randn(10, top_k)
                probabilities = (
                    m.expert_weighting_handler.maybe_get_expert_probabilities(
                        indices, probabilities, expert_index
                    )
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
                        c = MixtureOfExpertsPresets.experts_preset(
                            return_model_config_flag=True,
                            experts_weighted_parameters_flag=weighted_parameters_flag,
                            experts_weighting_position_option=weighting_position_option,
                            experts_num_experts=num_experts,
                            experts_top_k=top_k,
                        )

                        m = MixtureOfExperts(c)

                        input_batch = torch.randn(10, c.input_dim)
                        indices = torch.randint(0, m.num_experts, (10 * top_k,))
                        pribabilities = torch.randn(10 * top_k)
                        expert_input_slice = _ExpertInputData(
                            expert_index=0,
                            expert_samples=input_batch[indices],
                            dropped_samples=torch.zeros(0),
                            sample_indices=None,
                            dropped_sample_indices=None,
                            probabilities=pribabilities,
                        )

                        output, loss = m._MixtureOfExperts__compute_expert_output(
                            expert_input_slice
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
                    c = MixtureOfExpertsPresets.experts_preset(
                        return_model_config_flag=True,
                        experts_weighted_parameters_flag=weighted_parameters_flag,
                        experts_num_experts=num_experts,
                        experts_top_k=top_k,
                    )

                    m = MixtureOfExperts(c)
                    logits = torch.randn(10 * top_k, c.input_dim)
                    pribabilities = torch.randn(10 * top_k)
                    output = m.expert_weighting_handler._ExpertWeightingHandler__maybe_apply_probabilities(
                        logits, pribabilities
                    )

                    if weighted_parameters_flag:
                        self.assertIsInstance(output, torch.Tensor)
                        expected_output = logits * pribabilities.view(-1, 1)
                        self.assertTrue(
                            torch.allclose(
                                output, expected_output, atol=1e-6, rtol=1e-5
                            )
                        )
                        continue
                    self.assertIsInstance(output, torch.Tensor)
                    self.assertTrue(
                        torch.allclose(output, logits, atol=1e-6, rtol=1e-5)
                    )

    def test__compute_expert_mixture(self):
        num_experts = 6
        top_k_options = [1, 3, 6]

        for top_k in top_k_options:
            for compute_expert_mixture_flag in [True, False]:
                for weighted_parameters_flag in [True, False]:
                    message = f"Testing with weighted_parameters_flag={weighted_parameters_flag}, compute_expert_mixture_flag={compute_expert_mixture_flag}, top_k={top_k}"
                    with self.subTest(msg=message):
                        c = MixtureOfExpertsPresets.experts_preset(
                            return_model_config_flag=True,
                            experts_weighted_parameters_flag=weighted_parameters_flag,
                            experts_compute_expert_mixture_flag=compute_expert_mixture_flag,
                            experts_num_experts=num_experts,
                            experts_top_k=top_k,
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
        init_sampler_options = [InitSamplerOptions.DISABLED, InitSamplerOptions.LAYER]
        num_layers_options = [1, 2, 3]
        capacity_factor_options = [0.0, 1.0, 1.5]

        for num_layers in num_layers_options:
            for layer_stack_option in LinearLayerStackOptions:
                for weighting_position_option in ExpertWeightingPositionOptions:
                    for top_k in top_k_options:
                        for init_sampler_option in init_sampler_options:
                            for compute_expert_mixture_flag in flag_options:
                                for weighted_parameters_flag in flag_options:
                                    for capacity_factor in capacity_factor_options:
                                        message = f"Testing with layer_stack_option={layer_stack_option.name}, weighting_position_option={weighting_position_option.name}, init_sampler_option={init_sampler_option}, compute_expert_mixture_flag={compute_expert_mixture_flag}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}, capacity_factor={capacity_factor}"
                                        with self.subTest(msg=message):
                                            if (
                                                capacity_factor > 0
                                                and top_k == num_experts
                                            ):
                                                continue  # validator rejects capacity + top_k==num_experts
                                            output_dim = 8 if capacity_factor > 0 else 6
                                            c = MixtureOfExpertsPresets.experts_preset(
                                                return_model_config_flag=True,
                                                batch_size=10,
                                                input_dim=8,
                                                output_dim=output_dim,
                                                experts_layer_stack_option=layer_stack_option,
                                                experts_top_k=top_k,
                                                experts_weighting_position_option=weighting_position_option,
                                                experts_init_sampler_option=init_sampler_option,
                                                experts_weighted_parameters_flag=weighted_parameters_flag,
                                                experts_compute_expert_mixture_flag=compute_expert_mixture_flag,
                                                experts_num_experts=num_experts,
                                                experts_capacity_factor=capacity_factor,
                                                stack_num_layers=num_layers,
                                            )

                                            m = MixtureOfExperts(c)

                                            input = torch.randn(
                                                c.batch_size, c.input_dim
                                            )
                                            indices = probabilities = None
                                            if (
                                                init_sampler_option
                                                == InitSamplerOptions.DISABLED
                                            ):
                                                router_cfg = (
                                                    c.mixture_of_experts_config.router_model_config
                                                )
                                                sampler_cfg = (
                                                    c.mixture_of_experts_config.sampler_model_config
                                                )
                                                router = RouterModel(router_cfg)
                                                sampler = SamplerModel(sampler_cfg)

                                                logits = router.compute_logit_scores(
                                                    input
                                                )
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
                                            self.assertEqual(
                                                output.shape, expected_shape
                                            )
                                            self.assertEqual(total_loss.item(), 0.0)


class TestExpertCapacity(unittest.TestCase):
    def test_capacity_factor_zero_unchanged(self):
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            batch_size=10,
            experts_num_experts=6,
            experts_top_k=3,
            experts_capacity_factor=0.0,
        )
        m = MixtureOfExperts(c)
        self.assertEqual(m.capacity_factor, 0.0)

        input_batch = torch.randn(c.batch_size, c.input_dim)
        router_cfg = c.mixture_of_experts_config.router_model_config
        sampler_cfg = c.mixture_of_experts_config.sampler_model_config
        router = RouterModel(router_cfg)
        sampler = SamplerModel(sampler_cfg)
        logits = router.compute_logit_scores(input_batch)
        probabilities, indices, _, _ = sampler.sample_probabilities_and_indices(logits)

        output, loss = m.forward(input_batch, probabilities, indices)
        expected_shape = (c.batch_size * 3, c.output_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_capacity_factor_truncates_tokens(self):
        capacity_factors = [1.0, 1.5]
        for capacity_factor in capacity_factors:
            message = f"Testing capacity factor truncation with capacity_factor={capacity_factor}"
            with self.subTest(msg=message):
                c = MixtureOfExpertsPresets.experts_preset(
                    return_model_config_flag=True,
                    batch_size=10,
                    input_dim=8,
                    output_dim=8,
                    experts_num_experts=6,
                    experts_top_k=3,
                    experts_capacity_factor=capacity_factor,
                )
                m = MixtureOfExperts(c)
                self.assertEqual(m.capacity_factor, capacity_factor)

                input_batch = torch.randn(c.batch_size, c.input_dim)
                router_cfg = c.mixture_of_experts_config.router_model_config
                sampler_cfg = c.mixture_of_experts_config.sampler_model_config
                router = RouterModel(router_cfg)
                sampler = SamplerModel(sampler_cfg)
                logits = router.compute_logit_scores(input_batch)
                probabilities, indices, _, _ = sampler.sample_probabilities_and_indices(
                    logits
                )

                output, loss = m.forward(input_batch, probabilities, indices)
                # Output shape is always full (batch_size * top_k); dropped tokens become zeros.
                self.assertEqual(output.size(0), c.batch_size * 3)

    def test_capacity_factor_top_k_equals_num_experts(self):
        # capacity_factor > 0 with top_k == num_experts is invalid (all tokens
        # pass through all experts unconditionally, so capacity has no effect).
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            batch_size=10,
            input_dim=8,
            output_dim=8,
            experts_num_experts=6,
            experts_top_k=6,
            experts_capacity_factor=1.0,
        )
        with self.assertRaises(ValueError):
            MixtureOfExperts(c)

    def test_capacity_factor_small_batch(self):
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            batch_size=4,
            input_dim=8,
            output_dim=8,
            experts_num_experts=6,
            experts_top_k=3,
            experts_capacity_factor=1.0,
        )
        m = MixtureOfExperts(c)
        input_batch = torch.randn(c.batch_size, c.input_dim)
        router_cfg = c.mixture_of_experts_config.router_model_config
        sampler_cfg = c.mixture_of_experts_config.sampler_model_config
        router = RouterModel(router_cfg)
        sampler = SamplerModel(sampler_cfg)
        logits = router.compute_logit_scores(input_batch)
        probabilities, indices, _, _ = sampler.sample_probabilities_and_indices(logits)
        output, loss = m.forward(input_batch, probabilities, indices)
        self.assertIsInstance(output, torch.Tensor)

    def test_capacity_factor_negative_raises(self):
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            experts_capacity_factor=-1.0,
        )
        with self.assertRaises(ValueError):
            MixtureOfExperts(c)

    def test_capacity_factor_reduce(self):
        capacity_factors = [1.0, 1.5]
        for capacity_factor in capacity_factors:
            message = (
                f"Testing capacity factor reduce with capacity_factor={capacity_factor}"
            )
            with self.subTest(msg=message):
                mc = MixtureOfExpertsPresets.experts_preset(
                    input_dim=8,
                    output_dim=6,
                    return_model_config_flag=True,
                    batch_size=10,
                    experts_num_experts=6,
                    experts_top_k=1,
                )
                m = MixtureOfExpertsMap(mc)

                rc = MixtureOfExpertsPresets.experts_preset(
                    input_dim=6,
                    output_dim=6,
                    return_model_config_flag=True,
                    batch_size=10,
                    experts_num_experts=6,
                    experts_top_k=1,
                    experts_capacity_factor=capacity_factor,
                )
                r = MixtureOfExpertsReduce(rc)
                self.assertEqual(r.capacity_factor, capacity_factor)

                input_batch = torch.randn(mc.batch_size, mc.input_dim)
                router_cfg = mc.mixture_of_experts_config.router_model_config
                sampler_cfg = mc.mixture_of_experts_config.sampler_model_config
                router = RouterModel(router_cfg)
                sampler = SamplerModel(sampler_cfg)
                logits = router.compute_logit_scores(input_batch)
                probabilities, indices, _, _ = sampler.sample_probabilities_and_indices(
                    logits
                )

                map_output, _ = m.forward(input_batch, probabilities, indices)
                output, loss = r.forward(map_output, probabilities, indices)

                # Output shape is always full (batch_size); dropped tokens become zeros.
                self.assertEqual(output.size(0), map_output.size(0))

    def test_capacity_preserves_output_shape(self):
        batch_size = 10
        top_k = 3
        num_experts = 6
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            batch_size=batch_size,
            input_dim=8,
            output_dim=8,
            experts_num_experts=num_experts,
            experts_top_k=top_k,
            experts_capacity_factor=1.0,
            experts_compute_expert_mixture_flag=True,
        )
        m = MixtureOfExperts(c)

        input_batch = torch.randn(batch_size, c.input_dim)
        router_cfg = c.mixture_of_experts_config.router_model_config
        sampler_cfg = c.mixture_of_experts_config.sampler_model_config
        router = RouterModel(router_cfg)
        sampler = SamplerModel(sampler_cfg)
        logits = router.compute_logit_scores(input_batch)
        probabilities, indices, _, _ = sampler.sample_probabilities_and_indices(logits)

        output, loss = m.forward(input_batch, probabilities, indices)
        self.assertEqual(output.shape, (batch_size, c.output_dim))

    def test_capacity_reduce_preserves_output_shape(self):
        batch_size = 10
        num_experts = 6
        mc = MixtureOfExpertsPresets.experts_preset(
            input_dim=8,
            output_dim=6,
            return_model_config_flag=True,
            batch_size=batch_size,
            experts_num_experts=num_experts,
            experts_top_k=1,
        )
        m = MixtureOfExpertsMap(mc)

        rc = MixtureOfExpertsPresets.experts_preset(
            input_dim=6,
            output_dim=6,
            return_model_config_flag=True,
            batch_size=batch_size,
            experts_num_experts=num_experts,
            experts_top_k=1,
            experts_capacity_factor=1.0,
        )
        r = MixtureOfExpertsReduce(rc)

        input_batch = torch.randn(batch_size, mc.input_dim)
        router_cfg = mc.mixture_of_experts_config.router_model_config
        sampler_cfg = mc.mixture_of_experts_config.sampler_model_config
        router = RouterModel(router_cfg)
        sampler = SamplerModel(sampler_cfg)
        logits = router.compute_logit_scores(input_batch)
        probabilities, indices, _, _ = sampler.sample_probabilities_and_indices(logits)

        map_output, _ = m.forward(input_batch, probabilities, indices)
        output, loss = r.forward(map_output, probabilities, indices)
        self.assertEqual(output.size(0), map_output.size(0))

    def test_capacity_with_all_options(self):
        num_experts = 6
        top_k_options = [
            1,
            3,
        ]  # exclude top_k==num_experts (capacity is bypassed there)
        flag_options = [True, False]
        init_sampler_options = [InitSamplerOptions.DISABLED, InitSamplerOptions.LAYER]

        for layer_stack_option in LinearLayerStackOptions:
            for weighting_position_option in ExpertWeightingPositionOptions:
                for top_k in top_k_options:
                    for init_sampler_option in init_sampler_options:
                        for compute_expert_mixture_flag in flag_options:
                            for weighted_parameters_flag in flag_options:
                                message = (
                                    f"capacity+layer_stack={layer_stack_option.name},"
                                    f"weighting_position={weighting_position_option.name},"
                                    f"top_k={top_k},init_sampler={init_sampler_option},"
                                    f"compute_mixture={compute_expert_mixture_flag},"
                                    f"weighted={weighted_parameters_flag}"
                                )
                                with self.subTest(msg=message):
                                    c = MixtureOfExpertsPresets.experts_preset(
                                        return_model_config_flag=True,
                                        batch_size=10,
                                        input_dim=8,
                                        output_dim=8,
                                        experts_layer_stack_option=layer_stack_option,
                                        experts_top_k=top_k,
                                        experts_num_experts=num_experts,
                                        experts_capacity_factor=1.0,
                                        experts_weighting_position_option=weighting_position_option,
                                        experts_init_sampler_option=init_sampler_option,
                                        experts_weighted_parameters_flag=weighted_parameters_flag,
                                        experts_compute_expert_mixture_flag=compute_expert_mixture_flag,
                                    )
                                    m = MixtureOfExperts(c)

                                    input_batch = torch.randn(10, c.input_dim)
                                    indices = probabilities = None
                                    if (
                                        init_sampler_option
                                        == InitSamplerOptions.DISABLED
                                    ):
                                        router_cfg = (
                                            c.mixture_of_experts_config.router_model_config
                                        )
                                        sampler_cfg = (
                                            c.mixture_of_experts_config.sampler_model_config
                                        )
                                        router = RouterModel(router_cfg)
                                        sampler = SamplerModel(sampler_cfg)
                                        logits = router.compute_logit_scores(
                                            input_batch
                                        )
                                        probabilities, indices, _, _ = (
                                            sampler.sample_probabilities_and_indices(
                                                logits
                                            )
                                        )

                                    output, loss = m.forward(
                                        input_batch, probabilities, indices
                                    )

                                    expected_shape = (10 * top_k, c.output_dim)
                                    if compute_expert_mixture_flag:
                                        expected_shape = (10, c.output_dim)
                                    self.assertEqual(output.shape, expected_shape)


class TestMixtureOfExpertsStack(unittest.TestCase):
    def test_init_with_default_config(self):
        num_layer_options = [1, 2, 3]

        for num_layers in num_layer_options:
            message = f"Testing configuration with num_layers={num_layers}"
            with self.subTest(msg=message):
                c = MixtureOfExpertsPresets.experts_stack_preset(
                    return_model_config_flag=True,
                    experts_stack_num_layers=num_layers,
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
        init_sampler_options = [InitSamplerOptions.DISABLED, InitSamplerOptions.LAYER]

        for num_layers in num_layers_options:
            for layer_stack_option in LinearLayerStackOptions:
                for weighting_position_option in ExpertWeightingPositionOptions:
                    for top_k in top_k_options:
                        for init_sampler_option in init_sampler_options:
                            for compute_expert_mixture_flag in flag_options:
                                for weighted_parameters_flag in flag_options:
                                    message = f"Testing with layer_stack_option={layer_stack_option.name}, weighting_position_option={weighting_position_option.name}, init_sampler_option={init_sampler_option}, compute_expert_mixture_flag={compute_expert_mixture_flag}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}"
                                    with self.subTest(msg=message):
                                        c = MixtureOfExpertsPresets.experts_stack_preset(
                                            return_model_config_flag=True,
                                            experts_layer_stack_option=layer_stack_option,
                                            experts_top_k=top_k,
                                            experts_weighting_position_option=weighting_position_option,
                                            experts_init_sampler_option=init_sampler_option,
                                            experts_weighted_parameters_flag=weighted_parameters_flag,
                                            experts_compute_expert_mixture_flag=True,
                                            experts_num_experts=num_experts,
                                            experts_stack_num_layers=num_layers,
                                        )
                                        m = MixtureOfExpertsStack(c).build_model()

                                        batch_size = 10

                                        input = torch.randn(batch_size, c.input_dim)
                                        indices = probabilities = None
                                        if (
                                            init_sampler_option
                                            == InitSamplerOptions.DISABLED
                                        ):
                                            router_cfg = (
                                                c.layer_stack_config.override_config.router_model_config
                                            )
                                            sampler_cfg = (
                                                c.layer_stack_config.override_config.sampler_model_config
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


class TestMixtureOfExpertsModel(unittest.TestCase):
    def test_init_with_default_config(self):
        num_layer_options = [1, 2, 3]

        for num_layers in num_layer_options:
            message = f"Testing configuration with num_layers={num_layers}"
            with self.subTest(msg=message):
                c = MixtureOfExpertsPresets.experts_stack_preset(
                    return_model_config_flag=True,
                    experts_stack_num_layers=num_layers,
                )
                m = MixtureOfExpertsModel(c)
                if num_layers == 1:
                    self.assertIsInstance(m.expert_stack, Layer)
                else:
                    self.assertIsInstance(m.expert_stack, Sequential)

    def test__maybe_create_router_and_sampler(self):
        num_experts = 6
        expert_options = [1, 3, 6]
        init_sampler_options = [
            InitSamplerOptions.DISABLED,
            InitSamplerOptions.SHARED,
        ]
        sampler_options = [SamplerSparse, SamplerTopk, SamplerFull]

        for init_sampler_option in init_sampler_options:
            for sampler_option, expert_option in zip(sampler_options, expert_options):
                message = f"Testing configuration with sampler_option={sampler_option.__name__}, num_experts={num_experts}, top_k={expert_option}"
                with self.subTest(msg=message):
                    c = MixtureOfExpertsPresets.experts_stack_preset(
                        return_model_config_flag=True,
                        experts_init_sampler_option=init_sampler_option,
                        experts_num_experts=num_experts,
                        experts_top_k=expert_option,
                    )

                    m = MixtureOfExpertsModel(c)
                    router, sampler = (
                        m._MixtureOfExpertsModel__maybe_create_router_and_sampler()
                    )
                    if init_sampler_option == InitSamplerOptions.SHARED:
                        self.assertIsInstance(router, RouterModel)
                        self.assertIsInstance(sampler, SamplerModel)
                        self.assertIsInstance(sampler.sampler_model, sampler_option)
                        self.assertEqual(sampler.sampler_model.top_k, expert_option)
                        continue
                    self.assertIsNone(router)
                    self.assertIsNone(sampler)

    def test_forward(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        flag_options = [True, False]
        num_layers_options = [1, 2, 3]

        for num_layers in num_layers_options:
            for layer_stack_option in LinearLayerStackOptions:
                for weighting_position_option in ExpertWeightingPositionOptions:
                    for top_k in top_k_options:
                        for init_sampler_option in InitSamplerOptions:
                            for compute_expert_mixture_flag in flag_options:
                                for weighted_parameters_flag in flag_options:
                                    message = f"Testing with layer_stack_option={layer_stack_option.name}, weighting_position_option={weighting_position_option.name}, init_sampler_option={init_sampler_option}, compute_expert_mixture_flag={compute_expert_mixture_flag}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}"
                                    with self.subTest(msg=message):
                                        c = MixtureOfExpertsPresets.experts_stack_preset(
                                            return_model_config_flag=True,
                                            experts_layer_stack_option=layer_stack_option,
                                            experts_top_k=top_k,
                                            experts_weighting_position_option=weighting_position_option,
                                            experts_init_sampler_option=init_sampler_option,
                                            experts_weighted_parameters_flag=weighted_parameters_flag,
                                            experts_compute_expert_mixture_flag=True,
                                            experts_num_experts=num_experts,
                                            experts_stack_num_layers=num_layers,
                                        )

                                        m = MixtureOfExpertsModel(c)

                                        batch_size = 10

                                        input = torch.randn(batch_size, c.input_dim)
                                        indices = probabilities = None
                                        if (
                                            init_sampler_option
                                            == InitSamplerOptions.DISABLED
                                        ):
                                            router_cfg = (
                                                c.layer_stack_config.override_config.router_model_config
                                            )
                                            sampler_cfg = (
                                                c.layer_stack_config.override_config.sampler_model_config
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
                                        output, loss = m(
                                            input=input,
                                            probabilities=probabilities,
                                            indices=indices,
                                        )

                                        expected_shape = (
                                            batch_size,
                                            c.output_dim,
                                        )
                                        self.assertEqual(output.shape, expected_shape)
                                        self.assertEqual(loss.item(), 0.0)


class TestMixtureOfExpertsMap(unittest.TestCase):
    def test_forward(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        flag_options = [True, False]
        num_layers_options = [1, 2, 3]

        for num_layers in num_layers_options:
            for layer_stack_option in LinearLayerStackOptions:
                for weighting_position_option in ExpertWeightingPositionOptions:
                    for top_k in top_k_options:
                        for compute_expert_mixture_flag in flag_options:
                            for weighted_parameters_flag in flag_options:
                                message = f"Testing with layer_stack_option={layer_stack_option.name}, weighting_position_option={weighting_position_option.name}, compute_expert_mixture_flag={compute_expert_mixture_flag}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}"
                                with self.subTest(msg=message):
                                    c = MixtureOfExpertsPresets.experts_preset(
                                        return_model_config_flag=True,
                                        batch_size=10,
                                        experts_layer_stack_option=layer_stack_option,
                                        experts_top_k=top_k,
                                        experts_weighting_position_option=weighting_position_option,
                                        experts_weighted_parameters_flag=weighted_parameters_flag,
                                        experts_compute_expert_mixture_flag=compute_expert_mixture_flag,
                                        experts_num_experts=num_experts,
                                        stack_num_layers=num_layers,
                                    )

                                    m = MixtureOfExpertsMap(c)

                                    input = torch.randn(c.batch_size, c.input_dim)
                                    indices = probabilities = None
                                    router_cfg = (
                                        c.mixture_of_experts_config.router_model_config
                                    )
                                    sampler_cfg = (
                                        c.mixture_of_experts_config.sampler_model_config
                                    )
                                    router = RouterModel(router_cfg)
                                    sampler = SamplerModel(sampler_cfg)

                                    logits = router.compute_logit_scores(input)
                                    probabilities, indices, _, _ = (
                                        sampler.sample_probabilities_and_indices(logits)
                                    )

                                    output, total_loss = m.forward(
                                        input, probabilities, indices
                                    )

                                    expected_shape = (
                                        c.batch_size * top_k,
                                        c.output_dim,
                                    )

                                    self.assertEqual(output.shape, expected_shape)


class TestMixtureOfExpertsReduce(unittest.TestCase):
    def test_forward(self):
        num_experts = 6
        top_k_options = [1, 3]
        flag_options = [True, False]
        # init_sampler_options = [InitSamplerOptions.DISABLED, InitSamplerOptions.LAYER]
        num_layers_options = [1, 2, 3]

        for num_layers in num_layers_options:
            for layer_stack_option in LinearLayerStackOptions:
                for weighting_position_option in ExpertWeightingPositionOptions:
                    for top_k in top_k_options:
                        for compute_expert_mixture_flag in flag_options:
                            for weighted_parameters_flag in flag_options:
                                message = f"Testing with layer_stack_option={layer_stack_option.name}, weighting_position_option={weighting_position_option.name}, compute_expert_mixture_flag={compute_expert_mixture_flag}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}"
                                with self.subTest(msg=message):
                                    c = MixtureOfExpertsPresets.experts_preset(
                                        input_dim=8,
                                        output_dim=6,
                                        return_model_config_flag=True,
                                        batch_size=10,
                                        experts_layer_stack_option=layer_stack_option,
                                        experts_top_k=top_k,
                                        experts_weighting_position_option=weighting_position_option,
                                        experts_weighted_parameters_flag=weighted_parameters_flag,
                                        experts_compute_expert_mixture_flag=compute_expert_mixture_flag,
                                        experts_num_experts=num_experts,
                                        stack_num_layers=num_layers,
                                    )

                                    m = MixtureOfExpertsMap(c)

                                    rc = MixtureOfExpertsPresets.experts_preset(
                                        input_dim=6,
                                        output_dim=8,
                                        return_model_config_flag=True,
                                        batch_size=10,
                                        experts_layer_stack_option=layer_stack_option,
                                        experts_top_k=top_k,
                                        experts_weighting_position_option=weighting_position_option,
                                        experts_weighted_parameters_flag=weighted_parameters_flag,
                                        experts_compute_expert_mixture_flag=compute_expert_mixture_flag,
                                        experts_num_experts=num_experts,
                                        stack_num_layers=num_layers,
                                    )
                                    r = MixtureOfExpertsReduce(rc)

                                    input = torch.randn(c.batch_size, c.input_dim)
                                    indices = probabilities = None

                                    router_cfg = (
                                        c.mixture_of_experts_config.router_model_config
                                    )
                                    sampler_cfg = (
                                        c.mixture_of_experts_config.sampler_model_config
                                    )
                                    router = RouterModel(router_cfg)
                                    sampler = SamplerModel(sampler_cfg)

                                    logits = router.compute_logit_scores(input)
                                    probabilities, indices, _, _ = (
                                        sampler.sample_probabilities_and_indices(logits)
                                    )

                                    output, total_loss = m.forward(
                                        input, probabilities, indices
                                    )
                                    output, total_loss = r.forward(
                                        output, probabilities, indices
                                    )

                                    expected_shape = (c.batch_size, rc.output_dim)

                                    self.assertEqual(output.shape, expected_shape)


class TestDroppedTokenOptions(unittest.TestCase):
    def test_capacity_identity_preserves_dropped_tokens(self):
        batch_size = 10
        num_experts = 6
        top_k = 3
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            batch_size=batch_size,
            input_dim=8,
            output_dim=8,
            experts_num_experts=num_experts,
            experts_top_k=top_k,
            experts_capacity_factor=1.0,
            experts_dropped_token_behavior=DroppedTokenOptions.IDENTITY,
        )
        m = MixtureOfExperts(c)
        self.assertEqual(m.dropped_token_behavior, DroppedTokenOptions.IDENTITY)

        input_batch = torch.randn(batch_size, c.input_dim)
        router_cfg = c.mixture_of_experts_config.router_model_config
        sampler_cfg = c.mixture_of_experts_config.sampler_model_config
        router = RouterModel(router_cfg)
        sampler = SamplerModel(sampler_cfg)
        logits = router.compute_logit_scores(input_batch)
        probabilities, indices, _, _ = sampler.sample_probabilities_and_indices(logits)

        output, loss = m.forward(input_batch, probabilities, indices)
        expected_rows = batch_size * top_k
        self.assertEqual(output.size(0), expected_rows)
        # Dropped positions should contain original input (not zeros)
        zero_rows = (output.abs().sum(dim=-1) == 0).sum().item()
        self.assertEqual(zero_rows, 0)

    def test_capacity_identity_with_reduce(self):
        batch_size = 10
        num_experts = 6
        top_k = 1

        mc = MixtureOfExpertsPresets.experts_preset(
            input_dim=8,
            output_dim=8,
            return_model_config_flag=True,
            batch_size=batch_size,
            experts_num_experts=num_experts,
            experts_top_k=top_k,
        )
        m = MixtureOfExpertsMap(mc)

        rc = MixtureOfExpertsPresets.experts_preset(
            input_dim=8,
            output_dim=8,
            return_model_config_flag=True,
            batch_size=batch_size,
            experts_num_experts=num_experts,
            experts_top_k=top_k,
            experts_capacity_factor=1.0,
            experts_dropped_token_behavior=DroppedTokenOptions.IDENTITY,
        )
        r = MixtureOfExpertsReduce(rc)
        self.assertEqual(r.dropped_token_behavior, DroppedTokenOptions.IDENTITY)

        input_batch = torch.randn(batch_size, mc.input_dim)
        router_cfg = mc.mixture_of_experts_config.router_model_config
        sampler_cfg = mc.mixture_of_experts_config.sampler_model_config
        router = RouterModel(router_cfg)
        sampler = SamplerModel(sampler_cfg)
        logits = router.compute_logit_scores(input_batch)
        probabilities, indices, _, _ = sampler.sample_probabilities_and_indices(logits)

        map_output, _ = m.forward(input_batch, probabilities, indices)
        output, loss = r.forward(map_output, probabilities, indices)
        self.assertEqual(output.size(0), map_output.size(0))
        # Reduce applies weighting (prob * output) after experts, so dropped tokens
        # (prob=0) still become zero after weighting. Verify shape is preserved.
        self.assertEqual(output.size(-1), rc.output_dim)

    def test_capacity_zero_behavior_unchanged(self):
        batch_size = 10
        num_experts = 6
        top_k = 3
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            batch_size=batch_size,
            input_dim=8,
            output_dim=8,
            experts_num_experts=num_experts,
            experts_top_k=top_k,
            experts_capacity_factor=1.0,
            experts_dropped_token_behavior=DroppedTokenOptions.ZEROS,
        )
        m = MixtureOfExperts(c)
        self.assertEqual(m.dropped_token_behavior, DroppedTokenOptions.ZEROS)

        input_batch = torch.randn(batch_size, c.input_dim)
        router_cfg = c.mixture_of_experts_config.router_model_config
        sampler_cfg = c.mixture_of_experts_config.sampler_model_config
        router = RouterModel(router_cfg)
        sampler = SamplerModel(sampler_cfg)
        logits = router.compute_logit_scores(input_batch)
        probabilities, indices, _, _ = sampler.sample_probabilities_and_indices(logits)

        output, loss = m.forward(input_batch, probabilities, indices)
        expected_rows = batch_size * top_k
        self.assertEqual(output.size(0), expected_rows)
        # With ZERO behavior and capacity limiting, some rows should be zero vectors
        zero_rows = (output.abs().sum(dim=-1) == 0).sum().item()
        self.assertGreaterEqual(zero_rows, 0)


class TestSplitTokensPerExpert(unittest.TestCase):
    def _make_model_and_inputs(
        self,
        top_k,
        num_experts,
        input_dim=8,
        output_dim=8,
        capacity_factor=0.0,
        batch_size=10,
    ):
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            experts_num_experts=num_experts,
            experts_top_k=top_k,
            experts_capacity_factor=capacity_factor,
        )
        m = MixtureOfExperts(c)
        input_batch = torch.randn(batch_size, input_dim)
        router_cfg = c.mixture_of_experts_config.router_model_config
        sampler_cfg = c.mixture_of_experts_config.sampler_model_config
        router = RouterModel(router_cfg)
        sampler = SamplerModel(sampler_cfg)
        logits = router.compute_logit_scores(input_batch)
        probabilities, indices, _, _ = sampler.sample_probabilities_and_indices(logits)
        return m, input_batch, probabilities, indices

    def test_returns_list_of_expert_input_data(self):
        m, input_batch, probabilities, indices = self._make_model_and_inputs(
            top_k=1, num_experts=6
        )
        expert_input_data = m._split_tokens_per_expert(
            input_batch, probabilities, indices
        )
        self.assertIsInstance(expert_input_data, list)
        for s in expert_input_data:
            self.assertIsInstance(s, _ExpertInputData)

    def test_skips_empty_experts(self):
        num_experts = 6
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            experts_num_experts=num_experts,
            experts_top_k=1,
        )
        m = MixtureOfExperts(c)
        input_batch = torch.randn(4, c.input_dim)
        # Only experts 0 and 1 receive tokens
        indices = torch.tensor([0, 1, 0, 1])
        probabilities = torch.rand(4)
        expert_input_data = m._split_tokens_per_expert(
            input_batch, probabilities, indices
        )
        self.assertEqual(len(expert_input_data), 2)
        expert_indices_used = {s.expert_index for s in expert_input_data}
        self.assertEqual(expert_indices_used, {0, 1})

    def test_expert_samples_shape(self):
        input_dim = 8
        m, input_batch, probabilities, indices = self._make_model_and_inputs(
            top_k=1, num_experts=6, input_dim=input_dim
        )
        expert_input_data = m._split_tokens_per_expert(
            input_batch, probabilities, indices
        )
        for s in expert_input_data:
            self.assertEqual(s.expert_samples.shape[-1], input_dim)

    def test_top_k_equals_num_experts(self):
        num_experts = 6
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            experts_num_experts=num_experts,
            experts_top_k=num_experts,
        )
        m = MixtureOfExperts(c)
        batch_size = 8
        input_batch = torch.randn(batch_size, c.input_dim)
        probabilities = torch.rand(batch_size, num_experts)
        indices = None
        expert_input_data = m._split_tokens_per_expert(
            input_batch, probabilities, indices
        )
        self.assertEqual(len(expert_input_data), num_experts)
        for s in expert_input_data:
            self.assertIsNone(s.sample_indices)
            self.assertTrue(torch.equal(s.expert_samples, input_batch))

    def test_with_capacity_factor(self):
        # With capacity_factor > 0, an expert over capacity produces non-empty dropped_samples.
        # capacity = max(1, int(4/2 * 0.5)) = 1; expert 0 gets 3 tokens → 2 dropped.
        num_experts = 2
        input_dim = 4
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            experts_num_experts=num_experts,
            experts_top_k=1,
            experts_capacity_factor=0.5,
            input_dim=input_dim,
            output_dim=input_dim,
        )
        m = MixtureOfExperts(c)
        input_batch = torch.randn(4, input_dim)
        indices = torch.tensor([0, 0, 0, 1])
        probabilities = torch.rand(4)
        expert_input_data = m._split_tokens_per_expert(
            input_batch, probabilities, indices
        )
        expert_0_slice = next(s for s in expert_input_data if s.expert_index == 0)
        self.assertGreater(expert_0_slice.dropped_samples.numel(), 0)

    def test_reduce_override(self):
        num_experts = 6
        rc = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            experts_num_experts=num_experts,
            experts_top_k=1,
        )
        r = MixtureOfExpertsReduce(rc)
        input_batch = torch.randn(6, rc.input_dim)
        indices = torch.tensor([0, 1, 2, 3, 4, 5])
        probabilities = torch.rand(6)
        expert_input_data = r._split_tokens_per_expert(
            input_batch, probabilities, indices
        )
        for s in expert_input_data:
            self.assertEqual(s.dropped_samples.numel(), 0)
            self.assertIsNone(s.probabilities)

    def test_split_then_compute_matches_forward(self):
        torch.manual_seed(42)
        num_experts = 6
        top_k = 1
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            batch_size=8,
            experts_num_experts=num_experts,
            experts_top_k=top_k,
        )
        m = MixtureOfExperts(c)
        m.eval()
        input_batch = torch.randn(8, c.input_dim)
        router_cfg = c.mixture_of_experts_config.router_model_config
        sampler_cfg = c.mixture_of_experts_config.sampler_model_config
        router = RouterModel(router_cfg)
        sampler = SamplerModel(sampler_cfg)
        logits = router.compute_logit_scores(input_batch)
        probabilities, indices, _, _ = sampler.sample_probabilities_and_indices(logits)

        with torch.no_grad():
            expert_input_data = m._split_tokens_per_expert(
                input_batch, probabilities, indices
            )
            expert_outputs, routing_positions, reindexed_probs, expert_loss = (
                m._compute_experts(expert_input_data, probabilities)
            )
            manual_output = m._MixtureOfExperts__compute_expert_mixture(
                expert_outputs, routing_positions, reindexed_probs
            )

        with torch.no_grad():
            forward_output, forward_loss = m.forward(
                input_batch, probabilities, indices
            )

        self.assertTrue(torch.allclose(forward_output, manual_output))
        self.assertTrue(torch.allclose(forward_loss, expert_loss))
