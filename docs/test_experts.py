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
    MixtureOfExpertsConfig,
    _ExpertInputData,
)
from emperor.experts.utils._expert_capacity import _ExpertCapacityHandler
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
                    elif init_sampler_option == InitSamplerOptions.DISABLED:
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
                            expected_capacity = max(
                                1, int(batch_size / num_experts * capacity_factor)
                            )
                            self.assertLess(sample_indices.size(0), total)
                            self.assertEqual(sample_indices.size(0), expected_capacity)
                            self.assertEqual(
                                dropped_indices.size(0), total - expected_capacity
                            )

    def test_get_expert_routing_positions(self):
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
                        if top_k == 1:
                            indices = indices.squeeze(-1)

                        m._get_expert_token_indices(indices, expert_index)
                        sample_positions, dropped_positions = (
                            m._get_expert_routing_positions(indices, expert_index)
                        )

                        self.assertIsInstance(sample_positions, torch.Tensor)
                        self.assertIsInstance(dropped_positions, torch.Tensor)

                        total = sample_positions.size(0) + dropped_positions.size(0)
                        if capacity_factor > 0 and dropped_positions.size(0) > 0:
                            expected_capacity = max(
                                1, int(batch_size / num_experts * capacity_factor)
                            )
                            self.assertLess(sample_positions.size(0), total)
                            self.assertEqual(
                                sample_positions.size(0), expected_capacity
                            )
                            self.assertEqual(
                                dropped_positions.size(0), total - expected_capacity
                            )

    def test_maybe_get_expert_probabilities(self):
        num_experts = 6
        top_k_options = [1, 3, num_experts]

        for weighting_position_option in ExpertWeightingPositionOptions:
            for top_k in top_k_options:
                for expert_index in range(num_experts):
                    message = f"Testing with weighting_position_option={weighting_position_option.name}, top_k={top_k}, expert_index={expert_index}"
                    with self.subTest(msg=message):
                        c = MixtureOfExpertsPresets.experts_preset(
                            return_model_config_flag=True,
                            experts_num_experts=num_experts,
                            experts_top_k=top_k,
                            experts_weighting_position_option=weighting_position_option,
                        )

                        m = MixtureOfExperts(c)

                        batch_size = 10
                        probabilities = torch.randn(batch_size, top_k)

                        if top_k == num_experts:
                            indices = None
                        else:
                            indices = torch.randperm(batch_size * top_k)[:batch_size]

                        result = (
                            m.expert_weighting_handler.maybe_get_expert_probabilities(
                                indices, probabilities, expert_index
                            )
                        )

                        if (
                            weighting_position_option
                            == ExpertWeightingPositionOptions.AFTER_EXPERTS
                        ):
                            self.assertIsNone(result)
                        elif (
                            weighting_position_option
                            == ExpertWeightingPositionOptions.BEFORE_EXPERTS
                        ):
                            self.assertIsInstance(result, torch.Tensor)
                            if top_k == num_experts:
                                self.assertTrue(
                                    torch.equal(result, probabilities[:, expert_index])
                                )
                            else:
                                self.assertTrue(
                                    torch.equal(
                                        result, probabilities.flatten()[indices]
                                    )
                                )

    def test_select_expert_and_dropped_samples(self):
        num_experts = 6
        top_k = 3
        dropped_indices_options = [
            torch.tensor([2, 5, 8]),
            torch.tensor([], dtype=torch.long),
        ]

        for dropped_token_behavior in DroppedTokenOptions:
            for dropped_indices in dropped_indices_options:
                message = f"Testing with dropped_token_behavior={dropped_token_behavior.name}, dropped_indices_size={dropped_indices.size(0)}"
                with self.subTest(msg=message):
                    input_dim = 8
                    c = MixtureOfExpertsPresets.experts_preset(
                        return_model_config_flag=True,
                        input_dim=input_dim,
                        output_dim=input_dim,
                        experts_num_experts=num_experts,
                        experts_top_k=top_k,
                        experts_dropped_token_behavior=dropped_token_behavior,
                    )

                    m = MixtureOfExperts(c)

                    batch_size = 10
                    input_batch = torch.randn(batch_size, input_dim)
                    indices = torch.randperm(batch_size)[:top_k]

                    expert_samples, dropped_samples = (
                        m.capacity_handler.select_expert_and_dropped_samples(
                            input_batch, indices, dropped_indices
                        )
                    )

                    self.assertIsInstance(expert_samples, torch.Tensor)
                    self.assertIsInstance(dropped_samples, torch.Tensor)
                    self.assertTrue(torch.equal(expert_samples, input_batch[indices]))

                    if dropped_token_behavior == DroppedTokenOptions.ZEROS:
                        self.assertTrue(
                            torch.equal(
                                dropped_samples,
                                torch.zeros_like(input_batch[dropped_indices]),
                            )
                        )
                    else:
                        self.assertTrue(
                            torch.equal(dropped_samples, input_batch[dropped_indices])
                        )

    def test__build_routed_expert_inputs(self):
        num_experts = 6
        top_k_options = [1, 3]
        capacity_factor_options = [0.0, 1.0, 1.5, 2.0]

        for top_k in top_k_options:
            for capacity_factor in capacity_factor_options:
                for weighting_position_option in ExpertWeightingPositionOptions:
                    message = f"Testing with top_k={top_k}, capacity_factor={capacity_factor}, weighting_position_option={weighting_position_option.name}"
                    with self.subTest(msg=message):
                        input_dim = 8
                        c = MixtureOfExpertsPresets.experts_preset(
                            return_model_config_flag=True,
                            input_dim=input_dim,
                            output_dim=input_dim,
                            experts_num_experts=num_experts,
                            experts_top_k=top_k,
                            experts_capacity_factor=capacity_factor,
                            experts_weighting_position_option=weighting_position_option,
                        )

                        m = MixtureOfExperts(c)

                        batch_size = 30
                        rows = []
                        for _ in range(batch_size):
                            row = torch.randperm(num_experts)[:top_k]
                            rows.append(row)
                        indices = torch.stack(rows)
                        if top_k == 1:
                            indices = indices.squeeze(-1)

                        probabilities = torch.rand(batch_size, top_k)
                        input_batch = torch.randn(batch_size, input_dim)

                        result = m._MixtureOfExperts__build_routed_expert_inputs(
                            input_batch, probabilities, indices
                        )

                        self.assertIsInstance(result, list)
                        self.assertLessEqual(len(result), num_experts)
                        seen_expert_indices = [item.expert_index for item in result]
                        self.assertEqual(
                            len(seen_expert_indices), len(set(seen_expert_indices))
                        )

                        for item in result:
                            self.assertIsInstance(item, _ExpertInputData)
                            self.assertGreaterEqual(item.expert_index, 0)
                            self.assertLess(item.expert_index, num_experts)

                            self.assertIsInstance(item.expert_samples, torch.Tensor)
                            self.assertGreater(item.expert_samples.numel(), 0)
                            self.assertEqual(item.expert_samples.shape[-1], input_dim)

                            self.assertIsInstance(item.dropped_samples, torch.Tensor)
                            if capacity_factor == 0.0:
                                self.assertEqual(item.dropped_samples.numel(), 0)

                            self.assertIsInstance(
                                item.expert_routing_positions, torch.Tensor
                            )
                            self.assertIsInstance(
                                item.dropped_routing_positions, torch.Tensor
                            )

                            if (
                                weighting_position_option
                                == ExpertWeightingPositionOptions.BEFORE_EXPERTS
                            ):
                                self.assertIsInstance(item.probabilities, torch.Tensor)
                            else:
                                self.assertIsNone(item.probabilities)

    def test__build_routed_expert_inputs_skips_empty_experts(self):
        input_dim = 8
        num_experts = 6
        c = MixtureOfExpertsPresets.experts_preset(
            return_model_config_flag=True,
            input_dim=input_dim,
            output_dim=input_dim,
            experts_num_experts=num_experts,
            experts_top_k=1,
        )

        m = MixtureOfExperts(c)

        assigned_experts = {0, 2, 4}
        indices = torch.tensor([0, 2, 4, 0, 2, 4])
        probabilities = torch.rand(indices.size(0), 1)
        input_batch = torch.randn(indices.size(0), input_dim)

        result = m._MixtureOfExperts__build_routed_expert_inputs(
            input_batch, probabilities, indices
        )

        result_expert_indices = {item.expert_index for item in result}
        self.assertEqual(len(result), len(assigned_experts))
        self.assertEqual(result_expert_indices, assigned_experts)
        for item in result:
            self.assertGreater(item.expert_samples.numel(), 0)

    def test__build_dense_expert_inputs(self):
        num_experts = 6
        input_dim = 8
        batch_size = 10

        for weighting_position_option in ExpertWeightingPositionOptions:
            message = f"Testing with weighting_position_option={weighting_position_option.name}"
            with self.subTest(msg=message):
                c = MixtureOfExpertsPresets.experts_preset(
                    return_model_config_flag=True,
                    input_dim=input_dim,
                    output_dim=input_dim,
                    experts_num_experts=num_experts,
                    experts_top_k=num_experts,
                    experts_weighting_position_option=weighting_position_option,
                )

                m = MixtureOfExperts(c)

                input_batch = torch.randn(batch_size, input_dim)
                probabilities = torch.rand(batch_size, num_experts)

                result = m._MixtureOfExperts__build_dense_expert_inputs(
                    input_batch, probabilities
                )

                self.assertIsInstance(result, list)
                self.assertEqual(len(result), num_experts)

                for expert_index, item in enumerate(result):
                    self.assertIsInstance(item, _ExpertInputData)
                    self.assertEqual(item.expert_index, expert_index)

                    self.assertTrue(torch.equal(item.expert_samples, input_batch))

                    self.assertIsInstance(item.dropped_samples, torch.Tensor)
                    self.assertEqual(item.dropped_samples.numel(), 0)

                    self.assertIsNone(item.expert_routing_positions)
                    self.assertIsNone(item.dropped_routing_positions)

                    if (
                        weighting_position_option
                        == ExpertWeightingPositionOptions.BEFORE_EXPERTS
                    ):
                        assert isinstance(item.probabilities, torch.Tensor)
                        self.assertTrue(
                            torch.equal(
                                item.probabilities, probabilities[:, expert_index]
                            )
                        )
                    else:
                        self.assertIsNone(item.probabilities)

    def test__split_tokens_per_expert(self):
        num_experts = 6
        input_dim = 8
        batch_size = 10
        top_k_options = [1, 3, num_experts]

        for top_k in top_k_options:
            message = f"Testing with top_k={top_k}"
            with self.subTest(msg=message):
                c = MixtureOfExpertsPresets.experts_preset(
                    return_model_config_flag=True,
                    input_dim=input_dim,
                    output_dim=input_dim,
                    experts_num_experts=num_experts,
                    experts_top_k=top_k,
                )

                m = MixtureOfExperts(c)

                input_batch = torch.randn(batch_size, input_dim)
                probabilities = torch.rand(batch_size, top_k)

                if top_k == num_experts:
                    indices = None
                else:
                    rows = [
                        torch.randperm(num_experts)[:top_k] for _ in range(batch_size)
                    ]
                    indices = torch.stack(rows)
                    if top_k == 1:
                        indices = indices.squeeze(-1)

                result = m._split_tokens_per_expert(input_batch, probabilities, indices)

                self.assertIsInstance(result, list)

                if top_k == num_experts:
                    self.assertEqual(len(result), num_experts)
                    for item in result:
                        self.assertTrue(torch.equal(item.expert_samples, input_batch))
                        self.assertIsNone(item.expert_routing_positions)
                        self.assertIsNone(item.dropped_routing_positions)
                else:
                    self.assertLessEqual(len(result), num_experts)
                    for item in result:
                        self.assertGreater(item.expert_samples.numel(), 0)
                        self.assertIsInstance(
                            item.expert_routing_positions, torch.Tensor
                        )
                        self.assertIsInstance(
                            item.dropped_routing_positions, torch.Tensor
                        )

    def test__compute_experts_output(self):
        num_experts = 6
        batch_size = 10
        top_k_options = [1, 3, 6]
        weighted_parameters_flag_options = [True, False]

        for layer_stack_option in LinearLayerStackOptions:
            for weighting_position_option in ExpertWeightingPositionOptions:
                for top_k in top_k_options:
                    for weighted_parameters_flag in weighted_parameters_flag_options:
                        message = f"Testing configuration with layer_stack_option={layer_stack_option}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, weighting_position={weighting_position_option}"
                        with self.subTest(msg=message):
                            c = MixtureOfExpertsPresets.experts_preset(
                                return_model_config_flag=True,
                                experts_layer_stack_option=layer_stack_option,
                                experts_weighted_parameters_flag=weighted_parameters_flag,
                                experts_weighting_position_option=weighting_position_option,
                                experts_num_experts=num_experts,
                                experts_top_k=top_k,
                            )

                            m = MixtureOfExperts(c)

                            assert c.input_dim is not None
                            assert c.output_dim is not None

                            num_samples = batch_size * top_k
                            expert_samples = torch.randn(num_samples, c.input_dim)
                            probabilities = torch.randperm(num_samples).float()
                            expert_input_slice = _ExpertInputData(
                                expert_index=0,
                                expert_samples=expert_samples,
                                dropped_samples=torch.zeros(0),
                                expert_routing_positions=None,
                                dropped_routing_positions=None,
                                probabilities=probabilities,
                            )

                            output, loss = m._MixtureOfExperts__compute_expert_output(  # type: ignore[operator]
                                expert_input_slice
                            )

                            self.assertIsInstance(output, torch.Tensor)
                            self.assertEqual(output.shape, (num_samples, c.output_dim))
                            self.assertTrue(torch.isfinite(output).all())
                            self.assertEqual(loss.item(), 0.0)

                            applies_before = (
                                weighting_position_option
                                == ExpertWeightingPositionOptions.BEFORE_EXPERTS
                            )
                            if applies_before and weighted_parameters_flag:
                                zero_probs_slice = _ExpertInputData(
                                    expert_index=0,
                                    expert_samples=expert_samples,
                                    dropped_samples=torch.zeros(0),
                                    expert_routing_positions=None,
                                    dropped_routing_positions=None,
                                    probabilities=torch.zeros(num_samples),
                                )
                                zero_output, _ = (
                                    m._MixtureOfExperts__compute_expert_output(  # type: ignore[operator]
                                        zero_probs_slice
                                    )
                                )
                                expert_model: torch.nn.Module = m.expert_modules[0]  # type: ignore[assignment]
                                expected = expert_model(
                                    torch.zeros_like(expert_samples)
                                )
                                self.assertTrue(torch.allclose(zero_output, expected))

    def test__compute_expert_mixture(self):
        num_experts = 6
        top_k_options = [1, 3, num_experts]
        flag_options = [True, False]

        for top_k in top_k_options:
            for compute_expert_mixture_flag in flag_options:
                for weighted_parameters_flag in flag_options:
                    for weighting_position_option in ExpertWeightingPositionOptions:
                        message = (
                            f"Testing with weighted_parameters_flag={weighted_parameters_flag}, "
                            f"compute_expert_mixture_flag={compute_expert_mixture_flag}, "
                            f"top_k={top_k}, "
                            f"weighting_position_option={weighting_position_option}"
                        )
                        with self.subTest(msg=message):
                            c = MixtureOfExpertsPresets.experts_preset(
                                return_model_config_flag=True,
                                experts_weighted_parameters_flag=weighted_parameters_flag,
                                experts_compute_expert_mixture_flag=compute_expert_mixture_flag,
                                experts_weighting_position_option=weighting_position_option,
                                experts_num_experts=num_experts,
                                experts_top_k=top_k,
                            )

                            m = MixtureOfExperts(c)

                            batch_size = 8
                            experts_output = torch.randn(
                                batch_size * top_k, c.output_dim
                            )
                            sample_indices = torch.cat(
                                [
                                    torch.randperm(num_experts)[:top_k]
                                    for _ in range(batch_size)
                                ]
                            )
                            probabilities = torch.softmax(
                                torch.randn(batch_size * top_k), dim=-1
                            )

                            output = m._MixtureOfExperts__compute_expert_mixture(
                                experts_output, sample_indices, probabilities
                            )

                            expected = experts_output.clone()
                            if top_k != num_experts:
                                _, sort_order = sample_indices.sort(dim=0)
                                expected = expected[sort_order]

                            applies_after = (
                                weighted_parameters_flag
                                and weighting_position_option
                                == ExpertWeightingPositionOptions.AFTER_EXPERTS
                            )
                            if applies_after:
                                expected = expected * probabilities.reshape(-1, 1)

                            if compute_expert_mixture_flag and top_k > 1:
                                expected = expected.view(-1, top_k, c.output_dim).sum(
                                    dim=1
                                )

                            self.assertEqual(output.shape, expected.shape)
                            self.assertTrue(torch.allclose(output, expected))

    def test__compute_expert_mixture_sorting_correctness(self):
        output_dim = 4
        batch_size = 2
        top_k = 3
        indices = torch.tensor([2, 0, 1, 1, 2, 0])
        num_experts_options = [6, 3]

        for num_experts in num_experts_options:
            should_sort = top_k != num_experts
            with self.subTest(
                msg=f"sorting num_experts={num_experts}, should_sort={should_sort}"
            ):
                c = MixtureOfExpertsPresets.experts_preset(
                    return_model_config_flag=True,
                    experts_weighted_parameters_flag=False,
                    experts_compute_expert_mixture_flag=False,
                    experts_num_experts=num_experts,
                    experts_top_k=top_k,
                    output_dim=output_dim,
                )
                m = MixtureOfExperts(c)

                experts_output = torch.arange(
                    batch_size * top_k * output_dim, dtype=torch.float
                ).view(batch_size * top_k, output_dim)

                output = m._MixtureOfExperts__compute_expert_mixture(
                    experts_output, indices, probabilities=None
                )

                if should_sort:
                    _, sort_order = indices.sort(dim=0)
                    expected_output = experts_output[sort_order]
                else:
                    expected_output = experts_output
                self.assertTrue(torch.equal(output, expected_output))

    def test_forward(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        flag_options = [True, False]
        init_sampler_options = [InitSamplerOptions.DISABLED, InitSamplerOptions.LAYER]
        num_layers_options = [1, 2, 3]
        capacity_factor_options = [0.0, 1.0, 1.5]
        dropped_token_behavior_options = [
            DroppedTokenOptions.ZEROS,
            DroppedTokenOptions.IDENTITY,
        ]

        for num_layers in num_layers_options:
            for layer_stack_option in LinearLayerStackOptions:
                for weighting_position_option in ExpertWeightingPositionOptions:
                    for top_k in top_k_options:
                        for init_sampler_option in init_sampler_options:
                            for compute_expert_mixture_flag in flag_options:
                                for weighted_parameters_flag in flag_options:
                                    for capacity_factor in capacity_factor_options:
                                        for (
                                            dropped_token_behavior
                                        ) in dropped_token_behavior_options:
                                            message = f"Testing with layer_stack_option={layer_stack_option.name}, weighting_position_option={weighting_position_option.name}, init_sampler_option={init_sampler_option}, compute_expert_mixture_flag={compute_expert_mixture_flag}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}, capacity_factor={capacity_factor}, dropped_token_behavior={dropped_token_behavior}"
                                            with self.subTest(msg=message):
                                                if (
                                                    capacity_factor > 0
                                                    and top_k == num_experts
                                                ):
                                                    continue  # validator rejects capacity + top_k==num_experts
                                                output_dim = (
                                                    8 if capacity_factor > 0 else 6
                                                )
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
                                                    experts_dropped_token_behavior=dropped_token_behavior,
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
                                                    router_cfg = c.mixture_of_experts_config.router_model_config
                                                    sampler_cfg = c.mixture_of_experts_config.sampler_model_config
                                                    router = RouterModel(router_cfg)
                                                    sampler = SamplerModel(sampler_cfg)

                                                    logits = (
                                                        router.compute_logit_scores(
                                                            input
                                                        )
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


class TestMixtureOfExpertsStack(unittest.TestCase):
    def test_init_with_default_config(self):
        num_layer_options = [1, 2, 3]

        for layer_stack_option in LinearLayerStackOptions:
            for num_layers in num_layer_options:
                message = f"Testing configuration with layer_stack_option={layer_stack_option.name}, num_layers={num_layers}"
                with self.subTest(msg=message):
                    c = MixtureOfExpertsPresets.experts_stack_preset(
                        return_model_config_flag=True,
                        experts_stack_num_layers=num_layers,
                        experts_layer_stack_option=layer_stack_option,
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
                            for weighted_parameters_flag in flag_options:
                                message = f"Testing with layer_stack_option={layer_stack_option.name}, weighting_position_option={weighting_position_option.name}, init_sampler_option={init_sampler_option}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}"
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
                                        router_cfg = c.layer_stack_config.override_config.router_model_config
                                        sampler_cfg = c.layer_stack_config.override_config.sampler_model_config
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

        for layer_stack_option in LinearLayerStackOptions:
            for num_layers in num_layer_options:
                message = f"Testing configuration with num_layers={num_layers}"
                with self.subTest(msg=message):
                    c = MixtureOfExpertsPresets.experts_stack_preset(
                        return_model_config_flag=True,
                        experts_stack_num_layers=num_layers,
                        experts_layer_stack_option=layer_stack_option,
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

        for layer_stack_option in LinearLayerStackOptions:
            for init_sampler_option in init_sampler_options:
                for sampler_option, expert_option in zip(
                    sampler_options, expert_options
                ):
                    message = f"Testing configuration with layer_stack_option={layer_stack_option.name}, sampler_option={sampler_option.__name__}, num_experts={num_experts}, top_k={expert_option}"
                    with self.subTest(msg=message):
                        c = MixtureOfExpertsPresets.experts_stack_preset(
                            return_model_config_flag=True,
                            experts_init_sampler_option=init_sampler_option,
                            experts_num_experts=num_experts,
                            experts_top_k=expert_option,
                            router_model_layer_stack_option=layer_stack_option,
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
            for router_layer_stack_option in LinearLayerStackOptions:
                for expert_layer_stack_option in LinearLayerStackOptions:
                    for weighting_position_option in ExpertWeightingPositionOptions:
                        for top_k in top_k_options:
                            for init_sampler_option in InitSamplerOptions:
                                for compute_expert_mixture_flag in flag_options:
                                    for weighted_parameters_flag in flag_options:
                                        message = f"Testing with router_layer_stack_option={router_layer_stack_option.name}, expert_layer_stack_option={expert_layer_stack_option.name}, weighting_position_option={weighting_position_option.name}, init_sampler_option={init_sampler_option}, compute_expert_mixture_flag={compute_expert_mixture_flag}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}"
                                        with self.subTest(msg=message):
                                            c = MixtureOfExpertsPresets.experts_stack_preset(
                                                return_model_config_flag=True,
                                                router_model_layer_stack_option=router_layer_stack_option,
                                                experts_layer_stack_option=expert_layer_stack_option,
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
                                                router_cfg = c.layer_stack_config.override_config.router_model_config
                                                sampler_cfg = c.layer_stack_config.override_config.sampler_model_config
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
                                            self.assertEqual(
                                                output.shape, expected_shape
                                            )
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


class TestExpertCapacityHandler(unittest.TestCase):
    def test_token_indices_within_capacity_returns_all(self):
        capacity_factors = [0, 1.0, 1.5, 2.0]
        token_indices = torch.tensor([3, 7])
        for capacity_factor in capacity_factors:
            message = f"Testing with capacity_factor={capacity_factor}"
            with self.subTest(msg=message):
                cfg = MixtureOfExpertsConfig(
                    capacity_factor=capacity_factor,
                    num_experts=5,
                    top_k=2,
                    dropped_token_behavior=DroppedTokenOptions.ZEROS,
                )
                handler = _ExpertCapacityHandler(cfg)
                expert_tokens, dropped_tokens = (
                    handler.maybe_apply_capacity_limit_token_indices(
                        token_indices, batch_size=10
                    )
                )
                self.assertTrue(torch.equal(expert_tokens, token_indices))
                self.assertEqual(dropped_tokens.numel(), 0)
                self.assertIsNone(handler.shuffle_indices)

    def test_token_indices_exceeds_capacity_drops_tokens(self):
        capacity_factors = [1.0, 1.5]
        for capacity_factor in capacity_factors:
            message = f"Testing with capacity_factor={capacity_factor}"
            with self.subTest(msg=message):
                cfg = MixtureOfExpertsConfig(
                    capacity_factor=capacity_factor,
                    num_experts=5,
                    top_k=2,
                    dropped_token_behavior=DroppedTokenOptions.ZEROS,
                )
                handler = _ExpertCapacityHandler(cfg)
                capacity = max(1, int(10 / 5 * capacity_factor))
                tokens_to_drop = 2
                token_indices = torch.arange(capacity + tokens_to_drop)
                expert_tokens, dropped_tokens = (
                    handler.maybe_apply_capacity_limit_token_indices(
                        token_indices, batch_size=10
                    )
                )
                self.assertEqual(expert_tokens.numel(), capacity)
                self.assertEqual(dropped_tokens.numel(), tokens_to_drop)
                all_returned = torch.cat([expert_tokens, dropped_tokens]).sort().values
                self.assertTrue(torch.equal(all_returned, token_indices.sort().values))
                self.assertIsInstance(handler.shuffle_indices, torch.Tensor)

    def test_routing_positions_no_shuffle_returns_unchanged(self):
        cfg = MixtureOfExpertsConfig(
            capacity_factor=1.0,
            num_experts=5,
            top_k=2,
            dropped_token_behavior=DroppedTokenOptions.ZEROS,
        )
        handler = _ExpertCapacityHandler(cfg)
        self.assertIsNone(handler.shuffle_indices)
        token_indices = torch.tensor([0, 1, 2, 3])
        result, dropped = handler.maybe_apply_capacity_limit_routing_positions(
            token_indices, batch_size=10
        )
        self.assertTrue(torch.equal(result, token_indices))
        self.assertEqual(dropped.numel(), 0)

    def test_routing_positions_uses_stored_shuffle_indices(self):
        cfg = MixtureOfExpertsConfig(
            capacity_factor=1.0,
            num_experts=5,
            top_k=2,
            dropped_token_behavior=DroppedTokenOptions.ZEROS,
        )
        handler = _ExpertCapacityHandler(cfg)
        handler.shuffle_indices = torch.tensor([1, 0, 2, 3])
        token_indices = torch.tensor([10, 20, 30, 40])
        result, dropped = handler.maybe_apply_capacity_limit_routing_positions(
            token_indices, batch_size=10
        )
        self.assertEqual(result.numel(), 2)
        self.assertEqual(dropped.numel(), 2)
        self.assertTrue(torch.equal(result, torch.tensor([20, 10])))
        self.assertTrue(torch.equal(dropped, torch.tensor([30, 40])))

    def test_select_expert_and_dropped_samples(self):
        input_batch = torch.randn(4, 8)
        indices = torch.tensor([0, 1])
        dropped_indices = torch.tensor([2, 3])
        expected_dropped = {
            DroppedTokenOptions.ZEROS: torch.zeros(2, 8),
            DroppedTokenOptions.IDENTITY: input_batch[dropped_indices],
        }
        for behavior in DroppedTokenOptions:
            message = f"Testing with behavior={behavior.name}"
            with self.subTest(msg=message):
                cfg = MixtureOfExpertsConfig(
                    capacity_factor=1.0,
                    num_experts=5,
                    top_k=2,
                    dropped_token_behavior=behavior,
                )
                handler = _ExpertCapacityHandler(cfg)
                expert_samples, dropped = handler.select_expert_and_dropped_samples(
                    input_batch, indices, dropped_indices
                )
                self.assertTrue(torch.equal(expert_samples, input_batch[indices]))
                self.assertTrue(torch.equal(dropped, expected_dropped[behavior]))

