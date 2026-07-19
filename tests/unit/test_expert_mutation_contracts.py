import unittest

import torch

from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    RoutingInitializationMode,
)
from emperor.experts._layers.map import MixtureOfExpertsMap
from emperor.experts._layers.mixture import MixtureOfExperts
from emperor.experts._layers.reduce import MixtureOfExpertsReduce
from emperor.experts._routing.capacity import ExpertCapacityHandler
from emperor.experts._routing.weighting import ExpertWeightingHandler
from emperor.experts._state import MixtureOfExpertsLayerState
from emperor.experts._validation.mixture import MixtureOfExpertsValidator
from emperor.experts._validation.model import MixtureOfExpertsModelValidator
from emperor.halting import HaltingHiddenStateModeOptions, StickBreakingConfig
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, SamplerConfig
from tests.unit.test_expert_behavioral_contracts import (
    _linear_stack,
    _mixture_config,
    _mixture_model_config,
    _sampler_config,
)


def _copy_expert_weights(
    model: MixtureOfExperts,
    weights: tuple[torch.Tensor, ...],
) -> None:
    with torch.no_grad():
        for expert_stack, weight in zip(
            model.expert_modules,
            weights,
            strict=True,
        ):
            expert_stack[0].model.weight_params.copy_(weight)


def _router_config(input_dim: int = 7, num_experts: int = 2) -> RouterConfig:
    return RouterConfig(
        input_dim=input_dim,
        num_experts=num_experts,
        noisy_topk_flag=False,
        model_config=_linear_stack(input_dim, num_experts),
    )


def _owned_sampler_config(
    *,
    input_dim: int = 7,
    top_k: int = 1,
    num_experts: int = 2,
) -> SamplerConfig:
    config = _sampler_config(top_k=top_k, num_experts=num_experts)
    config.router_config = _router_config(input_dim, num_experts)
    return config


def _halting_expert_stack(dim: int = 2) -> LayerStackConfig:
    halting_gate_config = LayerStackConfig(
        input_dim=dim,
        hidden_dim=dim,
        output_dim=2,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DISABLED,
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
    return LayerStackConfig(
        input_dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        num_layers=2,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        shared_halting_config=StickBreakingConfig(
            input_dim=dim,
            threshold=1.0,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=halting_gate_config,
        ),
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=False),
        ),
    )


class ExpertMutationContractTests(unittest.TestCase):
    def assert_exact_error(
        self,
        exception_type: type[Exception],
        expected_message: str,
        callback,
    ) -> None:
        with self.assertRaises(exception_type) as error:
            callback()
        self.assertEqual(str(error.exception), expected_message)

    def test_dense_map_uses_sample_major_order_and_forces_default_overrides(
        self,
    ) -> None:
        config = _mixture_config()
        config.routing_initialization_mode = RoutingInitializationMode.LAYER
        model = MixtureOfExpertsMap(config)
        weights = (
            torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
            torch.tensor([[0.0, 1.0], [3.0, 0.0]]),
        )
        _copy_expert_weights(model, weights)
        inputs = torch.tensor([[1.0, 2.0], [-1.0, 3.0]], requires_grad=True)
        probabilities = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
        expected = torch.stack(
            (
                inputs.detach()[0] @ weights[0],
                inputs.detach()[0] @ weights[1],
                inputs.detach()[1] @ weights[0],
                inputs.detach()[1] @ weights[1],
            )
        )

        output, _skip_mask, loss = model(
            inputs, probabilities=probabilities, indices=None
        )

        torch.testing.assert_close(output, expected)
        self.assertFalse(model.weighted_parameters_flag)
        self.assertFalse(model.compute_expert_mixture_flag)
        self.assertEqual(
            model.routing_initialization_mode,
            RoutingInitializationMode.DISABLED,
        )
        self.assertIsNone(model.routing_positions)
        self.assertIsNone(model.sample_probabilities)
        self.assertEqual(loss.item(), 0.0)
        output.square().sum().backward()
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)

        explicit_config = _mixture_config()
        explicit_config.routing_initialization_mode = RoutingInitializationMode.LAYER
        explicit = MixtureOfExpertsMap(
            explicit_config,
            MixtureOfExpertsConfig(
                routing_initialization_mode=RoutingInitializationMode.LAYER
            ),
        )
        self.assertEqual(
            explicit.routing_initialization_mode,
            RoutingInitializationMode.DISABLED,
        )

    def test_reduce_forces_defaults_but_preserves_unforced_explicit_overrides(
        self,
    ) -> None:
        config = _mixture_config()
        config.weighted_parameters_flag = False
        config.compute_expert_mixture_flag = False
        config.weighting_position_option = ExpertWeightingPositionOptions.BEFORE_EXPERTS
        config.routing_initialization_mode = RoutingInitializationMode.LAYER
        model = MixtureOfExpertsReduce(
            config,
            MixtureOfExpertsConfig(input_dim=3, output_dim=3),
        )

        self.assertEqual(model.input_dim, 3)
        self.assertEqual(model.output_dim, 3)
        self.assertTrue(model.weighted_parameters_flag)
        self.assertTrue(model.compute_expert_mixture_flag)
        self.assertEqual(
            model.weighting_position_option,
            ExpertWeightingPositionOptions.AFTER_EXPERTS,
        )
        self.assertEqual(
            model.routing_initialization_mode,
            RoutingInitializationMode.DISABLED,
        )

        default_model = MixtureOfExpertsReduce(config)
        self.assertTrue(default_model.weighted_parameters_flag)
        self.assertTrue(default_model.compute_expert_mixture_flag)
        self.assertEqual(
            default_model.weighting_position_option,
            ExpertWeightingPositionOptions.AFTER_EXPERTS,
        )
        self.assertEqual(
            default_model.routing_initialization_mode,
            RoutingInitializationMode.DISABLED,
        )

    def test_sparse_reduce_restores_routing_order_before_exact_weighted_sum(
        self,
    ) -> None:
        model = MixtureOfExpertsReduce(_mixture_config(top_k=2, num_experts=3))
        weights = (
            torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
            torch.tensor([[0.0, 1.0], [3.0, 0.0]]),
            torch.tensor([[2.0, -1.0], [1.0, 1.0]]),
        )
        _copy_expert_weights(model, weights)
        flattened_map_output = torch.tensor(
            [[1.0, 2.0], [4.0, -2.0], [-1.0, 3.0], [2.0, 5.0]],
            requires_grad=True,
        )
        probabilities = torch.tensor([[0.2, 0.8], [0.3, 0.7]])
        indices = torch.tensor([[2, 0], [1, 2]])
        expected_rows = torch.stack(
            [
                flattened_map_output.detach()[row] @ weights[expert_index.item()]
                for row, expert_index in enumerate(indices.flatten())
            ]
        )
        expected = (
            (expected_rows * probabilities.flatten()[:, None])
            .reshape(2, 2, 2)
            .sum(dim=1)
        )

        output, _skip_mask, loss = model(
            flattened_map_output,
            probabilities=probabilities,
            indices=indices,
        )

        torch.testing.assert_close(output, expected)
        self.assertEqual(loss.item(), 0.0)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(flattened_map_output.grad).all())
        self.assertGreater(flattened_map_output.grad.abs().sum().item(), 0.0)

    def test_sparse_reduce_does_not_execute_an_unselected_expert(self) -> None:
        model = MixtureOfExpertsReduce(_mixture_config(top_k=1, num_experts=3))
        weights = (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
            torch.tensor([[4.0, 0.0], [0.0, 5.0]]),
        )
        _copy_expert_weights(model, weights)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        output, _skip_mask, loss = model(
            inputs,
            probabilities=torch.ones(2),
            indices=torch.tensor([0, 1]),
        )

        expected = torch.stack((inputs.detach()[0], inputs.detach()[1] @ weights[1]))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(loss.item(), 0.0)
        output.sum().backward()
        for selected_expert in model.expert_modules[:2]:
            gradient = selected_expert[0].model.weight_params.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assertIsNone(model.expert_modules[2][0].model.weight_params.grad)

    def test_sparse_forward_rejects_duplicate_expert_ids_per_sample(self) -> None:
        model = MixtureOfExperts(_mixture_config(top_k=2, num_experts=3))

        self.assert_exact_error(
            ValueError,
            "Input Error: 'indices' must contain distinct expert ids for each "
            "input sample in sparse MixtureOfExperts routing, received duplicate "
            "expert ids in sample rows [0].",
            lambda: model(
                torch.tensor([[1.0, 2.0]]),
                probabilities=torch.tensor([[0.25, 0.75]]),
                indices=torch.tensor([[0, 0]]),
            ),
        )

    def test_sparse_forward_detects_a_duplicate_after_the_first_route(self) -> None:
        model = MixtureOfExperts(_mixture_config(top_k=3, num_experts=4))

        self.assert_exact_error(
            ValueError,
            "Input Error: 'indices' must contain distinct expert ids for each "
            "input sample in sparse MixtureOfExperts routing, received duplicate "
            "expert ids in sample rows [0].",
            lambda: model(
                torch.tensor([[1.0, 2.0]]),
                probabilities=torch.tensor([[0.2, 0.3, 0.5]]),
                indices=torch.tensor([[0, 2, 2]]),
            ),
        )

    def test_sparse_forward_rejects_an_empty_input_batch(self) -> None:
        model = MixtureOfExperts(_mixture_config(top_k=1, num_experts=2))

        self.assert_exact_error(
            ValueError,
            "Input Error: MixtureOfExperts requires at least one input sample, "
            "received input shape (0, 2).",
            lambda: model(
                torch.empty(0, 2),
                probabilities=torch.empty(0),
                indices=torch.empty(0, dtype=torch.long),
            ),
        )

    def test_top_k_one_column_routing_preserves_samples_and_gradients(self) -> None:
        config = _mixture_config(
            input_dim=1,
            output_dim=1,
            top_k=1,
            num_experts=3,
        )
        config.capacity_factor = 1.0
        model = MixtureOfExperts(config)
        _copy_expert_weights(
            model,
            (
                torch.tensor([[2.0]]),
                torch.tensor([[3.0]]),
                torch.tensor([[5.0]]),
            ),
        )
        inputs = torch.tensor([[2.0], [5.0]], requires_grad=True)

        output, _skip_mask, loss = model(
            inputs,
            probabilities=torch.tensor([[0.25], [0.75]]),
            indices=torch.tensor([[1], [0]]),
        )

        torch.testing.assert_close(
            output,
            torch.tensor([[1.5], [7.5]]),
            rtol=0,
            atol=0,
        )
        self.assertEqual(loss.item(), 0.0)

        output.sum().backward()

        torch.testing.assert_close(
            inputs.grad,
            torch.tensor([[0.75], [1.5]]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model.expert_modules[0][0].model.weight_params.grad,
            torch.tensor([[3.75]]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model.expert_modules[1][0].model.weight_params.grad,
            torch.tensor([[0.5]]),
            rtol=0,
            atol=0,
        )
        self.assertIsNone(model.expert_modules[2][0].model.weight_params.grad)

    def test_capacity_boundary_keeps_one_and_public_forward_keeps_dropped_row(
        self,
    ) -> None:
        handler = ExpertCapacityHandler(
            MixtureOfExpertsConfig(
                capacity_factor=0.5,
                num_experts=4,
                top_k=1,
                dropped_token_behavior=DroppedTokenOptions.ZEROS,
            )
        )
        torch.manual_seed(7)
        kept, dropped = handler.maybe_apply_capacity_limit_token_indices(
            torch.tensor([0, 1]),
            batch_size=4,
        )
        self.assertEqual(handler.top_k, 1)
        self.assertEqual(kept.numel(), 1)
        self.assertEqual(dropped.numel(), 1)
        self.assertEqual(
            set(torch.cat((kept, dropped)).tolist()),
            {0, 1},
        )

        config = _mixture_config(
            input_dim=1,
            output_dim=1,
            top_k=1,
            num_experts=2,
        )
        config.capacity_factor = 1.0
        model = MixtureOfExperts(config)
        _copy_expert_weights(model, (torch.tensor([[2.0]]), torch.tensor([[3.0]])))
        torch.manual_seed(11)
        output, _, _ = model(
            torch.tensor([[2.0], [5.0]]),
            probabilities=torch.ones(2),
            indices=torch.zeros(2, dtype=torch.long),
        )

        self.assertEqual(output.shape, (2, 1))
        self.assertEqual(torch.count_nonzero(output).item(), 1)
        self.assertIn(output.abs().sum().item(), (4.0, 10.0))

    def test_capacity_shuffle_state_is_scoped_to_one_token_routing_pair(
        self,
    ) -> None:
        capacity = ExpertCapacityHandler(
            MixtureOfExpertsConfig(
                capacity_factor=1.0,
                num_experts=2,
                top_k=1,
                dropped_token_behavior=DroppedTokenOptions.ZEROS,
            )
        )
        torch.manual_seed(13)
        capacity.maybe_apply_capacity_limit_token_indices(
            torch.tensor([0, 1, 2]),
            batch_size=2,
        )
        self.assertIsNotNone(capacity.shuffle_indices)

        kept, dropped = capacity.maybe_apply_capacity_limit_token_indices(
            torch.tensor([7]),
            batch_size=2,
        )

        self.assertEqual(kept.tolist(), [7])
        self.assertEqual(dropped.numel(), 0)
        self.assertIsNone(capacity.shuffle_indices)
        routing, dropped_routing = (
            capacity.maybe_apply_capacity_limit_routing_positions(
                torch.tensor([5]),
                batch_size=2,
            )
        )
        self.assertEqual(routing.tolist(), [5])
        self.assertEqual(dropped_routing.numel(), 0)

    def test_capacity_keeps_a_later_within_capacity_expert_aligned(self) -> None:
        config = _mixture_config(
            input_dim=1,
            output_dim=1,
            top_k=1,
            num_experts=3,
        )
        config.capacity_factor = 1.0
        model = MixtureOfExperts(config)
        _copy_expert_weights(
            model,
            (
                torch.tensor([[2.0]]),
                torch.tensor([[3.0]]),
                torch.tensor([[5.0]]),
            ),
        )
        torch.manual_seed(11)

        output, _skip_mask, loss = model(
            torch.tensor([[2.0], [5.0], [7.0]]),
            probabilities=torch.tensor([0.1, 0.9, 0.5]),
            indices=torch.tensor([0, 0, 1]),
        )

        torch.testing.assert_close(
            output,
            torch.tensor([[0.0], [9.0], [10.5]]),
            rtol=0,
            atol=0,
        )
        self.assertEqual(loss.item(), 0.0)

    def test_capacity_shuffle_keeps_probabilities_aligned_after_restoring_order(
        self,
    ) -> None:
        config = _mixture_config(
            input_dim=1,
            output_dim=1,
            top_k=1,
            num_experts=2,
        )
        config.capacity_factor = 1.0
        model = MixtureOfExperts(config)
        _copy_expert_weights(model, (torch.tensor([[1.0]]), torch.tensor([[1.0]])))
        torch.manual_seed(11)

        output, _, _ = model(
            torch.tensor([[2.0], [5.0]]),
            probabilities=torch.tensor([0.1, 0.9]),
            indices=torch.tensor([0, 0]),
        )

        self.assertIsNone(model.capacity_handler.shuffle_indices)
        torch.testing.assert_close(
            output,
            torch.tensor([[0.0], [4.5]]),
            rtol=0,
            atol=0,
        )

    def test_capacity_empty_outputs_and_shuffle_preserve_meta_device(self) -> None:
        capacity = ExpertCapacityHandler(
            MixtureOfExpertsConfig(
                capacity_factor=0.0,
                num_experts=2,
                top_k=1,
                dropped_token_behavior=DroppedTokenOptions.ZEROS,
            )
        )
        meta_indices = torch.empty(0, dtype=torch.long, device="meta")
        kept, dropped = capacity.maybe_apply_capacity_limit_token_indices(
            meta_indices,
            batch_size=0,
        )
        self.assertEqual(kept.device.type, "meta")
        self.assertEqual(dropped.device.type, "meta")
        kept, dropped = capacity.maybe_apply_capacity_limit_routing_positions(
            meta_indices,
            batch_size=0,
        )
        self.assertEqual(kept.device.type, "meta")
        self.assertEqual(dropped.device.type, "meta")
        self.assertEqual(kept.dtype, torch.long)
        self.assertEqual(dropped.dtype, torch.long)

        capacity.capacity_factor = 1.0
        capacity.num_experts = 4
        torch.manual_seed(5)
        kept, dropped = capacity.maybe_apply_capacity_limit_token_indices(
            torch.empty(2, dtype=torch.long, device="meta"),
            batch_size=4,
        )
        self.assertEqual(kept.device.type, "meta")
        self.assertEqual(dropped.device.type, "meta")
        self.assertEqual(capacity.shuffle_indices.device.type, "meta")

    def test_private_split_records_preserve_dtype_and_device_metadata(self) -> None:
        dense = MixtureOfExperts(_mixture_config())
        dense_records = dense._split_tokens_per_expert(
            torch.ones(2, 2, dtype=torch.float64),
            torch.ones(2, 2, dtype=torch.float64),
            None,
        )
        for record in dense_records:
            self.assertEqual(record.dropped_samples.dtype, torch.float64)
            self.assertEqual(record.dropped_samples.device.type, "cpu")

        dense_meta_records = dense._split_tokens_per_expert(
            torch.empty(2, 2, device="meta"),
            torch.empty(2, 2, device="meta"),
            None,
        )
        for record in dense_meta_records:
            self.assertEqual(record.dropped_samples.device.type, "meta")

        reduced = MixtureOfExpertsReduce(_mixture_config())
        reduce_records = reduced._split_tokens_per_expert(
            torch.ones(4, 2, dtype=torch.float64),
            torch.ones(2, 2, dtype=torch.float64),
            None,
        )
        for expert_index, record in enumerate(reduce_records):
            self.assertEqual(record.dropped_samples.dtype, torch.float64)
            self.assertEqual(
                record.expert_routing_positions.tolist(),
                [expert_index, expert_index + 2],
            )
            self.assertEqual(record.dropped_routing_positions.dtype, torch.long)
            self.assertEqual(record.dropped_routing_positions.numel(), 0)

        reduce_meta_records = reduced._split_tokens_per_expert(
            torch.empty(4, 2, device="meta"),
            torch.empty(2, 2, device="meta"),
            None,
        )
        for record in reduce_meta_records:
            self.assertEqual(record.dropped_samples.device.type, "meta")
            self.assertEqual(record.expert_routing_positions.device.type, "meta")
            self.assertEqual(record.dropped_routing_positions.device.type, "meta")

        capacity = ExpertCapacityHandler(
            MixtureOfExpertsConfig(
                capacity_factor=0.0,
                num_experts=2,
                top_k=1,
                dropped_token_behavior=DroppedTokenOptions.ZEROS,
            )
        )
        meta_indices = torch.empty(0, dtype=torch.long, device="meta")
        kept, dropped = capacity.maybe_apply_capacity_limit_token_indices(
            meta_indices,
            batch_size=0,
        )
        self.assertEqual(kept.device.type, "meta")
        self.assertEqual(dropped.device.type, "meta")
        kept, dropped = capacity.maybe_apply_capacity_limit_routing_positions(
            meta_indices,
            batch_size=0,
        )
        self.assertEqual(kept.device.type, "meta")
        self.assertEqual(dropped.device.type, "meta")

        capacity.capacity_factor = 1.0
        capacity.num_experts = 4
        torch.manual_seed(5)
        kept, dropped = capacity.maybe_apply_capacity_limit_token_indices(
            torch.empty(2, dtype=torch.long, device="meta"),
            batch_size=4,
        )
        self.assertEqual(kept.device.type, "meta")
        self.assertEqual(dropped.device.type, "meta")
        self.assertEqual(capacity.shuffle_indices.device.type, "meta")

    def test_expert_stack_and_owned_router_use_outer_input_dimension(self) -> None:
        config = _mixture_config()
        config.expert_model_config = _linear_stack(7, 9)
        model = MixtureOfExperts(config)
        for expert in model.expert_modules:
            self.assertEqual(expert.input_dim, 2)
            self.assertEqual(expert.output_dim, 2)
            self.assertEqual(expert[0].input_dim, 2)
            self.assertEqual(expert[0].output_dim, 2)

        owned_config = _mixture_config(top_k=1, num_experts=2)
        owned_config.routing_initialization_mode = RoutingInitializationMode.LAYER
        owned_config.sampler_config = _owned_sampler_config()
        owned = MixtureOfExperts(owned_config)
        self.assertIsNotNone(owned.sampler)
        self.assertIsNotNone(owned.sampler.router)
        self.assertEqual(owned.sampler.router.input_dim, 2)

    def test_model_preserves_input_loss_and_shared_router_uses_model_input_dim(
        self,
    ) -> None:
        model = _mixture_model_config().build()
        self.assertEqual(model.output_dim, 2)
        state = MixtureOfExpertsLayerState(
            hidden=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            probabilities=torch.ones(2),
            indices=torch.tensor([0, 1]),
            loss=torch.tensor(2.0),
        )

        result = model(state)

        self.assertEqual(result.loss.item(), 2.0)

        shared_config = _mixture_model_config()
        shared_config.routing_initialization_mode = RoutingInitializationMode.SHARED
        shared_config.sampler_config = _owned_sampler_config()
        shared = shared_config.build()
        self.assertIsNotNone(shared.shared_sampler)
        self.assertIsNotNone(shared.shared_sampler.router)
        self.assertEqual(shared.shared_sampler.router.input_dim, 2)

    def test_nested_real_expert_auxiliary_losses_are_summed(self) -> None:
        outer_config = _mixture_config(top_k=1, num_experts=2)
        outer_config.expert_model_config = _halting_expert_stack()
        model = MixtureOfExperts(outer_config)
        model.eval()

        inputs = torch.tensor([[1.0, 2.0], [3.0, -1.0], [2.0, 0.5], [-2.0, 4.0]])
        indices = torch.tensor([0, 1, 0, 1])
        probabilities = torch.ones(4)
        expected_loss = inputs.new_zeros(())
        for expert_index, expert_stack in enumerate(model.expert_modules):
            expert_state = Layer.run_model_returning_state(
                expert_stack,
                inputs[indices == expert_index],
            )
            self.assertIsNotNone(expert_state.loss)
            self.assertGreater(expert_state.loss.item(), 0.0)
            expected_loss = expected_loss + expert_state.loss

        _, _, loss = model(
            inputs,
            probabilities=probabilities,
            indices=indices,
        )

        torch.testing.assert_close(loss, expected_loss)
        self.assertGreater(loss.item(), 0.0)

    def test_exact_configuration_validation_errors(self) -> None:
        model = MixtureOfExperts(_mixture_config(top_k=1, num_experts=2))

        model.expert_model_config = object()
        self.assert_exact_error(
            TypeError,
            "Configuration Error: 'expert_model_config' must be of type "
            "LayerStackConfig or RecurrentLayerConfig, received type object",
            lambda: MixtureOfExpertsValidator.validate_forward_reference_types(model),
        )
        model.expert_model_config = model.cfg.expert_model_config

        model.weighting_position_option = object()
        self.assert_exact_error(
            TypeError,
            "Configuration Error: 'weighting_position_option' must be of type "
            "ExpertWeightingPositionOptions, received type object",
            lambda: MixtureOfExpertsValidator.validate_forward_reference_types(model),
        )
        model.weighting_position_option = ExpertWeightingPositionOptions.AFTER_EXPERTS

        model.routing_initialization_mode = object()
        self.assert_exact_error(
            TypeError,
            "Configuration Error: 'routing_initialization_mode' must be of type "
            "RoutingInitializationMode, received type object",
            lambda: MixtureOfExpertsValidator.validate_forward_reference_types(model),
        )
        model.routing_initialization_mode = RoutingInitializationMode.DISABLED

        model.cfg.dropped_token_behavior = object()
        self.assert_exact_error(
            TypeError,
            "Configuration Error: 'dropped_token_behavior' must be of type "
            "DroppedTokenOptions, received type object",
            lambda: MixtureOfExpertsValidator.validate_forward_reference_types(model),
        )
        model.cfg.dropped_token_behavior = DroppedTokenOptions.ZEROS

        for field_name in ("input_dim", "output_dim", "top_k", "num_experts"):
            original = getattr(model, field_name)
            setattr(model, field_name, 0)
            self.assert_exact_error(
                ValueError,
                f"Configuration Error: '{field_name}' must be a positive integer, "
                "received 0.",
                lambda: MixtureOfExpertsValidator.validate_dimensions(model),
            )
            setattr(model, field_name, original)

        model.top_k = 3
        self.assert_exact_error(
            ValueError,
            "Configuration Error: 'top_k' cannot exceed 'num_experts' for "
            "MixtureOfExperts, received top_k=3, num_experts=2.",
            lambda: MixtureOfExpertsValidator.validate_dimensions(model),
        )
        model.top_k = 1

        model.capacity_factor = -0.25
        self.assert_exact_error(
            ValueError,
            "Configuration Error: 'capacity_factor' must be >= 0.0, received -0.25",
            lambda: MixtureOfExpertsValidator.validate_capacity_factor_is_non_negative(
                model
            ),
        )
        model.capacity_factor = 0.5
        model.top_k = model.num_experts
        self.assert_exact_error(
            ValueError,
            "Configuration Error: 'capacity_factor' cannot be > 0.0 when "
            "'top_k' equals 'num_experts'. When top_k == num_experts all tokens "
            "pass through all experts unconditionally, so capacity limiting has "
            "no effect and dropped tokens cannot occur.",
            lambda: (
                MixtureOfExpertsValidator.validate_capacity_factor_consistent_with_top_k(
                    model
                )
            ),
        )
        model.top_k = 1
        model.output_dim = 3
        self.assert_exact_error(
            ValueError,
            "Configuration Error: 'input_dim' must equal 'output_dim' when "
            "'capacity_factor' > 0.0, because dropped tokens pass through as "
            "identity and must match the expert output shape. Got "
            "input_dim=2, output_dim=3",
            lambda: MixtureOfExpertsValidator.validate_dims_match_when_capacity_enabled(
                model
            ),
        )

    def test_exact_owned_routing_and_model_validation_errors(self) -> None:
        model = MixtureOfExperts(_mixture_config(top_k=1, num_experts=2))
        model.routing_initialization_mode = RoutingInitializationMode.LAYER
        model.sampler_config = object()
        self.assert_exact_error(
            TypeError,
            "Configuration Error: 'sampler_config' must be of type SamplerConfig "
            "when 'routing_initialization_mode' is LAYER, received type object",
            lambda: MixtureOfExpertsValidator.validate_owned_routing_config_types(
                model
            ),
        )
        model.sampler_config = _sampler_config()
        self.assert_exact_error(
            TypeError,
            "Configuration Error: 'sampler_config.router_config' must be of type "
            "RouterConfig when 'routing_initialization_mode' is LAYER, received "
            "type NoneType",
            lambda: MixtureOfExpertsValidator.validate_owned_routing_config_types(
                model
            ),
        )

        model.routing_initialization_mode = RoutingInitializationMode.DISABLED
        self.assert_exact_error(
            ValueError,
            "Invalid configuration: `routing_initialization_mode` must be "
            "`RoutingInitializationMode.LAYER` to initialize the `RouterModel` and "
            "`SamplerModel` when `indices` are not provided. Current option: "
            "RoutingInitializationMode.DISABLED",
            lambda: MixtureOfExpertsValidator.validate_sampler_is_initialized(model),
        )
        model.sampler_config = None
        self.assert_exact_error(
            ValueError,
            "Configuration Error: `sampler_config` must be defined to properly "
            "initialize and utilize the sampler model in the mixture of experts "
            "layer.",
            lambda: MixtureOfExpertsValidator.validate_sampler_config_exists(model),
        )
        model.sampler_config = _sampler_config()
        self.assert_exact_error(
            ValueError,
            "Configuration Error: `sampler_config.router_config` must be defined "
            "to properly initialize and utilize the router model in the mixture "
            "of experts layer.",
            lambda: MixtureOfExpertsValidator.validate_router_config_exists(model),
        )

        wrapper = _mixture_model_config().build()
        wrapper.cfg = object()
        self.assert_exact_error(
            TypeError,
            "Configuration Error: `cfg` must be of type "
            "MixtureOfExpertsModelConfig, received type object",
            lambda: MixtureOfExpertsModelValidator.validate_cfg_type(wrapper),
        )
        wrapper.cfg = _mixture_model_config()
        wrapper.stack_config = object()
        self.assert_exact_error(
            TypeError,
            "Configuration Error: 'stack_config' must be of type LayerStackConfig, "
            "received type object",
            lambda: MixtureOfExpertsModelValidator.validate_stack_config_type(wrapper),
        )
        wrapper.routing_initialization_mode = RoutingInitializationMode.SHARED
        wrapper.sampler_config = object()
        self.assert_exact_error(
            TypeError,
            "Configuration Error: 'sampler_config' must be of type SamplerConfig "
            "when 'routing_initialization_mode' is SHARED, received type object",
            lambda: (
                MixtureOfExpertsModelValidator.validate_shared_routing_config_when_shared(
                    wrapper
                )
            ),
        )
        wrapper.sampler_config = _sampler_config()
        self.assert_exact_error(
            TypeError,
            "Configuration Error: 'sampler_config.router_config' must be of type "
            "RouterConfig when 'routing_initialization_mode' is SHARED, received "
            "type NoneType",
            lambda: (
                MixtureOfExpertsModelValidator.validate_shared_routing_config_when_shared(
                    wrapper
                )
            ),
        )

    def test_router_type_errors_report_the_actual_invalid_type(self) -> None:
        model = MixtureOfExperts(_mixture_config(top_k=1, num_experts=2))
        model.routing_initialization_mode = RoutingInitializationMode.LAYER
        model.sampler_config = _sampler_config()
        model.sampler_config.router_config = object()  # type: ignore[assignment]
        self.assert_exact_error(
            TypeError,
            "Configuration Error: 'sampler_config.router_config' must be of type "
            "RouterConfig when 'routing_initialization_mode' is LAYER, received "
            "type object",
            lambda: MixtureOfExpertsValidator.validate_owned_routing_config_types(
                model
            ),
        )

        wrapper = _mixture_model_config().build()
        wrapper.routing_initialization_mode = RoutingInitializationMode.SHARED
        wrapper.sampler_config = _sampler_config()
        wrapper.sampler_config.router_config = object()  # type: ignore[assignment]
        self.assert_exact_error(
            TypeError,
            "Configuration Error: 'sampler_config.router_config' must be of type "
            "RouterConfig when 'routing_initialization_mode' is SHARED, received "
            "type object",
            lambda: (
                MixtureOfExpertsModelValidator.validate_shared_routing_config_when_shared(
                    wrapper
                )
            ),
        )

    def test_exact_runtime_validation_errors_and_singleton_boundaries(self) -> None:
        model = MixtureOfExperts(_mixture_config(top_k=1, num_experts=2))
        self.assert_exact_error(
            TypeError,
            "Input Error: 'input_batch' must be a Tensor for MixtureOfExperts, "
            "received list.",
            lambda: MixtureOfExpertsValidator.validate_input_batch(model, []),
        )
        self.assert_exact_error(
            ValueError,
            "Input Error: MixtureOfExperts expects a 2D input tensor "
            "(batch_size, input_dim), received a 1D tensor with shape (2,).",
            lambda: MixtureOfExpertsValidator.validate_input_batch(
                model, torch.ones(2)
            ),
        )
        self.assert_exact_error(
            ValueError,
            "Input Error: input feature dimension must match 'input_dim' for "
            "MixtureOfExperts, received input_dim=2 and input shape (3, 4).",
            lambda: MixtureOfExpertsValidator.validate_input_batch(
                model, torch.ones(3, 4)
            ),
        )

        self.assert_exact_error(
            TypeError,
            "Input Error: 'probabilities' must be a Tensor for MixtureOfExperts, "
            "received list.",
            lambda: MixtureOfExpertsValidator.validate_probabilities(
                model, torch.ones(3, 2), []
            ),
        )
        self.assert_exact_error(
            ValueError,
            "Input Error: 'probabilities' batch dimension must match input_batch, "
            "received probabilities shape (2,) and input_batch shape (3, 2).",
            lambda: MixtureOfExpertsValidator.validate_probabilities(
                model, torch.ones(3, 2), torch.ones(2)
            ),
        )
        self.assert_exact_error(
            ValueError,
            "Input Error: 'probabilities' routing dimension must match top_k, "
            "received top_k=1 and probabilities shape (3, 2).",
            lambda: MixtureOfExpertsValidator.validate_probabilities(
                model, torch.ones(3, 2), torch.ones(3, 2)
            ),
        )
        MixtureOfExpertsValidator.validate_probabilities(
            model,
            torch.ones(3, 2),
            torch.ones(3, 1),
        )

        self.assert_exact_error(
            TypeError,
            "Input Error: 'indices' must contain integer expert ids for "
            "MixtureOfExperts, received dtype torch.float32.",
            lambda: MixtureOfExpertsValidator.validate_indices_dtype_and_range(
                model, torch.tensor([0.0])
            ),
        )
        self.assert_exact_error(
            ValueError,
            "Input Error: 'indices' values must be in [0, num_experts), received "
            "num_experts=2 and indices range [2, 2].",
            lambda: MixtureOfExpertsValidator.validate_indices_dtype_and_range(
                model, torch.tensor([2])
            ),
        )
        MixtureOfExpertsValidator.validate_indices_dtype_and_range(
            model,
            torch.empty(0, dtype=torch.long),
        )

        external_message = (
            "`probabilities` and `indices` must both be None when the "
            "MixtureOfExperts layer owns routing. Providing external routing "
            "inputs where they are not expected is not allowed."
        )
        self.assert_exact_error(
            ValueError,
            external_message,
            lambda: (
                MixtureOfExpertsValidator.validate_external_probabilities_are_not_given(
                    torch.ones(1), None
                )
            ),
        )
        self.assert_exact_error(
            ValueError,
            external_message,
            lambda: (
                MixtureOfExpertsValidator.validate_external_probabilities_are_not_given(
                    None, torch.zeros(1, dtype=torch.long)
                )
            ),
        )

        model.routing_initialization_mode = RoutingInitializationMode.LAYER
        self.assert_exact_error(
            ValueError,
            external_message,
            lambda: MixtureOfExpertsValidator.validate_external_routing_inputs(
                model, torch.ones(1), None
            ),
        )
        self.assert_exact_error(
            ValueError,
            external_message,
            lambda: MixtureOfExpertsValidator.validate_external_routing_inputs(
                model, None, torch.zeros(1, dtype=torch.long)
            ),
        )

        self.assert_exact_error(
            ValueError,
            "Missing input: `probabilities` must be supplied when `indices` are "
            "used to ensure accurate weighting and processing of inputs.",
            lambda: MixtureOfExpertsValidator.validate_probabilities_exist(None),
        )

    def test_exact_indices_and_reduce_delegated_validation_errors(self) -> None:
        model = MixtureOfExperts(_mixture_config(top_k=1, num_experts=2))
        inputs = torch.ones(3, 2)
        self.assert_exact_error(
            ValueError,
            "Input Error: 'indices' must be a 1D or 2D tensor for "
            "MixtureOfExperts, received a 3D tensor with shape (3, 1, 1).",
            lambda: MixtureOfExpertsValidator.validate_indices(
                model, inputs, torch.zeros(3, 1, 1, dtype=torch.long)
            ),
        )
        self.assert_exact_error(
            ValueError,
            "Input Error: 'indices' batch dimension must match input_batch, "
            "received indices shape (2,) and input_batch shape (3, 2).",
            lambda: MixtureOfExpertsValidator.validate_indices(
                model, inputs, torch.zeros(2, dtype=torch.long)
            ),
        )
        self.assert_exact_error(
            ValueError,
            "Input Error: 'indices' routing dimension must match top_k, received "
            "top_k=1 and indices shape (3, 2).",
            lambda: MixtureOfExpertsValidator.validate_indices(
                model, inputs, torch.zeros(3, 2, dtype=torch.long)
            ),
        )

        reduced = MixtureOfExpertsReduce(_mixture_config(top_k=2, num_experts=3))
        reduced_inputs = torch.ones(4, 2)
        self.assert_exact_error(
            ValueError,
            "Input Error: 'probabilities' must be a 1D or 2D tensor for "
            "MixtureOfExperts, received a 3D tensor with shape (2, 2, 1).",
            lambda: MixtureOfExpertsValidator.validate_reduce_forward_inputs(
                reduced,
                reduced_inputs,
                torch.ones(2, 2, 1),
                torch.tensor([[0, 1], [1, 2]]),
            ),
        )
        self.assert_exact_error(
            ValueError,
            "Input Error: 'probabilities' routing dimension must match top_k, "
            "received top_k=2 and probabilities shape (4, 1).",
            lambda: MixtureOfExpertsValidator.validate_reduce_forward_inputs(
                reduced,
                reduced_inputs,
                torch.ones(4, 1),
                torch.tensor([[0, 1], [1, 2]]),
            ),
        )
        self.assert_exact_error(
            ValueError,
            "Input Error: 'indices' must be a 1D or 2D tensor for "
            "MixtureOfExperts, received a 3D tensor with shape (2, 2, 1).",
            lambda: MixtureOfExpertsValidator.validate_reduce_forward_inputs(
                reduced,
                reduced_inputs,
                torch.ones(2, 2),
                torch.zeros(2, 2, 1, dtype=torch.long),
            ),
        )
        self.assert_exact_error(
            ValueError,
            "Input Error: 'indices' routing dimension must match top_k, received "
            "top_k=2 and indices shape (4, 1).",
            lambda: MixtureOfExpertsValidator.validate_reduce_forward_inputs(
                reduced,
                reduced_inputs,
                torch.ones(2, 2),
                torch.zeros(4, 1, dtype=torch.long),
            ),
        )

    def test_layer_owned_routing_rejects_each_single_external_input(self) -> None:
        model = MixtureOfExperts(_mixture_config(top_k=1, num_experts=2))
        model.routing_initialization_mode = RoutingInitializationMode.LAYER
        message = (
            "`probabilities` and `indices` must both be None when the "
            "MixtureOfExperts layer owns routing. Providing external routing "
            "inputs where they are not expected is not allowed."
        )
        self.assert_exact_error(
            ValueError,
            message,
            lambda: MixtureOfExpertsValidator.validate_external_routing_inputs(
                model, torch.ones(1), None
            ),
        )
        self.assert_exact_error(
            ValueError,
            message,
            lambda: MixtureOfExpertsValidator.validate_external_routing_inputs(
                model, None, torch.zeros(1, dtype=torch.long)
            ),
        )

    def test_exact_external_and_reduce_validation_errors(self) -> None:
        sparse = MixtureOfExperts(_mixture_config(top_k=1, num_experts=2))
        self.assert_exact_error(
            ValueError,
            "Missing input: 'probabilities' must be supplied when external routing "
            "is used by MixtureOfExperts.",
            lambda: MixtureOfExpertsValidator.validate_external_routing_inputs(
                sparse, None, torch.tensor([0])
            ),
        )
        self.assert_exact_error(
            ValueError,
            "Missing input: 'indices' must be supplied when external sparse routing "
            "is used by MixtureOfExperts.",
            lambda: MixtureOfExpertsValidator.validate_external_routing_inputs(
                sparse, torch.ones(1), None
            ),
        )
        dense = MixtureOfExperts(_mixture_config())
        self.assert_exact_error(
            ValueError,
            "Input Error: 'indices' must be None when 'top_k' equals 'num_experts' "
            "for dense MixtureOfExperts routing.",
            lambda: MixtureOfExpertsValidator.validate_external_routing_inputs(
                dense,
                torch.ones(1, 2),
                torch.tensor([[0, 1]]),
            ),
        )

        reduced = MixtureOfExpertsReduce(_mixture_config(top_k=2, num_experts=3))
        inputs = torch.ones(4, 2)
        self.assert_exact_error(
            ValueError,
            "Input Error: 'probabilities' must contain one routing weight per "
            "flattened reduce input sample, received probabilities shape (3, 2) "
            "and input_batch shape (4, 2).",
            lambda: MixtureOfExpertsValidator.validate_reduce_forward_inputs(
                reduced,
                inputs,
                torch.ones(3, 2),
                torch.tensor([[0, 1], [1, 2]]),
            ),
        )
        self.assert_exact_error(
            ValueError,
            "Input Error: 'indices' must contain one expert id per flattened "
            "reduce input sample, received indices shape (3, 2) and input_batch "
            "shape (4, 2).",
            lambda: MixtureOfExpertsValidator.validate_reduce_forward_inputs(
                reduced,
                inputs,
                torch.ones(2, 2),
                torch.tensor([[0, 1], [1, 2], [2, 0]]),
            ),
        )

        handler = ExpertWeightingHandler(_mixture_config(top_k=1, num_experts=2))
        self.assert_exact_error(
            ValueError,
            "Missing input: `probabilities` must be supplied when `indices` are "
            "used to ensure accurate weighting and processing of inputs.",
            lambda: handler.maybe_apply_probabilities_after(torch.ones(1, 2), None),
        )


if __name__ == "__main__":
    unittest.main()
