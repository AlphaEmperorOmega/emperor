import unittest

import torch
from emperor.experts import DroppedTokenOptions, MixtureOfExpertsConfig
from emperor.experts._layers.mixture import MixtureOfExperts
from emperor.experts._routing.capacity import ExpertCapacityHandler
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

from tests.unit.test_expert_behavioral_contracts import _mixture_config


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
