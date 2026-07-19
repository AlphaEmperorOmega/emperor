import unittest

import torch
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsLayerState,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
)
from emperor.experts._layers.mixture import MixtureOfExperts
from emperor.experts._model import MixtureOfExpertsModel
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, SamplerConfig


def _linear_stack(input_dim: int, output_dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
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
            layer_model_config=LinearLayerConfig(bias_flag=False),
        ),
    )


def _mixture_config(
    *,
    input_dim: int = 2,
    output_dim: int = 2,
    top_k: int = 2,
    num_experts: int = 2,
) -> MixtureOfExpertsConfig:
    return MixtureOfExpertsConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=top_k,
        num_experts=num_experts,
        capacity_factor=0.0,
        dropped_token_behavior=DroppedTokenOptions.ZEROS,
        compute_expert_mixture_flag=True,
        weighted_parameters_flag=True,
        weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
        routing_initialization_mode=RoutingInitializationMode.DISABLED,
        sampler_config=None,
        expert_model_config=_linear_stack(input_dim, output_dim),
    )


def _sampler_config(*, top_k: int = 1, num_experts: int = 2) -> SamplerConfig:
    return SamplerConfig(
        top_k=top_k,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        normalize_probabilities_flag=False,
        noisy_topk_flag=False,
        num_experts=num_experts,
        coefficient_of_variation_loss_weight=0.0,
        switch_loss_weight=0.0,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        router_config=None,
    )


def _owned_routing_mixture_config(
    routing_mode: RoutingInitializationMode,
    *,
    threshold: float = 0.0,
) -> MixtureOfExpertsConfig:
    config = _mixture_config(top_k=1, num_experts=2)
    sampler_config = _sampler_config(top_k=1, num_experts=2)
    sampler_config.threshold = threshold
    sampler_config.router_config = RouterConfig(
        input_dim=2,
        num_experts=2,
        noisy_topk_flag=False,
        model_config=_linear_stack(2, 2),
    )
    config.routing_initialization_mode = routing_mode
    config.sampler_config = sampler_config
    return config


def _routing_model_config(
    routing_mode: RoutingInitializationMode,
    *,
    num_layers: int = 2,
) -> MixtureOfExpertsModelConfig:
    mixture_config = _owned_routing_mixture_config(routing_mode)
    stack_config = LayerStackConfig(
        input_dim=2,
        hidden_dim=2,
        output_dim=2,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=MixtureOfExpertsLayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=mixture_config,
        ),
    )
    return MixtureOfExpertsModelConfig(
        input_dim=2,
        output_dim=2,
        top_k=1,
        routing_initialization_mode=routing_mode,
        sampler_config=mixture_config.sampler_config,
        stack_config=stack_config,
    )


class ExpertBehavioralContractTests(unittest.TestCase):
    def test_owned_sampler_receives_exact_mask_and_returns_its_update(self) -> None:
        model = MixtureOfExperts(
            _owned_routing_mixture_config(RoutingInitializationMode.LAYER)
        )
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        skip_mask = torch.tensor([[True], [False]])
        sampler_updated_mask = torch.tensor([[False], [False]])
        supplied_masks = []
        original_sample = model.sampler.sample_probabilities_and_indices

        def sample_with_mask_update(input_matrix, supplied_skip_mask=None):
            supplied_masks.append(supplied_skip_mask)
            probabilities, indices, _, loss = original_sample(
                input_matrix, supplied_skip_mask
            )
            return probabilities, indices, sampler_updated_mask, loss

        model.sampler.sample_probabilities_and_indices = sample_with_mask_update

        output, returned_mask, loss = model(inputs, skip_mask=skip_mask)

        self.assertEqual(len(supplied_masks), 1)
        self.assertIs(supplied_masks[0], skip_mask)
        self.assertIs(returned_mask, sampler_updated_mask)
        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(loss))

    def test_skip_mask_validation_covers_type_shape_batch_and_device(self) -> None:
        model = MixtureOfExperts(_mixture_config())
        inputs = torch.ones(2, 2)
        probabilities = torch.full((2, 2), 0.5)
        cases = (
            (TypeError, [1, 0]),
            (ValueError, torch.ones(2)),
            (ValueError, torch.ones(2, 2)),
            (ValueError, torch.ones(3, 1)),
            (ValueError, torch.ones(2, 1, device="meta")),
        )

        for error_type, skip_mask in cases:
            with self.subTest(error_type=error_type.__name__, skip_mask=skip_mask):
                with self.assertRaises(error_type):
                    model(
                        inputs,
                        probabilities=probabilities,
                        indices=None,
                        skip_mask=skip_mask,
                    )

    def test_default_and_explicit_none_mask_preserve_output_loss_and_rng(
        self,
    ) -> None:
        model = MixtureOfExperts(
            _owned_routing_mixture_config(RoutingInitializationMode.LAYER)
        ).eval()
        inputs = torch.tensor([[0.25, -0.5], [0.75, 1.0]])
        torch.manual_seed(91)
        initial_rng_state = torch.random.get_rng_state()

        default_output, default_mask, default_loss = model(inputs)
        default_rng_state = torch.random.get_rng_state()
        torch.random.set_rng_state(initial_rng_state)
        explicit_output, explicit_mask, explicit_loss = model(inputs, skip_mask=None)
        explicit_rng_state = torch.random.get_rng_state()

        torch.testing.assert_close(default_output, explicit_output)
        torch.testing.assert_close(default_loss, explicit_loss)
        self.assertIsNone(default_mask)
        self.assertIsNone(explicit_mask)
        self.assertTrue(torch.equal(default_rng_state, explicit_rng_state))

    def test_layer_routing_threads_sampler_updates_across_layers(self) -> None:
        model = MixtureOfExpertsModel(
            _routing_model_config(RoutingInitializationMode.LAYER)
        )
        initial_mask = torch.tensor([[1.0], [1.0], [0.0]])
        first_update = torch.tensor([[1.0], [0.0], [0.0]])
        second_update = torch.tensor([[0.0], [0.0], [0.0]])
        expected_inputs = (initial_mask, first_update)
        updates = (first_update, second_update)
        supplied_masks = []

        for expert_layer, expected_input, update in zip(
            model.expert_stack,
            expected_inputs,
            updates,
            strict=True,
        ):
            sampler = expert_layer.model.sampler
            original_sample = sampler.sample_probabilities_and_indices

            def sample_with_update(
                input_matrix,
                supplied_skip_mask=None,
                *,
                _original_sample=original_sample,
                _expected_input=expected_input,
                _update=update,
            ):
                supplied_masks.append(supplied_skip_mask)
                self.assertIs(supplied_skip_mask, _expected_input)
                probabilities, indices, _, loss = _original_sample(
                    input_matrix, supplied_skip_mask
                )
                return probabilities, indices, _update, loss

            sampler.sample_probabilities_and_indices = sample_with_update

        result = model(
            MixtureOfExpertsLayerState(
                hidden=torch.randn(3, 2),
                skip_mask=initial_mask,
            )
        )

        self.assertEqual(len(supplied_masks), 2)
        self.assertIs(result.skip_mask, second_update)

    def test_shared_routing_samples_once_and_leaf_layers_preserve_update(
        self,
    ) -> None:
        model = MixtureOfExpertsModel(
            _routing_model_config(RoutingInitializationMode.SHARED)
        )
        initial_mask = torch.tensor([[True], [True], [False]])
        shared_update = torch.tensor([[True], [False], [False]])
        shared_sampler_masks = []
        leaf_masks = []
        original_sample = model.shared_sampler.sample_probabilities_and_indices

        def sample_once(input_matrix, supplied_skip_mask=None):
            shared_sampler_masks.append(supplied_skip_mask)
            probabilities, indices, _, loss = original_sample(
                input_matrix, supplied_skip_mask
            )
            return probabilities, indices, shared_update, loss

        model.shared_sampler.sample_probabilities_and_indices = sample_once
        for expert_layer in model.expert_stack:
            expert_model = expert_layer.model
            original_forward = expert_model.forward

            def record_leaf_mask(
                input_batch,
                probabilities=None,
                indices=None,
                skip_mask=None,
                *,
                _original_forward=original_forward,
            ):
                leaf_masks.append(skip_mask)
                return _original_forward(input_batch, probabilities, indices, skip_mask)

            expert_model.forward = record_leaf_mask

        result = model(
            MixtureOfExpertsLayerState(
                hidden=torch.randn(3, 2),
                skip_mask=initial_mask,
            )
        )

        self.assertEqual(shared_sampler_masks, [initial_mask])
        self.assertEqual(len(leaf_masks), 2)
        self.assertTrue(all(mask is shared_update for mask in leaf_masks))
        self.assertIs(result.skip_mask, shared_update)

    def test_disabled_routing_preserves_mask_through_complete_stack(self) -> None:
        model = MixtureOfExpertsModel(
            _routing_model_config(RoutingInitializationMode.DISABLED)
        )
        skip_mask = torch.tensor([[1.0], [0.5], [0.0]])
        result = model(
            MixtureOfExpertsLayerState(
                hidden=torch.randn(3, 2),
                probabilities=torch.ones(3),
                indices=torch.tensor([0, 1, 0]),
                skip_mask=skip_mask,
            )
        )

        self.assertIs(result.skip_mask, skip_mask)

    def test_all_active_mask_matches_baseline_and_mixed_mask_keeps_gradients(
        self,
    ) -> None:
        config = _owned_routing_mixture_config(
            RoutingInitializationMode.LAYER,
            threshold=1e-8,
        )
        baseline_model = MixtureOfExperts(config).eval()
        active_mask_model = MixtureOfExperts(config).eval()
        mixed_mask_model = MixtureOfExperts(config).eval()
        active_mask_model.load_state_dict(baseline_model.state_dict())
        mixed_mask_model.load_state_dict(baseline_model.state_dict())
        baseline_inputs = torch.tensor(
            [[0.25, -0.5], [0.75, 1.0], [-0.25, 0.5]],
            requires_grad=True,
        )
        active_mask_inputs = baseline_inputs.detach().clone().requires_grad_()
        all_active_mask = torch.ones(3, 1, dtype=torch.bool)

        baseline_output, _, baseline_loss = baseline_model(baseline_inputs)
        active_output, returned_active_mask, active_loss = active_mask_model(
            active_mask_inputs,
            skip_mask=all_active_mask,
        )
        torch.testing.assert_close(active_output, baseline_output)
        torch.testing.assert_close(active_loss, baseline_loss)
        self.assertEqual(returned_active_mask.dtype, all_active_mask.dtype)
        torch.testing.assert_close(returned_active_mask, all_active_mask)

        (baseline_output.sum() + baseline_loss).backward()
        (active_output.sum() + active_loss).backward()
        torch.testing.assert_close(active_mask_inputs.grad, baseline_inputs.grad)

        mixed_inputs = baseline_inputs.detach().clone().requires_grad_()
        mixed_mask = torch.tensor([[1.0], [0.0], [0.5]])
        mixed_output, returned_mixed_mask, mixed_loss = mixed_mask_model(
            mixed_inputs,
            skip_mask=mixed_mask,
        )
        self.assertTrue(torch.isfinite(mixed_output).all())
        self.assertTrue(torch.isfinite(mixed_loss))
        self.assertEqual(returned_mixed_mask.dtype, mixed_mask.dtype)
        (mixed_output.sum() + mixed_loss).backward()
        self.assertTrue(torch.isfinite(mixed_inputs.grad).all())
        active_rows = mixed_mask.squeeze(1) != 0
        self.assertGreater(mixed_inputs.grad[active_rows].abs().sum().item(), 0.0)
