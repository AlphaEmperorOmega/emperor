import unittest

import torch

from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, SamplerConfig


def _linear_stack(input_dim: int = 2, output_dim: int = 2) -> LayerStackConfig:
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
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def _mixture_config(
    *,
    top_k: int = 2,
    num_experts: int = 3,
    weighting_position: ExpertWeightingPositionOptions = (
        ExpertWeightingPositionOptions.AFTER_EXPERTS
    ),
    expert_model_config=None,
) -> MixtureOfExpertsConfig:
    return MixtureOfExpertsConfig(
        input_dim=2,
        output_dim=2,
        top_k=top_k,
        num_experts=num_experts,
        capacity_factor=0.0,
        dropped_token_behavior=DroppedTokenOptions.ZEROS,
        compute_expert_mixture_flag=True,
        weighted_parameters_flag=True,
        weighting_position_option=weighting_position,
        routing_initialization_mode=RoutingInitializationMode.DISABLED,
        sampler_config=None,
        expert_model_config=expert_model_config or _linear_stack(),
    )


def _recurrent_linear_expert() -> RecurrentLayerConfig:
    return RecurrentLayerConfig(
        input_dim=2,
        output_dim=2,
        max_steps=2,
        recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
        block_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
        gate_config=None,
        residual_config=None,
        halting_config=None,
        memory_config=None,
    )


def _sampler_config(*, top_k: int = 2, num_experts: int = 2) -> SamplerConfig:
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
        router_config=RouterConfig(
            input_dim=2,
            num_experts=num_experts,
            noisy_topk_flag=False,
            model_config=_linear_stack(input_dim=2, output_dim=num_experts),
        ),
    )


def _routed_mixture_model_config(
    routing_mode: RoutingInitializationMode,
) -> MixtureOfExpertsModelConfig:
    sampler_config = _sampler_config()
    mixture_config = _mixture_config(top_k=2, num_experts=2)
    mixture_config.routing_initialization_mode = routing_mode
    mixture_config.sampler_config = sampler_config
    stack_config = LayerStackConfig(
        input_dim=2,
        hidden_dim=2,
        output_dim=2,
        num_layers=2,
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
        top_k=2,
        routing_initialization_mode=routing_mode,
        sampler_config=sampler_config,
        stack_config=stack_config,
    )


def _disabled_mixture_model_config() -> MixtureOfExpertsModelConfig:
    mixture_config = _mixture_config(top_k=2, num_experts=2)
    stack_config = LayerStackConfig(
        input_dim=2,
        hidden_dim=2,
        output_dim=2,
        num_layers=2,
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
        top_k=2,
        routing_initialization_mode=RoutingInitializationMode.DISABLED,
        sampler_config=None,
        stack_config=stack_config,
    )


def _asymmetric_affines(dtype: torch.dtype):
    return (
        (
            torch.tensor([[1.5, -0.5], [0.25, 2.0]], dtype=dtype),
            torch.tensor([0.3, -0.2], dtype=dtype),
        ),
        (
            torch.tensor([[-1.0, 1.25], [2.5, 0.75]], dtype=dtype),
            torch.tensor([-0.4, 0.6], dtype=dtype),
        ),
        (
            torch.tensor([[0.5, 1.5], [-2.0, 0.25]], dtype=dtype),
            torch.tensor([0.8, -0.7], dtype=dtype),
        ),
    )


def _set_linear_expert_affines(model, affines) -> None:
    with torch.no_grad():
        for expert_stack, (weight, bias) in zip(
            model.expert_modules,
            affines,
            strict=True,
        ):
            linear = expert_stack[0].model
            linear.weight_params.copy_(weight)
            linear.bias_params.copy_(bias)


def _expected_sparse_affine_mixture(
    inputs: torch.Tensor,
    probabilities: torch.Tensor,
    indices: torch.Tensor,
    affines,
    weighting_position: ExpertWeightingPositionOptions,
) -> torch.Tensor:
    sample_outputs = []
    for sample, sample_probabilities, sample_indices in zip(
        inputs,
        probabilities,
        indices,
        strict=True,
    ):
        route_outputs = []
        for probability, expert_index in zip(
            sample_probabilities,
            sample_indices,
            strict=True,
        ):
            weight, bias = affines[int(expert_index)]
            if weighting_position == ExpertWeightingPositionOptions.BEFORE_EXPERTS:
                route_output = (probability * sample) @ weight + bias
            else:
                route_output = probability * (sample @ weight + bias)
            route_outputs.append(route_output)
        sample_outputs.append(torch.stack(route_outputs).sum(dim=0))
    return torch.stack(sample_outputs)


class ExpertNumericalContractTests(unittest.TestCase):
    def test_weighting_positions_match_biased_affine_equations_for_float_dtypes(
        self,
    ) -> None:
        for dtype in (torch.float32, torch.float64):
            for weighting_position in ExpertWeightingPositionOptions:
                with self.subTest(dtype=dtype, weighting_position=weighting_position):
                    model = _mixture_config(
                        weighting_position=weighting_position
                    ).build()
                    model.to(dtype=dtype)
                    affines = _asymmetric_affines(dtype)
                    _set_linear_expert_affines(model, affines)
                    inputs = torch.tensor(
                        [[1.2, -0.7], [-0.4, 2.1]],
                        dtype=dtype,
                    )
                    probabilities = torch.tensor(
                        [[0.2, 0.7], [0.6, 0.1]],
                        dtype=dtype,
                    )
                    indices = torch.tensor([[0, 1], [1, 0]])
                    expected = _expected_sparse_affine_mixture(
                        inputs,
                        probabilities,
                        indices,
                        affines,
                        weighting_position,
                    )

                    output, skip_mask, auxiliary_loss = model(
                        inputs,
                        probabilities=probabilities,
                        indices=indices,
                    )

                    torch.testing.assert_close(output, expected)
                    self.assertIsNone(skip_mask)
                    torch.testing.assert_close(auxiliary_loss, inputs.new_zeros(()))

    def test_probability_boundaries_have_exact_finite_affine_jacobians(
        self,
    ) -> None:
        dtype = torch.float64
        probabilities_template = torch.tensor(
            [0.0, 1e-12, 1.0],
            dtype=dtype,
        )
        inputs_template = torch.tensor(
            [[1.2, -0.7], [-0.4, 2.1], [0.8, 0.3]],
            dtype=dtype,
        )
        weight, bias = _asymmetric_affines(dtype)[0]

        for weighting_position in ExpertWeightingPositionOptions:
            with self.subTest(weighting_position=weighting_position):
                model = _mixture_config(
                    top_k=1,
                    num_experts=1,
                    weighting_position=weighting_position,
                ).build()
                model.to(dtype=dtype)
                _set_linear_expert_affines(model, ((weight, bias),))
                inputs = inputs_template.clone().requires_grad_()
                probabilities = probabilities_template.clone().requires_grad_()

                output, _skip_mask, auxiliary_loss = model(
                    inputs,
                    probabilities=probabilities,
                    indices=None,
                )
                if weighting_position == ExpertWeightingPositionOptions.BEFORE_EXPERTS:
                    expected_output = (
                        probabilities.detach().reshape(-1, 1) * inputs.detach()
                    ) @ weight + bias
                    expected_probability_gradient = (inputs.detach() @ weight).sum(
                        dim=-1
                    )
                    expected_bias_gradient = torch.full_like(
                        bias,
                        inputs.shape[0],
                    )
                else:
                    expected_output = probabilities.detach().reshape(-1, 1) * (
                        inputs.detach() @ weight + bias
                    )
                    expected_probability_gradient = (
                        inputs.detach() @ weight + bias
                    ).sum(dim=-1)
                    expected_bias_gradient = torch.full_like(
                        bias,
                        probabilities.detach().sum(),
                    )

                objective = output.sum() + auxiliary_loss
                objective.backward()
                expected_input_gradient = probabilities.detach().reshape(
                    -1, 1
                ) * weight.sum(dim=-1)
                weighted_inputs = (
                    probabilities.detach().reshape(-1, 1) * inputs.detach()
                )
                expected_weight_gradient = (
                    weighted_inputs.sum(dim=0).reshape(-1, 1).expand_as(weight)
                )
                linear = model.expert_modules[0][0].model

                torch.testing.assert_close(output, expected_output)
                torch.testing.assert_close(inputs.grad, expected_input_gradient)
                torch.testing.assert_close(
                    probabilities.grad,
                    expected_probability_gradient,
                )
                torch.testing.assert_close(
                    linear.weight_params.grad,
                    expected_weight_gradient,
                )
                torch.testing.assert_close(
                    linear.bias_params.grad,
                    expected_bias_gradient,
                )
                for value in (
                    output,
                    inputs.grad,
                    probabilities.grad,
                    linear.weight_params.grad,
                    linear.bias_params.grad,
                ):
                    self.assertTrue(torch.isfinite(value).all())
