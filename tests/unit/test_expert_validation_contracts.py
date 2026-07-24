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
from emperor.experts._layers.map import MixtureOfExpertsMap
from emperor.experts._layers.mixture import MixtureOfExperts
from emperor.experts._layers.reduce import MixtureOfExpertsReduce
from emperor.experts._model import MixtureOfExpertsModel
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    MirroredLayerStackConfig,
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
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def _mixture_config(
    *,
    input_dim: int = 2,
    output_dim: int = 2,
    top_k: int = 1,
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


def _model_config(
    *,
    input_dim: int = 2,
    output_dim: int = 2,
    top_k: int = 1,
) -> MixtureOfExpertsModelConfig:
    mixture_config = _mixture_config(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=top_k,
    )
    return MixtureOfExpertsModelConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=top_k,
        routing_initialization_mode=RoutingInitializationMode.DISABLED,
        sampler_config=None,
        stack_config=LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=max(input_dim, output_dim),
            output_dim=output_dim,
            num_layers=1,
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
        ),
    )


def _sampler_config(
    *,
    input_dim: int = 2,
    top_k: int = 1,
    num_experts: int = 2,
) -> SamplerConfig:
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
            input_dim=input_dim,
            num_experts=num_experts,
            noisy_topk_flag=False,
            model_config=_linear_stack(input_dim, num_experts),
        ),
    )


class ExpertValidationContractTests(unittest.TestCase):
    def test_expert_constructors_reject_malformed_config_before_dereference(
        self,
    ) -> None:
        for module_type in (
            MixtureOfExperts,
            MixtureOfExpertsMap,
            MixtureOfExpertsReduce,
        ):
            with self.subTest(module_type=module_type.__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    "Configuration Error: `cfg` must be of type "
                    "MixtureOfExpertsConfig, received type object",
                ):
                    module_type(object())  # type: ignore[arg-type]

    def test_expert_constructors_reject_malformed_overrides_before_dereference(
        self,
    ) -> None:
        for module_type in (
            MixtureOfExperts,
            MixtureOfExpertsMap,
            MixtureOfExpertsReduce,
        ):
            with self.subTest(module_type=module_type.__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    "Configuration Error: `overrides` must be of type "
                    "MixtureOfExpertsConfig or None, received type object",
                ):
                    module_type(
                        _mixture_config(),
                        object(),  # type: ignore[arg-type]
                    )

    def test_model_config_build_rejects_the_first_missing_required_field(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "input_dim is required for MixtureOfExpertsModelConfig, received None",
        ):
            MixtureOfExpertsModelConfig().build()

    def test_model_config_build_rejects_each_missing_required_field(self) -> None:
        for field_name in (
            "input_dim",
            "output_dim",
            "top_k",
            "routing_initialization_mode",
            "stack_config",
        ):
            with self.subTest(field_name=field_name):
                config = _model_config()
                setattr(config, field_name, None)

                with self.assertRaisesRegex(
                    ValueError,
                    f"{field_name} is required for MixtureOfExpertsModelConfig",
                ):
                    config.build()

    def test_model_config_build_rejects_each_invalid_field_type(self) -> None:
        invalid_values = {
            "input_dim": "2",
            "output_dim": 2.0,
            "top_k": True,
            "routing_initialization_mode": object(),
            "stack_config": object(),
        }
        for field_name, invalid_value in invalid_values.items():
            with self.subTest(field_name=field_name):
                config = _model_config()
                setattr(config, field_name, invalid_value)

                with self.assertRaisesRegex(TypeError, field_name):
                    config.build()

    def test_model_config_build_rejects_non_positive_outer_dimensions(self) -> None:
        for field_name in ("input_dim", "output_dim", "top_k"):
            for invalid_value in (0, -1):
                with self.subTest(
                    field_name=field_name,
                    invalid_value=invalid_value,
                ):
                    config = _model_config()
                    setattr(config, field_name, invalid_value)

                    with self.assertRaisesRegex(
                        ValueError,
                        f"Configuration Error: '{field_name}' must be a positive integer",
                    ):
                        config.build()

    def test_model_constructor_rejects_a_malformed_config_before_rng_work(
        self,
    ) -> None:
        rng_state = torch.random.get_rng_state()

        with self.assertRaisesRegex(
            TypeError,
            "Configuration Error: `cfg` must be of type "
            "MixtureOfExpertsModelConfig, received type object",
        ):
            MixtureOfExpertsModel(object())  # type: ignore[arg-type]

        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))

    def test_model_constructor_rejects_malformed_overrides_before_rng_work(
        self,
    ) -> None:
        rng_state = torch.random.get_rng_state()

        with self.assertRaisesRegex(
            TypeError,
            "Configuration Error: `overrides` must be of type "
            "MixtureOfExpertsModelConfig or None, received type object",
        ):
            MixtureOfExpertsModel(
                _model_config(),
                object(),  # type: ignore[arg-type]
            )

        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))

    def test_mixture_config_build_rejects_non_finite_capacity(self) -> None:
        for invalid_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid_value=invalid_value):
                config = _mixture_config()
                config.capacity_factor = invalid_value

                with self.assertRaises(ValueError) as error:
                    config.build()
                self.assertEqual(
                    str(error.exception),
                    "Configuration Error: 'capacity_factor' must be finite, "
                    f"received {invalid_value}",
                )

    def test_model_build_rejects_outer_and_stack_boundary_dimension_mismatch(
        self,
    ) -> None:
        for dimension_name in ("input_dim", "output_dim"):
            with self.subTest(dimension_name=dimension_name):
                config = _model_config()
                setattr(config, dimension_name, 3)

                with self.assertRaisesRegex(
                    ValueError,
                    f"Configuration Error: model {dimension_name} must match "
                    f"stack_config.{dimension_name}",
                ):
                    config.build()

    def test_model_build_rejects_outer_and_leaf_top_k_mismatch(self) -> None:
        config = _model_config()
        config.top_k = 2

        with self.assertRaisesRegex(
            ValueError,
            "Configuration Error: model top_k must match the expert leaf top_k, "
            "received top_k=2 and leaf top_k=1",
        ):
            config.build()

    def test_model_build_rejects_non_expert_stack_and_leaf_configs(self) -> None:
        config = _model_config()
        config.stack_config.layer_config = _linear_stack(2, 2).layer_config
        with self.assertRaises(TypeError) as error:
            config.build()
        self.assertEqual(
            str(error.exception),
            "Configuration Error: 'stack_config.layer_config' must be of type "
            "MixtureOfExpertsLayerConfig, received type LayerConfig",
        )

        config = _model_config()
        config.stack_config.layer_config.layer_model_config = LinearLayerConfig(
            bias_flag=True
        )
        with self.assertRaises(TypeError) as error:
            config.build()
        self.assertEqual(
            str(error.exception),
            "Configuration Error: "
            "'stack_config.layer_config.layer_model_config' must be of type "
            "MixtureOfExpertsConfig, received type LinearLayerConfig",
        )

    def test_model_build_allows_layer_stack_to_override_leaf_dimensions(self) -> None:
        for leaf_input_dim, leaf_output_dim in ((11, 13), (None, None)):
            with self.subTest(
                leaf_input_dim=leaf_input_dim,
                leaf_output_dim=leaf_output_dim,
            ):
                config = _model_config(input_dim=2, output_dim=4)
                config.stack_config.hidden_dim = 3
                config.stack_config.num_layers = 2
                leaf_config = config.stack_config.layer_config.layer_model_config
                leaf_config.input_dim = leaf_input_dim
                leaf_config.output_dim = leaf_output_dim
                leaf_config.expert_model_config = _linear_stack(11, 13)
                model = config.build()
                state = MixtureOfExpertsLayerState(
                    hidden=torch.ones(2, 2),
                    probabilities=torch.ones(2),
                    indices=torch.tensor([0, 1]),
                )

                result = model(state)

                self.assertEqual(result.hidden.shape, (2, 4))
                self.assertEqual(leaf_config.input_dim, leaf_input_dim)
                self.assertEqual(leaf_config.output_dim, leaf_output_dim)

    def test_single_layer_model_ignores_the_unused_hidden_dimension(self) -> None:
        config = _model_config(input_dim=2, output_dim=2)
        config.stack_config.hidden_dim = 3
        config.stack_config.num_layers = 1
        leaf_config = config.stack_config.layer_config.layer_model_config
        leaf_config.capacity_factor = 1.0

        model = config.build()
        state = MixtureOfExpertsLayerState(
            hidden=torch.ones(2, 2),
            probabilities=torch.ones(2),
            indices=torch.tensor([0, 1]),
        )

        result = model(state)

        self.assertEqual(len(model.expert_stack), 1)
        self.assertEqual(model.expert_stack[0].input_dim, 2)
        self.assertEqual(model.expert_stack[0].output_dim, 2)
        self.assertEqual(result.hidden.shape, (2, 2))

    def test_model_preflights_and_builds_each_mirrored_layer_dimension(self) -> None:
        config = _model_config()
        layer_config = config.stack_config.layer_config
        config.stack_config = MirroredLayerStackConfig(
            input_dim=2,
            hidden_dim=3,
            output_dim=2,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=layer_config,
        )
        model = config.build()
        state = MixtureOfExpertsLayerState(
            hidden=torch.ones(2, 2),
            probabilities=torch.ones(2),
            indices=torch.tensor([0, 1]),
        )

        result = model(state)

        self.assertEqual(len(model.expert_stack), 4)
        self.assertEqual(result.hidden.shape, (2, 2))
