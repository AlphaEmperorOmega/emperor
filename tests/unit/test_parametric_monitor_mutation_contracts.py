import math
import unittest
from types import SimpleNamespace

import torch
from lightning import LightningModule

from emperor.parametric import (
    MatrixWeightsMixtureConfig,
    ParametricLayer,
    ParametricLayerMonitorCallback,
)
from emperor.parametric._monitoring import (
    _ParametricDiagnostics,
    _ParametricObservation,
    _ParametricTrackingContext,
)
from tests.unit.test_parametric_behavioral_contracts import (
    _mixture_kwargs,
    _parametric_config,
)


class _MalformedObservedModule(torch.nn.Module):
    def _generate_parameters(self, *args: object, **kwargs: object) -> str:
        return "malformed generated parameters"

    def _compute_affine_transformation_callback(
        self,
        *args: object,
        **kwargs: object,
    ) -> str:
        return "malformed affine output"

    def _ParametricLayer__sample_weight_probabilities_and_indices(
        self,
        *args: object,
        **kwargs: object,
    ) -> str:
        return "malformed router sample"


class _RecordingLightningModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.logged_names: list[str] = []

    def log(
        self,
        name: str,
        value: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.logged_names.append(name)


class ParametricMonitorMutationContractTests(unittest.TestCase):
    def test_constructor_defaults_and_validation_messages_are_exact(self) -> None:
        callback = ParametricLayerMonitorCallback()

        self.assertEqual(callback.log_every_n_steps, 100)
        self.assertEqual(callback.history_size, 128)
        self.assertIs(callback.log_per_slot_scalars, False)
        for kwargs, expected_message in (
            (
                {"log_every_n_steps": 0},
                "log_every_n_steps must be greater than 0.",
            ),
            (
                {"history_size": 0},
                "history_size must be greater than 0.",
            ),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError) as error:
                    ParametricLayerMonitorCallback(**kwargs)
                self.assertEqual(str(error.exception), expected_message)

    def test_clip_saturation_includes_the_exact_boundary(self) -> None:
        config = MatrixWeightsMixtureConfig(**_mixture_kwargs())
        config.clip_range = 2.0
        mixture = config.build()
        values = torch.tensor([-3.0, -2.0, 1.999, 2.0])

        saturation = _ParametricDiagnostics.clip_saturation(mixture, values)

        torch.testing.assert_close(saturation, torch.tensor(0.75))

    def test_router_entropy_matches_hand_calculated_asymmetric_rows(self) -> None:
        probabilities = torch.tensor([[0.3, 0.1], [0.1, 0.1]])
        expected = (
            -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) + math.log(2.0)
        ) / 2.0

        entropy = _ParametricDiagnostics.router_entropy(probabilities)

        torch.testing.assert_close(entropy, torch.tensor(expected))

    def test_top_one_router_entropy_is_zero_per_sample_and_vector_axis(self) -> None:
        matrix_probabilities = torch.tensor([0.9, 0.2, 0.4])
        vector_probabilities = torch.tensor([[0.9, 0.2, 0.4], [0.1, 0.8, 0.3]])

        torch.testing.assert_close(
            _ParametricDiagnostics.router_entropy(
                matrix_probabilities,
                top_k=1,
            ),
            torch.tensor(0.0),
        )
        torch.testing.assert_close(
            _ParametricDiagnostics.router_entropy(
                vector_probabilities,
                top_k=1,
            ),
            torch.tensor(0.0),
        )

    def test_sparse_utilization_counts_zero_and_dead_experts_exactly(self) -> None:
        indices = torch.tensor([[0, 2], [1, 2], [2, 1]])

        utilization = _ParametricDiagnostics.utilization(
            probabilities=torch.ones(3, 2),
            indices=indices,
            num_experts=4,
        )

        torch.testing.assert_close(
            utilization,
            torch.tensor([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 0.0]),
        )
        torch.testing.assert_close(
            _ParametricDiagnostics.utilization(
                probabilities=torch.ones(1, 1),
                indices=torch.tensor([[0]]),
                num_experts=1,
            ),
            torch.ones(1),
        )
        torch.testing.assert_close(
            _ParametricDiagnostics.utilization(
                probabilities=torch.ones(1, 1),
                indices=torch.tensor([[0]]),
                num_experts=2,
            ),
            torch.tensor([1.0, 0.0]),
        )

    def test_dense_utilization_normalizes_each_sample_before_averaging(self) -> None:
        probabilities = torch.tensor([[0.1, 0.2, 0.1], [0.0, 0.3, 0.1]])

        utilization = _ParametricDiagnostics.utilization(
            probabilities=probabilities,
            indices=None,
            num_experts=3,
        )

        torch.testing.assert_close(
            utilization,
            torch.tensor([0.125, 0.625, 0.25]),
        )

    def test_sparse_utilization_preserves_the_indices_device(self) -> None:
        indices = torch.tensor([[0, 1], [1, 1]], device="cpu")

        with torch.device("meta"):
            utilization = _ParametricDiagnostics.utilization(
                probabilities=None,
                indices=indices,
                num_experts=2,
            )

        self.assertIsNotNone(utilization)
        self.assertEqual(utilization.device, indices.device)
        torch.testing.assert_close(utilization, torch.tensor([0.25, 0.75]))

    def test_diagnostics_handle_disabled_clipping_and_unusable_routing(self) -> None:
        values = torch.tensor([[-2.0, 0.5], [1.0, 3.0]], dtype=torch.float64)

        without_range = _ParametricDiagnostics.clip_saturation(object(), values)
        zero_range = _ParametricDiagnostics.clip_saturation(
            SimpleNamespace(clip_range=0.0),
            values,
        )

        torch.testing.assert_close(without_range, values.new_zeros(()))
        torch.testing.assert_close(zero_range, values.new_zeros(()))
        self.assertEqual(without_range.dtype, torch.float64)
        self.assertEqual(without_range.device, values.device)

        self.assertIsNone(
            _ParametricDiagnostics.utilization(
                probabilities=None,
                indices=torch.tensor([], dtype=torch.long),
                num_experts=0,
            )
        )
        invalid_indices = torch.tensor([-2, 3, 4, 9])
        torch.testing.assert_close(
            _ParametricDiagnostics.utilization(
                probabilities=None,
                indices=invalid_indices,
                num_experts=3,
            ),
            torch.zeros(3),
        )
        self.assertIsNone(
            _ParametricDiagnostics.utilization(
                probabilities=None,
                indices=None,
                num_experts=3,
            )
        )
        self.assertIsNone(
            _ParametricDiagnostics.utilization(
                probabilities=torch.ones(2, 2),
                indices=None,
                num_experts=3,
            )
        )

    def test_malformed_wrapper_outputs_are_ignored_and_restored(self) -> None:
        callback = ParametricLayerMonitorCallback(log_every_n_steps=1)
        observed = _MalformedObservedModule()
        observation = _ParametricObservation()
        callback._observations[id(observed)] = observation
        callback._ParametricLayerMonitorCallback__record_generated_parameters(
            observation,
            (object(), object(), object(), object()),
        )

        callback._ParametricLayerMonitorCallback__wrap_generate_parameters(observed)
        callback._ParametricLayerMonitorCallback__wrap_affine_callback(observed)
        callback._ParametricLayerMonitorCallback__wrap_sampling_method(
            observed,
            "_ParametricLayer__sample_weight_probabilities_and_indices",
            "weight",
        )

        self.assertEqual(
            observed._generate_parameters(),
            "malformed generated parameters",
        )
        self.assertEqual(
            observed._compute_affine_transformation_callback(
                None,
                None,
                object(),
            ),
            "malformed affine output",
        )
        self.assertEqual(
            observed._ParametricLayer__sample_weight_probabilities_and_indices(),
            "malformed router sample",
        )
        self.assertEqual(observation, _ParametricObservation())

        callback._ParametricLayerMonitorCallback__cleanup()

        self.assertEqual(callback._wrapped_methods, [])
        self.assertEqual(callback._observations, {})

    def test_real_wrappers_forward_keyword_and_mixed_affine_calls(self) -> None:
        callback = ParametricLayerMonitorCallback(log_every_n_steps=1)
        layer = ParametricLayer(_parametric_config())
        observation = _ParametricObservation()
        callback._observations[id(layer)] = observation
        callback._ParametricLayerMonitorCallback__wrap_generate_parameters(layer)
        callback._ParametricLayerMonitorCallback__wrap_affine_callback(layer)
        inputs = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])

        try:
            generated = layer._generate_parameters(
                input=inputs,
                skip_mask=None,
            )
            self.assertIsNotNone(observation.generated_weights)
            torch.testing.assert_close(
                observation.generated_weights,
                generated[0],
            )

            keyword_output = layer._compute_affine_transformation_callback(
                weights=generated[0],
                bias=generated[1],
                input=inputs,
            )
            torch.testing.assert_close(observation.affine_input, inputs)
            torch.testing.assert_close(
                observation.affine_output,
                keyword_output,
            )

            observation.affine_input = None
            observation.affine_output = None
            mixed_output = layer._compute_affine_transformation_callback(
                generated[0],
                generated[1],
                input=inputs,
            )
            torch.testing.assert_close(observation.affine_input, inputs)
            torch.testing.assert_close(observation.affine_output, mixed_output)
        finally:
            callback._ParametricLayerMonitorCallback__cleanup()

    def test_partial_affine_observation_emits_no_metrics(self) -> None:
        callback = ParametricLayerMonitorCallback(log_every_n_steps=1)
        module = _RecordingLightningModule()
        context = _ParametricTrackingContext(
            pl_module=module,
            module_name="partial",
            parametric_layer=torch.nn.Module(),
            observation=_ParametricObservation(
                affine_input=torch.ones(2, 2),
                affine_output=None,
            ),
            weight_utilization=None,
            bias_utilization=None,
            experiment=None,
            global_step=0,
        )

        callback._ParametricLayerMonitorCallback__track_affine_relative_output_norm(
            context
        )
        callback._ParametricLayerMonitorCallback__track_affine_delta_norm(context)
        callback._ParametricLayerMonitorCallback__track_affine_relative_delta_norm(
            context
        )

        self.assertEqual(module.logged_names, [])

    def test_missing_sampling_methods_install_no_wrappers(self) -> None:
        callback = ParametricLayerMonitorCallback(log_every_n_steps=1)

        callback._ParametricLayerMonitorCallback__wrap_sampling_methods(
            torch.nn.Module()
        )

        self.assertEqual(callback._wrapped_methods, [])

    def test_missing_bias_mixture_reports_zero_experts(self) -> None:
        layer = ParametricLayer(_parametric_config(bias_config=None))

        num_experts = (
            ParametricLayerMonitorCallback._ParametricLayerMonitorCallback__num_experts
        )

        self.assertEqual(num_experts(layer, "weight"), 2)
        self.assertEqual(num_experts(layer, "bias"), 0)

    def test_router_sample_parser_preserves_only_tensor_fields(self) -> None:
        parse = ParametricLayerMonitorCallback._ParametricLayerMonitorCallback__parse_router_sample

        self.assertIsNone(parse("not a four-item tuple"))
        self.assertIsNone(parse([object(), object(), object(), object()]))
        sample = parse((object(), object(), None, object()))

        self.assertIsNotNone(sample)
        self.assertIsNone(sample.probabilities)
        self.assertIsNone(sample.indices)
        self.assertIsNone(sample.auxiliary_loss)

    def test_generated_parameter_recorder_rejects_non_tuple_sequences(
        self,
    ) -> None:
        observation = _ParametricObservation()

        ParametricLayerMonitorCallback._ParametricLayerMonitorCallback__record_generated_parameters(
            observation,
            [torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(())],
        )

        self.assertEqual(observation, _ParametricObservation())

    def test_empty_observation_emits_nothing_and_keeps_histories_empty(self) -> None:
        callback = ParametricLayerMonitorCallback(log_every_n_steps=1)
        observation = _ParametricObservation()
        context = _ParametricTrackingContext(
            pl_module=LightningModule(),
            module_name="empty",
            parametric_layer=torch.nn.Module(),
            observation=observation,
            weight_utilization=None,
            bias_utilization=None,
            experiment=None,
            global_step=0,
        )

        callback._ParametricLayerMonitorCallback__track_parametric_diagnostics(context)

        self.assertIsNone(
            callback._ParametricLayerMonitorCallback__calculate_utilization(
                context.parametric_layer,
                observation,
                "weight",
            )
        )
        self.assertEqual(callback._utilization_histories, {})


if __name__ == "__main__":
    unittest.main()
