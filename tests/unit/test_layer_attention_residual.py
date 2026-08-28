import unittest
from dataclasses import FrozenInstanceError

import torch
import torch.nn.functional as F

from emperor.layers import LayerState
from emperor.layers._composition.residual.base import (
    ResidualState,
    ResidualStateLifecycle,
)
from emperor.layers._composition.residual.config import AttentionResidualConfig
from emperor.layers._composition.residual.variants.attention import (
    AttentionResidual,
    AttentionResidualState,
)


def _paper_attention_residual(
    sources,
    query,
    norm_weight,
    epsilon,
):
    values = torch.stack(sources, dim=0)
    keys = F.rms_norm(
        values,
        normalized_shape=(values.shape[-1],),
        weight=norm_weight,
        eps=epsilon,
    )
    logits = torch.sum(keys * query, dim=-1)
    depth_weights = torch.softmax(logits, dim=0)
    return torch.sum(depth_weights.unsqueeze(-1) * values, dim=0)


def _apply_attention_residual(residual, current, state):
    return residual(
        current,
        current,
        residual_state=state,
    )


class TestAttentionResidual(unittest.TestCase):
    def test_residual_state_requires_only_an_explicit_branch_fork_contract(self):
        class MissingForkResidualState(ResidualState):
            pass

        with self.assertRaisesRegex(TypeError, "abstract method 'fork'"):
            MissingForkResidualState()

        class ForkOnlyResidualState(ResidualState):
            def fork(self):
                return self

        state = ForkOnlyResidualState()
        self.assertIs(state.fork(), state)
        self.assertFalse(hasattr(state, "fork_at_gradient_boundary"))
        self.assertFalse(hasattr(state, "fork_with_detached_generated_sources"))

    def test_attention_state_exposes_no_recurrent_gradient_lifecycle(self):
        state = AttentionResidualState(torch.zeros(1, 2), block_size=1)

        self.assertFalse(hasattr(state, "fork_at_gradient_boundary"))
        self.assertFalse(hasattr(state, "fork_with_detached_generated_sources"))

    def test_full_state_keeps_initial_and_raw_sources_separate(self):
        initial_source = torch.tensor([[1.0, 2.0]])
        first_raw_output = torch.tensor([[3.0, 4.0]])
        second_raw_output = torch.tensor([[5.0, 6.0]])
        state = AttentionResidualState(initial_source, block_size=1)

        state.append(first_raw_output)
        state.append(second_raw_output)

        self.assertEqual(len(state.sources), 3)
        self.assertIs(state.sources[0], initial_source)
        self.assertIs(state.sources[1], first_raw_output)
        self.assertIs(state.sources[2], second_raw_output)

    def test_block_state_sums_raw_outputs_within_each_block(self):
        initial_source = torch.tensor([[10.0, 20.0]])
        raw_outputs = [
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[3.0, 4.0]]),
            torch.tensor([[5.0, 6.0]]),
        ]
        state = AttentionResidualState(initial_source, block_size=2)

        state.append(raw_outputs[0])
        self.assertEqual(len(state.sources), 2)
        self.assertIs(state.sources[0], initial_source)
        self.assertIs(state.sources[1], raw_outputs[0])

        state.append(raw_outputs[1])
        self.assertEqual(len(state.sources), 2)
        torch.testing.assert_close(
            state.sources[1],
            raw_outputs[0] + raw_outputs[1],
        )

        state.append(raw_outputs[2])
        self.assertEqual(len(state.sources), 3)
        torch.testing.assert_close(
            state.sources[1],
            raw_outputs[0] + raw_outputs[1],
        )
        self.assertIs(state.sources[2], raw_outputs[2])

    def test_forked_partial_block_histories_evolve_independently(self):
        initial_source = torch.tensor([[10.0, 20.0]])
        common_partial = torch.tensor([[1.0, 2.0]])
        source_output = torch.tensor([[3.0, 4.0]])
        target_output = torch.tensor([[5.0, 6.0]])
        common_state = AttentionResidualState(initial_source, block_size=2)
        common_state.append(common_partial)

        source_state = common_state.fork()
        target_state = common_state.fork()
        source_state.append(source_output)
        target_state.append(target_output)

        self.assertIs(common_state.sources[1], common_partial)
        torch.testing.assert_close(
            source_state.sources[1],
            common_partial + source_output,
        )
        torch.testing.assert_close(
            target_state.sources[1],
            common_partial + target_output,
        )

    def test_forked_completed_block_histories_evolve_independently(self):
        initial_source = torch.tensor([[10.0, 20.0]])
        common_output = torch.tensor([[1.0, 2.0]])
        source_output = torch.tensor([[3.0, 4.0]])
        target_output = torch.tensor([[5.0, 6.0]])
        common_state = AttentionResidualState(initial_source, block_size=1)
        common_state.append(common_output)

        source_state = common_state.fork()
        target_state = common_state.fork()
        source_state.append(source_output)
        target_state.append(target_output)

        expected_histories = (
            (common_state.sources, (initial_source, common_output)),
            (
                source_state.sources,
                (initial_source, common_output, source_output),
            ),
            (
                target_state.sources,
                (initial_source, common_output, target_output),
            ),
        )
        for actual_sources, expected_sources in expected_histories:
            self.assertEqual(len(actual_sources), len(expected_sources))
            for actual, expected in zip(
                actual_sources,
                expected_sources,
                strict=True,
            ):
                self.assertIs(actual, expected)

    def test_state_rejects_invalid_block_sizes(self):
        for block_size in (0, -1, True, 1.5):
            with self.subTest(block_size=block_size):
                with self.assertRaisesRegex(
                    ValueError,
                    "block_size must be a positive integer",
                ):
                    AttentionResidualState(
                        torch.ones(1, 2),
                        block_size=block_size,
                    )

    def test_attention_owns_an_immutable_forward_state_lifecycle(self):
        residual = AttentionResidual(
            AttentionResidualConfig(
                residual_dim=2,
                block_size=3,
            )
        )

        lifecycle = residual.residual_state_lifecycle

        self.assertIsInstance(lifecycle, ResidualStateLifecycle)
        self.assertEqual(lifecycle.residual_dim, 2)
        self.assertEqual(lifecycle.block_size, 3)
        with self.assertRaises(FrozenInstanceError):
            lifecycle.block_size = 1

    def test_compatibility_state_creation_delegates_to_the_lifecycle(self):
        residual = AttentionResidual(
            AttentionResidualConfig(residual_dim=2, block_size=2)
        )
        initial_source = torch.ones(1, 2)

        lifecycle_state = residual.residual_state_lifecycle.create_state(initial_source)
        compatibility_state = residual.new_state(initial_source)

        self.assertIsInstance(lifecycle_state, AttentionResidualState)
        self.assertIsInstance(compatibility_state, AttentionResidualState)
        self.assertIs(lifecycle_state.initial_source, initial_source)
        self.assertIs(compatibility_state.initial_source, initial_source)
        self.assertEqual(lifecycle_state.block_size, 2)
        self.assertEqual(compatibility_state.block_size, 2)

    def test_state_aware_application_lazily_initializes_attention_history(self):
        residual = AttentionResidual(
            AttentionResidualConfig(residual_dim=2, block_size=1)
        )
        previous = torch.tensor([[2.0, 6.0]])
        current = torch.tensor([[4.0, 10.0]])
        layer_state = LayerState(hidden=current)

        result = residual.apply_to_layer_state(layer_state, previous)

        self.assertIs(result, layer_state)
        self.assertIsInstance(result.residual_state, AttentionResidualState)
        self.assertIs(result.residual_state.initial_source, previous)
        self.assertIs(result.residual_state.sources[1], current)
        torch.testing.assert_close(result.hidden, (previous + current) / 2.0)

    def test_state_aware_application_reuses_existing_attention_history(self):
        residual = AttentionResidual(
            AttentionResidualConfig(residual_dim=2, block_size=1)
        )
        initial_source = torch.tensor([[1.0, 3.0]])
        existing_state = residual.new_state(initial_source)
        layer_state = LayerState(
            hidden=torch.tensor([[5.0, 7.0]]),
            residual_state=existing_state,
        )

        residual.apply_to_layer_state(layer_state, initial_source)

        self.assertIs(layer_state.residual_state, existing_state)
        self.assertEqual(len(existing_state.sources), 2)

    def test_state_aware_application_rejects_incompatible_residual_state(self):
        residual = AttentionResidual(AttentionResidualConfig(residual_dim=2))
        layer_state = LayerState(
            hidden=torch.ones(1, 2),
            residual_state=object(),
        )

        with self.assertRaisesRegex(
            TypeError,
            "residual_state must be an AttentionResidualState",
        ):
            residual.apply_to_layer_state(layer_state, torch.ones(1, 2))

    def test_state_aware_application_keeps_history_off_the_module(self):
        residual = AttentionResidual(AttentionResidualConfig(residual_dim=2))
        layer_state = LayerState(hidden=torch.ones(1, 2))

        residual.apply_to_layer_state(layer_state, torch.ones(1, 2))

        self.assertFalse(
            any(
                isinstance(attribute, AttentionResidualState)
                for attribute in vars(residual).values()
            )
        )

    def test_zero_initialized_mixer_averages_all_depth_sources(self):
        residual = AttentionResidual(
            AttentionResidualConfig(
                residual_dim=2,
                block_size=1,
                rms_norm_epsilon=1e-6,
            )
        )
        initial_source = torch.tensor([[2.0, 4.0]])
        first_raw_output = torch.tensor([[4.0, 8.0]])
        second_raw_output = torch.tensor([[9.0, 3.0]])
        state = residual.new_state(initial_source)

        first_hidden = _apply_attention_residual(residual, first_raw_output, state)
        second_hidden = _apply_attention_residual(residual, second_raw_output, state)

        torch.testing.assert_close(
            first_hidden,
            torch.stack((initial_source, first_raw_output)).mean(dim=0),
        )
        torch.testing.assert_close(
            second_hidden,
            torch.stack((initial_source, first_raw_output, second_raw_output)).mean(
                dim=0
            ),
        )

    def test_mixer_matches_paper_depth_softmax_with_raw_values(self):
        config = AttentionResidualConfig(
            residual_dim=3,
            block_size=1,
            rms_norm_epsilon=1e-5,
        )
        residual = AttentionResidual(config)
        initial_source = torch.tensor([[[1.0, 2.0, 8.0], [3.0, -4.0, 1.0]]])
        first_raw_output = torch.tensor([[[5.0, 1.0, -2.0], [-2.0, 6.0, 3.0]]])
        second_raw_output = torch.tensor([[[2.0, -7.0, 4.0], [8.0, 1.0, -5.0]]])
        query = torch.tensor([0.7, -0.4, 1.2])
        norm_weight = torch.tensor([1.1, 0.6, 1.7])
        with torch.no_grad():
            residual.query.copy_(query)
            residual.key_norm.weight.copy_(norm_weight)
        state = residual.new_state(initial_source)

        _apply_attention_residual(residual, first_raw_output, state)
        actual = _apply_attention_residual(residual, second_raw_output, state)

        expected = _paper_attention_residual(
            (initial_source, first_raw_output, second_raw_output),
            query,
            norm_weight,
            config.rms_norm_epsilon,
        )
        torch.testing.assert_close(actual, expected)

    def test_mixer_rejects_invalid_residual_dimensions(self):
        for residual_dim in (None, 0, -1, True, 2.5):
            with self.subTest(residual_dim=residual_dim):
                with self.assertRaisesRegex(
                    ValueError,
                    "residual_dim must be a positive integer",
                ):
                    AttentionResidual(
                        AttentionResidualConfig(
                            residual_dim=residual_dim,
                            block_size=1,
                            rms_norm_epsilon=1e-6,
                        )
                    )

    def test_mixer_rejects_invalid_configured_block_sizes(self):
        for block_size in (0, -1, True, 1.5):
            with self.subTest(block_size=block_size):
                with self.assertRaisesRegex(
                    ValueError,
                    "block_size must be a positive integer",
                ):
                    AttentionResidual(
                        AttentionResidualConfig(
                            residual_dim=2,
                            block_size=block_size,
                            rms_norm_epsilon=1e-6,
                        )
                    )

    def test_mixer_rejects_non_positive_or_non_finite_epsilon(self):
        invalid_epsilons = (
            0.0,
            -1e-6,
            float("inf"),
            float("-inf"),
            float("nan"),
            True,
            "1e-6",
        )
        for epsilon in invalid_epsilons:
            with self.subTest(epsilon=epsilon):
                with self.assertRaisesRegex(
                    ValueError,
                    "rms_norm_epsilon must be a finite positive number",
                ):
                    AttentionResidual(
                        AttentionResidualConfig(
                            residual_dim=2,
                            block_size=1,
                            rms_norm_epsilon=epsilon,
                        )
                    )

    def test_mixer_rejects_non_floating_initial_sources(self):
        for initial_source in (torch.tensor([[1, 2]]), object()):
            with self.subTest(initial_source=initial_source):
                residual = AttentionResidual(AttentionResidualConfig(residual_dim=2))
                with self.assertRaisesRegex(
                    TypeError,
                    "attention residual sources must be floating-point tensors",
                ):
                    residual.new_state(initial_source)

    def test_mixer_rejects_wrong_initial_feature_dimension(self):
        for initial_source in (torch.ones(2, 3), torch.tensor(1.0)):
            with self.subTest(initial_source=initial_source):
                residual = AttentionResidual(AttentionResidualConfig(residual_dim=2))
                with self.assertRaisesRegex(
                    ValueError,
                    "source last dimension must equal residual_dim 2",
                ):
                    residual.new_state(initial_source)

    def test_mixer_requires_its_forward_local_state_type(self):
        residual = AttentionResidual(AttentionResidualConfig(residual_dim=2))

        with self.assertRaisesRegex(
            TypeError,
            "residual_state must be an AttentionResidualState",
        ):
            _apply_attention_residual(residual, torch.ones(1, 2), object())

    def test_mixer_rejects_state_from_a_different_block_variant(self):
        residual = AttentionResidual(
            AttentionResidualConfig(residual_dim=2, block_size=2)
        )
        mismatched_state = AttentionResidualState(
            torch.ones(1, 2),
            block_size=1,
        )

        with self.assertRaisesRegex(
            ValueError,
            "residual_state block_size 1 does not match configured block_size 2",
        ):
            _apply_attention_residual(residual, torch.ones(1, 2), mismatched_state)

    def test_mixer_validates_current_before_mutating_history(self):
        residual = AttentionResidual(AttentionResidualConfig(residual_dim=2))
        initial_source = torch.ones(1, 2)
        state = residual.new_state(initial_source)

        with self.assertRaisesRegex(
            TypeError,
            "attention residual sources must be floating-point tensors",
        ):
            _apply_attention_residual(
                residual,
                torch.ones(1, 2, dtype=torch.int64),
                state,
            )

        self.assertEqual(len(state.sources), 1)
        self.assertIs(state.sources[0], initial_source)

    def test_mixer_requires_identical_source_shapes(self):
        residual = AttentionResidual(AttentionResidualConfig(residual_dim=2))
        initial_source = torch.ones(2, 2)
        state = residual.new_state(initial_source)

        with self.assertRaisesRegex(
            ValueError,
            r"all attention residual sources must have shape \(2, 2\)",
        ):
            _apply_attention_residual(residual, torch.ones(1, 2), state)

        self.assertEqual(len(state.sources), 1)
        self.assertIs(state.sources[0], initial_source)

    def test_mixer_promotes_mixed_floating_source_dtypes(self):
        residual = AttentionResidual(AttentionResidualConfig(residual_dim=2))
        initial_source = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
        current = torch.tensor([[5.0, 7.0]], dtype=torch.bfloat16)
        state = residual.new_state(initial_source)

        actual = _apply_attention_residual(residual, current, state)

        expected = (initial_source + current.float()) / 2.0
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(state.sources), 2)
        self.assertIs(state.sources[0], initial_source)

    def test_mixer_requires_sources_on_one_device(self):
        residual = AttentionResidual(AttentionResidualConfig(residual_dim=2))
        state = residual.new_state(torch.ones(1, 2))

        with self.assertRaisesRegex(
            ValueError,
            "all attention residual sources must be on device cpu",
        ):
            _apply_attention_residual(
                residual,
                torch.ones(1, 2, device="meta"),
                state,
            )

    def test_mixer_preserves_gradients_to_every_source_and_parameter(self):
        residual = AttentionResidual(AttentionResidualConfig(residual_dim=3))
        with torch.no_grad():
            residual.query.copy_(torch.tensor([0.7, -0.2, 0.4]))
            residual.key_norm.weight.copy_(torch.tensor([1.0, 1.3, 0.8]))
        initial_source = torch.tensor(
            [[1.0, 2.0, -1.0]],
            requires_grad=True,
        )
        first_raw_output = torch.tensor(
            [[3.0, -2.0, 4.0]],
            requires_grad=True,
        )
        second_raw_output = torch.tensor(
            [[-4.0, 1.0, 2.0]],
            requires_grad=True,
        )
        state = residual.new_state(initial_source)

        _apply_attention_residual(residual, first_raw_output, state)
        output = _apply_attention_residual(residual, second_raw_output, state)
        output.square().sum().backward()

        for tensor in (
            initial_source,
            first_raw_output,
            second_raw_output,
            residual.query,
            residual.key_norm.weight,
        ):
            with self.subTest(tensor=tensor):
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(torch.isfinite(tensor.grad).all())
                self.assertGreater(torch.count_nonzero(tensor.grad), 0)

    def test_mixer_accumulates_low_precision_sources_in_float32(self):
        for source_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(source_dtype=source_dtype):
                epsilon = 1e-6
                residual = AttentionResidual(
                    AttentionResidualConfig(
                        residual_dim=3,
                        rms_norm_epsilon=epsilon,
                    )
                )
                query = torch.tensor([0.3, -0.8, 1.1])
                norm_weight = torch.tensor([1.2, 0.9, 1.4])
                with torch.no_grad():
                    residual.query.copy_(query)
                    residual.key_norm.weight.copy_(norm_weight)
                initial_source = torch.tensor(
                    [[1000.0, 0.125, -32.0]],
                    dtype=source_dtype,
                )
                current = torch.tensor(
                    [[-500.0, 8.0, 64.0]],
                    dtype=source_dtype,
                )
                state = residual.new_state(initial_source)

                actual = _apply_attention_residual(residual, current, state)

                expected = _paper_attention_residual(
                    (initial_source.float(), current.float()),
                    query,
                    norm_weight,
                    epsilon,
                ).to(source_dtype)
                self.assertEqual(actual.dtype, source_dtype)
                torch.testing.assert_close(actual, expected)

    def test_mixer_preserves_float64_computation(self):
        epsilon = 1e-9
        residual = AttentionResidual(
            AttentionResidualConfig(
                residual_dim=2,
                rms_norm_epsilon=epsilon,
            )
        )
        query = torch.tensor([0.25, -0.75], dtype=torch.float64)
        norm_weight = torch.tensor([1.4, 0.6], dtype=torch.float64)
        with torch.no_grad():
            residual.query.copy_(query)
            residual.key_norm.weight.copy_(norm_weight)
        initial_source = torch.tensor([[1.125, -3.75]], dtype=torch.float64)
        current = torch.tensor([[6.25, 2.5]], dtype=torch.float64)
        state = residual.new_state(initial_source)

        actual = _apply_attention_residual(residual, current, state)

        expected = _paper_attention_residual(
            (initial_source, current),
            query,
            norm_weight,
            epsilon,
        )
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual, expected)

    def test_state_is_ephemeral_and_parameters_use_paper_initializers(self):
        residual = AttentionResidual(AttentionResidualConfig(residual_dim=3))
        state = residual.new_state(torch.ones(2, 3))

        _apply_attention_residual(residual, torch.full((2, 3), 2.0), state)

        self.assertTupleEqual(
            tuple(residual.state_dict()),
            ("query", "key_norm.weight"),
        )
        torch.testing.assert_close(residual.query, torch.zeros(3))
        torch.testing.assert_close(residual.key_norm.weight, torch.ones(3))

    def test_block_mixer_routes_over_completed_and_partial_block_sums(self):
        epsilon = 1e-6
        residual = AttentionResidual(
            AttentionResidualConfig(
                residual_dim=2,
                block_size=2,
                rms_norm_epsilon=epsilon,
            )
        )
        query = torch.tensor([0.5, -0.25])
        norm_weight = torch.tensor([1.1, 0.7])
        with torch.no_grad():
            residual.query.copy_(query)
            residual.key_norm.weight.copy_(norm_weight)
        initial_source = torch.tensor([[2.0, -1.0]])
        raw_outputs = (
            torch.tensor([[1.0, 3.0]]),
            torch.tensor([[4.0, -2.0]]),
            torch.tensor([[-3.0, 5.0]]),
        )
        state = residual.new_state(initial_source)

        _apply_attention_residual(residual, raw_outputs[0], state)
        _apply_attention_residual(residual, raw_outputs[1], state)
        actual = _apply_attention_residual(residual, raw_outputs[2], state)

        expected = _paper_attention_residual(
            (
                initial_source,
                raw_outputs[0] + raw_outputs[1],
                raw_outputs[2],
            ),
            query,
            norm_weight,
            epsilon,
        )
        torch.testing.assert_close(actual, expected)

    def test_block_sum_preserves_each_raw_output_gradient(self):
        residual = AttentionResidual(
            AttentionResidualConfig(residual_dim=2, block_size=2)
        )
        initial_source = torch.tensor([[1.0, 2.0]], requires_grad=True)
        first_raw_output = torch.tensor([[3.0, 4.0]], requires_grad=True)
        second_raw_output = torch.tensor([[5.0, 6.0]], requires_grad=True)
        state = residual.new_state(initial_source)

        _apply_attention_residual(residual, first_raw_output, state)
        output = _apply_attention_residual(residual, second_raw_output, state)
        output.sum().backward()

        expected_gradient = torch.full((1, 2), 0.5)
        torch.testing.assert_close(initial_source.grad, expected_gradient)
        torch.testing.assert_close(first_raw_output.grad, expected_gradient)
        torch.testing.assert_close(second_raw_output.grad, expected_gradient)


if __name__ == "__main__":
    unittest.main()
